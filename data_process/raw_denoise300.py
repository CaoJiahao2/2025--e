# -*- coding: utf-8 -*-
"""
批量：包络谱分析 + 去噪 + 可视化 + 报表 + 镜像保存
- 输入根目录：默认 /home/user1/data/learn/sumo/data/raw_resampled_12kHz/
- 输出镜像：默认 <in_root> 同级目录生成 env_denoised_12kHz（保持子目录结构）
- 图保存：reports/envelope_figs
- 报表：reports/envelope_report_12kHz_*.csv（UTF-8 with BOM）
- 采样率：固定 12 kHz

依赖：
    pip install numpy scipy matplotlib

与原 MATLAB 逻辑一致：
- 自动选带：fc∈[1.5,5.5]kHz，带宽≈1kHz（±0.5kHz），选择带通后**峭度最大**的频带
- 去噪：Butterworth 4阶 bandpass + filtfilt（sosfiltfilt），Hilbert包络取模，移动中位数平滑
- 包络谱：Welch（Hamming，nfft≤16384）
- 频谱峰：0~1000 Hz 内取前三峰；并评估 fr/BPFO/BPFI/BSF 处幅值与 SBR
- 额外：输出完整**时域/频域/时频域**特征
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 优先名单（你机器上有哪个就用哪个）
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Source Han Sans SC", "Source Han Sans CN",  # 思源黑体
    "SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei"
]

# 已安装字体名集合
installed = {f.name for f in font_manager.fontManager.ttflist}

# 尝试自动匹配
for name in CJK_CANDIDATES:
    if name in installed:
        plt.rcParams["font.family"] = name
        break
else:
    # 如果上面都没有，但你知道某个字体文件路径，可在此处手动加入：
    # 例：font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    # plt.rcParams["font.family"] = "Noto Sans CJK SC"
    pass

# 解决坐标轴负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False

from scipy.io import loadmat, savemat
from scipy.signal import butter, sosfiltfilt, hilbert, welch, medfilt, find_peaks, stft, get_window
from scipy.stats import kurtosis as stat_kurtosis, skew as stat_skew


# ───────────────────── 参数 ─────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量包络去噪 + 可视化 + 报表（12 kHz）。")
    parser.add_argument("--in-root", type=Path,
                        default=Path("/home/user1/data/learn/sumo/data/raw_resampled_12kHz/"),
                        help="输入根目录（已统一到12kHz）")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="输出镜像根目录（默认与 in-root 同级：env_denoised_12kHz）")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"),
                        help="报表输出根目录（会自动创建）")
    parser.add_argument("--fig-subdir", type=str, default="envelope_figs",
                        help="报表目录下保存图片的子目录名")
    parser.add_argument("--fs", type=float, default=12_000.0, help="采样率 Hz（默认 12000）")
    parser.add_argument("--guard", type=float, default=60.0, help="与 0/Nyquist 的保护带 Hz")
    parser.add_argument("--fc-start-khz", type=float, default=1.5, help="候选中心频率起点（kHz）")
    parser.add_argument("--fc-stop-khz", type=float, default=5.5, help="候选中心频率终点（kHz）")
    parser.add_argument("--fc-step-khz", type=float, default=0.5, help="候选中心频率步长（kHz）")
    parser.add_argument("--bw-half", type=float, default=500.0, help="半带宽 Hz（默认 ±500Hz）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    return parser.parse_args()


# ───────────────────── 工具 ─────────────────────

def next_pow2(n: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, n))))

def write_csv_utf8_bom(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def parse_fs_from_path(_folder: str) -> float:
    # 这里固定 12k（输入已统一），留函数占位以便未来扩展
    return 12_000.0

def parse_rpm_from_filename(fname: str) -> float:
    m = re.search(r"(\d{3,5})\s*rpm", fname, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")

def rpm_to_hz(rpm: float) -> float:
    return float(rpm) / 60.0 if np.isfinite(rpm) else float("nan")

def sensor_from_name(vname: str) -> str:
    if vname.endswith("_DE_time"): return "DE"
    if vname.endswith("_FE_time"): return "FE"
    if vname.endswith("_BA_time"): return "BA"
    return "DE"

def sensor_if_missing_use_DE(sensor: str) -> str:
    return sensor if sensor in ("DE", "FE", "BA") else "DE"

@dataclass
class BearingGeom:
    Nd: int
    d: float
    D: float
    thetaDeg: float = 0.0

def bearing_freqs(fr: float, g: BearingGeom) -> Tuple[float, float, float]:
    if not np.isfinite(fr): return (float("nan"),)*3
    Nd, d, D, c = g.Nd, g.d, g.D, math.cos(math.radians(g.thetaDeg))
    bpfo = fr * (Nd/2.0) * (1.0 - (d/D)*c)
    bpfi = fr * (Nd/2.0) * (1.0 + (d/D)*c)
    bsf  = fr * (D/(2.0*d)) * (1.0 - ((d/D)*c)**2)
    return bpfo, bpfi, bsf

def ensure_out_path(in_root: Path, out_root: Path, full_folder: Path) -> Path:
    rel_folder = full_folder.relative_to(in_root)
    out_folder = out_root / rel_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder

def nz(arr: np.ndarray | List[float] | Tuple[float, ...], i: int) -> float:
    try:
        return float(arr[i-1])
    except Exception:
        return float("nan")


# ───────────────── 选带 + 去噪链路 ─────────────────

def auto_select_band(X: np.ndarray, fs: float,
                     fc_list_hz: np.ndarray, bw_half: float, guard: float) -> Tuple[float, float, float]:
    """在候选频带中，选择‘带通后信号峭度’最大的频带（以第1列代表）"""
    x1 = X[:, 0] if X.ndim == 2 else X.reshape(-1)
    kmax, flo_best, fhi_best = -np.inf, 800.0, 1800.0
    nyq = fs / 2.0
    for fc in fc_list_hz:
        flo = max(guard, fc - bw_half)
        fhi = min(nyq - guard, fc + bw_half)
        if fhi <= flo + 10:
            continue
        try:
            sos = butter(4, [flo/(nyq), fhi/(nyq)], btype="band", output="sos")
            xf = sosfiltfilt(sos, x1.astype(float))
            k  = stat_kurtosis(xf, fisher=False, bias=False)  # 与 MATLAB 默认一致（Fisher=False）
            if np.isfinite(k) and k > kmax:
                kmax, flo_best, fhi_best = float(k), float(flo), float(fhi)
        except Exception:
            continue
    return flo_best, fhi_best, kmax

def process_one(x: np.ndarray, fs: float, flo: float, fhi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """带通→包络→平滑→包络谱。返回：x_bandpassed, env_smoothed, f_env, P_env"""
    x = np.asarray(x, dtype=float).reshape(-1)
    nyq = fs/2.0
    sos = butter(4, [flo/nyq, fhi/nyq], btype="band", output="sos")
    xb  = sosfiltfilt(sos, x)

    env = np.abs(hilbert(xb))
    # 平滑窗口：约 5ms，且取奇数，>=3
    win = max(3, int(round(0.005*fs)))
    if win % 2 == 0:
        win += 1
    env_sm = medfilt(env, kernel_size=win)

    # 包络谱（Welch）
    nfft = 1 << int(np.floor(np.log2(min(len(env_sm), 16384))))
    nfft = max(nfft, 512)
    nperseg = max(256, nfft//4)
    noverlap = nperseg//2
    f_env, P_env = welch(env_sm - np.mean(env_sm),
                         fs=fs, window="hamming",
                         nperseg=nperseg, noverlap=noverlap,
                         nfft=nfft, detrend="constant", return_onesided=True, scaling="density")
    return xb, env_sm, f_env, P_env

def top_env_peaks(f: np.ndarray, P: np.ndarray, N: int = 3) -> Tuple[List[float], List[float]]:
    """0~1000 Hz 包络谱前三峰"""
    idx = (f >= 0) & (f <= 1000.0)
    if not np.any(idx):
        return [np.nan]*N, [np.nan]*N
    ff, pp = f[idx], P[idx]
    df = np.median(np.diff(ff))
    distance_bins = max(1, int(round(2.0/df)))  # 约 2 Hz 的最小峰距
    prom = np.median(pp)*3.0
    try:
        peaks, _ = find_peaks(pp, prominence=prom, distance=distance_bins)
    except Exception:
        # 回退：纯排序
        order = np.argsort(pp)[::-1][:N]
        freqs = ff[order].tolist()
        amps  = pp[order].tolist()
        return freqs + [np.nan]*(N-len(freqs)), amps + [np.nan]*(N-len(amps))

    if peaks.size == 0:
        return [np.nan]*N, [np.nan]*N
    pks_sorted = peaks[np.argsort(pp[peaks])[::-1][:N]]
    freqs = ff[pks_sorted].tolist()
    amps  = pp[pks_sorted].tolist()
    return freqs + [np.nan]*(N-len(freqs)), amps + [np.nan]*(N-len(amps))

def amp_sbr_at(f: np.ndarray, P: np.ndarray, f0: float) -> Tuple[float, float]:
    """在 f0±bw 内的峰幅，SBR=峰/局部背景（dB）"""
    if not (np.isfinite(f0) and f0 > 0):
        return (float("nan"), float("nan"))
    df = np.median(np.diff(f))
    if not np.isfinite(df) or df <= 0:
        df = 0.5
    bw = max(3*df, 5.0)  # ±5 Hz
    idx = (f >= max(0.0, f0-bw)) & (f <= f0+bw)
    if not np.any(idx):
        return (float("nan"), float("nan"))
    ii = np.argmax(P[idx])
    A = float(P[idx][ii])
    # 背景：更宽邻域去掉峰核心
    idx_bg = (f >= max(0.0, f0-5*bw)) & (f <= f0+5*bw)
    bg = np.median(P[idx_bg]) if np.any(idx_bg) else np.median(P)
    bg = max(float(bg), np.finfo(float).eps)
    SBRdB = 20.0 * math.log10(A / bg)
    return A, SBRdB


# ─────────────────── 特征集 ───────────────────

def time_features(x: np.ndarray) -> Dict[str, float]:
    """时域特征（对原始 x0）"""
    x = np.asarray(x, dtype=float).reshape(-1)
    N = len(x)
    xm = np.mean(x)
    xd = np.std(x, ddof=0)
    xrms = math.sqrt(np.mean(x*x))
    xmax, xmin = float(np.max(x)), float(np.min(x))
    xpp = xmax - xmin
    xabs = np.abs(x)
    mean_abs = float(np.mean(xabs))
    mean_sqrt_abs = float(np.mean(np.sqrt(xabs))) if np.any(xabs>0) else np.nan
    skew = float(stat_skew(x, bias=False)) if N >= 3 else np.nan
    kurt = float(stat_kurtosis(x, fisher=False, bias=False)) if N >= 4 else np.nan
    crest = (xmax / xrms) if xrms > 0 else np.nan
    impulse = (xmax / mean_abs) if mean_abs > 0 else np.nan
    shape = (xrms / mean_abs) if mean_abs > 0 else np.nan
    clearance = (xmax / (mean_sqrt_abs**2)) if (mean_sqrt_abs and not np.isnan(mean_sqrt_abs)) else np.nan
    return {
        "时域_均值": xm, "时域_STD": xd, "时域_RMS": xrms, "时域_峰峰值": xpp,
        "时域_最大": xmax, "时域_最小": xmin,
        "时域_偏斜度": skew, "时域_峭度": kurt,
        "时域_波形指标": shape, "时域_峰值指标": crest,
        "时域_脉冲指标": impulse, "时域_裕度指标": clearance
    }

def freq_features_env(f: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    """频域特征（基于 0~1000 Hz 的包络谱）"""
    idx = (f >= 0) & (f <= 1000.0)
    if not np.any(idx):
        return {k: np.nan for k in [
            "频域_谱质心Hz","频域_谱带宽Hz","频域_谱熵","频域_主峰Hz","频域_主峰幅","频域_中值频Hz","频域_95滚降Hz"
        ]}
    ff, pp = f[idx], P[idx]
    ps = pp / (np.sum(pp) + np.finfo(float).eps)
    centroid = float(np.sum(ff * ps))
    spread = float(np.sqrt(np.sum(((ff-centroid)**2) * ps)))
    sent = float(-np.sum(ps * np.log(ps + 1e-16)))
    pk = int(np.argmax(pp))
    fpk = float(ff[pk]); Apk = float(pp[pk])
    # 中值频 & 95% roll-off
    cdf = np.cumsum(ps)
    fmed = float(np.interp(0.5, cdf, ff))
    f95  = float(np.interp(0.95, cdf, ff))
    return {
        "频域_谱质心Hz": centroid,
        "频域_谱带宽Hz": spread,
        "频域_谱熵": sent,
        "频域_主峰Hz": fpk,
        "频域_主峰幅": Apk,
        "频域_中值频Hz": fmed,
        "频域_95滚降Hz": f95
    }

def tf_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """
    时频域特征（STFT）：
      - 频点谱峭度最大值/对应频率
      - 帧谱熵均值/STD
      - 三分段能量比的时间CV（低/中/高频段能量占比随时间的变异）
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    # 窗长自适应（与 MATLAB 近似）
    wlen = 2048
    if wlen > len(x):
        wlen = 2 ** int(np.floor(np.log2(max(len(x), 256))))
    wlen = max(wlen, 256)
    olap = int(0.5 * wlen)
    nfft = max(4096, 1 << int(np.ceil(np.log2(wlen))))
    f, t, Z = stft(x, fs=fs, window=get_window("hamming", wlen),
                   nperseg=wlen, noverlap=olap, nfft=nfft, boundary=None, padded=False)
    P = np.abs(Z)**2  # [freq, time]

    # 1) 频点谱峭度（对时间方向）
    if P.shape[1] >= 4:
        sk = stat_kurtosis(P, axis=1, fisher=False, bias=False)
        imax = int(np.nanargmax(sk))
        sk_max = float(sk[imax])
        sk_f = float(f[imax])
    else:
        sk_max, sk_f = (np.nan, np.nan)

    # 2) 帧谱熵（对每帧）
    Pn = P / (np.sum(P, axis=0, keepdims=True) + 1e-16)
    frame_entropy = -np.sum(Pn * np.log(Pn + 1e-16), axis=0)
    fe_mean = float(np.mean(frame_entropy)) if frame_entropy.size else np.nan
    fe_std  = float(np.std(frame_entropy))  if frame_entropy.size else np.nan

    # 3) 三分段能量比的时间CV（0-200 / 200-400 / 400-1000 Hz，超出范围自动裁剪）
    bands = [(0, 200), (200, 400), (400, 1000)]
    ratios = []
    for lo, hi in bands:
        idx = (f >= lo) & (f < hi)
        e  = np.sum(P[idx, :], axis=0)
        Et = np.sum(P, axis=0) + 1e-16
        ratios.append(e / Et)
    ratios = np.vstack(ratios)  # [3, T]
    cv = np.std(ratios, axis=1) / (np.mean(ratios, axis=1) + 1e-16)
    return {
        "时频_谱峭度最大值": sk_max,
        "时频_谱峭度最大值对应频率Hz": sk_f,
        "时频_帧谱熵均值": fe_mean,
        "时频_帧谱熵STD": fe_std,
        "时频_三段能量比CV_0_200": float(cv[0]),
        "时频_三段能量比CV_200_400": float(cv[1]),
        "时频_三段能量比CV_400_1000": float(cv[2]),
    }


# ─────────────────── 绘图 ───────────────────

def plot_envelope_figure(x0: np.ndarray, xDen: np.ndarray, env: np.ndarray, fs: float,
                         fEnv: np.ndarray, Penv: np.ndarray, flo: float, fhi: float,
                         fr: float, bpfo: float, bpfi: float, bsf: float, figPath: Path) -> None:
    Tshow = min(1.0, len(x0)/fs)
    nshow = max(1, int(round(Tshow*fs)))
    t = np.arange(nshow) / fs

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2,2,1)
    ax.plot(t, x0[:nshow]); ax.grid(True)
    ax.set_title("原始信号（片段）"); ax.set_xlabel("t (s)"); ax.set_ylabel("x")

    ax = fig.add_subplot(2,2,2)
    ax.plot(t, xDen[:nshow]); ax.grid(True)
    ax.set_title(f"带通去噪（{flo:.0f}–{fhi:.0f} Hz）"); ax.set_xlabel("t (s)"); ax.set_ylabel("x_bp")

    ax = fig.add_subplot(2,2,3)
    ax.plot(t, env[:nshow]); ax.grid(True)
    ax.set_title("包络（平滑后）"); ax.set_xlabel("t (s)"); ax.set_ylabel("|Hilbert|")

    ax = fig.add_subplot(2,2,4)
    idx = (fEnv >= 0) & (fEnv <= 300.0)
    ax.plot(fEnv[idx], Penv[idx])
    ax.set_xlim(0, 300) 
    ax.grid(True)
    ax.set_title("包络谱（0–1000 Hz）"); ax.set_xlabel("f (Hz)"); ax.set_ylabel("P_env")
    for name, f0 in [("fr", fr), ("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf)]:
        if np.isfinite(f0) and 0 < f0 < 1000:
            ax.axvline(f0, linestyle="--", color="k")
            ax.text(f0, ax.get_ylim()[1]*0.95, name, rotation=90, va="top", ha="right", fontsize=9)

    fig.tight_layout()
    figPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figPath, dpi=150)
    plt.close(fig)


# ─────────────────── 主流程 ───────────────────

def main():
    args = parse_args()
    fs = float(args.fs)
    nyq = fs/2.0

    in_root: Path = args.in_root
    if not in_root.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{in_root}")

    out_root = args.out_root or (in_root.parent / "env_denoised_12kHz")
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    report_dir: Path = args.report_dir; report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir: Path = report_dir / args.fig_subdir; fig_dir.mkdir(parents=True, exist_ok=True)

    # 轴承几何参数（与原 MATLAB 保持一致）
    geom = {
        "DE": BearingGeom(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),   # SKF6205
        "FE": BearingGeom(Nd=9, d=0.2656, D=1.122, thetaDeg=0.0),   # SKF6203
        "BA": BearingGeom(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),
    }

    # 报表头
    header = [
        "文件路径","输出路径","变量名","测点","样本数","时长s",
        "选带低Hz","选带高Hz","带内峭度","去噪方法",
        "env峰1_Hz","env峰1_幅","env峰2_Hz","env峰2_幅","env峰3_Hz","env峰3_幅",
        "fr_Hz","BPFO_Hz","BPFI_Hz","BSF_Hz",
        "Amp@fr","SBR@fr_dB","Amp@BPFO","SBR@BPFO_dB","Amp@BPFI","SBR@BPFI_dB","Amp@BSF","SBR@BSF_dB",
        # 时域
        "时域_均值","时域_STD","时域_RMS","时域_峰峰值","时域_最大","时域_最小",
        "时域_偏斜度","时域_峭度","时域_波形指标","时域_峰值指标","时域_脉冲指标","时域_裕度指标",
        # 频域（包络谱）
        "频域_谱质心Hz","频域_谱带宽Hz","频域_谱熵","频域_主峰Hz","频域_主峰幅","频域_中值频Hz","频域_95滚降Hz",
        # 时频域
        "时频_谱峭度最大值","时频_谱峭度最大值对应频率Hz","时频_帧谱熵均值","时频_帧谱熵STD",
        "时频_三段能量比CV_0_200","时频_三段能量比CV_200_400","时频_三段能量比CV_400_1000",
    ]

    # 候选带参数
    fc_list = np.arange(args.fc_start_khz, args.fc_stop_khz + 1e-9, args.fc_step_khz) * 1e3
    bw_half = float(args.bw_half)
    guard = float(args.guard)

    files = list(in_root.rglob("*.mat"))
    if not files:
        warnings.warn(f"没有发现 .mat 文件：{in_root}")
        return
    print(f"发现 {len(files)} 个 .mat 文件，开始处理...")

    rows: List[List[Any]] = []
    nOK = nFail = 0

    for k, f in enumerate(files, 1):
        in_path = f
        try:
            # 输出镜像路径（保持相对结构）
            out_folder = ensure_out_path(in_root, out_root, f.parent)
            out_path = out_folder / f.name

            # 加载 mat
            Sraw = loadmat(str(in_path), squeeze_me=False, struct_as_record=False)
            S = {k2: v for k2, v in Sraw.items() if not k2.startswith("__")}
            names = list(S.keys())
            sig_vars = [n for n in names if n.endswith(("_DE_time","_FE_time","_BA_time")) and isinstance(S[n], np.ndarray)]
            if len(sig_vars) == 0:
                # 也复制以保持镜像
                if not args.dry_run:
                    savemat(str(out_path), S, do_compression=False)
                nOK += 1
                continue

            # RPM/转频
            rpm = parse_rpm_from_filename(f.name)
            if not np.isfinite(rpm):
                rpm_vars = [n for n in names if n.endswith("RPM")]
                if rpm_vars:
                    try:
                        rv = float(np.ravel(S[rpm_vars[0]])[0])
                        rpm = rv
                    except Exception:
                        rpm = float("nan")
            fr = rpm_to_hz(rpm)

            # 每个变量处理
            file_changed = False
            for vname in sig_vars:
                sensor = sensor_from_name(vname)
                x0 = np.asarray(S[vname])
                x0_class = x0.dtype
                X = x0.reshape(-1,1) if x0.ndim == 1 else x0
                nSamp = X.shape[0]; durS = nSamp / fs

                # 自动选带
                flo, fhi, kmax = auto_select_band(X, fs, fc_list, bw_half, guard)
                denoise_method = f"bandpass[{flo:.0f},{fhi:.0f}]+hilbert+movmedian"

                # 去噪链路与包络谱
                if x0.ndim == 1:
                    xDen, env, fEnv, Penv = process_one(X[:,0], fs, flo, fhi)
                    # 保存变量
                    if not args.dry_run:
                        S[f"{vname}_denoised"] = xDen.astype(x0_class).reshape(x0.shape)
                        S[f"{vname}_env"] = env.astype(float).reshape(x0.shape)
                        S[f"{vname}_env_band_Hz"] = np.array([flo, fhi], dtype=float)
                    file_changed = True
                else:
                    Xden = np.zeros_like(X, dtype=float)
                    ENV  = np.zeros_like(X, dtype=float)
                    # 逐列处理；fEnv/Penv 用第一列的
                    for c in range(X.shape[1]):
                        xd, ev, fEnv, Penv = process_one(X[:,c], fs, flo, fhi)
                        Xden[:,c] = xd; ENV[:,c] = ev
                    if not args.dry_run:
                        S[f"{vname}_denoised"] = Xden.astype(x0_class).reshape(x0.shape)
                        S[f"{vname}_env"] = ENV.astype(float).reshape(x0.shape)
                        S[f"{vname}_env_band_Hz"] = np.array([flo, fhi], dtype=float)
                    file_changed = True

                # 包络谱峰&特征频率
                pkF, pkA = top_env_peaks(fEnv, Penv, 3)
                g = geom.get(sensor_if_missing_use_DE(sensor), geom["DE"])
                bpfo, bpfi, bsf = bearing_freqs(fr, g)
                A_fr, S_fr = amp_sbr_at(fEnv, Penv, fr)
                A_bo, S_bo = amp_sbr_at(fEnv, Penv, bpfo)
                A_bi, S_bi = amp_sbr_at(fEnv, Penv, bpfi)
                A_bs, S_bs = amp_sbr_at(fEnv, Penv, bsf)

                # 指标（时域/频域/时频域）
                # 时域对原始 x0（若矩阵，用第一列代表）
                x_for_td = X[:,0] if X.ndim == 2 else X.reshape(-1)
                td = time_features(x_for_td)
                fd = freq_features_env(fEnv, Penv)
                tf = tf_features(x_for_td, fs)

                # 可视化
                fig_path = fig_dir / f"{in_path.stem}_{vname}.png"
                try:
                    xDen_plot = xDen if x0.ndim == 1 else Xden[:,0]
                    env_plot  = env  if x0.ndim == 1 else ENV[:,0]
                    plot_envelope_figure(x_for_td, xDen_plot, env_plot, fs, fEnv, Penv,
                                         flo, fhi, fr, bpfo, bpfi, bsf, fig_path)
                except Exception as e:
                    warnings.warn(f"绘图失败：{fig_path} ({e})")

                # 报表行
                row = [
                    str(in_path), str(out_path), vname, sensor, nSamp, durS,
                    flo, fhi, kmax, denoise_method,
                    nz(pkF,1), nz(pkA,1), nz(pkF,2), nz(pkA,2), nz(pkF,3), nz(pkA,3),
                    fr, bpfo, bpfi, bsf,
                    A_fr, S_fr, A_bo, S_bo, A_bi, S_bi, A_bs, S_bs,
                    td["时域_均值"], td["时域_STD"], td["时域_RMS"], td["时域_峰峰值"], td["时域_最大"], td["时域_最小"],
                    td["时域_偏斜度"], td["时域_峭度"], td["时域_波形指标"], td["时域_峰值指标"], td["时域_脉冲指标"], td["时域_裕度指标"],
                    fd["频域_谱质心Hz"], fd["频域_谱带宽Hz"], fd["频域_谱熵"], fd["频域_主峰Hz"], fd["频域_主峰幅"], fd["频域_中值频Hz"], fd["频域_95滚降Hz"],
                    tf["时频_谱峭度最大值"], tf["时频_谱峭度最大值对应频率Hz"], tf["时频_帧谱熵均值"], tf["时频_帧谱熵STD"],
                    tf["时频_三段能量比CV_0_200"], tf["时频_三段能量比CV_200_400"], tf["时频_三段能量比CV_400_1000"],
                ]
                rows.append(row)

            # 保存镜像文件
            if not args.dry_run:
                if file_changed:
                    savemat(str(out_path), S, do_compression=False)
                elif not out_path.exists():
                    savemat(str(out_path), S, do_compression=False)

            nOK += 1
            if nOK % 50 == 0:
                print(f"... 已处理 {nOK}/{len(files)}")

        except Exception as e:
            nFail += 1
            warnings.warn(f"处理失败：{in_path} ({e})")

    # 写报表
    csv_path = report_dir / f"envelope_report_12kHz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not args.dry_run:
        write_csv_utf8_bom(csv_path, header, rows)
        print(f"✅ 完成：成功 {nOK}，失败 {nFail}\n📄 报表：{csv_path}\n📁 输出：{out_root}")
    else:
        print(f"[DRY-RUN] 计划生成 {len(rows)} 行；不写报表。")


if __name__ == "__main__":
    main()
