# -*- coding: utf-8 -*-
"""
批量：包络谱分析 + 去噪 + 可视化 + 报表 + 镜像保存
输入根目录（已统一到12kHz，默认来自题目）：/home/user1/data/learn/sumo/data/raw_resampled_12kHz/
输出根目录（可改）：同级创建 envelope_denoised_12kHz
报表/图片目录：reports/（自动创建）
——
流程（与原 MATLAB 思路一致）：
1) 自动选带（候选中心频率 1.5:0.5:5.5 kHz，带宽±0.5k，Butterworth-4 阶，filtfilt）
2) 带通 → Hilbert 包络 → 中位数平滑（~5ms）
3) Welch 包络谱（0–1000 Hz为主）+ 峰值/特征频率（可选从文件名解析 rpm）
4) 保存：*_bandpassed / *_env / *_env_band_Hz 到镜像 .mat
5) 可视化：每变量一张 2×2 图（原始、带通、包络、包络谱）
6) 报表：含完整时域/频域/时频域特征（多列）
"""

from __future__ import annotations
import os, re, csv, sys, warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import argparse
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, hilbert, welch, get_window, spectrogram, medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========================= 参数与入口 =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量包络去噪+可视化+报表（12kHz数据）")
    parser.add_argument("--in-root", type=Path,
                        default=Path("/home/user1/data/learn/sumo/data/raw_resampled_12kHz/"),
                        help="输入根目录（递归 .mat）")
    parser.add_argument("--out-root", type=Path,
                        default=None,
                        help="输出（镜像）根目录；未指定则自动用同级 envelope_denoised_12kHz")
    parser.add_argument("--report-dir", type=Path,
                        default=Path("reports"),
                        help="报表与图片输出目录（默认 reports）")
    parser.add_argument("--fs", type=float, default=12_000.0, help="采样率 Hz（默认 12000）")
    parser.add_argument("--guard", type=float, default=60.0, help="带通设计保护带 Hz（默认60）")
    parser.add_argument("--fc-start-khz", type=float, default=1.5, help="候选中心频率起点 kHz")
    parser.add_argument("--fc-stop-khz", type=float, default=5.5, help="候选中心频率终点 kHz")
    parser.add_argument("--fc-step-khz", type=float, default=0.5, help="候选中心频率步进 kHz")
    parser.add_argument("--bw-half", type=float, default=500.0, help="半带宽 Hz（默认 500 => 带宽约1k）")
    parser.add_argument("--glob", type=str, default="*.mat", help='文件匹配（默认 "*.mat"）')
    parser.add_argument("--fig-subdir", type=str, default="包络谱图", help="图片子目录名")
    parser.add_argument("--dry-run", action="store_true", help="试跑：不写文件")
    return parser.parse_args()


# ========================= 主流程 =========================

def main():
    args = parse_args()
    in_root: Path = args.in_root
    if not in_root.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{in_root}")

    out_root = args.out_root or in_root.parent / "envelope_denoised_12kHz"
    out_root.mkdir(parents=True, exist_ok=True)

    report_dir: Path = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / args.fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    fs = float(args.fs)
    nyq = fs / 2.0

    # 候选中心频率列表（Hz）
    fc_list = np.arange(args.fc_start_khz, args.fc_stop_khz + 1e-9, args.fc_step_khz) * 1000.0
    bw_half = float(args.bw_half)
    guard = float(args.guard)

    # 搜索文件
    mat_files = list(in_root.rglob(args.glob))
    if not mat_files:
        warnings.warn(f"在 {in_root} 未找到 {args.glob} 文件。")
        return
    print(f"发现 {len(mat_files)} 个 .mat 文件，开始处理...")

    # 报表 CSV
    csv_path = report_dir / f"envelope_report_12kHz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    header = build_report_header()
    rows: List[List[Any]] = []

    n_ok = n_fail = 0

    for idx, in_path in enumerate(mat_files, 1):
        try:
            # 输出镜像路径（保持相对目录）
            rel_folder = str(in_path.parent.relative_to(in_root)) if in_path.parent != in_root else ""
            out_folder = out_root / rel_folder
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / in_path.name

            # 读取 .mat（去掉元字段）
            S = loadmat(str(in_path), squeeze_me=False, struct_as_record=False)
            S = {k: v for k, v in S.items() if not k.startswith("__")}

            # 信号变量：以 _DE_time/_FE_time/_BA_time 结尾，或“非标量数值型矩阵”
            sig_vars = [k for k in S.keys() if k.endswith(("_DE_time", "_FE_time", "_BA_time")) and isinstance(S[k], np.ndarray)]
            if not sig_vars:
                # 兜底：找所有“非标量数值型”
                for k, v in S.items():
                    if isinstance(v, np.ndarray) and v.size > 1 and np.issubdtype(v.dtype, np.number):
                        sig_vars.append(k)

            # RPM（可选）
            rpm = parse_rpm_from_filename(in_path.name)
            if not np.isfinite(rpm):
                # 试找 *RPM 变量
                rpm_keys = [k for k in S.keys() if k.endswith("RPM")]
                if rpm_keys:
                    try:
                        rv = np.asarray(S[rpm_keys[0]], dtype=float).ravel()
                        if rv.size > 0:
                            rpm = float(rv[0])
                    except Exception:
                        rpm = np.nan
            fr = rpm / 60.0 if np.isfinite(rpm) else np.nan

            file_changed = False

            if not sig_vars:
                # 无信号变量：原样镜像保存（不修改）
                if not args.dry_run:
                    savemat(str(out_path), S, do_compression=False)
                # 也写一行报表，至少记录文件级信息
                rows.append(file_level_row(header, str(in_path), "(no-signal-var)", fs))
                n_ok += 1
                continue

            for vname in sig_vars:
                X0 = np.asarray(S[vname])
                xclass = X0.dtype

                # 统一列向量/矩阵（时间轴=行）
                if X0.ndim == 1:
                    X = X0.reshape(-1, 1)
                elif X0.shape[0] < X0.shape[1]:
                    # 习惯上把时间放在行；若列数>行数，做转置
                    X = X0.T
                else:
                    X = X0

                n_samp = X.shape[0]
                dur_s = n_samp / fs

                # —— 自动选带（以第1通道为代表，按带后峭度最大）——
                flo, fhi, kmax = auto_select_band(X[:, 0], fs, fc_list, bw_half, guard)
                denoise_method = f"bandpass[{int(flo)},{int(fhi)}]+hilbert+movmedian"

                # —— 设计带通 ——（Butterworth 4阶）
                b, a = butter(4, [max(guard, flo) / nyq, min(nyq - guard, fhi) / nyq], btype="band")

                # —— 去噪：带通→Hilbert包络→平滑 ——（逐通道）
                X_bp = np.empty_like(X, dtype=float)
                ENV = np.empty_like(X, dtype=float)
                for c in range(X.shape[1]):
                    xf = filtfilt(b, a, X[:, c].astype(float, copy=False))
                    env = np.abs(hilbert(xf))
                    # 平滑窗口：约 5ms（>=3 且奇数，便于 medfilt）
                    win = max(3, int(round(0.005 * fs)))
                    if win % 2 == 0:
                        win += 1
                    env_sm = medfilt(env, kernel_size=win)
                    X_bp[:, c] = xf
                    ENV[:, c] = env_sm

                # —— 包络谱（取第一通道代表，Welch）——
                f_env, P_env = envelope_welch(ENV[:, 0], fs)

                # —— 包络谱峰值（0–1000 Hz）——
                pk_f, pk_a = top_env_peaks(f_env, P_env, N=3, fmax=1000.0)

                # —— 可选：特征频率评估 ——（若 fr 有效，可估 BPFO/BPFI/BSF）
                # 若你知道具体几何参数，可在此处替换/读取；这里给一组默认（6205）
                geom = dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0)
                bpfo = bpfi = bsf = np.nan
                A_fr = S_fr = A_bo = S_bo = A_bi = S_bi = A_bs = S_bs = np.nan
                if np.isfinite(fr) and fr > 0:
                    bpfo, bpfi, bsf = bearing_freqs(fr, geom)
                    A_fr, S_fr = amp_sbr_at(f_env, P_env, fr)
                    A_bo, S_bo = amp_sbr_at(f_env, P_env, bpfo)
                    A_bi, S_bi = amp_sbr_at(f_env, P_env, bpfi)
                    A_bs, S_bs = amp_sbr_at(f_env, P_env, bsf)

                # —— 特征：时域/频域/时频域 ——（按第一通道计算；矩阵可改成聚合）
                td = time_features(X[:, 0])
                fd = freq_features(X[:, 0], fs)
                tfd = tf_features(X[:, 0], fs)

                # —— 保存到结构 ——（变量命名与 MATLAB 对齐）
                if not args.dry_run:
                    # 恢复到与原变量同形状
                    if X0.ndim == 1:
                        S[f"{vname}_bandpassed"] = X_bp.ravel().astype(xclass, copy=False)
                        S[f"{vname}_env"] = ENV.ravel().astype(float, copy=False)
                    else:
                        # 调回原 shape
                        shp = X0.T.shape if (X0.ndim == 2 and X0.shape[0] < X0.shape[1]) else X0.shape
                        S[f"{vname}_bandpassed"] = X_bp.reshape(shp).astype(xclass, copy=False)
                        S[f"{vname}_env"] = ENV.reshape(shp).astype(float, copy=False)
                    S[f"{vname}_env_band_Hz"] = np.array([float(flo), float(fhi)], dtype=float)

                file_changed = True

                # —— 可视化 ——（每变量一张图）
                fig_path = (report_dir / args.fig_subdir / f"{in_path.stem}_{vname}.png")
                try:
                    if not args.dry_run:
                        plot_envelope_figure(
                            x0=X0 if X0.ndim == 1 else (X0[:, 0] if X0.shape[0] >= X0.shape[1] else X0[0, :]),
                            x_bp=X_bp[:, 0],
                            env=ENV[:, 0],
                            fs=fs,
                            f_env=f_env,
                            P_env=P_env,
                            flo=flo, fhi=fhi,
                            fr=fr, bpfo=bpfo, bpfi=bpfi, bsf=bsf,
                            save_path=str(fig_path)
                        )
                except Exception as e:
                    warnings.warn(f"绘图失败：{fig_path} ({e})")

                # —— 报表一行 ——（字段很多，按 header 顺序组织）
                row = report_row(
                    header=header,
                    in_path=str(in_path),
                    var_name=vname,
                    n_samp=n_samp,
                    dur_s=dur_s,
                    flo=flo, fhi=fhi, kmax=kmax,
                    denoise=denoise_method,
                    pkF=pk_f, pkA=pk_a,
                    fr=fr, bpfo=bpfo, bpfi=bpfi, bsf=bsf,
                    A_fr=A_fr, S_fr=S_fr, A_bo=A_bo, S_bo=S_bo, A_bi=A_bi, S_bi=S_bi, A_bs=A_bs, S_bs=S_bs,
                    out_path=str(out_path),
                    td=td, fd=fd, tfd=tfd
                )
                rows.append(row)

            # —— 写出镜像 .mat ——（若有修改）
            if file_changed and not args.dry_run:
                savemat(str(out_path), S, do_compression=False)
            elif not args.dry_run and (not out_path.exists()):
                # 没有修改也保证镜像存在
                savemat(str(out_path), S, do_compression=False)

            n_ok += 1
            if n_ok % 50 == 0:
                print(f"... 已处理 {n_ok}/{len(mat_files)}")

        except Exception as e:
            n_fail += 1
            warnings.warn(f"处理失败：{in_path} ({e})")

    # —— 写 CSV 报表（UTF-8 BOM）——
    if not args.dry_run:
        write_csv_utf8_bom(csv_path, header, rows)
    print(f"✅ 完成：成功 {n_ok}，失败 {n_fail}\n📄 报表：{csv_path}\n📁 输出：{out_root}\n🖼️ 图片：{fig_dir}")


# ========================= 特征与分析函数 =========================

def auto_select_band(x: np.ndarray, fs: float, fc_list: np.ndarray, bw_half: float, guard: float) -> Tuple[float, float, float]:
    """
    在候选频带中选择“带通后峭度最大”的频带。返回 (flo, fhi, kmax)
    """
    x = np.asarray(x, dtype=float).ravel()
    nyq = fs / 2.0
    best = (-np.inf, fc_list[0] - bw_half, fc_list[0] + bw_half)  # (kurt, flo, fhi)
    for fc in fc_list:
        lo = max(guard, fc - bw_half)
        hi = min(nyq - guard, fc + bw_half)
        if hi <= lo + 10:
            continue
        try:
            b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
            xf = filtfilt(b, a, x)
            k = kurtosis_np(xf)
            if np.isfinite(k) and (k > best[0]):
                best = (k, lo, hi)
        except Exception:
            pass
    kmax, flo, fhi = best
    return float(flo), float(fhi), float(kmax)


def kurtosis_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2) + 1e-12
    k = np.mean((x - m) ** 4) / (s2 ** 2)
    return float(k)


def envelope_welch(env: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch 包络谱（参数与稳健性参考 MATLAB：汉明窗、分段/重叠按数据长度自适应）
    """
    env = np.asarray(env, dtype=float).ravel()
    n = env.size
    nfft = 1 << int(np.floor(np.log2(min(n, 16384))))
    if nfft < 256:
        nfft = min(1024, n)
    win_len = max(256, nfft // 4)
    win = get_window("hamming", win_len, fftbins=True)
    f, P = welch(env - np.mean(env), fs=fs, window=win, nperseg=win_len, noverlap=win_len // 2, nfft=nfft, return_onesided=True, detrend="constant")
    return f, P


def top_env_peaks(f: np.ndarray, P: np.ndarray, N: int = 3, fmax: float = 1000.0) -> Tuple[List[float], List[float]]:
    idx = (f >= 0) & (f <= fmax)
    ff, pp = f[idx], P[idx]
    if ff.size == 0:
        return [np.nan] * N, [np.nan] * N
    # 简单回退策略：直接取最大 N 个 bin
    ord_idx = np.argsort(pp)[::-1][:N]
    pk_f = ff[ord_idx].tolist()
    pk_a = pp[ord_idx].tolist()
    # 填满N
    while len(pk_f) < N:
        pk_f.append(np.nan)
        pk_a.append(np.nan)
    return pk_f, pk_a


def amp_sbr_at(f: np.ndarray, P: np.ndarray, f0: float) -> Tuple[float, float]:
    if not np.isfinite(f0) or f0 <= 0:
        return np.nan, np.nan
    df = np.median(np.diff(f))
    if not np.isfinite(df) or df <= 0:
        df = 0.5
    bw = max(3 * df, 5.0)  # ±5 Hz 或 3*df
    idx = (f >= max(0.0, f0 - bw)) & (f <= f0 + bw)
    if not np.any(idx):
        return np.nan, np.nan
    ii = np.argmax(P[idx])
    ii_glob = np.where(idx)[0][ii]
    A = float(P[ii_glob])
    # 背景：更宽邻域，去掉核心 bin
    idx_bg = (f >= max(0.0, f0 - 5 * bw)) & (f <= f0 + 5 * bw)
    if np.any(idx_bg):
        idx_bg[np.where(idx)[0][ii]] = False
        B = float(np.median(P[idx_bg])) if np.any(idx_bg) else float(np.median(P))
    else:
        B = float(np.median(P))
    SBR = 20.0 * np.log10(max(A, 1e-18) / max(B, 1e-18))
    return A, SBR


def bearing_freqs(fr: float, g: Dict[str, float]) -> Tuple[float, float, float]:
    Nd, d, D, theta = g["Nd"], g["d"], g["D"], g["thetaDeg"]
    c = np.cos(np.deg2rad(theta))
    bpfo = fr * (Nd / 2.0) * (1.0 - (d / D) * c)
    bpfi = fr * (Nd / 2.0) * (1.0 + (d / D) * c)
    bsf  = fr * (D / (2.0 * d)) * (1.0 - ((d / D) * c) ** 2)
    return float(bpfo), float(bpfi), float(bsf)


# ========================= 特征集合 =========================

def time_features(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0) + 1e-18)
    rms = float(np.sqrt(np.mean(x ** 2)))
    maxv = float(np.max(x))
    minv = float(np.min(x))
    ptp = maxv - minv
    sk = float(np.mean(((x - mean) / std) ** 3))
    ku = float(np.mean(((x - mean) / std) ** 4))
    cf = float(max(abs(maxv), abs(minv)) / (rms + 1e-18))         # 峰值因子
    if std > 0:
        if_r = float(np.max(np.abs(x)) / (np.mean(np.abs(x)) + 1e-18))  # 脉冲指标
    else:
        if_r = np.nan
    sf = float(rms / (np.mean(np.abs(x)) + 1e-18))                # 波形指标
    mf = float(np.max(np.abs(x)) / (np.mean(np.sqrt(np.abs(x))) + 1e-18))  # 裕度指标
    return {
        "td_mean": mean, "td_std": std, "td_rms": rms, "td_max": maxv, "td_min": minv, "td_ptp": ptp,
        "td_skew": sk, "td_kurt": ku, "td_crest": cf, "td_impulse": if_r, "td_shape": sf, "td_margin": mf
    }


def freq_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    # Welch
    f, P = welch(x - np.mean(x), fs=fs, window="hamming", nperseg=min(4096, max(256, len(x)//4)),
                 noverlap=None, nfft=None, detrend="constant", return_onesided=True)
    P = np.maximum(P, 1e-18)
    # 归一化谱作为概率
    Pn = P / np.sum(P)
    f_mean = float(np.sum(f * Pn))
    f_bw = float(np.sqrt(np.sum(((f - f_mean) ** 2) * Pn)))
    f_peak = float(f[np.argmax(P)])
    spec_entropy = float(-np.sum(Pn * np.log(Pn)))
    # 频带能量比（0-1k、1-2k、2-3k、3-nyq）
    bands = [(0, 1000), (1000, 2000), (2000, 3000), (3000, fs/2)]
    band_energies = []
    for lo, hi in bands:
        idx = (f >= lo) & (f < hi)
        band_energies.append(float(np.sum(P[idx])))
    total_e = float(np.sum(P))
    ratios = [be / (total_e + 1e-18) for be in band_energies]
    return {
        "fd_centroid": f_mean, "fd_bandwidth": f_bw, "fd_peakfreq": f_peak, "fd_spec_entropy": spec_entropy,
        "fd_band0_1k": ratios[0], "fd_band1_2k": ratios[1], "fd_band2_3k": ratios[2], "fd_band3_nyq": ratios[3]
    }


def tf_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """
    STFT-based: 谱峭度最大值/频率；帧谱熵均值/STD；三分段能量比时间CV（与之前思路一致）
    """
    x = np.asarray(x, dtype=float).ravel()
    # 自适应窗口：~ 2048，最小 256
    wlen = 2048
    if wlen > len(x):
        wlen = 1 << int(np.floor(np.log2(max(len(x), 256))))
    wlen = max(wlen, 256)
    olap = int(0.5 * wlen)
    nfft = max(4096, 1 << int(np.ceil(np.log2(wlen))))
    f, t, Sxx = spectrogram(x - np.mean(x), fs=fs, window=("hamming", wlen), nperseg=wlen,
                            noverlap=olap, nfft=nfft, detrend=False, mode="magnitude")
    P = Sxx ** 2 + 1e-18  # 功率
    # 谱峭度（对频率维）
    sk = np.mean(((P - np.mean(P, axis=1, keepdims=True)) ** 4), axis=1) / \
         (np.mean(((P - np.mean(P, axis=1, keepdims=True)) ** 2), axis=1) ** 2 + 1e-18)
    sk_max = float(np.max(sk))
    sk_f = float(f[np.argmax(sk)])
    # 帧谱熵
    Pn = P / np.sum(P, axis=0, keepdims=True)
    spec_ent = -np.sum(Pn * np.log(Pn), axis=0)
    se_mean = float(np.mean(spec_ent))
    se_std = float(np.std(spec_ent))
    # 三分段能量比时间 CV
    thirds = np.array_split(np.arange(P.shape[0]), 3)
    ratios = []
    for g in thirds:
        ratios.append(np.sum(P[g, :], axis=0) / (np.sum(P, axis=0) + 1e-18))
    ratios = np.vstack(ratios)  # (3, T)
    cv = float(np.std(ratios, axis=1).mean() / (np.mean(ratios, axis=1).mean() + 1e-18))
    return {
        "tf_skurt_max": sk_max, "tf_skurt_freq": sk_f,
        "tf_specent_mean": se_mean, "tf_specent_std": se_std,
        "tf_3band_ratio_cv": cv
    }


# ========================= 可视化与报表组织 =========================

def plot_envelope_figure(x0: np.ndarray, x_bp: np.ndarray, env: np.ndarray, fs: float,
                         f_env: np.ndarray, P_env: np.ndarray,
                         flo: float, fhi: float,
                         fr: float, bpfo: float, bpfi: float, bsf: float,
                         save_path: str):
    x0 = np.asarray(x0).ravel()
    nshow = int(min(len(x0), fs * 1.0))  # 展示 1s
    t = np.arange(nshow) / fs

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t, x0[:nshow])
    plt.title("原始信号（片段）")
    plt.xlabel("t (s)"); plt.ylabel("x"); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, x_bp[:nshow])
    plt.title(f"带通去噪（{int(flo)}–{int(fhi)} Hz）")
    plt.xlabel("t (s)"); plt.ylabel("x_bp"); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, env[:nshow])
    plt.title("包络（平滑后）")
    plt.xlabel("t (s)"); plt.ylabel("|Hilbert|"); plt.grid(True)

    plt.subplot(2, 2, 4)
    idx = f_env <= 1000.0
    plt.plot(f_env[idx], P_env[idx]); plt.grid(True)
    plt.title("包络谱（0–1000 Hz）")
    plt.xlabel("f (Hz)"); plt.ylabel("P_env")
    # 标注 fr/BPFO/BPFI/BSF
    def mark(f0, name):
        if np.isfinite(f0) and (0 < f0 < 1000):
            plt.axvline(f0, linestyle="--"); plt.text(f0, plt.ylim()[1]*0.85, name, rotation=90, va="top")
    mark(fr, "fr"); mark(bpfo, "BPFO"); mark(bpfi, "BPFI"); mark(bsf, "BSF")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_report_header() -> List[str]:
    # 基础 + 包络谱指标 + 机理频率 + SBR + 输出 + 时/频/时频域特征
    base = ["文件路径", "变量名", "样本数", "时长(s)",
            "选带低(Hz)", "选带高(Hz)", "带内峭度", "去噪方法",
            "env峰1_Hz", "env峰1_幅", "env峰2_Hz", "env峰2_幅", "env峰3_Hz", "env峰3_幅",
            "fr_Hz", "BPFO_Hz", "BPFI_Hz", "BSF_Hz",
            "Amp@fr", "SBR@fr_dB", "Amp@BPFO", "SBR@BPFO_dB",
            "Amp@BPFI", "SBR@BPFI_dB", "Amp@BSF", "SBR@BSF_dB",
            "输出文件"]
    td_keys = ["td_mean","td_std","td_rms","td_max","td_min","td_ptp","td_skew","td_kurt","td_crest","td_impulse","td_shape","td_margin"]
    fd_keys = ["fd_centroid","fd_bandwidth","fd_peakfreq","fd_spec_entropy","fd_band0_1k","fd_band1_2k","fd_band2_3k","fd_band3_nyq"]
    tfd_keys = ["tf_skurt_max","tf_skurt_freq","tf_specent_mean","tf_specent_std","tf_3band_ratio_cv"]
    return base + td_keys + fd_keys + tfd_keys


def report_row(header: List[str], in_path: str, var_name: str,
               n_samp: int, dur_s: float,
               flo: float, fhi: float, kmax: float, denoise: str,
               pkF: List[float], pkA: List[float],
               fr: float, bpfo: float, bpfi: float, bsf: float,
               A_fr: float, S_fr: float, A_bo: float, S_bo: float, A_bi: float, S_bi: float, A_bs: float, S_bs: float,
               out_path: str,
               td: Dict[str, float], fd: Dict[str, float], tfd: Dict[str, float]) -> List[Any]:
    base = [
        in_path, var_name, int(n_samp), float(dur_s),
        float(flo), float(fhi), float(kmax), denoise,
        safe_get(pkF, 0), safe_get(pkA, 0),
        safe_get(pkF, 1), safe_get(pkA, 1),
        safe_get(pkF, 2), safe_get(pkA, 2),
        fr, bpfo, bpfi, bsf,
        A_fr, S_fr, A_bo, S_bo, A_bi, S_bi, A_bs, S_bs,
        out_path
    ]
    td_keys = ["td_mean","td_std","td_rms","td_max","td_min","td_ptp","td_skew","td_kurt","td_crest","td_impulse","td_shape","td_margin"]
    fd_keys = ["fd_centroid","fd_bandwidth","fd_peakfreq","fd_spec_entropy","fd_band0_1k","fd_band1_2k","fd_band2_3k","fd_band3_nyq"]
    tfd_keys = ["tf_skurt_max","tf_skurt_freq","tf_specent_mean","tf_specent_std","tf_3band_ratio_cv"]
    feat_vals = [td.get(k, np.nan) for k in td_keys] + [fd.get(k, np.nan) for k in fd_keys] + [tfd.get(k, np.nan) for k in tfd_keys]
    return base + feat_vals


def safe_get(a: List[float], i: int) -> float:
    try:
        return float(a[i])
    except Exception:
        return np.nan


def parse_rpm_from_filename(fname: str) -> float:
    m = re.search(r"(\d{2,5})\s*rpm", fname, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def write_csv_utf8_bom(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ========================= 入口 =========================

if __name__ == "__main__":
    main()
