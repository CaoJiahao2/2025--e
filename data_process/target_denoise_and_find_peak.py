# -*- coding: utf-8 -*-
"""
Batch envelope denoise + report for target domain (GPU-optional)
- 输入根：/home/user1/data/learn/sumo/data/target_resample_12kHz
- 输出扁平：/home/user1/data/learn/sumo/data/target_envelope_denoised_12kHz
- 报表：./reports/target_envelope_report_12kHz_YYYYmmdd_HHMMSS.csv
- 特征：计算6个特征频率（BPFO/I/SF及其2x谐波）的最近峰距离和幅值
"""
from __future__ import annotations
import os
import re
import csv
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Any

import argparse
import numpy as np
import matplotlib
from scipy.io import loadmat, savemat
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks, spectrogram, medfilt

# 绘图后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体配置
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Source Han Sans SC", "Source Han Sans CN",
    "SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei"
]
installed = {f.name for f in font_manager.fontManager.ttflist}
for name in CJK_CANDIDATES:
    if name in installed:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

# 可选依赖
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

USE_GPU = False
try:
    import cupy as cp
    from cupyx.scipy.signal import (firwin as cp_firwin, lfilter as cp_lfilter, 
                                    hilbert as cp_hilbert, welch as cp_welch, 
                                    spectrogram as cp_spectrogram)
    from cupyx.scipy.ndimage import median_filter as cp_medfilt
    from cupyx.scipy.fft import rfft, irfft
    USE_GPU = True
except ImportError:
    USE_GPU = False

# ───────── 配置 ─────────
BASE_DIR = Path("/home/user1/data/learn/sumo")
IN_ROOT = BASE_DIR / "data/target_resample_12kHz"
OUT_ROOT = BASE_DIR / "data/target_envelope_denoised_12kHz2"
REPORT_DIR = Path("./reports2")
PLOT_DIR = REPORT_DIR / "target-包络谱图"

FS = 12000.0
NYQ = FS / 2
GUARD_HZ = 60.0

# 候选中心频率（Hz）与半带宽（~±0.5 kHz）
FC_LIST = np.arange(1500.0, 5500.0 + 1e-6, 500.0)
BW_HALF = 500.0

# 轴承几何
GEOM = {
    "DE": dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),
    "FE": dict(Nd=9, d=0.2656, D=1.122, thetaDeg=0.0),
}

# =======================================================
# 基础I/O
# =======================================================
def ensure_dirs():
    for d in [OUT_ROOT, REPORT_DIR, PLOT_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

def list_mat_files(root: Path) -> List[Path]:
    return sorted(list(root.glob("*.mat")))

def load_mat_any(path: Path) -> Dict[str, np.ndarray]:
    try:
        return loadmat(str(path), squeeze_me=False, struct_as_record=False)
    except Exception:
        if not HAS_H5PY:
            raise
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("#"):
                out[k] = f[k][()]
    return out

def savemat_safe(path: Path, dict_vars: Dict[str, Any]):
    clean = {k: v for k, v in dict_vars.items() if not k.startswith("__")}
    savemat(str(path), clean, do_compression=True)

# =======================================================
# 杂项
# =======================================================
def rpm_to_hz(rpm: float | None) -> float:
    return float(rpm) / 60.0 if (rpm is not None and np.isfinite(rpm) and rpm > 0) else np.nan

def parse_rpm_from_filename(fname: str) -> float:
    m = re.search(r"(\d{3,5})\s*rpm", fname, flags=re.I)
    return float(m.group(1)) if m else np.nan

def get_rpm_from_vars(d: Dict[str, Any]) -> float:
    for k in d.keys():
        if "rpm" in k.lower() and isinstance(d[k], np.ndarray) and d[k].size == 1:
            return float(d[k].item())
    return np.nan

def find_signal_vars(S: dict) -> List[str]:
    """选择所有 '非标量且数值型 ndarray' 的变量名。"""
    keep = []
    for k, v in S.items():
        if isinstance(v, np.ndarray) and v.size > 1 and np.issubdtype(v.dtype, np.number):
            keep.append(k)
    return keep

def sensor_from_vname(vname: str) -> str:
    v = vname.lower()
    if "_de" in v: return "DE"
    if "_fe" in v: return "FE"
    # 目标域数据变量名可能不规范，基于文件名猜测
    if "b" in v: return "DE" # B.mat -> DE
    if "g" in v: return "FE" # G.mat -> FE
    return "DE" # 默认

def bearing_freqs(fr: float, g: Dict[str, float]) -> Dict[str, float]:
    if not (np.isfinite(fr) and fr > 0):
        return {"BPFO": np.nan, "BPFI": np.nan, "BSF": np.nan}
    Nd, d, D, theta_rad = g['Nd'], g['d'], g['D'], np.deg2rad(g['thetaDeg'])
    c = np.cos(theta_rad)
    bpfo = fr * (Nd / 2.0) * (1.0 - (d / D) * c)
    bpfi = fr * (Nd / 2.0) * (1.0 + (d / D) * c)
    bsf = fr * (D / (2.0 * d)) * (1.0 - ((d / D) * c) ** 2)
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}

# =======================================================
# GPU/CPU 计算适配层
# =======================================================
def as_xp(a):
    return cp.asarray(a) if (USE_GPU and not isinstance(a, cp.ndarray)) else a

def to_numpy(a):
    return cp.asnumpy(a) if (USE_GPU and isinstance(a, cp.ndarray)) else np.asarray(a)

def butter_bp(lo, hi, fs, order=4):
    if USE_GPU:
        return cp_firwin(order + 1, [lo, hi], pass_zero=False, fs=fs), cp.array([1.0])
    else:
        return butter(order, [lo, hi], btype="band", fs=fs)

def filt_filt(b, a, x):
    if USE_GPU:
        return cp_lfilter(b, a, x)
    else:
        return filtfilt(b, a, x)

def hilbert_env(x):
    if USE_GPU:
        return cp.abs(cp_hilbert(as_xp(x)))
    else:
        return np.abs(hilbert(x))

def medfilt_1d(x, win):
    if USE_GPU:
        return cp_medfilt(as_xp(x), size=win)
    else:
        return medfilt(x, kernel_size=win)

def welch_psd(x, fs, nperseg):
    if USE_GPU:
        f, P = cp_welch(as_xp(x), fs=fs, nperseg=nperseg)
        return to_numpy(f), to_numpy(P)
    else:
        return welch(x, fs=fs, nperseg=nperseg)

def spectro(x, fs, wlen, olap, nfft):
    if USE_GPU:
        f, t, Sxx = cp_spectrogram(as_xp(x), fs=fs, nperseg=wlen, noverlap=olap, nfft=nfft)
        return to_numpy(f), to_numpy(t), to_numpy(Sxx)
    else:
        return spectrogram(x, fs=fs, nperseg=wlen, noverlap=olap, nfft=nfft)

# =======================================================
# 信号处理
# =======================================================
def select_band_by_kurtosis(x, fs, fc_list, bw_half, guard=60.0):
    x1 = np.atleast_1d(np.asarray(x, dtype=float))
    if x1.ndim > 1: x1 = x1.ravel()
    best_k, best_fc = -np.inf, fc_list[0]
    flo, fhi = 0, 0
    for fc in fc_list:
        flo, fhi = fc - bw_half, fc + bw_half
        if flo < guard: continue
        if fhi > fs / 2 - guard: continue
        b, a = butter_bp(flo, fhi, fs, order=4)
        y = filt_filt(b, a, as_xp(x1))
        k = kurtosis(to_numpy(y), fisher=False)
        if k > best_k:
            best_k, best_fc = k, fc
    flo, fhi = best_fc - bw_half, best_fc + bw_half
    b, a = butter_bp(flo, fhi, fs, order=4)
    x_bp = filt_filt(b, a, as_xp(x1))
    return to_numpy(x_bp), flo, fhi, best_k

def envelope_and_spectrum(x_bp, fs):
    env = hilbert_env(as_xp(x_bp))
    env = medfilt_1d(env, int(round(0.005 * fs)) or 3)
    env = to_numpy(env)
    nfft = int(2 ** np.floor(np.log2(min(len(env), 16384))))
    if nfft < 256: return None, None, None
    f, P = welch_psd(env - np.mean(env), fs=fs, nperseg=max(256, nfft // 4))
    return env, f, P

def find_closest_peak(target_f, peak_fs, peak_amps):
    if not (np.isfinite(target_f) and target_f > 0) or peak_fs.size == 0:
        return np.nan, np.nan
    dist = np.abs(peak_fs - target_f)
    idx = np.argmin(dist)
    return dist[idx], peak_amps[idx]

# =======================================================
# 绘图
# =======================================================
def plot_and_save_envelope(
    f_env: np.ndarray,
    P_env: np.ndarray,
    peak_fs: np.ndarray,
    peak_amps: np.ndarray,
    fault_freqs: Dict[str, float],
    rpm: float,
    base_filename: str,
    var_name: str,
    sensor: str,
    plot_dir: Path,
):
    """绘制并保存包络谱图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制包络谱
    ax.plot(f_env, P_env, label="包络谱", color="c", linewidth=0.8)
    ax.scatter(peak_fs, peak_amps, color='red', s=20, marker='x', label="谱峰", zorder=5)

    # 绘制故障频率线
    colors = {"BPFO": "r", "BPFI": "g", "BSF": "b"}
    styles = {1: "-", 2: "--"}
    
    if np.isfinite(rpm) and rpm > 0:
        for name, f0 in fault_freqs.items():
            if not np.isfinite(f0): continue
            for h in [1, 2]:
                fh = f0 * h
                ax.axvline(x=fh, color=colors.get(name, "k"), linestyle=styles[h], 
                           linewidth=1.2, label=f"{name} {h}x ({fh:.2f} Hz)")

    ax.set_title(f"包络谱 - {base_filename} ({var_name}@{sensor}, {rpm:.0f} RPM)")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("功率谱密度")
    
    # 特定文件使用特定X轴范围
    if base_filename.upper().startswith('D.MAT'):
        ax.set_xlim(0, 200)
    else:
        ax.set_xlim(0, max(500, np.max(f_env) if f_env.size > 0 else 500))
        
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    
    # 避免图例重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    plt.tight_layout()
    
    # 保存图像
    plot_filename = f"{Path(base_filename).stem}_{var_name}.png"
    fig.savefig(plot_dir / plot_filename, dpi=150)
    plt.close(fig)

# =======================================================
# 特征
# =======================================================
def time_features(x):
    x = np.asarray(x).astype(float); N = len(x)
    if N==0: return {k: np.nan for k in "mean,std,var,rms,max,min,p2p,skew,kurt,crest,impulse,shape,clearance".split(",")}
    mean_ = float(np.mean(x))
    std_  = float(np.std(x, ddof=1)) if N>1 else 0.0
    var_  = float(np.var(x, ddof=1)) if N>1 else 0.0
    rms   = float(np.sqrt(np.mean(x**2)))
    max_  = float(np.max(x)); min_ = float(np.min(x)); p2p = max_-min_
    skw   = float(skew(x, bias=False)) if N>2 else np.nan
    kur   = float(kurtosis(x, fisher=False, bias=False)) if N>3 else np.nan
    ma    = float(np.mean(np.abs(x))) if N>0 else np.nan
    msa   = float(np.mean(np.sqrt(np.abs(x)))) if N>0 else np.nan
    crest = (max_/rms) if rms>0 else np.nan
    impulse = (max_/ma) if ma>0 else np.nan
    shape   = (rms/ma) if ma>0 else np.nan
    clearance = (max_/(msa**2)) if msa>0 else np.nan
    return dict(mean=mean_, std=std_, var=var_, rms=rms, max=max_, min=min_, p2p=p2p,
                skew=skw, kurt=kur, crest=crest, impulse=impulse, shape=shape, clearance=clearance)

def freq_features(x, fs):
    x = np.asarray(x).astype(float)
    nper = 1024 if len(x)>=256 else max(64, 2**int(np.ceil(np.log2(len(x)))))
    f, P = welch(x - np.mean(x), fs=fs, nperseg=nper)
    P = np.maximum(P, 0.0)
    Psum = np.sum(P)
    Pn = P/Psum if Psum>0 else P
    centroid = float(np.sum(f*Pn))
    bw = float(np.sqrt(np.sum(((f-centroid)**2)*Pn)))
    spec_entropy = float(-np.sum(Pn*np.log(Pn+1e-12))/np.log(len(Pn))) if len(Pn)>1 else np.nan
    return dict(psd_centroid=centroid, psd_bandwidth=bw, psd_entropy=spec_entropy)

def tf_features(x, fs):
    x = np.asarray(x).astype(float)
    wlen = 2048
    if wlen > len(x):
        wlen = 2**int(np.floor(np.log2(len(x))))
    if wlen < 64: return dict(sk_max=np.nan, sk_f_hz=np.nan, tf_entropy_mean=np.nan, tf_entropy_std=np.nan, tf_energy3_cv=np.nan)
    olap = wlen//2
    nfft = max(4096, 2**int(np.ceil(np.log2(wlen))))
    f, t, Sxx = spectro(x, fs, wlen, olap, nfft)
    P = (Sxx**2)
    if P.shape[1] >= 4:
        sk = kurtosis(P, fisher=True, axis=0)
        kmax = float(np.nanmax(sk))
        f_at_kmax = float(f[np.nanargmax(sk)])
    else:
        kmax, f_at_kmax = np.nan, np.nan
    Psum = np.sum(P, axis=0, keepdims=True)
    Pn = P/(Psum+1e-12)
    frame_entropy = -np.sum(Pn*np.log(Pn+1e-12), axis=0)/np.log(P.shape[0]) if P.shape[0]>1 else np.nan
    ent_mean = float(np.nanmean(frame_entropy)) if np.ndim(frame_entropy)>0 else np.nan
    ent_std  = float(np.nanstd(frame_entropy))  if np.ndim(frame_entropy)>0 else np.nan
    N = len(x); thirds = np.array_split(np.arange(N), 3)
    energies = [float(np.sum(x[idx]**2)) for idx in thirds]
    cv_energy = float(np.std(energies, ddof=1)/np.mean(energies)) if np.mean(energies)>0 and len(energies)>=2 else np.nan
    return dict(sk_max=kmax, sk_f_hz=f_at_kmax, tf_entropy_mean=ent_mean, tf_entropy_std=ent_std, tf_energy3_cv=cv_energy)

# =======================================================
# 主流程
# =======================================================
def main():
    if USE_GPU:
        print("INFO: Using GPU via CuPy.")
    else:
        print("INFO: Using CPU via NumPy/SciPy.")
    ensure_dirs()
    if not IN_ROOT.is_dir():
        warnings.warn(f"输入目录未找到: {IN_ROOT}")
        return

    mats = list_mat_files(IN_ROOT)
    if not mats:
        warnings.warn(f"在 {IN_ROOT} 中未找到 .mat 文件")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"target_envelope_report_12kHz_{ts}.csv"

    header = [
        "文件", "变量", "传感器", "转速rpm", "转频Hz",
        "最优带通下限Hz", "最优带通上限Hz", "最优带通峭度",
        "BPFO_1x_dist", "BPFO_1x_amp", "BPFI_1x_dist", "BPFI_1x_amp", "BSF_1x_dist", "BSF_1x_amp",
        "BPFO_2x_dist", "BPFO_2x_amp", "BPFI_2x_dist", "BPFI_2x_amp", "BSF_2x_dist", "BSF_2x_amp",
        "raw_mean", "raw_std", "raw_var", "raw_rms", "raw_max", "raw_min", "raw_p2p",
        "raw_skew", "raw_kurt", "raw_crest", "raw_impulse", "raw_shape", "raw_clearance",
        "raw_psd_centroid", "raw_psd_bandwidth", "raw_psd_entropy",
        "raw_sk_max", "raw_sk_f_hz", "raw_tf_entropy_mean", "raw_tf_entropy_std", "raw_tf_energy3_cv",
        "env_mean", "env_std", "env_var", "env_rms", "env_max", "env_min", "env_p2p",
        "env_skew", "env_kurt", "env_crest", "env_impulse", "env_shape", "env_clearance"
    ]
    rows = []

    print(f"找到 {len(mats)} 个文件，开始处理...")
    for i, in_path in enumerate(mats):
        try:
            out_path = OUT_ROOT / in_path.name
            S_in = load_mat_any(in_path)
            S_out = S_in.copy()

            # 确定转速
            rpm = parse_rpm_from_filename(in_path.name)
            if not np.isfinite(rpm):
                rpm = get_rpm_from_vars(S_in)
            fr = rpm_to_hz(rpm)

            sig_vars = find_signal_vars(S_in)
            if not sig_vars:
                warnings.warn(f"无信号变量, 跳过: {in_path.name}")
                continue

            for v_name in sig_vars:
                x0 = S_in[v_name].ravel().astype(np.float64)
                sensor = sensor_from_vname(v_name)
                if sensor not in GEOM:
                    warnings.warn(f"未知传感器 '{sensor}' for {v_name} in {in_path.name}, 跳过")
                    continue

                # 1. 包络去噪
                x_bp, flo, fhi, k_bp = select_band_by_kurtosis(x0, FS, FC_LIST, BW_HALF, GUARD_HZ)
                env, f_env, P_env = envelope_and_spectrum(x_bp, FS)
                if env is None:
                    warnings.warn(f"包络谱计算失败, 跳过: {in_path.name} / {v_name}")
                    continue

                # 2. 寻找谱峰
                prom = np.median(P_env) * 3.0
                if not (np.isfinite(prom) and prom > 0): prom = 1e-6
                peak_indices, _ = find_peaks(P_env, prominence=prom)
                peak_fs = f_env[peak_indices]
                peak_amps = P_env[peak_indices]

                # 3. 计算特征频率
                fault_freqs = bearing_freqs(fr, GEOM[sensor])
                
                # 4. 计算6个特征频率的距离和幅值
                dist_amp_vals = {}
                for name, f0 in fault_freqs.items():
                    for harmonic in [1, 2]:
                        h_name = f"{name}_{harmonic}x"
                        dist, amp = find_closest_peak(f0 * harmonic, peak_fs, peak_amps)
                        dist_amp_vals[f"{h_name}_dist"] = dist
                        dist_amp_vals[f"{h_name}_amp"] = amp

                # 5. 计算时/频/时频域特征
                raw_t_feats = time_features(x0)
                raw_f_feats = freq_features(x0, FS)
                raw_tf_feats = tf_features(x0, FS)
                env_t_feats = time_features(env)

                # 6. 记录
                row = {
                    "文件": in_path.name, "变量": v_name, "传感器": sensor,
                    "转速rpm": rpm, "转频Hz": fr,
                    "最优带通下限Hz": flo, "最优带通上限Hz": fhi, "最优带通峭度": k_bp,
                }
                row.update(dist_amp_vals)
                row.update({f"raw_{k}": v for k, v in raw_t_feats.items()})
                row.update({f"raw_{k}": v for k, v in raw_f_feats.items()})
                row.update({f"raw_{k}": v for k, v in raw_tf_feats.items()})
                row.update({f"env_{k}": v for k, v in env_t_feats.items()})
                rows.append([row.get(h, "") for h in header])

                # 7. 绘图
                plot_and_save_envelope(
                    f_env, P_env, peak_fs, peak_amps, fault_freqs,
                    rpm, in_path.name, v_name, sensor, PLOT_DIR
                )

                # 更新输出 .mat
                S_out[f"{v_name}_env_denoised"] = env
                S_out[f"{v_name}_env_spectrum_f"] = f_env
                S_out[f"{v_name}_env_spectrum_P"] = P_env

                # 7. 绘图
                plot_and_save_envelope(f_env, P_env, peak_fs, peak_amps, fault_freqs, rpm, in_path.stem, v_name, sensor, REPORT_DIR)

            savemat_safe(out_path, S_out)
            if (i + 1) % 10 == 0:
                print(f"... 已处理 {i+1}/{len(mats)}")

        except Exception as e:
            warnings.warn(f"处理失败: {in_path.name} ({type(e).__name__}: {e})")

    # 写报表
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ 完成! 共处理 {len(rows)} 条记录。")
    print(f"📄 报表: {csv_path}")
    print(f"📁 输出: {OUT_ROOT}")

if __name__ == "__main__":
    main()
