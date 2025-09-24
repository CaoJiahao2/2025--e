# -*- coding: utf-8 -*-
"""
Batch envelope denoise + report (GPU-optional)
- 输入根：/home/user1/data/learn/sumo + data/raw_resampled_12kHz
- 输出镜像：/home/user1/data/learn/sumo/data/peak-1-envelope_denoised_12kHz
- 报表：./reports/envelope_report_12kHz_YYYYmmdd_HHMMSS.csv
- 图：./reports/peak-1-包络谱图/*.png   （第4幅仅 0–200 Hz）
- 新增：最近峰唯一分配、6个新指标、三色标注BSF/BPFO/BPFI
"""
import os, re, sys, math, time, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks, spectrogram, medfilt
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

# -------------------- 可选：MAT v7.3 --------------------
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

# -------------------- 可选：GPU --------------------
USE_GPU = False
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.signal as cpx_signal
    from cupyx.scipy.fft import rfft, irfft  # 可能备用
    USE_GPU = True
except Exception:
    USE_GPU = False

# ───────── 配置 ─────────
BASE_DIR   = "/home/user1/data/learn/sumo"
IN_REL     = "data/raw_resampled_12kHz"
IN_ROOT    = os.path.join(BASE_DIR, IN_REL)
OUT_ROOT   = os.path.join(BASE_DIR, "data/peak-1-envelope_denoised_12kHz")
REPORT_DIR = os.path.abspath("./reports")
FIG_DIR    = os.path.join(REPORT_DIR, "peak-1-包络谱图")

FS   = 12000.0
NYQ  = FS/2
GUARD_HZ = 60.0

# 候选中心频率（Hz）与半带宽（~±0.5 kHz）
FC_LIST  = np.arange(1500.0, 5500.0+1e-6, 500.0)
BW_HALF  = 500.0

# 唯一分配：最小间隔与优先窗
ASSIGN_MIN_SEP_HZ = 2.0      # 可调：5.0 更强不重合
ASSIGN_WIN_HZ     = 20.0     # 优先在±20 Hz内分配

# 轴承几何
GEOM = {
    "DE": dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),
    "FE": dict(Nd=9, d=0.2656, D=1.122, thetaDeg=0.0),
    "BA": dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),
}

# 颜色映射（绘图）
COLOR_MAP = {"BSF":"tab:red", "BPFO":"tab:orange", "BPFI":"tab:green"}

# =======================================================
# 基础I/O
# =======================================================
def ensure_dirs():
    for d in [OUT_ROOT, REPORT_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

def list_mat_files(root):
    mats = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".mat"):
                mats.append(os.path.join(r, fn))
    return mats

def load_mat_any(path):
    # 先试 scipy (v7-)
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k:v for k,v in data.items() if not k.startswith("__")}
    except Exception:
        pass
    # 再试 h5py (v7.3)
    if not HAS_H5PY:
        raise RuntimeError("需要 h5py 读取 MAT v7.3：{}".format(path))
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                out[k] = np.array(obj)
    return out

def savemat_safe(path, dict_vars):
    clean = {}
    for k, v in dict_vars.items():
        if isinstance(v, (int, float, np.integer, np.floating, str, np.ndarray)):
            clean[k] = v
    sio.savemat(path, clean, do_compression=True)

# =======================================================
# 杂项
# =======================================================
def rpm_to_hz(rpm):
    return float(rpm)/60.0 if (rpm is not None and np.isfinite(rpm) and rpm>0) else np.nan

def parse_rpm_from_filename(fname):
    m = re.search(r"(\d{3,5})\s*rpm", fname, flags=re.I)
    if m:
        try: return float(m.group(1))
        except Exception: return np.nan
    return np.nan

def get_rpm_from_vars(d):
    for k in d.keys():
        if "rpm" in k.lower():
            try: return float(np.atleast_1d(d[k]).ravel()[0])
            except Exception: pass
    return np.nan

def sensor_from_vname(vname):
    v = vname.lower()
    if v.endswith("_de_time"): return "DE"
    if v.endswith("_fe_time"): return "FE"
    if v.endswith("_ba_time"): return "BA"
    return "DE"

def bearing_freqs(fr, g):
    if not (np.isfinite(fr) and fr > 0): return np.nan, np.nan, np.nan
    Nd, d, D, c = g["Nd"], g["d"], g["D"], math.cos(math.radians(g["thetaDeg"]))
    bpfo = fr*(Nd/2.0)*(1.0 - (d/D)*c)
    bpfi = fr*(Nd/2.0)*(1.0 + (d/D)*c)
    bsf  = fr*(D/(2.0*d))*(1.0 - ((d/D)*c)**2)
    return bpfo, bpfi, bsf

# =======================================================
# GPU/CPU 计算适配层
# =======================================================
def as_xp(a):
    return cp.asarray(a) if (USE_GPU and not isinstance(a, cp.ndarray)) else a

def to_numpy(a):
    return cp.asnumpy(a) if (USE_GPU and isinstance(a, cp.ndarray)) else np.asarray(a)

def butter_bp(lo, hi, fs, order=4):
    if USE_GPU:
        return cpx_signal.butter(order, [lo/(fs/2.0), hi/(fs/2.0)], btype="band")
    else:
        return butter(order, [lo/(fs/2.0), hi/(fs/2.0)], btype="band")

def filt_filt(b, a, x):
    if USE_GPU:
        return cpx_signal.filtfilt(b, a, x, method="gust")
    else:
        return filtfilt(b, a, x, method="gust")

def hilbert_env(x):
    if USE_GPU:
        hx = cupyx.scipy.signal.hilbert(x)
        return cp.abs(hx)
    else:
        return np.abs(hilbert(x))

def medfilt_1d(x, win):
    if USE_GPU:
        # cupyx 没有直接的 medfilt1d，这里退回CPU做中值滤波后再上GPU（影响可忽略）
        xx = to_numpy(x)
        if win % 2 == 0: win += 1
        y = medfilt(xx, kernel_size=win)
        return as_xp(y)
    else:
        if win % 2 == 0: win += 1
        return medfilt(x, kernel_size=win)

def welch_psd(x, fs, nperseg):
    if USE_GPU:
        f, P = cpx_signal.welch(x, fs=fs, nperseg=nperseg)
        return to_numpy(f), to_numpy(P)
    else:
        return welch(x, fs=fs, nperseg=nperseg)

def spectro(x, fs, wlen, olap, nfft):
    if USE_GPU:
        f, t, Sxx = cpx_signal.spectrogram(x, fs=fs, nperseg=wlen, noverlap=olap,
                                           nfft=nfft, scaling="spectrum", mode="magnitude")
        return to_numpy(f), to_numpy(t), to_numpy(Sxx)
    else:
        return spectrogram(x, fs=fs, nperseg=wlen, noverlap=olap,
                           nfft=nfft, scaling="spectrum", mode="magnitude")

# =======================================================
# 信号处理
# =======================================================
def select_band_by_kurtosis(x, fs, fc_list, bw_half, guard=60.0):
    x1 = np.atleast_1d(np.asarray(x, dtype=float))
    if x1.ndim>1: x1 = x1[:,0]
    best_k = -np.inf
    flo, fhi = 800.0, 1800.0
    for fc in fc_list:
        lo = max(guard, fc - bw_half)
        hi = min(fs/2.0 - guard, fc + bw_half)
        if hi <= lo + 10.0: 
            continue
        try:
            b,a = butter_bp(lo, hi, fs, order=4)
            xf = filt_filt(b, a, as_xp(x1.astype(float)))
            xf = to_numpy(xf)
            k  = kurtosis(xf, fisher=False, bias=False)
            if np.isfinite(k) and k > best_k:
                best_k, flo, fhi = k, lo, hi
        except Exception:
            continue
    return flo, fhi, best_k

def envelope_and_spectrum(x_bp, fs):
    # Hilbert + 中值平滑（~5 ms）
    env = hilbert_env(as_xp(x_bp))
    env = medfilt_1d(env, int(round(0.005*fs)) or 3)
    env = to_numpy(env)
    # Welch
    nfft = int(2**np.floor(np.log2(min(len(env), 16384))))
    if nfft < 256: nfft = 256
    f, P = welch_psd(env - np.mean(env), fs=fs, nperseg=max(256, nfft//4))
    return env, f, P

def list_env_peaks(f_env, P_env, fmax=200.0, prom_scale=3.0):
    mask = (f_env >= 0.0) & (f_env <= fmax)
    if not np.any(mask):
        return np.array([]), np.array([])
    ff, pp = f_env[mask], P_env[mask]
    prom = np.median(pp) * prom_scale
    if not (np.isfinite(prom) and prom > 0):
        prom = np.percentile(pp, 90)*0.1
    idx, _ = find_peaks(pp, prominence=prom)
    if idx.size == 0:
        k = min(8, len(pp))
        idx = np.argsort(pp)[-k:]
    return ff[idx], pp[idx]

def assign_unique_peaks(target_f, pk_f, pk_a, min_sep_hz, win_hz):
    n = len(target_f)
    f_out = [np.nan]*n
    a_out = [np.nan]*n
    if pk_f.size == 0: return f_out, a_out
    avail = np.ones(len(pk_f), dtype=bool)
    # 每个目标的候选
    cand = []
    for f0 in target_f:
        if not (np.isfinite(f0) and f0>0):
            cand.append([]); continue
        dist = np.abs(pk_f - f0)
        inwin = np.where(dist <= win_hz)[0]
        if inwin.size == 0:
            inwin = np.array([int(np.argmin(dist))])
        cand.append(list(inwin))
    # 贪心唯一匹配
    while True:
        best = None
        for i, idxs in enumerate(cand):
            if len(idxs)==0 or np.isfinite(f_out[i]): continue
            idxs2 = [j for j in idxs if avail[j]]
            if not idxs2: continue
            d2 = np.abs(pk_f[idxs2] - target_f[i])
            j  = idxs2[int(np.argmin(d2))]
            item = (np.abs(pk_f[j]-target_f[i]), i, j)
            if (best is None) or (item[0]<best[0]): best = item
        if best is None: break
        _, i, j = best
        f_out[i] = float(pk_f[j]); a_out[i] = float(pk_a[j])
        avail[np.abs(pk_f - pk_f[j]) <= min_sep_hz] = False
    return f_out, a_out

def amp_sbr_at(f, P, f0):
    if not (np.isfinite(f0) and f0>0): return np.nan, np.nan
    df = np.median(np.diff(f)); 
    if not (np.isfinite(df) and df>0): df = 0.5
    bw = max(3*df, 5.0)
    idx = (f >= max(0.0, f0-bw)) & (f <= f0+bw)
    if not np.any(idx): return np.nan, np.nan
    sub = np.where(idx)[0]
    ii  = sub[np.argmax(P[idx])]
    A   = float(P[ii])
    idxBg = (f >= max(0.0,f0-5*bw)) & (f <= f0+5*bw)
    idxBg[ii] = False
    B = np.median(P[idxBg]) if np.any(idxBg) else np.median(P)
    if not (np.isfinite(B) and B>0): B = np.median(P[P>0]) if np.any(P>0) else 1.0
    SBRdB = 20.0*np.log10(A/max(B, np.finfo(float).eps))
    return A, SBRdB

# =======================================================
# 特征
# =======================================================
def time_features(x):
    x = np.asarray(x).astype(float); N = len(x)
    if N==0: return {}
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
        wlen = 2**int(np.floor(np.log2(max(len(x),256))))
        wlen = max(wlen, 256)
    olap = wlen//2
    nfft = max(4096, 2**int(np.ceil(np.log2(wlen))))
    f, t, Sxx = spectro(x, fs, wlen, olap, nfft)
    P = (Sxx**2)
    if P.shape[1] >= 4:
        sk = kurtosis(P, axis=1, fisher=False, bias=False)
        kmax = float(np.nanmax(sk))
        f_at_kmax = float(f[np.nanargmax(sk)]) if np.any(np.isfinite(sk)) else np.nan
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
# 绘图
# =======================================================
def plot_envelope_figure(x0, x_bp, env, fs, f_env, P_env,
                         flo, fhi, fr, bpfo, bpfi, bsf,
                         fpk_bsf, Apk_bsf, fpk_bpfo, Apk_bpfo, fpk_bpfi, Apk_bpfi,
                         figPath):
    Tshow = min(1.0, len(x0)/fs); nshow = max(1, int(round(Tshow*fs)))
    t = np.arange(nshow)/fs

    plt.figure(figsize=(12,8))
    gs = plt.GridSpec(2,2, hspace=0.25, wspace=0.15)

    ax1 = plt.subplot(gs[0,0]); ax1.plot(t, x0[:nshow]); 
    ax1.set_title("原始信号（片段）"); ax1.set_xlabel("t (s)"); ax1.set_ylabel("x"); ax1.grid(True)

    ax2 = plt.subplot(gs[0,1]); ax2.plot(t, x_bp[:nshow]); 
    ax2.set_title(f"带通去噪（{flo:.0f}–{fhi:.0f} Hz）"); ax2.set_xlabel("t (s)"); ax2.set_ylabel("x_bp"); ax2.grid(True)

    ax3 = plt.subplot(gs[1,0]); ax3.plot(t, env[:nshow]); 
    ax3.set_title("包络（中值平滑后）"); ax3.set_xlabel("t (s)"); ax3.set_ylabel("|Hilbert|"); ax3.grid(True)

    ax4 = plt.subplot(gs[1,1])
    idx = (f_env>=0.0) & (f_env<=200.0)
    ax4.plot(f_env[idx], P_env[idx]); ax4.grid(True)
    ax4.set_title("包络谱（0–200 Hz）"); ax4.set_xlabel("f (Hz)"); ax4.set_ylabel("P_env")

    # fr 灰色
    if np.isfinite(fr) and 0.0 <= fr <= 200.0:
        ax4.axvline(fr, ls="--", color="0.5")
        ax4.text(fr, ax4.get_ylim()[1]*0.88, "fr", rotation=90, va="top", color="0.3")

    # 三条特征（不同颜色）+ 已分配峰点
    items = [("BSF", bsf,  fpk_bsf,  Apk_bsf),
             ("BPFO",bpfo, fpk_bpfo, Apk_bpfo),
             ("BPFI",bpfi, fpk_bpfi, Apk_bpfi)]
    for name, f0, fpk, Apk in items:
        cc = COLOR_MAP[name]
        if np.isfinite(f0) and 0.0 <= f0 <= 200.0:
            ax4.axvline(f0, ls="--", color=cc)
            ax4.text(f0, ax4.get_ylim()[1]*0.85, name, rotation=90, va="top", color=cc)
        if np.isfinite(fpk) and 0.0 <= fpk <= 200.0:
            ax4.plot([fpk], [Apk], marker="o", ms=6, color=cc)

    plt.tight_layout()
    plt.savefig(figPath, dpi=150)
    plt.close()

# =======================================================
# 主流程
# =======================================================
def main():
    if USE_GPU:
        print(">> Using GPU (CuPy) for signal processing.")
    else:
        print(">> Using CPU.")
    ensure_dirs()
    if not os.path.isdir(IN_ROOT):
        print(f"[ERROR] 未找到输入目录：{IN_ROOT}")
        sys.exit(1)

    mats = list_mat_files(IN_ROOT)
    if len(mats)==0:
        print("[WARN] 未发现 .mat 文件")
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_DIR, f"envelope_report_12kHz_{ts}.csv")

    # 表头（原有 + 新增6列，列名含中英文）
    hdr = [
        "文件路径/file_path","变量名/var_name","测点/sensor","样本数/n_samples","时长(s)/duration_s",
        "选带低(Hz)/band_lo_Hz","选带高(Hz)/band_hi_Hz","带内峭度/kurtosis_inband","去噪方法/denoise_method",
        "env峰1_Hz/peak1_Hz","env峰1_幅/peak1_amp","env峰2_Hz/peak2_Hz","env峰2_幅/peak2_amp","env峰3_Hz/peak3_Hz","env峰3_幅/peak3_amp",
        "fr_Hz/fr_Hz","BPFO_Hz/BPFO_Hz","BPFI_Hz/BPFI_Hz","BSF_Hz/BSF_Hz",
        "Amp@fr/Amp@fr","SBR@fr_dB/SBR@fr_dB","Amp@BPFO/Amp@BPFO","SBR@BPFO_dB/SBR@BPFO_dB",
        "Amp@BPFI/Amp@BPFI","SBR@BPFI_dB/SBR@BPFI_dB","Amp@BSF/Amp@BSF","SBR@BSF_dB/SBR@BSF_dB",
        # 时域
        "均值/mean","标准差/std","方差/variance","均方根/rms","最大值/max","最小值/min","峰峰值/peak_to_peak",
        "偏斜度/skewness","峭度/kurtosis","峰值指标/crest_factor","脉冲指标/impulse_factor","波形指标/shape_factor","裕度指标/margin_factor",
        # 频域
        "频谱质心(Hz)/f_centroid_Hz","谱带宽(Hz)/bandwidth_Hz","频谱熵/spec_entropy",
        # 时频
        "谱峭度最大值/sk_max","谱峭度对应频率(Hz)/sk_f_Hz","帧谱熵均值/stft_spec_entropy_mean","帧谱熵标准差/stft_spec_entropy_std","三分段能量CV/E_time3_cv",
        # 新增6列
        "BPFO-最近包络峰频差(Hz)/dF_env_BPFO_Hz",
        "BPFI-最近包络峰频差(Hz)/dF_env_BPFI_Hz",
        "BSF-最近包络峰频差(Hz)/dF_env_BSF_Hz",
        "邻峰/转频幅之比(BPFO)/peak_over_fr_BPFO",
        "邻峰/转频幅之比(BPFI)/peak_over_fr_BPFI",
        "邻峰/转频幅之比(BSF)/peak_over_fr_BSF",
        "输出文件/out_file"
    ]
    rows = []

    nOK = nFail = 0
    t0 = time.time()

    for idx, in_path in enumerate(mats, 1):
        rel = os.path.relpath(os.path.dirname(in_path), IN_ROOT)
        out_folder = os.path.join(OUT_ROOT, rel)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, os.path.basename(in_path))

        try:
            D = load_mat_any(in_path)
        except Exception as e:
            nFail += 1; print(f"[FAIL-READ] {in_path} -> {e}")
            continue

        keys = list(D.keys())
        sig_vars = [k for k in keys if k.lower().endswith(("_de_time","_fe_time","_ba_time"))]

        rpm = parse_rpm_from_filename(os.path.basename(in_path))
        if not np.isfinite(rpm): rpm = get_rpm_from_vars(D)
        fr = rpm_to_hz(rpm)

        out_dict = {}
        out_dict.update(D)
        if len(sig_vars)==0:
            try:
                savemat_safe(out_path, out_dict); nOK += 1
            except Exception as e:
                nFail += 1; print(f"[FAIL-SAVE] {in_path} -> {e}")
            continue

        changed = False

        for vname in sig_vars:
            x0 = np.asarray(D[vname]).astype(float)
            X  = x0.reshape(-1,1) if x0.ndim==1 else x0
            nSamp = X.shape[0]; durS = nSamp/FS; sensor = sensor_from_vname(vname)
            geom  = GEOM.get(sensor, GEOM["DE"])

            flo, fhi, kmax = select_band_by_kurtosis(X[:,0], FS, FC_LIST, BW_HALF, GUARD_HZ)
            b,a = butter_bp(flo, fhi, FS, order=4)
            denoise_method = f"bandpass[{flo:.0f},{fhi:.0f}]+hilbert+medfilt"

            if X.shape[1]==1:
                xb = filt_filt(b, a, as_xp(X[:,0]))
                xb = to_numpy(xb)
                env_s, f_env, P_env = envelope_and_spectrum(xb, FS)

                # 所有峰（0–200 Hz）
                pkF_all, pkA_all = list_env_peaks(f_env, P_env, fmax=200.0, prom_scale=3.0)
                df_env = np.median(np.diff(f_env)); 
                min_sep = max(ASSIGN_MIN_SEP_HZ, 2*df_env)
                f_assigned, a_assigned = assign_unique_peaks(
                    [GEOM_F for GEOM_F in (bearing_freqs(fr, geom)[2], bearing_freqs(fr, geom)[0], bearing_freqs(fr, geom)[1])],
                    pkF_all, pkA_all, min_sep, ASSIGN_WIN_HZ
                )
                # 注意：上面为了按 [BSF, BPFO, BPFI] 顺序匹配，我们重复调用 bearing_freqs，或单独先算：
                bpfo, bpfi, bsf = bearing_freqs(fr, geom)  # 再显式获得
                # 重新以明确顺序赋值
                fpk_bsf, fpk_bpfo, fpk_bpfi = f_assigned
                Apk_bsf, Apk_bpfo, Apk_bpfi = a_assigned

                # 峰表（用于报CSV前三峰：仍用0–200Hz前三大，与你之前一致）
                # 这里复用 pkF_all/pkA_all 排序（从大到小取前三）
                if pkA_all.size:
                    ord3 = np.argsort(pkA_all)[::-1][:3]
                    topF = pkF_all[ord3]; topA = pkA_all[ord3]
                else:
                    topF = np.array([np.nan, np.nan, np.nan]); topA = np.array([np.nan, np.nan, np.nan])

                # 在 fr/BPFO/BPFI/BSF 处的幅值与SBR
                A_fr, S_fr = amp_sbr_at(f_env, P_env, fr)
                A_bo, S_bo = amp_sbr_at(f_env, P_env, bpfo)
                A_bi, S_bi = amp_sbr_at(f_env, P_env, bpfi)
                A_bs, S_bs = amp_sbr_at(f_env, P_env, bsf)

                # 6个新增指标
                def _absdiff(a,b): return float(abs(a-b)) if (np.isfinite(a) and np.isfinite(b)) else np.nan
                def _ratio(a,b):   return float(a/b) if (np.isfinite(a) and np.isfinite(b) and b>0) else np.nan
                dF_env_BPFO = _absdiff(fpk_bpfo, bpfo)
                dF_env_BPFI = _absdiff(fpk_bpfi, bpfi)
                dF_env_BSF  = _absdiff(fpk_bsf , bsf )
                R_env_BPFO_over_fr = _ratio(Apk_bpfo, A_fr)
                R_env_BPFI_over_fr = _ratio(Apk_bpfi, A_fr)
                R_env_BSF_over_fr  = _ratio(Apk_bsf , A_fr)

                # 保存新变量
                out_dict[vname + "_denoised"]    = xb.reshape(x0.shape)
                out_dict[vname + "_env"]         = env_s.reshape(x0.shape).astype(float)
                out_dict[vname + "_env_band_Hz"] = np.array([flo, fhi], dtype=float)
                changed = True

                # 绘图
                fig_png = os.path.join(FIG_DIR, f"{os.path.splitext(os.path.basename(in_path))[0]}_{vname}.png")
                try:
                    plot_envelope_figure(x0, xb, env_s, FS, f_env, P_env,
                                         flo, fhi, fr, bpfo, bpfi, bsf,
                                         fpk_bsf, Apk_bsf, fpk_bpfo, Apk_bpfo, fpk_bpfi, Apk_bpfi,
                                         fig_png)
                except Exception as e:
                    print(f"[WARN-PLOT] {fig_png} -> {e}")

                # 其它特征
                feats_time = time_features(x0)
                feats_freq = freq_features(x0, FS)
                feats_tf   = tf_features(x0, FS)

                row = [
                    in_path, vname, sensor, nSamp, durS,
                    flo, fhi, kmax, denoise_method,
                    _nz(topF,0), _nz(topA,0), _nz(topF,1), _nz(topA,1), _nz(topF,2), _nz(topA,2),
                    fr, bpfo, bpfi, bsf,
                    A_fr, S_fr, A_bo, S_bo, A_bi, S_bi, A_bs, S_bs,
                    feats_time.get("mean"), feats_time.get("std"), feats_time.get("var"), feats_time.get("rms"),
                    feats_time.get("max"), feats_time.get("min"), feats_time.get("p2p"),
                    feats_time.get("skew"), feats_time.get("kurt"), feats_time.get("crest"),
                    feats_time.get("impulse"), feats_time.get("shape"), feats_time.get("clearance"),
                    feats_freq.get("psd_centroid"), feats_freq.get("psd_bandwidth"), feats_freq.get("psd_entropy"),
                    feats_tf.get("sk_max"), feats_tf.get("sk_f_hz"), feats_tf.get("tf_entropy_mean"),
                    feats_tf.get("tf_entropy_std"), feats_tf.get("tf_energy3_cv"),
                    dF_env_BPFO, dF_env_BPFI, dF_env_BSF,
                    R_env_BPFO_over_fr, R_env_BPFI_over_fr, R_env_BSF_over_fr,
                    out_path
                ]
                rows.append(row)

            else:
                # 多列：逐列滤波/包络；峰/图按第1列
                Xbp = np.zeros_like(X, dtype=float)
                ENV = np.zeros_like(X, dtype=float)
                for c in range(X.shape[1]):
                    xb = filt_filt(b, a, as_xp(X[:,c])); Xbp[:,c] = to_numpy(xb)
                    env_s, f_env, P_env = envelope_and_spectrum(Xbp[:,c], FS)
                    ENV[:,c] = env_s
                out_dict[vname + "_denoised"]    = Xbp.reshape(x0.shape)
                out_dict[vname + "_env"]         = ENV.reshape(x0.shape).astype(float)
                out_dict[vname + "_env_band_Hz"] = np.array([flo, fhi], dtype=float)
                changed = True

                # 用第1列做指标
                bpfo, bpfi, bsf = bearing_freqs(fr, geom)
                pkF_all, pkA_all = list_env_peaks(f_env, P_env, fmax=200.0, prom_scale=3.0)
                df_env = np.median(np.diff(f_env)); 
                min_sep = max(ASSIGN_MIN_SEP_HZ, 2*df_env)
                f_assigned, a_assigned = assign_unique_peaks([bsf, bpfo, bpfi],
                                                             pkF_all, pkA_all, min_sep, ASSIGN_WIN_HZ)
                fpk_bsf, fpk_bpfo, fpk_bpfi = f_assigned
                Apk_bsf, Apk_bpfo, Apk_bpfi = a_assigned

                if pkA_all.size:
                    ord3 = np.argsort(pkA_all)[::-1][:3]
                    topF = pkF_all[ord3]; topA = pkA_all[ord3]
                else:
                    topF = np.array([np.nan, np.nan, np.nan]); topA = np.array([np.nan, np.nan, np.nan])

                A_fr, S_fr = amp_sbr_at(f_env, P_env, fr)
                A_bo, S_bo = amp_sbr_at(f_env, P_env, bpfo)
                A_bi, S_bi = amp_sbr_at(f_env, P_env, bpfi)
                A_bs, S_bs = amp_sbr_at(f_env, P_env, bsf)

                def _absdiff(a,b): return float(abs(a-b)) if (np.isfinite(a) and np.isfinite(b)) else np.nan
                def _ratio(a,b):   return float(a/b) if (np.isfinite(a) and np.isfinite(b) and b>0) else np.nan
                dF_env_BPFO = _absdiff(fpk_bpfo, bpfo)
                dF_env_BPFI = _absdiff(fpk_bpfi, bpfi)
                dF_env_BSF  = _absdiff(fpk_bsf , bsf )
                R_env_BPFO_over_fr = _ratio(Apk_bpfo, A_fr)
                R_env_BPFI_over_fr = _ratio(Apk_bpfi, A_fr)
                R_env_BSF_over_fr  = _ratio(Apk_bsf , A_fr)

                fig_png = os.path.join(FIG_DIR, f"{os.path.splitext(os.path.basename(in_path))[0]}_{vname}.png")
                try:
                    plot_envelope_figure(X[:,0], Xbp[:,0], ENV[:,0], FS, f_env, P_env,
                                         flo, fhi, fr, bpfo, bpfi, bsf,
                                         fpk_bsf, Apk_bsf, fpk_bpfo, Apk_bpfo, fpk_bpfi, Apk_bpfi,
                                         fig_png)
                except Exception as e:
                    print(f"[WARN-PLOT] {fig_png} -> {e}")

                feats_time = time_features(X[:,0])
                feats_freq = freq_features(X[:,0], FS)
                feats_tf   = tf_features(X[:,0], FS)

                row = [
                    in_path, vname, sensor, nSamp, durS,
                    flo, fhi, kmax, denoise_method,
                    _nz(topF,0), _nz(topA,0), _nz(topF,1), _nz(topA,1), _nz(topF,2), _nz(topA,2),
                    fr, bpfo, bpfi, bsf,
                    A_fr, S_fr, A_bo, S_bo, A_bi, S_bi, A_bs, S_bs,
                    feats_time.get("mean"), feats_time.get("std"), feats_time.get("var"), feats_time.get("rms"),
                    feats_time.get("max"), feats_time.get("min"), feats_time.get("p2p"),
                    feats_time.get("skew"), feats_time.get("kurt"), feats_time.get("crest"),
                    feats_time.get("impulse"), feats_time.get("shape"), feats_time.get("clearance"),
                    feats_freq.get("psd_centroid"), feats_freq.get("psd_bandwidth"), feats_freq.get("psd_entropy"),
                    feats_tf.get("sk_max"), feats_tf.get("sk_f_hz"), feats_tf.get("tf_entropy_mean"),
                    feats_tf.get("tf_entropy_std"), feats_tf.get("tf_energy3_cv"),
                    dF_env_BPFO, dF_env_BPFI, dF_env_BSF,
                    R_env_BPFO_over_fr, R_env_BPFI_over_fr, R_env_BSF_over_fr,
                    out_path
                ]
                rows.append(row)

        # 保存镜像
        try:
            savemat_safe(out_path, out_dict if changed else D); nOK += 1
        except Exception as e:
            nFail += 1; print(f"[FAIL-SAVE] {in_path} -> {e}")

        if idx % 20 == 0:
            print(f"... 已处理 {idx}/{len(mats)}")

    # 写报表（UTF-8 BOM）
    df = pd.DataFrame(rows, columns=hdr)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    dt = time.time()-t0
    print(f"✅ 完成：成功 {nOK}，失败 {nFail} | 用时 {dt:.1f}s")
    print(f"📄 报表：{csv_path}")
    print(f"📁 输出：{OUT_ROOT}")
    print(f"🖼️ 图像：{FIG_DIR}")

def _nz(arr, i):
    try: return float(arr[i])
    except Exception: return np.nan

if __name__ == "__main__":
    main()
