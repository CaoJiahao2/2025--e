# -*- coding: utf-8 -*-
"""
Batch envelope denoise + report (Python版)
- 输入根：/home/user1/data/learn/sumo + data/raw_resampled_12kHz/
- 输出镜像：/home/user1/data/learn/sumo/data/envelope_denoised_12kHz/
- 报表：./reports/envelope_report_12kHz_YYYYmmdd_HHMMSS.csv
- 图：./reports/包络谱图/*.png   （第4幅仅画0–200 Hz）
"""
import os, re, sys, math, traceback, shutil, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, welch, find_peaks, spectrogram, medfilt
from scipy.stats import skew, kurtosis
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

try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

# ───────── 配置 ─────────
BASE_DIR   = "/home/user1/data/learn/sumo"
IN_REL     = "data/raw_resampled_12kHz"
IN_ROOT    = os.path.join(BASE_DIR, IN_REL)
OUT_ROOT   = os.path.join(BASE_DIR, "data/peak-envelope_denoised_12kHz")
REPORT_DIR = os.path.abspath("./reports")
FIG_DIR    = os.path.join(REPORT_DIR, "peak-包络谱图")

FS   = 12000.0
NYQ  = FS/2
GUARD_HZ = 60.0

# 候选中心频率（Hz）与半带宽（~±0.5 kHz）
FC_LIST  = np.arange(1500.0, 5500.0+1e-6, 500.0)
BW_HALF  = 500.0

# 轴承几何（与题意一致）
GEOM = {
    "DE": dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),  # SKF6205
    "FE": dict(Nd=9, d=0.2656, D=1.122, thetaDeg=0.0),  # SKF6203
    "BA": dict(Nd=9, d=0.3126, D=1.537, thetaDeg=0.0),  # 近似按DE
}

# ───────── 工具函数 ─────────
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
    """
    加载 .mat：
    - 先试 scipy.io.loadmat（v7 及以下）
    - 若失败再试 h5py（v7.3）
    返回：dict（仅保留基础数值数组）
    """
    # 1) scipy 读取
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        # 清理mat内置键
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data
    except Exception:
        pass

    # 2) h5py 读取（v7.3）
    if not HAS_H5PY:
        raise RuntimeError("无法读取MAT v7.3（缺少h5py）。")
    out = {}
    with h5py.File(path, "r") as f:
        def read_obj(o):
            if isinstance(o, h5py.Dataset):
                arr = np.array(o)
                # 去掉冗余维
                return np.array(arr)
            elif isinstance(o, h5py.Group):
                # 不递归复杂对象，这里仅返回None
                return None
            else:
                return None
        for k in f.keys():
            v = read_obj(f[k])
            if v is not None:
                out[k] = v
    return out

def savemat_safe(path, dict_vars):
    # 仅保存可序列化为数值数组/标量/字符串的键
    clean = {}
    for k, v in dict_vars.items():
        if isinstance(v, (int, float, np.integer, np.floating, str, np.ndarray)):
            clean[k] = v
    sio.savemat(path, clean, do_compression=True)

def sensor_from_vname(vname):
    v = vname.lower()
    if v.endswith("_de_time"): return "DE"
    if v.endswith("_fe_time"): return "FE"
    if v.endswith("_ba_time"): return "BA"
    return "DE"

def rpm_to_hz(rpm):
    return float(rpm)/60.0 if (rpm is not None and np.isfinite(rpm) and rpm>0) else np.nan

def parse_rpm_from_filename(fname):
    m = re.search(r"(\d{3,5})\s*rpm", fname, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan

def get_rpm_from_vars(d):
    # 任一包含"RPM"的变量名都尝试
    for k in d.keys():
        if "rpm" in k.lower():
            v = d[k]
            try:
                v = float(np.atleast_1d(v).ravel()[0])
                return v
            except Exception:
                continue
    return np.nan

def butter_bp(lo, hi, fs, order=4):
    return butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")

def filt_filt(b, a, x):
    return filtfilt(b, a, x, method="gust")

def select_band_by_kurtosis(x, fs, fc_list, bw_half, guard=60.0):
    x1 = np.atleast_1d(x)
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
            xf = filt_filt(b,a, x1.astype(float))
            k  = kurtosis(xf, fisher=False, bias=False)
            if np.isfinite(k) and k > best_k:
                best_k = k
                flo, fhi = lo, hi
        except Exception:
            continue
    return flo, fhi, best_k

def envelope_and_spectrum(x_bp, fs):
    # Hilbert 包络
    env = np.abs(hilbert(x_bp))
    # 移动中值平滑（~5 ms 窗）：窗口点数需奇数
    win = max(3, int(round(0.005*fs)))
    if win % 2 == 0: win += 1
    env_s = medfilt(env, kernel_size=win)
    # Welch 包络谱
    nfft = int(2**np.floor(np.log2(min(len(env_s), 16384))))
    if nfft < 256:
        nfft = 256
    f, P = welch(env_s - np.mean(env_s), fs=fs, nperseg=max(256, nfft//4))
    return env_s, f, P

def amp_sbr_at(f, P, f0):
    """在 f0±bw 内找峰：返回幅值A与SBR(dB)"""
    if not (np.isfinite(f0) and f0>0): 
        return np.nan, np.nan
    df = np.median(np.diff(f))
    if not (np.isfinite(df) and df>0): df = 0.5
    bw = max(3*df, 5.0)
    idx = (f >= max(0.0, f0-bw)) & (f <= f0+bw)
    if not np.any(idx):
        return np.nan, np.nan
    loc = np.argmax(P[idx])
    A   = float(P[idx][loc])
    # 背景：去掉核心±bw，扩大到±5*bw
    idx_bg = (f >= max(0.0, f0-5*bw)) & (f <= f0+5*bw)
    idx_bg[np.where(idx)[0][loc]] = False
    B = np.median(P[idx_bg]) if np.any(idx_bg) else np.median(P)
    if not (np.isfinite(B) and B>0): B = np.median(P[P>0]) if np.any(P>0) else 1.0
    SBRdB = 20.0*np.log10(A / max(B, np.finfo(float).eps))
    return A, SBRdB

def bearing_freqs(fr, g):
    """fr[Hz]; g={Nd,d,D,thetaDeg}"""
    if not (np.isfinite(fr) and fr>0):
        return np.nan, np.nan, np.nan
    Nd = g["Nd"]; d = g["d"]; D = g["D"]; c = math.cos(math.radians(g["thetaDeg"]))
    bpfo = fr*(Nd/2.0)*(1.0 - (d/D)*c)
    bpfi = fr*(Nd/2.0)*(1.0 + (d/D)*c)
    bsf  = fr*(D/(2.0*d))*(1.0 - ((d/D)*c)**2)
    return bpfo, bpfi, bsf

def top_env_peaks(f, P, N=3, fmax=1000.0):
    idx = (f>=0.0) & (f<=fmax)
    if not np.any(idx):
        return [np.nan]*N, [np.nan]*N
    ff, pp = f[idx], P[idx]
    # 用 prominence 自适应
    prom = np.median(pp)*3.0
    if not (np.isfinite(prom) and prom>0):
        prom = np.percentile(pp, 90)*0.1
    locs, props = find_peaks(pp, prominence=prom, distance=max(1, int(round(2.0/np.median(np.diff(ff))))))
    if len(locs)==0:
        # 回退：直接取前三大
        ord_idx = np.argsort(pp)[::-1][:N]
        return list(ff[ord_idx]), list(pp[ord_idx])
    amps = pp[locs]
    order = np.argsort(amps)[::-1]
    locs = locs[order][:N]
    return list(ff[locs]), list(pp[locs])

def plot_envelope_figure(x0, x_bp, env, fs, f_env, P_env,
                         flo, fhi, fr, bpfo, bpfi, bsf, figPath):
    Tshow = min(1.0, len(x0)/fs)
    nshow = max(1, int(round(Tshow*fs)))
    t = np.arange(nshow)/fs

    # 事先算最近峰（仅用于画图点）
    fpk_bpfo, Apk_bpfo = nearest_env_peak(f_env, P_env, bpfo, fmax=200.0)
    fpk_bpfi, Apk_bpfi = nearest_env_peak(f_env, P_env, bpfi, fmax=200.0)
    fpk_bsf , Apk_bsf  = nearest_env_peak(f_env, P_env, bsf , fmax=200.0)

    # 颜色（按要求三条线区分）
    color_map = {"BPFO": "tab:orange", "BPFI": "tab:green", "BSF": "tab:red"}

    plt.figure(figsize=(12,8))
    gs = plt.GridSpec(2,2, hspace=0.25, wspace=0.15)

    ax1 = plt.subplot(gs[0,0]); ax1.plot(t, x0[:nshow])
    ax1.set_title("原始信号（片段）"); ax1.set_xlabel("t (s)"); ax1.set_ylabel("x"); ax1.grid(True)

    ax2 = plt.subplot(gs[0,1]); ax2.plot(t, x_bp[:nshow])
    ax2.set_title(f"带通去噪（{flo:.0f}–{fhi:.0f} Hz）"); ax2.set_xlabel("t (s)"); ax2.set_ylabel("x_bp"); ax2.grid(True)

    ax3 = plt.subplot(gs[1,0]); ax3.plot(t, env[:nshow])
    ax3.set_title("包络（中值平滑后）"); ax3.set_xlabel("t (s)"); ax3.set_ylabel("|Hilbert|"); ax3.grid(True)

    ax4 = plt.subplot(gs[1,1])
    idx = (f_env>=0.0) & (f_env<=200.0)
    ax4.plot(f_env[idx], P_env[idx]); ax4.grid(True)
    ax4.set_title("包络谱（0–200 Hz）"); ax4.set_xlabel("f (Hz)"); ax4.set_ylabel("P_env")

    # fr 用灰色虚线
    if np.isfinite(fr) and 0.0 <= fr <= 200.0:
        ax4.axvline(fr, ls="--", color="0.5")
        ax4.text(fr, ax4.get_ylim()[1]*0.88, "fr", rotation=90, va="top", color="0.3")

    # 三条特征频率+最近峰（不同颜色）
    marks = [
        ("BSF", bsf,  fpk_bsf,  Apk_bsf,  color_map["BSF"]),
        ("BPFO",bpfo, fpk_bpfo, Apk_bpfo, color_map["BPFO"]),
        ("BPFI",bpfi, fpk_bpfi, Apk_bpfi, color_map["BPFI"]),
    ]
    for name, f0, fpk, Apk, cc in marks:
        if np.isfinite(f0) and 0.0 <= f0 <= 200.0:
            ax4.axvline(f0, ls="--", color=cc)
            ax4.text(f0, ax4.get_ylim()[1]*0.85, name, rotation=90, va="top", color=cc)
        if np.isfinite(fpk) and 0.0 <= fpk <= 200.0:
            ax4.plot([fpk], [Apk], marker="o", ms=6, color=cc)

    plt.tight_layout()
    plt.savefig(figPath, dpi=150)
    plt.close()


# ── 时域/频域/时频域特征 ─────────────────────────
def time_features(x):
    x = np.asarray(x).astype(float)
    N = len(x)
    if N==0: 
        return dict()
    mean_ = float(np.mean(x))
    std_  = float(np.std(x, ddof=1)) if N>1 else 0.0
    var_  = float(np.var(x, ddof=1)) if N>1 else 0.0
    rms   = float(np.sqrt(np.mean(x**2)))
    max_  = float(np.max(x))
    min_  = float(np.min(x))
    p2p   = max_ - min_
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
    """对原始信号做功率谱，提取一些通用频域统计"""
    x = np.asarray(x).astype(float)
    if len(x)<256:
        nper = max(64, 2**int(np.ceil(np.log2(len(x)))))
    else:
        nper = 1024
    f, P = welch(x - np.mean(x), fs=fs, nperseg=nper)
    P = np.maximum(P, 0.0)
    # 频谱质心、带宽、谱熵
    Pn = P/np.sum(P) if np.sum(P)>0 else P
    centroid = float(np.sum(f*Pn))
    bw = float(np.sqrt(np.sum(((f-centroid)**2)*Pn)))
    spec_entropy = float(-np.sum(Pn*np.log(Pn+1e-12))/np.log(len(Pn))) if len(Pn)>1 else np.nan
    return dict(psd_centroid=centroid, psd_bandwidth=bw, psd_entropy=spec_entropy)

def tf_features(x, fs):
    """
    STFT特征：
    - 谱峭度最大值及对应频率
    - 帧谱熵（对每帧归一化）的均值/标准差
    - 时间三分段能量比的变异系数（CV）
    """
    x = np.asarray(x).astype(float)
    wlen = 2048
    if wlen > len(x):
        wlen = 2**int(np.floor(np.log2(max(len(x),256))))
        wlen = max(wlen, 256)
    olap = wlen//2
    nfft = max(4096, 2**int(np.ceil(np.log2(wlen))))
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=wlen, noverlap=olap, nfft=nfft, scaling="spectrum", mode="magnitude")
    P = (Sxx**2)  # 功率
    # 谱峭度（沿时间对每个频率序列求峭度）
    if P.shape[1] >= 4:
        sk = kurtosis(P, axis=1, fisher=False, bias=False)
        kmax = float(np.nanmax(sk))
        f_at_kmax = float(f[np.nanargmax(sk)]) if np.any(np.isfinite(sk)) else np.nan
    else:
        kmax, f_at_kmax = np.nan, np.nan
    # 帧谱熵
    Psum = np.sum(P, axis=0, keepdims=True)
    Pn = P/(Psum + 1e-12)
    frame_entropy = -np.sum(Pn*np.log(Pn+1e-12), axis=0)/np.log(P.shape[0]) if P.shape[0]>1 else np.nan
    ent_mean = float(np.nanmean(frame_entropy)) if np.ndim(frame_entropy)>0 else np.nan
    ent_std  = float(np.nanstd(frame_entropy))  if np.ndim(frame_entropy)>0 else np.nan
    # 时间三分段能量
    N = len(x)
    thirds = np.array_split(np.arange(N), 3)
    energies = [float(np.sum(x[idx]**2)) for idx in thirds]
    cv_energy = float(np.std(energies, ddof=1)/np.mean(energies)) if np.mean(energies)>0 and len(energies)>=2 else np.nan
    return dict(sk_max=kmax, sk_f_hz=f_at_kmax, tf_entropy_mean=ent_mean, tf_entropy_std=ent_std, tf_energy3_cv=cv_energy)


def nearest_env_peak(f_env, P_env, f0, fmax=200.0):
    """
    在 0~fmax Hz 内找到距离 f0 最近的局部峰。
    返回：f_pk, A_pk（峰频率与幅值）。若没有可用峰，回退到最近频点。
    """
    if not (np.isfinite(f0) and f0 > 0):
        return np.nan, np.nan
    mask = (f_env >= 0.0) & (f_env <= fmax)
    if not np.any(mask):
        return np.nan, np.nan
    ff = f_env[mask]
    pp = P_env[mask]
    # 自适应prominence
    prom = np.median(pp) * 3.0
    if not (np.isfinite(prom) and prom > 0):
        prom = np.percentile(pp, 90) * 0.1
    peaks, _ = find_peaks(pp, prominence=prom)  # 可能为空
    if peaks.size == 0:
        # 回退：用最近bin
        i = int(np.argmin(np.abs(ff - f0)))
        return float(ff[i]), float(pp[i])
    # 选距离f0最近的峰
    i = int(np.argmin(np.abs(ff[peaks] - f0)))
    pidx = peaks[i]
    return float(ff[pidx]), float(pp[pidx])

# ───────── 主流程 ─────────
def main():
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

    # 报表表头（在原Matlab基础上，补充时/频/时频域指标）
    hdr = [
        "文件路径","变量名","测点","样本数","时长(s)",
        "选带低(Hz)","选带高(Hz)","带内峭度","去噪方法",
        "env峰1_Hz","env峰1_幅","env峰2_Hz","env峰2_幅","env峰3_Hz","env峰3_幅",
        "fr_Hz","BPFO_Hz","BPFI_Hz","BSF_Hz",
        "Amp@fr","SBR@fr_dB","Amp@BPFO","SBR@BPFO_dB","Amp@BPFI","SBR@BPFI_dB","Amp@BSF","SBR@BSF_dB",
        # 时域
        "mean","std","var","rms","max","min","p2p","skew","kurt","crest","impulse","shape","clearance",
        # 频域
        "psd_centroid","psd_bandwidth","psd_entropy",
        # 时频域
        "sk_max","sk_f_hz","tf_entropy_mean","tf_entropy_std","tf_energy3_cv",
        # 新增：三大特征频率到最近包络峰的频差 + 峰幅/转频幅 比值
        "dF_env_BPFO_Hz/BPFO-最近包络峰频差(Hz)",
        "dF_env_BPFI_Hz/BPFI-最近包络峰频差(Hz)",
        "dF_env_BSF_Hz/BSF-最近包络峰频差(Hz)",
        "peak_over_fr_BPFO/邻峰幅与fr幅之比(BPFO)",
        "peak_over_fr_BPFI/邻峰幅与fr幅之比(BPFI)",
        "peak_over_fr_BSF/邻峰幅与fr幅之比(BSF)",
        "输出文件"
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
            nFail += 1
            print(f"[FAIL-READ] {in_path} -> {e}")
            continue

        # 找到时域变量
        keys = list(D.keys())
        sig_vars = [k for k in keys if k.lower().endswith(("_de_time","_fe_time","_ba_time"))]

        # 解析rpm
        rpm = parse_rpm_from_filename(os.path.basename(in_path))
        if not np.isfinite(rpm):
            rpm = get_rpm_from_vars(D)
        fr = rpm_to_hz(rpm)

        # 输出字典（尽量“镜像+新增”）
        out_dict = {}
        out_dict.update(D)  # 把能序列化的原变量带上，非数值对象会在保存时过滤

        if len(sig_vars)==0:
            # 无时域信号，也复制一份（保持镜像）
            try:
                savemat_safe(out_path, out_dict)
                nOK += 1
            except Exception as e:
                nFail += 1
                print(f"[FAIL-SAVE] {in_path} -> {e}")
            continue

        changed = False

        for vname in sig_vars:
            x0 = np.asarray(D[vname]).astype(float)
            if x0.ndim == 1:
                X = x0.reshape(-1,1)
            else:
                X = x0
            nSamp = X.shape[0]
            durS  = nSamp/FS
            sensor = sensor_from_vname(vname)
            geom   = GEOM.get(sensor, GEOM["DE"])

            # 自动选带（用第一列代表）
            flo, fhi, kmax = select_band_by_kurtosis(X[:,0], FS, FC_LIST, BW_HALF, GUARD_HZ)
            b,a = butter_bp(flo, fhi, FS, order=4)
            denoise_method = f"bandpass[{flo:.0f},{fhi:.0f}]+hilbert+medfilt"

            if X.shape[1]==1:
                x_bp = filt_filt(b, a, X[:,0])
                env_s, f_env, P_env = envelope_and_spectrum(x_bp, FS)

                # 特征频率与SBR
                bpfo, bpfi, bsf = bearing_freqs(fr, geom)
                A_fr, S_fr = amp_sbr_at(f_env, P_env, fr)
                A_bo, S_bo = amp_sbr_at(f_env, P_env, bpfo)
                A_bi, S_bi = amp_sbr_at(f_env, P_env, bpfi)
                A_bs, S_bs = amp_sbr_at(f_env, P_env, bsf)

                # 峰
                pkF, pkA = top_env_peaks(f_env, P_env, N=3, fmax=200.0)  # 峰表基于0–200Hz
                
                # —— 新增：三大特征频率到“最近包络峰”的频差 + 峰幅/转频幅 比值 ——
                fpk_bpfo, Apk_bpfo = nearest_env_peak(f_env, P_env, bpfo, fmax=200.0)
                fpk_bpf_i, Apk_bpf_i = nearest_env_peak(f_env, P_env, bpfi, fmax=200.0)
                fpk_bsf , Apk_bsf  = nearest_env_peak(f_env, P_env, bsf , fmax=200.0)

                # 频差取绝对值（单位Hz）；比值=邻峰幅 / A_fr
                def _abs_diff(a, b):
                    return float(abs(a - b)) if (np.isfinite(a) and np.isfinite(b)) else np.nan
                def _ratio(apk, afr):
                    return float(apk/afr) if (np.isfinite(apk) and np.isfinite(afr) and afr>0) else np.nan

                dF_env_BPFO = _abs_diff(fpk_bpfo, bpfo)
                dF_env_BPFI = _abs_diff(fpk_bpf_i, bpfi)
                dF_env_BSF  = _abs_diff(fpk_bsf , bsf )

                R_env_BPFO_over_fr = _ratio(Apk_bpfo, A_fr)
                R_env_BPFI_over_fr = _ratio(Apk_bpf_i, A_fr)
                R_env_BSF_over_fr  = _ratio(Apk_bsf , A_fr)


                # 保存新变量（尽量保持原形状）
                out_dict[vname + "_denoised"]     = x_bp.reshape(x0.shape)
                out_dict[vname + "_env"]          = env_s.reshape(x0.shape).astype(float)
                out_dict[vname + "_env_band_Hz"]  = np.array([flo, fhi], dtype=float)
                changed = True

                # 可视化
                fig_png = os.path.join(FIG_DIR, f"{os.path.splitext(os.path.basename(in_path))[0]}_{vname}.png")
                try:
                    plot_envelope_figure(x0, x_bp, env_s, FS, f_env, P_env, flo, fhi, fr, bpfo, bpfi, bsf, fig_png)
                except Exception as e:
                    print(f"[WARN-PLOT] {fig_png} -> {e}")

                # 特征（时域/频域/时频域）
                feats_time = time_features(x0)
                feats_freq = freq_features(x0, FS)
                feats_tf   = tf_features(x0, FS)

                row = [
                    in_path, vname, sensor, nSamp, durS,
                    flo, fhi, kmax, denoise_method,
                    _nz(pkF,0), _nz(pkA,0), _nz(pkF,1), _nz(pkA,1), _nz(pkF,2), _nz(pkA,2),
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
                # 多列：逐列带通与包络，谱/图按第1列展示
                Xbp = np.zeros_like(X, dtype=float)
                ENV = np.zeros_like(X, dtype=float)
                for c in range(X.shape[1]):
                    xbpc = filt_filt(b,a,X[:,c])
                    envc, f_env, P_env = envelope_and_spectrum(xbpc, FS)
                    Xbp[:,c] = xbpc
                    ENV[:,c] = envc
                out_dict[vname + "_denoised"]     = Xbp.reshape(x0.shape)
                out_dict[vname + "_env"]          = ENV.reshape(x0.shape).astype(float)
                out_dict[vname + "_env_band_Hz"]  = np.array([flo, fhi], dtype=float)
                changed = True

                # 第1列用于指标与图
                bpfo, bpfi, bsf = bearing_freqs(fr, geom)
                A_fr, S_fr = amp_sbr_at(f_env, P_env, fr)
                A_bo, S_bo = amp_sbr_at(f_env, P_env, bpfo)
                A_bi, S_bi = amp_sbr_at(f_env, P_env, bpfi)
                A_bs, S_bs = amp_sbr_at(f_env, P_env, bsf)
                pkF, pkA = top_env_peaks(f_env, P_env, N=3, fmax=200.0)

                fig_png = os.path.join(FIG_DIR, f"{os.path.splitext(os.path.basename(in_path))[0]}_{vname}.png")
                try:
                    plot_envelope_figure(X[:,0], Xbp[:,0], ENV[:,0], FS, f_env, P_env, flo, fhi, fr, bpfo, bpfi, bsf, fig_png)
                except Exception as e:
                    print(f"[WARN-PLOT] {fig_png} -> {e}")

                feats_time = time_features(X[:,0])
                feats_freq = freq_features(X[:,0], FS)
                feats_tf   = tf_features(X[:,0], FS)

                row = [
                    in_path, vname, sensor, nSamp, durS,
                    flo, fhi, kmax, denoise_method,
                    _nz(pkF,0), _nz(pkA,0), _nz(pkF,1), _nz(pkA,1), _nz(pkF,2), _nz(pkA,2),
                    fr, bpfo, bpfi, bsf,
                    A_fr, S_fr, A_bo, S_bo, A_bi, S_bi, A_bs, S_bs,
                    feats_time.get("mean"), feats_time.get("std"), feats_time.get("var"), feats_time.get("rms"),
                    feats_time.get("max"), feats_time.get("min"), feats_time.get("p2p"),
                    feats_time.get("skew"), feats_time.get("kurt"), feats_time.get("crest"),
                    feats_time.get("impulse"), feats_time.get("shape"), feats_time.get("clearance"),
                    feats_freq.get("psd_centroid"), feats_freq.get("psd_bandwidth"), feats_freq.get("psd_entropy"),
                    feats_tf.get("sk_max"), feats_tf.get("sk_f_hz"), feats_tf.get("tf_entropy_mean"),
                    feats_tf.get("tf_entropy_std"), feats_tf.get("tf_energy3_cv"),
                    out_path
                ]
                rows.append(row)

        # 保存镜像文件
        try:
            savemat_safe(out_path, out_dict if changed else D)
            nOK += 1
        except Exception as e:
            nFail += 1
            print(f"[FAIL-SAVE] {in_path} -> {e}")

        if idx % 20 == 0:
            print(f"... 已处理 {idx}/{len(mats)}")

    # 写报表
    df = pd.DataFrame(rows, columns=hdr)
    # UTF-8 BOM 便于中文Excel
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    dt = time.time()-t0
    print(f"✅ 完成：成功 {nOK}，失败 {nFail} | 用时 {dt:.1f}s")
    print(f"📄 报表：{csv_path}")
    print(f"📁 输出：{OUT_ROOT}")
    print(f"🖼️ 图像：{FIG_DIR}")

def _nz(lst, i):
    try:
        return float(lst[i])
    except Exception:
        return np.nan

if __name__ == "__main__":
    main()
