# -*- coding: utf-8 -*-
"""
批量将源域 .mat 文件统一到目标采样率（默认 12 kHz）
- 从路径段解析原始 fs（<digits>kHz）
- 若 fs == target：仅复制（保持原含kHz目录段不变）
- 若 fs != target：重采样到 target，并把“含kHz目录段”改名为“原名 + _{target_tag}”
- 生成 UTF-8 BOM 的 CSV 报表

依赖：
    pip install numpy scipy
用法示例：
    python batch_resample_to_12kHz.py \
        --in-root "/Users/caojiahao/Downloads/数据集/源域数据集" \
        --out-root "/Users/caojiahao/Downloads/数据集/统一采样率的源域数据集_12kHz" \
        --report-dir "/Users/caojiahao/Downloads/数据集/报表" \
        --target-fs 12000 \
        --suffixes "_DE_time,_FE_time,_BA_time"
"""

from __future__ import annotations
import os
import re
import csv
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from fractions import Fraction
from typing import Dict, Tuple, List, Any

import argparse
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly, firwin, lfilter
from scipy.interpolate import PchipInterpolator


# ───────── argparse & 入口 ─────────

def parse_args() -> argparse.Namespace:
    def parse_suffixes(s: str) -> Tuple[str, ...]:
        # 逗号/空白分隔均可
        parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
        return tuple(parts)

    default_base = Path("./data")
    parser = argparse.ArgumentParser(
        description="将 .mat 文件统一重采样到目标采样率（默认 12kHz），镜像输出并生成报表。"
    )
    parser.add_argument("--in-root", type=Path,
                        default=default_base / "raw_data",
                        help="输入根目录（递归查找 .mat）")
    parser.add_argument("--out-root", type=Path,
                        default=default_base / "raw_resampled_12kHz",
                        help="输出根目录（镜像写入）")
    parser.add_argument("--report-dir", type=Path,
                        default=Path("./reports"),
                        help="报表输出目录")
    parser.add_argument("--target-fs", type=float, default=12_000.0,
                        help="目标采样率（Hz），默认 12000")
    parser.add_argument("--target-tag", type=str, default=None,
                        help="目标采样率标签，用于目录改名后缀；默认根据 target-fs 自动生成，如 12kHz")
    parser.add_argument("--suffixes", type=parse_suffixes,
                        default=("_DE_time", "_FE_time", "_BA_time"),
                        help='需要重采样/记录的变量名后缀，逗号分隔，默认 "_DE_time,_FE_time,_BA_time"')
    parser.add_argument("--glob", type=str, default="*.mat",
                        help='匹配模式（相对 in-root 递归），默认 "*.mat"')
    parser.add_argument("--max-den", type=int, default=1_000_000,
                        help="L/M 近似的分母上限（用于 Fraction.limit_denominator）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划操作，不写文件/报表")
    return parser.parse_args()


# ───────── 主流程 ─────────

def batch_resample_to_target(in_root: Path,
                             out_root: Path,
                             report_dir: Path,
                             target_fs: float,
                             sig_suffix: Tuple[str, ...],
                             glob_pat: str = "*.mat",
                             target_tag: str | None = None,
                             max_den: int = 1_000_000,
                             dry_run: bool = False) -> None:

    target_tag = target_tag or format_target_tag(target_fs)

    if not in_root.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{in_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = list(in_root.rglob(glob_pat))
    if not files:
        warnings.warn(f"在 {in_root} 未找到 {glob_pat} 文件。")
        return

    print(f"共发现 {len(files)} 个文件（{glob_pat}），目标采样率统一为 {target_tag}。")

    csv_path = report_dir / f"resample_report_{target_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rep_hdr = ["文件路径", "输出路径", "变量名", "原fs(Hz)", "新fs(Hz)", "L", "M", "方法", "原样本数", "新样本数", "备注"]
    rep_rows: List[List[Any]] = []

    n_ok = n_fail = n_changed = 0

    for idx, in_path in enumerate(files, 1):
        try:
            fs_in = parse_fs_from_path(str(in_path.parent))
            if not np.isfinite(fs_in) or fs_in <= 0:
                warnings.warn(f"无法从路径解析采样率，跳过：{in_path}")
                continue

            out_folder, which_rate_dir = make_out_folder(in_root, out_root, in_path.parent, fs_in, target_fs, target_tag)
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / in_path.name

            # 载入 mat
            S = loadmat(str(in_path), squeeze_me=False, struct_as_record=False)
            S = {k: v for k, v in S.items() if not k.startswith("__")}

            # 选出信号变量
            sig_vars = [name for name in S.keys()
                        if name.endswith(sig_suffix) and isinstance(S[name], np.ndarray)]

            if len(sig_vars) == 0:
                if abs(fs_in - target_fs) < 1e-6:
                    if dry_run:
                        print(f"[COPY] (no-signal-var) {in_path} -> {out_path}")
                    else:
                        shutil.copy2(str(in_path), str(out_path))
                    rep_rows.append([str(in_path), str(out_path), "(no-signal-var)", fs_in, target_fs, 1, 1,
                                     "copy", "", "", which_rate_dir])
                else:
                    if dry_run:
                        print(f"[TAG ONLY] (no-signal-var) {in_path} -> {out_path} (+fs_resampled_Hz)")
                    else:
                        Sout = dict(S)
                        Sout["fs_resampled_Hz"] = np.array([[target_fs]], dtype=float)
                        savemat(str(out_path), Sout, do_compression=False)
                    rep_rows.append([str(in_path), str(out_path), "(no-signal-var)", fs_in, target_fs, "", "",
                                     "no-signal-but-marked", "", "", which_rate_dir])
                    n_changed += 1

                n_ok += 1
                continue

            # 已经 target：仅复制，但逐变量记录
            if abs(fs_in - target_fs) < 1e-6:
                if dry_run:
                    print(f"[COPY] {in_path} -> {out_path}")
                else:
                    shutil.copy2(str(in_path), str(out_path))
                for vn in sig_vars:
                    n0 = length_along_time(S[vn])
                    rep_rows.append([str(in_path), str(out_path), vn, fs_in, target_fs, 1, 1, "copy", n0, n0, which_rate_dir])

                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"... 已处理 {n_ok}/{len(files)}")
                continue

            # 需要重采样
            L, M = rational_ratio(target_fs / fs_in, max_den=max_den)
            method = f"resample_poly(L={L},M={M})"
            Sout = dict(S)
            file_changed = False

            for vn in sig_vars:
                X0 = S[vn]
                orig_dtype = X0.dtype
                if is_vector(X0):
                    y, n0, n1 = resample_vector_keep_shape(X0, fs_in, target_fs, L, M)
                    if not dry_run:
                        Sout[vn] = y.astype(orig_dtype, copy=False)
                else:
                    Y, n0, n1 = resample_matrix_rows(X0, fs_in, target_fs, L, M)
                    if not dry_run:
                        Sout[vn] = Y.astype(orig_dtype, copy=False)

                rep_rows.append([str(in_path), str(out_path), vn, fs_in, target_fs, L, M, method, n0, n1, which_rate_dir])
                file_changed = True

            if not dry_run:
                Sout["fs_resampled_Hz"] = np.array([[target_fs]], dtype=float)
                savemat(str(out_path), Sout, do_compression=False)

            if file_changed:
                n_changed += 1
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"... 已处理 {n_ok}/{len(files)}")

        except Exception as e:
            n_fail += 1
            warnings.warn(f"处理失败：{in_path} ({e})")

    # 报表
    if dry_run:
        print("[DRY-RUN] 不写报表。")
    else:
        write_csv_utf8_bom(csv_path, rep_hdr, rep_rows)
        print(
            f"✅ 完成：成功 {n_ok}，失败 {n_fail}；实际重采样的文件数 {n_changed}\n"
            f"📄 报表：{csv_path}\n📁 输出：{out_root}"
        )


# ───────── 工具函数 ─────────

def format_target_tag(fs: float) -> str:
    # 12kHz / 12.8kHz / 9600Hz 等
    if abs(fs - round(fs / 1000) * 1000) < 1e-9:
        return f"{int(round(fs/1000))}kHz"
    return f"{int(round(fs))}Hz"


def parse_fs_from_path(folder_path: str) -> float:
    """从路径段抓取 '<数字>kHz' 并转 Hz；若有多个匹配，用最后一个。"""
    matches = re.findall(r"(\d+)\s*kHz", folder_path, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1]) * 1000.0
    return float("nan")


def make_out_folder(in_root: Path, out_root: Path, in_folder: Path,
                    fs_in: float, fs_out: float, target_tag: str) -> Tuple[Path, str]:
    """生成输出镜像目录与命中的“含kHz目录段名”."""
    rel_folder = str(in_folder.relative_to(in_root))
    parts = rel_folder.split(os.sep)

    idx_hz = None
    for i in range(len(parts) - 1, -1, -1):
        if re.match(r"^\s*\d+\s*kHz.*$", parts[i], flags=re.IGNORECASE):
            idx_hz = i
            break

    if idx_hz is None:
        which_rate_dir = "(no-kHz-dir)"
        out_folder = out_root / rel_folder
        return out_folder, which_rate_dir

    which_rate_dir = parts[idx_hz]
    if abs(fs_in - fs_out) < 1e-6:
        new_rate_dir = which_rate_dir
    else:
        new_rate_dir = f"{which_rate_dir}_{target_tag}"
    parts[idx_hz] = new_rate_dir

    out_folder = out_root / Path(*parts)
    return out_folder, which_rate_dir


def rational_ratio(r: float, max_den: int = 1_000_000) -> Tuple[int, int]:
    """将比例 r 近似为有理数 L/M（类似 MATLAB rat）。"""
    frac = Fraction(r).limit_denominator(max_den)
    return frac.numerator, frac.denominator


def is_vector(X: np.ndarray) -> bool:
    """判断是否为 MATLAB 意义的向量（任一维为 1 的 2D 数组）。"""
    return X.ndim == 2 and (X.shape[0] == 1 or X.shape[1] == 1)


def length_along_time(X: np.ndarray) -> int:
    """返回时间轴长度：向量取非 1 的那一维；矩阵认为时间在行（axis=0）。"""
    if is_vector(X):
        return max(X.shape)
    return X.shape[0]


def resample_vector_keep_shape(X: np.ndarray, fs_in: float, fs_out: float, L: int, M: int) -> Tuple[np.ndarray, int, int]:
    """
    向量重采样：保持原 shape（1,n）或（n,1）。
    """
    n0 = length_along_time(X)
    x1d = np.asarray(X.ravel(order="F"), dtype=np.float64)  # F 顺序兼容 (1,n)/(n,1)
    y1d = resample_1d(x1d, fs_in, fs_out, L, M)
    n1 = y1d.size

    if X.shape[0] == 1:      # 行向量
        Y = y1d.reshape(1, n1, order="F")
    else:                    # 列向量
        Y = y1d.reshape(n1, 1, order="F")

    return Y, n0, n1


def resample_matrix_rows(X: np.ndarray, fs_in: float, fs_out: float, L: int, M: int) -> Tuple[np.ndarray, int, int]:
    """
    矩阵重采样：按“时间在行”的约定，沿 axis=0 重采样。
    优先一次性 resample_poly；若失败则逐列回退（抗混叠 + PCHIP）
    """
    n0, C = X.shape[0], X.shape[1]
    Xf = np.asarray(X, dtype=np.float64)

    try:
        Y = resample_poly(Xf, up=L, down=M, axis=0)
        n1 = Y.shape[0]
        return Y, n0, n1
    except Exception:
        # 逐列回退
        cols = []
        n1_ref = None
        for c in range(C):
            yc = resample_1d(Xf[:, c], fs_in, fs_out, L, M)
            if n1_ref is None:
                n1_ref = yc.size
            elif yc.size != n1_ref:
                yc = pad_or_trim(yc, n1_ref)
            cols.append(yc.reshape(-1, 1))
        Y = np.hstack(cols)
        n1 = Y.shape[0]
        return Y, n0, n1


def resample_1d(x: np.ndarray, fs_in: float, fs_out: float, L: int, M: int) -> np.ndarray:
    """
    单通道 1D 重采样。
    - 首选：polyphase 抗混叠 resample_poly
    - 回退：若降采样，先低通 FIR 再用 PCHIP 到新时间网格；若升采样，直接 PCHIP
    """
    try:
        return resample_poly(x, up=L, down=M)
    except Exception:
        r = fs_out / fs_in
        n_out = max(1, int(round(x.size * r)))
        t_in = np.arange(x.size) / fs_in
        t_out = np.arange(n_out) / fs_out

        if fs_out < fs_in:
            # 低通到 0.45*fs_out（归一化到 Nyquist）
            fc = 0.45 * fs_out / (fs_in / 2.0)
            fc = float(np.clip(fc, 0.01, 0.98))
            numtaps = max(31, 2 * int(np.ceil(6 / fc)) + 1)
            b = firwin(numtaps, cutoff=fc)
            x_f = lfilter(b, [1.0], x)
            y = PchipInterpolator(t_in, x_f, extrapolate=True)(t_out)
        else:
            y = PchipInterpolator(t_in, x, extrapolate=True)(t_out)

        return np.asarray(y, dtype=np.float64)


def pad_or_trim(x: np.ndarray, n1: int) -> np.ndarray:
    """将向量 x 调整为长度 n1（尾部截断或 0 填充）"""
    nx = x.size
    if nx == n1:
        return x
    if nx > n1:
        return x[:n1]
    y = np.zeros(n1, dtype=x.dtype)
    y[:nx] = x
    return y


def write_csv_utf8_bom(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """写 CSV（UTF-8+BOM）"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ───────── main ─────────

def main():
    args = parse_args()
    batch_resample_to_target(
        in_root=args.in_root,
        out_root=args.out_root,
        report_dir=args.report_dir,
        target_fs=args.target_fs,
        sig_suffix=args.suffixes,
        glob_pat=args.glob,
        target_tag=args.target_tag,
        max_den=args.max_den,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
