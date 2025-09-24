# -*- coding: utf-8 -*-
"""
目标域数据集 -> 统一到 12 kHz（镜像扁平输出 + 报表）
- 兼容：若路径中含 "<digits>kHz" 则按其解析；否则默认 fsIn=32 kHz
- 若 fsIn==12kHz：仅复制
- 若 fsIn!=12kHz：重采样到 12kHz（优先 resample_poly）
- 输出：out_root 下（不保留子目录）
- 报表：report_dir/resample_target_report_12kHz_*.csv

依赖：
    pip install numpy scipy
用法示例：
    python batch_resample_target_to_12kHz.py \
        --in-root "./data/target_data" \
        --out-root "./data/target_resample_12kHz" \
        --report-dir "./report" \
        --default-fs 32000 \
        --target-fs 12000
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
from typing import List, Tuple, Any

import argparse
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly, firwin, lfilter
from scipy.interpolate import PchipInterpolator


# ───────── argparse ─────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将目标域 .mat 文件统一重采样到 12kHz（或自定义目标），扁平输出并生成报表。"
    )
    parser.add_argument("--in-root", type=Path, default=Path("./data/target_data"),
                        help="输入根目录（递归查找 .mat）")
    parser.add_argument("--out-root", type=Path, default=Path("./data/target_resample_12kHz"),
                        help="输出根目录（不保留子目录，所有文件扁平落在此处）")
    parser.add_argument("--report-dir", type=Path, default=Path("./reports"),
                        help="报表输出目录")
    parser.add_argument("--target-fs", type=float, default=12_000.0,
                        help="目标采样率 Hz（默认 12000）")
    parser.add_argument("--default-fs", type=float, default=32_000.0,
                        help="若路径无法解析出 kHz 时默认的输入采样率 Hz（默认 32000）")
    parser.add_argument("--glob", type=str, default="*.mat",
                        help='匹配模式（相对 in-root 递归），默认 "*.mat"')
    parser.add_argument("--max-den", type=int, default=1_000_000,
                        help="L/M 近似的分母上限（Fraction.limit_denominator）")
    parser.add_argument("--dry-run", action="store_true",
                        help="试跑模式：只打印计划，不写任何文件")
    return parser.parse_args()


# ───────── 主流程 ─────────

def main():
    args = parse_args()

    in_root    = args.in_root
    out_root   = args.out_root
    report_dir = args.report_dir
    target_fs  = float(args.target_fs)
    default_fs = float(args.default_fs)
    glob_pat   = args.glob
    max_den    = int(args.max_den)
    dry_run    = bool(args.dry_run)

    if not in_root.is_dir():
        raise FileNotFoundError(f"未找到输入目录：{in_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = list(in_root.rglob(glob_pat))
    if not files:
        warnings.warn(f"在 {in_root} 未找到 {glob_pat} 文件。")
        return

    target_tag = format_target_tag(target_fs)
    print(f"共发现 {len(files)} 个 .mat 文件，目标采样率统一为 {target_tag}（默认输入 {int(default_fs)} Hz）。")

    csv_path = report_dir / f"resample_target_report_{target_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rep_hdr  = ["文件路径","输出路径","变量名","原fs(Hz)","新fs(Hz)","L","M","方法","原样本数","新样本数","备注"]
    rep_rows: List[List[Any]] = []

    n_ok = n_fail = n_changed = 0

    for k, in_path in enumerate(files, 1):
        try:
            # 1) 解析输入 fs
            fs_in = parse_fs_from_path(str(in_path.parent))
            remark = ""
            if not np.isfinite(fs_in) or fs_in <= 0:
                fs_in = default_fs
                remark = "assume-32k" if int(default_fs) == 32000 else f"assume-{int(default_fs)}"

            # 2) 扁平输出路径
            out_path = out_root / in_path.name

            # 3) 读取 mat
            S = loadmat(str(in_path), squeeze_me=False, struct_as_record=False)
            S = {k: v for k, v in S.items() if not k.startswith("__")}

            # 4) 选择信号变量（非标量的数值数组）
            sig_vars = find_signal_vars(S)

            # 5) 无信号变量：也镜像保存/标记
            if len(sig_vars) == 0:
                if abs(fs_in - target_fs) < 1e-6:
                    if dry_run:
                        print(f"[COPY] (no-signal-var) {in_path} -> {out_path}")
                    else:
                        shutil.copy2(str(in_path), str(out_path))
                    rep_rows.append([str(in_path), str(out_path), "(no-signal-var)", fs_in, target_fs, 1, 1, "copy", "", "", remark])
                else:
                    if dry_run:
                        print(f"[TAG ONLY] (no-signal-var) {in_path} -> {out_path} (+fs_resampled_Hz)")
                    else:
                        Sout = dict(S)
                        Sout["fs_resampled_Hz"] = np.array([[target_fs]], dtype=float)
                        savemat(str(out_path), Sout, do_compression=False)
                    rep_rows.append([str(in_path), str(out_path), "(no-signal-var)", fs_in, target_fs, "", "", "no-signal-but-marked", "", "", remark])
                    n_changed += 1
                n_ok += 1
                continue

            # 6) 已经 target：仅复制并记录
            if abs(fs_in - target_fs) < 1e-6:
                if dry_run:
                    print(f"[COPY] {in_path} -> {out_path}")
                else:
                    shutil.copy2(str(in_path), str(out_path))
                for vn in sig_vars:
                    n0 = length_along_time(S[vn])
                    rep_rows.append([str(in_path), str(out_path), vn, fs_in, target_fs, 1, 1, "copy", n0, n0, remark])
                n_ok += 1
                if n_ok % 20 == 0:
                    print(f"... 已处理 {n_ok}/{len(files)}")
                continue

            # 7) 需要重采样
            L, M = rational_ratio(target_fs / fs_in, max_den=max_den)   # 32k->12k ≈ 3/8
            method = f"resample_poly(L={L},M={M})"
            file_changed = False
            Sout = dict(S)

            for vn in sig_vars:
                X0 = S[vn]
                orig_dtype = X0.dtype
                if is_vector(X0):
                    # 保持原方向
                    Y, n0, n1 = resample_vector_keep_shape(X0, fs_in, target_fs, L, M)
                    if not dry_run:
                        Sout[vn] = Y.astype(orig_dtype, copy=False)
                else:
                    # 矩阵：按列为通道、行是时间（与实现一致），逐列重采样并对齐长度
                    Y, n0, n1 = resample_matrix_rows(X0, fs_in, target_fs, L, M)
                    if not dry_run:
                        Sout[vn] = Y.astype(orig_dtype, copy=False)

                rep_rows.append([str(in_path), str(out_path), vn, fs_in, target_fs, L, M, method, n0, n1, remark])
                file_changed = True

            # 8) 标记并保存
            if not dry_run:
                Sout["fs_resampled_Hz"] = np.array([[target_fs]], dtype=float)
                savemat(str(out_path), Sout, do_compression=False)

            if file_changed:
                n_changed += 1
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"... 已处理 {n_ok}/{len(files)}")

        except Exception as e:
            n_fail += 1
            warnings.warn(f"处理失败：{in_path} ({e})")

    # 9) 写报表
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
    if abs(fs - round(fs/1000)*1000) < 1e-9:
        return f"{int(round(fs/1000))}kHz"
    return f"{int(round(fs))}Hz"


def parse_fs_from_path(folder_path: str) -> float:
    """从路径抓取 '<数字>kHz' 并转 Hz；若多处匹配取最后一个；找不到返回 NaN。"""
    m = re.findall(r"(\d+)\s*kHz", folder_path, flags=re.IGNORECASE)
    if m:
        return float(m[-1]) * 1000.0
    return float("nan")


def find_signal_vars(S: dict) -> List[str]:
    """选择所有 '非标量且数值型 ndarray' 的变量名。"""
    keep = []
    for k, v in S.items():
        if isinstance(v, np.ndarray) and v.size > 1 and np.issubdtype(v.dtype, np.number):
            keep.append(k)
    return keep


def is_vector(X: np.ndarray) -> bool:
    """是否为向量（2D 且有一维为 1）"""
    return X.ndim == 2 and (X.shape[0] == 1 or X.shape[1] == 1)


def length_along_time(X: np.ndarray) -> int:
    """时间轴长度：向量取非 1 的那一维；矩阵认为时间在行（axis=0）。"""
    if is_vector(X):
        return max(X.shape)
    return X.shape[0]


def rational_ratio(r: float, max_den: int = 1_000_000) -> Tuple[int, int]:
    """r ≈ L/M（类似 MATLAB rat）"""
    frac = Fraction(r).limit_denominator(max_den)
    return frac.numerator, frac.denominator


def resample_vector_keep_shape(X: np.ndarray, fs_in: float, fs_out: float, L: int, M: int) -> Tuple[np.ndarray, int, int]:
    """向量重采样并保持 (1,n)/(n,1) 方向不变。"""
    n0 = length_along_time(X)
    x1d = np.asarray(X.ravel(order="F"), dtype=np.float64)
    y1d = resample_1d(x1d, fs_in, fs_out, L, M)
    n1 = y1d.size
    if X.shape[0] == 1:  # 行向量
        Y = y1d.reshape(1, n1, order="F")
    else:                # 列向量
        Y = y1d.reshape(n1, 1, order="F")
    return Y, n0, n1


def resample_matrix_rows(X: np.ndarray, fs_in: float, fs_out: float, L: int, M: int) -> Tuple[np.ndarray, int, int]:
    """
    矩阵重采样：按“行是时间、列为通道”的约定，沿 axis=0 重采样。
    优先一次性 resample_poly；失败则逐列回退（抗混叠 + PCHIP）
    """
    n0, C = X.shape[0], X.shape[1]
    Xf = np.asarray(X, dtype=np.float64)
    try:
        Y = resample_poly(Xf, up=L, down=M, axis=0)
        n1 = Y.shape[0]
        return Y, n0, n1
    except Exception:
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
    单通道 1D 重采样：
      - 首选：polyphase 抗混叠 resample_poly
      - 回退：降采样先 FIR 低通再 PCHIP；升采样直接 PCHIP
    """
    try:
        return resample_poly(x, up=L, down=M)
    except Exception:
        r = fs_out / fs_in
        n_out = max(1, int(round(x.size * r)))
        t_in  = np.arange(x.size) / fs_in
        t_out = np.arange(n_out)   / fs_out

        if fs_out < fs_in:
            fc = 0.45 * fs_out / (fs_in / 2.0)  # 归一化至 Nyquist
            fc = float(np.clip(fc, 0.01, 0.98))
            numtaps = max(31, 2 * int(np.ceil(6 / fc)) + 1)
            b = firwin(numtaps, cutoff=fc)
            x_f = lfilter(b, [1.0], x)
            y = PchipInterpolator(t_in, x_f, extrapolate=True)(t_out)
        else:
            y = PchipInterpolator(t_in, x, extrapolate=True)(t_out)

        return np.asarray(y, dtype=np.float64)


def pad_or_trim(x: np.ndarray, n1: int) -> np.ndarray:
    """将 1D 向量 x 调整到长度 n1（截断/零填充）"""
    nx = x.size
    if nx == n1:
        return x
    if nx > n1:
        return x[:n1]
    y = np.zeros(n1, dtype=x.dtype)
    y[:nx] = x
    return y


def write_csv_utf8_bom(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """写 CSV（UTF-8+BOM），便于 Excel 直接打开"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
