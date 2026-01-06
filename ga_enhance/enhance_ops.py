# ga_enhance/enhance_ops.py
# =========================================================
# ✅ 完整版：DIM=6 (新增锐化权重 w_sharp 控制)
# ✅ 包含：色彩校正 (Eq. 1-2)、多特征生成 (Eq. 3-4)、权重评估 (Eq. 5-6)、金字塔融合
# =========================================================

from __future__ import annotations
from pathlib import Path
import sys

import cv2
import numpy as np

# 🟢 容错处理：禁用 OpenCL
try:
    if hasattr(cv2, 'setUseOpenCL'):
        cv2.setUseOpenCL(False)
    elif hasattr(cv2, 'ocl') and hasattr(cv2.ocl, 'setUseOpenCL'):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# 1. 维度改为 7
DIM = 7

def decode_params(chrom) -> dict:
    chrom = np.asarray(chrom, dtype=float).reshape(-1)
    # 增加 g6 对应红光增益
    g0, g1, g2, g3, g4, g5, g6 = chrom.tolist()
    return {
        "eta1": g0,
        "eta2": g1,
        "gamma1": 0.8 + g2 * 3.0,
        "gamma2": 1.0 + g3 * 2.0,
        "gamma3": 1.0 + g4 * 2.0,
        "w_sharp": g5,
        "red_gain": 1.0 + g6 * 1.5   # 🟢 新增：红光增益范围 [1.0, 2.5]
    }


# --- 金字塔融合辅助函数 (必须保留) ---

def _get_gaussian_pyramid(img, levels):
    pyramid = [img]
    temp = img.copy()
    for _ in range(levels - 1):
        temp = cv2.pyrDown(temp)
        pyramid.append(temp)
    return pyramid


def _get_laplacian_pyramid(img, levels):
    gauss = _get_gaussian_pyramid(img, levels)
    pyramid = []
    for i in range(levels - 1):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        expanded = cv2.pyrUp(gauss[i + 1], dstsize=size)
        if expanded.shape != gauss[i].shape:
            expanded = cv2.resize(expanded, (gauss[i].shape[1], gauss[i].shape[0]))
        pyramid.append(gauss[i] - expanded)
    pyramid.append(gauss[-1])
    return pyramid


# --- 主增强函数 ---

def apply_enhancement(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    # 归一化输入
    img = img_bgr.astype(np.float32) / 255.0

    # 🟢 新增：红色通道预补偿 (针对水下环境)
    # 在 BGR 格式中，索引 2 是红色通道
    r_gain = params.get("red_gain", 1.0)
    img[..., 2] = np.clip(img[..., 2] * r_gain, 0.0, 1.0)

    # 1. 色彩校正 (Color Correction Eq. 1-2) [cite: 106-109]
    means = np.mean(img, axis=(0, 1))
    idxs = np.argsort(means)
    idx_low, idx_med, idx_high = idxs[0], idxs[1], idxs[2]

    # 保持之前验证过的温和补偿
    eta1, eta2 = float(params["eta1"]) * 0.5, float(params["eta2"]) * 0.5

    img_corr = img.copy()
    denom_med = (means[idx_high] + means[idx_med] + 1e-6)
    img_corr[..., idx_med] += eta1 * ((means[idx_high] - means[idx_med]) / denom_med) * img[..., idx_high]

    denom_low = (means[idx_high] + means[idx_low] + 1e-6)
    img_corr[..., idx_low] += eta2 * ((means[idx_high] - means[idx_low]) / denom_low) * img[..., idx_high]

    img_corr = np.clip(img_corr, 0.0, 1.0)

    # 2. 多特征生成 (Multi-feature Generation Eq. 3-4) [cite: 123-127]
    inputs = []

    # 2.1 锐化图 I_s：由 GA 基因 w_sharp 控制强度
    # 如果 GA 认为锐化伤害分数，它会把 w_s 搜向 0
    w_s = params.get("w_sharp", 0.5)
    blur = cv2.GaussianBlur(img_corr, (0, 0), 5)
    details = img_corr - blur
    d_min, d_max = details.min(), details.max()
    details_norm = (details - d_min) / (d_max - d_min + 1e-6)
    inputs.append(np.clip(img_corr * (1 - w_s) + details_norm * w_s, 0, 1))

    # 2.2 3张 Gamma 曝光图
    for k in ["gamma1", "gamma2", "gamma3"]:
        inputs.append(np.power(img_corr, params[k]))

    # 3. 权重评估 (Weighting Eq. 5-6) [cite: 133-134]
    weights = []
    sigma = 0.2
    for inp in inputs:
        gray = cv2.cvtColor((inp * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # 曝光权重 (Exposure Map Eq. 5)
        E = np.exp(-((gray - 0.5) ** 2) / (2 * sigma ** 2))
        # 对比度权重 (Contrast Map Eq. 6)
        C = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        weights.append(E * C + 1e-6)

    w_sum = np.sum(weights, axis=0)
    norm_weights = [w / w_sum for w in weights]

    # 4. 多尺度金字塔融合 (Pyramid Fusion Stage)
    levels = 5
    input_laps = [_get_laplacian_pyramid(inp, levels) for inp in inputs]
    weight_gauss = [_get_gaussian_pyramid(w, levels) for w in norm_weights]

    fused_pyramid = []
    for l in range(levels):
        fused_l = np.zeros_like(input_laps[0][l])
        for i in range(len(inputs)):
            fused_l += weight_gauss[i][l][..., np.newaxis] * input_laps[i][l]
        fused_pyramid.append(fused_l)

    # 重构图像
    res = fused_pyramid[-1]
    for l in range(levels - 2, -1, -1):
        size = (fused_pyramid[l].shape[1], fused_pyramid[l].shape[0])
        res = cv2.pyrUp(res, dstsize=size)
        if res.shape != fused_pyramid[l].shape:
            res = cv2.resize(res, (fused_pyramid[l].shape[1], fused_pyramid[l].shape[0]))
        res += fused_pyramid[l]

    return (np.clip(res, 0.0, 1.0) * 255.0).astype(np.uint8)


def enhance_val_images(src_img_dir: Path, dst_img_dir: Path, params: dict, quiet: bool = True) -> int:
    src_img_dir, dst_img_dir = Path(src_img_dir), Path(dst_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in src_img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    total = len(img_paths)
    if not quiet:
        print(f"\n[增强开始] 目标: {dst_img_dir.name}, 共 {total} 张")

    n_ok = 0
    for i, p in enumerate(img_paths, 1):
        img = cv2.imread(str(p))
        if img is None: continue
        out = apply_enhancement(img, params)
        cv2.imwrite(str(dst_img_dir / p.name), out)
        n_ok += 1
        if not quiet and i % 10 == 0:
            sys.stdout.write(f"\r >> 进度: {i}/{total} ({(i / total) * 100:.1f}%) ")
            sys.stdout.flush()

    if not quiet: print(f"\n[完成] 成功处理 {n_ok} 张")
    return n_ok