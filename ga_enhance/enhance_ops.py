# ga_enhance/enhance_ops.py
from __future__ import annotations
from pathlib import Path
import sys
import cv2
import numpy as np

# 🟢 维度保持为 8，对应 PIAFR 算子的 8 个可调参数
DIM = 8


def decode_params(chrom: np.ndarray) -> dict:
    """
    解码函数：将 [0, 1] 空间的基因映射为 PIAFR 算子的物理修补参数
    """
    # 确保输入是平铺的浮点数组
    chrom = np.asarray(chrom, dtype=float).reshape(-1)

    return {
        'red_gain': float(chrom[0] * 0.4 + 1.05),  # 红光补偿 [1.05, 1.45]
        'blue_gain': float(chrom[1] * 0.2 + 0.9),  # 蓝光平衡 [0.9, 1.1]
        'gamma': float(chrom[2] * 1.2 + 0.6),  # 亮度校正 [0.6, 1.8]
        'contrast': float(chrom[3] * 0.6 + 0.9),  # 对比度增益 [0.9, 1.5]
        'saturation': float(chrom[4] * 0.8 + 0.7),  # 色彩饱和度 [0.7, 1.5]
        'sharp_strength': float(chrom[5] * 0.04),  # 边缘锐化强度 [0, 0.04]
        'denoise_h': float(chrom[6] * 2.5 + 0.5),  # 噪声平滑强度 [0.5, 3.0]
        'beta_blend': float(chrom[7] * 0.3 + 0.1)  # 物理特征融合比例 [0.1, 0.4]
    }


def apply_enhancement(img: np.ndarray, params: dict) -> np.ndarray:
    """
    PIAFR: 基于物理启发的水下特征自适应修补算子 (高性能版)
    """
    # 1. 物理层：红/蓝光补偿 (BGR 空间)
    res = img.astype(np.float32)
    res[:, :, 2] *= params['red_gain']
    res[:, :, 0] *= params['blue_gain']
    res = np.clip(res, 0, 255).astype(np.uint8)

    # 2. 亮度层：对比度与亮度校正 (YUV 空间)
    yuv = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32) / 255.0
    # 执行 Gamma 变换与对比度拉伸
    y = np.power(y, params['gamma']) * params['contrast']
    yuv[:, :, 0] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    res = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 3. 🟢 语义保护层：双边滤波 (Bilateral Filter)
    # 替代了极慢的 NL-Means，速度提升约 10-20 倍
    sigma = float(params['denoise_h'] * 15)  # 映射到 [7.5, 45.0]
    res = cv2.bilateralFilter(res, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    # 4. 色彩层：饱和度调节 (HSV 空间)
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * params['saturation'], 0, 255).astype(np.uint8)
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 5. 边缘层：轻量级锐化 (Unsharp Masking)
    if params['sharp_strength'] > 0.005:
        gaussian = cv2.GaussianBlur(res, (0, 0), 2)
        res = cv2.addWeighted(res, 1.0 + params['sharp_strength'], gaussian, -params['sharp_strength'], 0)

    # 6. 物理融合：将增强图与原图按比例融合，保证图像自然度
    return cv2.addWeighted(img, params['beta_blend'], res, 1.0 - params['beta_blend'], 0)


def enhance_val_images(src_img_dir: Path, dst_img_dir: Path, params: dict, quiet: bool = True) -> int:
    """
    批量图像增强函数
    """
    src_img_dir, dst_img_dir = Path(src_img_dir), Path(dst_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in src_img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    n_ok = 0
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        out = apply_enhancement(img, params)
        cv2.imwrite(str(dst_img_dir / p.name), out)
        n_ok += 1
    return n_ok