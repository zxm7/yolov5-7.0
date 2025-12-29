# ga_enhance/enhance_ops.py
# =========================================================
# ✅ 这里是“增强参数”的唯一权威
# 你以后要：
# - 改维度 DIM
# - 改基因到真实参数的范围 decode_params()
# - 改增强算法 apply_enhancement()
# 都只改这一个文件即可。
# ga_main / eval_yolo 都不会再维护重复的一套逻辑。
# =========================================================

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

# -------------------------------
# 1) 染色体维度（唯一权威）
# -------------------------------
DIM = 5

def decode_params(chrom) -> dict:
    """
    将 GA 染色体 chrom（[0,1]^DIM）解码到真实增强参数（可解释参数）。

    注意：
    - chrom 必须长度等于 DIM
    - 返回 dict，键名顺序也会影响 CSV 列顺序（建议固定）
    """
    chrom = np.asarray(chrom, dtype=float).reshape(-1)
    if len(chrom) != DIM:
        raise ValueError(f"[decode_params] 期望维度 DIM={DIM}，但收到 len(chrom)={len(chrom)}")

    g0, g1, g2, g3, g4 = chrom.tolist()

    # 下面范围你可以随时改（论文/直觉/你自己实验）
    # 只要改这里，整个项目会同步生效
    bias     = -0.05 + 0.10 * g0           # [-0.05, 0.05]
    gamma    =  0.90 + 0.30 * g1           # [0.90, 1.20]
    contrast =  0.80 + 0.50 * g2           # [0.80, 1.30]
    gain_r   =  0.85 + 0.35 * g3           # [0.85, 1.20]
    gain_b   =  0.85 + 0.35 * g4           # [0.85, 1.20]

    return {
        "bias": bias,
        "gamma": gamma,
        "contrast": contrast,
        "gain_r": gain_r,
        "gain_b": gain_b,
    }

def apply_enhancement(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    对单张图像执行增强（你要的“真实增强流程”）。

    当前增强链路：
    1) R/B 增益（对 BGR 的 R/B 通道做增益）
    2) bias（整体加偏置）
    3) contrast（围绕 0.5 做线性拉伸）
    4) gamma（幂次变换）

    你以后想加：
    - CLAHE
    - 去雾（dark channel / guided filter）
    - 白平衡
    都可以在这里扩展。只改这一处就行。
    """
    img = img_bgr.astype(np.float32) / 255.0

    # 1) R/B gain（注意 OpenCV 读入是 BGR）
    gain_r = float(params["gain_r"])
    gain_b = float(params["gain_b"])
    img[..., 2] *= gain_r  # R
    img[..., 0] *= gain_b  # B

    # 2) bias
    bias = float(params["bias"])
    img = img + bias

    # 3) contrast around 0.5
    c = float(params["contrast"])
    img = (img - 0.5) * c + 0.5

    # 4) gamma
    g = float(params["gamma"])
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / max(g, 1e-6))

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def enhance_val_images(src_img_dir: Path, dst_img_dir: Path, params: dict, quiet: bool = True) -> int:
    """
    把 src_img_dir 下所有图片增强后写入 dst_img_dir（同名覆盖写）。

    返回：
        n_ok: 成功处理的图片数量
    """
    src_img_dir = Path(src_img_dir)
    dst_img_dir = Path(dst_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in src_img_dir.glob("*")
                        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if not quiet:
        print(f"[增强] 共找到 {len(img_paths)} 张图片，开始增强...")

    n_ok = 0
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = apply_enhancement(img, params)
        cv2.imwrite(str(dst_img_dir / p.name), out)
        n_ok += 1

    if not quiet:
        print(f"[增强] 完成：{n_ok}/{len(img_paths)} 张")
    return n_ok