# ga_enhance/eval_on_test.py
from pathlib import Path
import shutil
import yaml
import subprocess
import time

# 确保导入的是最新的 8 参数增强逻辑
from .enhance_ops import enhance_val_images

# 1. 设置绝对路径
ROOT = Path("/home/zhangxu/yolov5-7.0")

# 2. ✅ 保持你 WOA 跑出的 PIAFR (DIM=8) 最优参数
BEST = {
    'red_gain': 1.05,
    'blue_gain': 1.0574326634407043,
    'gamma': 1.0756906628608705,
    'contrast': 0.9,
    'saturation': 0.9273251295089722,
    'sharp_strength': 0.003267453908920288,
    'denoise_h': 3.0,
    'beta_blend': 0.4
}

def main():
    # ---------- 测试集路径 ----------
    SRC_IMG = ROOT / "datasets" / "UTDAC2020" / "images" / "test"
    SRC_LAB = ROOT / "datasets" / "UTDAC2020" / "labels" / "test"

    if not SRC_IMG.exists():
        print(f"错误：找不到测试图片路径: {SRC_IMG}")
        return

    # 3. 输出增强后的测试集
    tag = time.strftime("%Y%m%d_%H%M%S")
    OUT = ROOT / "datasets" / "UTDAC2020_woa_test" / f"test_piafr_final_{tag}"
    IMG_OUT = OUT / "images"
    LAB_OUT = OUT / "labels"

    # 4. 执行增强处理
    print(f"正在增强测试集图片(PIAFR DIM=8)...")
    enhance_val_images(src_img_dir=SRC_IMG, dst_img_dir=IMG_OUT, params=BEST, quiet=False)

    # 5. 复制标签
    LAB_OUT.mkdir(parents=True, exist_ok=True)
    for p in SRC_LAB.glob("*.txt"):
        shutil.copy(p, LAB_OUT / p.name)

    # 6. 生成临时数据配置
    data_tmp = ROOT / "datasets" / "UTDAC2020_woa_test" / f"data_piafr_final_{tag}.yaml"
    base_yaml = ROOT / "data" / "utdac.yaml"
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) if base_yaml.exists() else {}
    cfg["val"] = str(IMG_OUT)
    data_tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 7. ✅【核心修改】：加载你刚刚训练完成的最优权重
    # 这里指向你 500 轮训练出来的那个路径
    weights = ROOT / "runs" / "piafr_train" / "piafr_finetune_500e" / "weights" / "best.pt"

    if not weights.exists():
        print(f"错误：找不到权重文件: {weights}")
        return

    # 8. 调用 val.py 跑最终分数
    cmd = [
        "python", "val.py",
        "--data", str(data_tmp),
        "--weights", str(weights),
        "--imgsz", "640",
        "--task", "val",  # 在测试集上评估
        "--device", "",    # 自动选择设备
        "--project", str(ROOT / "runs" / "piafr_final_test"),
        "--name", f"final_result_{tag}",
        "--exist-ok",
    ]

    print(f"\n[RUNNING FINAL TEST EVALUATION]: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)

if __name__ == "__main__":
    main()