# ga_enhance/eval_on_test.py
from pathlib import Path
import shutil
import yaml
import subprocess
import time

from .enhance_ops import enhance_val_images

# 1. 设置绝对路径
ROOT = Path("/home/zhangxu/yolov5-7.0")

# 2. 这里的参数等你跑完 ga_main.py 后，从 ga_logs/runX/best.json 复制过来替换
BEST = {
    'bias': 0.004146116971969607,
    'gamma': 0.9241198897361755,
    'contrast': 0.9498559415340424,
    'gain_r': 1.018371468782425,
    'gain_b': 0.9979405105113983,
}


def main():
    # ---------- 修正后的测试集路径 (images/test) ----------
    SRC_IMG = ROOT / "datasets" / "UTDAC2020" / "images" / "test"
    SRC_LAB = ROOT / "datasets" / "UTDAC2020" / "labels" / "test"

    assert SRC_IMG.exists(), f"找不到测试图片路径: {SRC_IMG}"
    assert SRC_LAB.exists(), f"找不到测试标签路径: {SRC_LAB}"

    # 3. 输出增强后的测试集（带时间戳，防止重名）
    tag = time.strftime("%Y%m%d_%H%M%S")
    OUT = ROOT / "datasets" / "UTDAC2020_ga_test" / f"test_best_{tag}"
    IMG_OUT = OUT / "images"
    LAB_OUT = OUT / "labels"

    # 4. 执行增强处理
    print(f"正在增强测试集图片...")
    enhance_val_images(src_img_dir=SRC_IMG, dst_img_dir=IMG_OUT, params=BEST, quiet=False)

    # 5. 复制标签（不做增强，直接搬运）
    LAB_OUT.mkdir(parents=True, exist_ok=True)
    for p in SRC_LAB.glob("*.txt"):
        shutil.copy(p, LAB_OUT / p.name)

    # 6. 生成临时数据配置
    data_tmp = ROOT / "datasets" / "UTDAC2020_ga_test" / f"data_test_{tag}.yaml"
    base_yaml = ROOT / "data" / "utdac.yaml"
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    cfg["val"] = str(IMG_OUT)  # 告诉 val.py 去读增强后的测试图
    data_tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 7. 调用 val.py 跑最终分数
    weights = ROOT / "runs" / "train" / "yolov5n_scratch_baseline" / "weights" / "best.pt"

    cmd = [
        "python", "val.py",
        "--data", str(data_tmp),
        "--weights", str(weights),
        "--imgsz", "640",
        "--task", "val",
        "--device", "0",
        "--project", str(ROOT / "runs" / "ga_final_test"),
        "--name", f"result_{tag}",
        "--exist-ok",
    ]
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()