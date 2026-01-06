# ga_enhance/eval_on_test.py
from pathlib import Path
import shutil
import yaml
import subprocess
import time

# 确保导入的是最新的增强逻辑
from .enhance_ops import enhance_val_images

# 1. 设置绝对路径
ROOT = Path("/home/zhangxu/yolov5-7.0")

# 2. ✅ 更新为 DIM=6 的最优参数
# 来自你 run4 的结果
BEST = {
    'eta1': 0.21892237663269043,
    'eta2': 0.20907165110111237,
    'gamma1': 0.9429175734519959,
    'gamma2': 1.4881336688995361,
    'gamma3': 1.185707926750183,
    'w_sharp': 0.02922987937927246
}


def main():
    # ---------- 测试集路径 ----------
    SRC_IMG = ROOT / "datasets" / "UTDAC2020" / "images" / "test"
    SRC_LAB = ROOT / "datasets" / "UTDAC2020" / "labels" / "test"

    if not SRC_IMG.exists():
        print(f"错误：找不到测试图片路径: {SRC_IMG}")
        return

    # 3. 输出增强后的测试集（带时间戳）
    tag = time.strftime("%Y%m%d_%H%M%S")
    OUT = ROOT / "datasets" / "UTDAC2020_ga_test" / f"test_best_{tag}"
    IMG_OUT = OUT / "images"
    LAB_OUT = OUT / "labels"

    # 4. 执行增强处理
    print(f"正在使用最优参数(DIM=6)增强测试集图片...")
    # 注意：这里会调用你修改后的 apply_enhancement，包含 w_sharp 逻辑
    enhance_val_images(src_img_dir=SRC_IMG, dst_img_dir=IMG_OUT, params=BEST, quiet=False)

    # 5. 复制标签
    LAB_OUT.mkdir(parents=True, exist_ok=True)
    for p in SRC_LAB.glob("*.txt"):
        shutil.copy(p, LAB_OUT / p.name)

    # 6. 生成临时数据配置
    data_tmp = ROOT / "datasets" / "UTDAC2020_ga_test" / f"data_test_{tag}.yaml"
    base_yaml = ROOT / "data" / "utdac.yaml"

    if base_yaml.exists():
        cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    else:
        # 如果找不到基础 yaml，手动构建一个简单的
        cfg = {"names": ["holothurian", "echinus", "scallop", "starfish"]}  # 根据你数据集修改

    cfg["val"] = str(IMG_OUT)
    data_tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 7. 调用 val.py 跑最终分数
    weights = ROOT / "runs" / "train" / "yolov5n_scratch_baseline" / "weights" / "best.pt"

    cmd = [
        "python", "val.py",
        "--data", str(data_tmp),
        "--weights", str(weights),
        "--imgsz", "640",
        "--task", "val",  # 虽然是测试集，但 val 任务能出完整的 mAP 报告
        "--device", "0",
        "--project", str(ROOT / "runs" / "ga_final_test"),
        "--name", f"result_{tag}",
        "--exist-ok",
    ]

    print(f"\n[RUNNING TEST EVALUATION]: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()