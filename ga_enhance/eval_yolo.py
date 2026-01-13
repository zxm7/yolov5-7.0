# msf_woa_enhance/eval_yolo.py
from __future__ import annotations

import os
import re
import time
import shutil
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import yaml
import numpy as np

# 从你新版 enhance_ops 导入
from .enhance_ops import decode_params, enhance_val_images

# -------------------------------
# 1) 路径配置（请根据实际服务器环境微调）
# -------------------------------
ROOT = Path("/home/zhangxu/yolov5-7.0")
YOLO_DIR = ROOT
SPLIT_DIR = ROOT / "datasets" / "UTDAC2020"
VAL_NAME = "val"
BASE_DATA_YAML = ROOT / "data" / "utdac.yaml"

# 建议使用 Baseline 权重进行对比实验
WEIGHTS = ROOT / "runs" / "train" / "yolov5n_scratch_baseline" / "weights" / "best.pt"

# 更新缓存目录名为 msf_cache
CACHE_DIR = ROOT / "msf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 2) YOLO 验证参数
# -------------------------------
IMG_SIZE = 640
CONF_THRES = 0.001
IOU_THRES = 0.6
DEVICE = ""  # 指定显卡，如果报错可尝试 ""
YOLO_WORKERS = 0
YOLO_BATCH_SIZE = 16
YOLO_HALF = True

# -------------------------------
# 3) 缓存与日志控制
# -------------------------------
KEEP_RECENT_N = 60
MAX_CACHE_GB = 30
DEFAULT_QUIET_ENHANCE = True
DEFAULT_VERBOSE = True


@dataclass
class EvalResult:
    ok: int  # 1成功 / 0失败
    map50: float  # mAP@.5
    map50_95: float  # mAP@.5:.95
    time_sec: float
    cache_hit: int  # 1=缓存命中 / 0=实际执行
    work_dir: str
    msg: str


# -------------------------------
# 4) 指标解析工具
# -------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _read_metrics_from_csv(save_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """从 YOLOv5 的 results.csv 中提取指标"""
    candidates = [save_dir / "results.csv", save_dir.parent / "results.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None: return None, None

    import csv as _csv
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
            if not rows: return None, None
            row = rows[-1]
            m50, m95 = None, None
            for k, v in row.items():
                k_clean = k.strip()
                if k_clean in ["metrics/mAP_0.5", "mAP@0.5", "metrics/mAP50"]:
                    m50 = float(v)
                if k_clean in ["metrics/mAP_0.5:0.95", "mAP@0.5:0.95", "metrics/mAP50-95"]:
                    m95 = float(v)
            return m50, m95
    except:
        return None, None


def _read_metrics_from_stdout(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text: return None, None
    text = _ANSI_RE.sub("", text)  # 去除颜色代码

    # 更加宽松的正则匹配
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("all") or "all" in line:
            parts = line.split()
            # 标准 YOLOv5 输出行: all, images, instances, P, R, mAP50, mAP50-95
            # 总共 7 个元素，mAP50 在索引 5
            if len(parts) >= 7 and parts[0] == "all":
                try:
                    m50 = float(parts[5])
                    m95 = float(parts[6])
                    return m50, m95
                except:
                    continue
    return None, None


# -------------------------------
# 5) 核心评估函数 (并行安全)
# -------------------------------
def _stable_hash(chrom: np.ndarray) -> str:
    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)
    q = np.round(chrom, 6)
    key = f"{VAL_NAME}_msf|" + ",".join([f"{x:.6f}" for x in q.tolist()])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def evaluate_params(
        chrom,
        eval_tag: str,
        force_rebuild: bool = False,
        quiet_enhance: bool = DEFAULT_QUIET_ENHANCE,
        verbose: bool = DEFAULT_VERBOSE,
        wait_lock_sec: int = 3600,
) -> EvalResult:
    t0 = time.time()
    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)

    # 1. 解码与 Hash
    try:
        params = decode_params(chrom)
    except Exception as e:
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, "", f"Decode error: {e}")

    h = _stable_hash(chrom)
    work_dir = CACHE_DIR / f"work_{h}"
    done_flag, fail_flag, lock_path = work_dir / "DONE", work_dir / "FAIL", work_dir / "LOCK"

    # 2. 检查缓存
    if done_flag.exists() and not force_rebuild:
        try:
            m50 = float((work_dir / "map50.txt").read_text().strip())
            m95 = float((work_dir / "map50_95.txt").read_text().strip())
            return EvalResult(1, m50, m95, time.time() - t0, 1, str(work_dir), "cache_hit")
        except:
            pass

    # 3. 抢占原子锁 (并行安全)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # 等待其他进程完成
        wait_t0 = time.time()
        while time.time() - wait_t0 < wait_lock_sec:
            if done_flag.exists():
                return evaluate_params(chrom, eval_tag, False)  # 递归读缓存
            if fail_flag.exists():
                return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), "Other process failed")
            time.sleep(2)
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), "Lock timeout")

    # 4. 执行增强与验证 (持有锁)
    try:
        src_img_dir = SPLIT_DIR / "images" / VAL_NAME
        src_lab_dir = SPLIT_DIR / "labels" / VAL_NAME
        dst_img_dir, dst_lab_dir = work_dir / "images", work_dir / "labels"

        # 增强图片
        if not (dst_img_dir / "DONE_ENHANCE").exists():
            enhance_val_images(src_img_dir, dst_img_dir, params, quiet=quiet_enhance)
            (dst_img_dir / "DONE_ENHANCE").touch()

        # 复制标签
        if not any(dst_lab_dir.glob("*.txt")):
            dst_lab_dir.mkdir(parents=True, exist_ok=True)
            for p in src_lab_dir.glob("*.txt"): shutil.copy2(p, dst_lab_dir / p.name)

        # 准备 YAML
        data_cfg = yaml.safe_load(BASE_DATA_YAML.read_text())
        data_cfg["val"] = str(dst_img_dir)
        tmp_yaml = work_dir / f"data_tmp.yaml"
        tmp_yaml.write_text(yaml.safe_dump(data_cfg))

        # 运行 YOLO val.py
        yolo_out = work_dir / "yolo_val"
        cmd = [
            "python", "val.py", "--data", str(tmp_yaml), "--weights", str(WEIGHTS),
            "--imgsz", str(IMG_SIZE), "--conf-thres", str(CONF_THRES),
            "--device", DEVICE, "--project", str(yolo_out), "--name", eval_tag, "--exist-ok"
        ]
        if YOLO_HALF: cmd.append("--half")

        # 🟢 修改这里：捕获输出并写入文件
        proc = subprocess.run(cmd, cwd=str(YOLO_DIR), capture_output=True, text=True)
        (work_dir / "val_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (work_dir / "val_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

        # 解析结果
        save_dir = yolo_out / eval_tag
        m50, m95 = _read_metrics_from_csv(save_dir)

        # 🟢 关键修改：同时合并 stdout 和 stderr 进行解析
        combined_output = (proc.stdout or "") + (proc.stderr or "")

        if m50 is None:
            m50, m95 = _read_metrics_from_stdout(combined_output)

        if m50 is not None:
            # 记录成功
            (work_dir / "map50.txt").write_text(f"{m50:.6f}", encoding="utf-8")
            (work_dir / "map50_95.txt").write_text(f"{m95:.6f}", encoding="utf-8")
            done_flag.write_text("ok", encoding="utf-8")
            # 如果有 FAIL 标记则删除
            if fail_flag.exists(): fail_flag.unlink()
            return EvalResult(1, m50, m95, time.time() - t0, 0, str(work_dir), "ok")
        else:
            # 记录解析失败
            fail_flag.write_text("parse_failed", encoding="utf-8")
            (work_dir / "error.txt").write_text("Could not find 'all' metrics line in output.", encoding="utf-8")
            return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), "Parse failed")

    except Exception as e:
        # 🟢 记录代码运行异常
        (work_dir / "error.txt").write_text(str(e), encoding="utf-8")
        fail_flag.touch()
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), str(e))
    finally:
        if lock_path.exists(): lock_path.unlink()