# ga_enhance/eval_yolo.py
# =========================================================
# ✅ 评估函数（并行安全，100% 保留原有原子锁与清理框架）
# =========================================================

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
from .enhance_ops import decode_params, enhance_val_images

# -------------------------------
# 路径配置 (原封不动)
# -------------------------------
ROOT = Path("/home/zhangxu/yolov5-7.0")
YOLO_DIR = ROOT
SPLIT_DIR = ROOT / "datasets" / "UTDAC2020"
VAL_NAME = "val"
BASE_DATA_YAML = ROOT / "data" / "utdac.yaml"
WEIGHTS = ROOT / "runs" / "train" / "yolov5n_scratch_baseline" / "weights" / "best.pt"
CACHE_DIR = ROOT / "ga_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
CONF_THRES = 0.001
IOU_THRES = 0.6
DEVICE = ""
YOLO_WORKERS = 0
YOLO_BATCH_SIZE = 16
YOLO_HALF = True

# -------------------------------
# ✅ 针对你的顾虑，调低清理阈值 (更激进的防爆盘)
# -------------------------------
KEEP_RECENT_N = 10  # 🟢 原来是 80，现在只保留最近 10 个，大幅节省空间
MAX_CACHE_GB = 5  # 🟢 原来是 25GB，现在超过 5GB 就强制清理

DEFAULT_QUIET_ENHANCE = True
DEFAULT_VERBOSE = True


@dataclass
class EvalResult:
    ok: int
    map50: float
    map50_95: float
    time_sec: float
    cache_hit: int
    work_dir: str
    msg: str


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _log(msg: str, verbose: bool = True):
    if verbose: print(msg, flush=True)


def _write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s if s is not None else "", encoding="utf-8", errors="ignore")


def _stable_hash(chrom: np.ndarray) -> str:
    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)
    q = np.round(chrom, 6)
    key = f"{VAL_NAME}|" + ",".join([f"{x:.6f}" for x in q.tolist()])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# 🟢 方案优化：更鲁棒的 CSV 解析
def _read_metrics_from_results_csv(save_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        csv_path = save_dir.parent / "results.csv"
    if not csv_path.exists(): return None, None

    import csv as _csv
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
            if not rows: return None, None
            row = {k.strip(): v for k, v in rows[-1].items()}  # 🟢 增加 strip() 防止空格干扰

            m50 = next((row[k] for k in ["metrics/mAP_0.5", "mAP@0.5", "map50", "metrics/mAP50"] if k in row), None)
            m50_95 = next(
                (row[k] for k in ["metrics/mAP_0.5:0.95", "mAP@0.5:0.95", "map50-95", "metrics/mAP50_95"] if k in row),
                None)

            if m50 is not None and m50_95 is not None:
                return float(m50), float(m50_95)
    except:
        pass
    return None, None


# 🟢 方案优化：抛弃脆弱的正则，改用字符串分割法
def _read_metrics_from_stdout(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if text is None: return None, None
    text = _ANSI_RE.sub("", text)

    for line in text.splitlines():
        # 寻找包含 "all" 的那一行结果
        parts = line.split()
        if len(parts) >= 7 and parts[0] == 'all':
            try:
                # YOLO 输出顺序: all, images, instances, P, R, mAP50, mAP50-95
                m50 = float(parts[5])
                m50_95 = float(parts[6])
                return m50, m50_95
            except:
                continue
    return None, None


# -------------------------------
# 缓存控制功能 (100% 保留原有逻辑)
# -------------------------------
def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists(): return 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except:
                pass
    return total


def prune_cache(cache_dir: Path = CACHE_DIR, keep_recent_n: int = KEEP_RECENT_N, max_cache_gb: float = MAX_CACHE_GB,
                verbose: bool = False):
    if not cache_dir.exists(): return
    works = sorted(cache_dir.glob("work_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not works: return
    total_bytes = sum(_dir_size_bytes(w) for w in works)
    max_bytes = int(max_cache_gb * (1024 ** 3))
    protected = set(works[:keep_recent_n])
    if total_bytes <= max_bytes: return
    for w in reversed(works):
        if w in protected: continue
        if any(w.glob("RUNNING_*")) or (w / "LOCK").exists(): continue
        sz = _dir_size_bytes(w)
        try:
            shutil.rmtree(w, ignore_errors=True)
            total_bytes -= sz
            if verbose: _log(f"[缓存清理] 删除 {w.name} 释放 {sz / 1024 ** 3:.2f} GB", True)
        except:
            pass
        if total_bytes <= max_bytes: break


# =========================================================
# ✅ 评估函数 (100% 保留原子锁及并行逻辑)
# =========================================================
def evaluate_params(chrom, eval_tag: str, force_rebuild: bool = False, quiet_enhance: bool = DEFAULT_QUIET_ENHANCE,
                    verbose: bool = DEFAULT_VERBOSE, wait_lock_sec: int = 3600) -> EvalResult:
    t0 = time.time()
    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)

    try:
        params: Dict[str, Any] = decode_params(chrom)
    except Exception as e:
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, "", f"[解码失败] {e}")

    h = _stable_hash(chrom)
    work_dir = CACHE_DIR / f"work_{h}"
    done_flag, fail_flag, lock_path = work_dir / "DONE", work_dir / "FAIL", work_dir / "LOCK"

    # 缓存命中逻辑
    if done_flag.exists() and (not force_rebuild):
        try:
            m50 = float((work_dir / "map50.txt").read_text(encoding="utf-8").strip())
            m50_95 = 0.0
            if (work_dir / "map50_95.txt").exists():
                m50_95 = float((work_dir / "map50_95.txt").read_text(encoding="utf-8").strip())
            return EvalResult(1, m50, m50_95, time.time() - t0, 1, str(work_dir), "cache_hit")
        except:
            pass

    work_dir.mkdir(parents=True, exist_ok=True)

    # 原子锁机制 (保护并行)
    got_lock = False
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        got_lock = True
    except FileExistsError:
        got_lock = False

    if not got_lock and (not force_rebuild):
        _log(f"[等待] {work_dir.name} 正在被计算...", verbose)
        wait_t0 = time.time()
        while time.time() - wait_t0 < wait_lock_sec:
            if done_flag.exists():
                try:
                    m50 = float((work_dir / "map50.txt").read_text().strip())
                    m50_95 = 0.0
                    if (work_dir / "map50_95.txt").exists():
                        m50_95 = float((work_dir / "map50_95.txt").read_text().strip())
                    return EvalResult(1, m50, m50_95, time.time() - t0, 1, str(work_dir), "cache_wait_hit")
                except:
                    break
            if fail_flag.exists():
                return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), "FAIL")
            time.sleep(1.0)
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), "[超时]")

    running_flag = work_dir / f"RUNNING_{os.getpid()}"
    running_flag.write_text(eval_tag, encoding="utf-8")

    try:
        # 准备数据 (完全保留原有逻辑)
        if force_rebuild and work_dir.exists():
            for child in work_dir.iterdir():
                if child.name == "LOCK": continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)

        src_img_dir = SPLIT_DIR / "images" / VAL_NAME
        src_label_dir = SPLIT_DIR / "labels" / VAL_NAME
        dst_img_dir, dst_label_dir = work_dir / "images", work_dir / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        enhance_val_images(src_img_dir, dst_img_dir, params, quiet=quiet_enhance)
        for p in src_label_dir.glob("*.txt"): shutil.copy2(p, dst_label_dir / p.name)

        data = yaml.safe_load(BASE_DATA_YAML.read_text(encoding="utf-8"))
        data["val"] = str(dst_img_dir)
        tmp_yaml = work_dir / f"data_tmp_{eval_tag}.yaml"
        tmp_yaml.write_text(yaml.safe_dump(data))

        # 运行 YOLO (完全保留)
        yolo_out_root = work_dir / "yolo_val"
        yolo_out_root.mkdir(parents=True, exist_ok=True)
        cmd = ["python", "val.py", "--data", str(tmp_yaml), "--weights", str(WEIGHTS), "--imgsz", str(IMG_SIZE),
               "--conf-thres", str(CONF_THRES), "--iou-thres", str(IOU_THRES), "--device", DEVICE,
               "--project", str(yolo_out_root), "--name", eval_tag, "--exist-ok", "--half"]

        proc = subprocess.run(cmd, cwd=str(YOLO_DIR), capture_output=True, text=True)
        _write_text(work_dir / "val_stdout.txt", proc.stdout)
        _write_text(work_dir / "val_stderr.txt", proc.stderr)

        if proc.returncode != 0: raise RuntimeError("val.py failed")

        # 🟢 指标解析核心逻辑 (采用字符串分割法)
        save_dir = yolo_out_root / eval_tag
        m50, m50_95 = _read_metrics_from_results_csv(save_dir)
        if m50 is None:
            # 关键修改：同时搜索 stdout 和 stderr，因为 YOLO 输出流不固定
            m50, m50_95 = _read_metrics_from_stdout(proc.stdout)
            if m50 is None:
                m50, m50_95 = _read_metrics_from_stdout(proc.stderr)

        if m50 is None: raise RuntimeError("Parse metrics failed")

        # 成功保存
        (work_dir / "map50.txt").write_text(f"{float(m50):.6f}")
        (work_dir / "map50_95.txt").write_text(f"{float(m50_95):.6f}")
        done_flag.write_text("ok")

        _log(f"[完成] {eval_tag} | mAP50={m50:.4f}, m50_95={m50_95:.4f}")
        prune_cache()

        return EvalResult(1, float(m50), float(m50_95), time.time() - t0, 0, str(work_dir), "ok")

    except Exception as e:
        _write_text(work_dir / "error.txt", str(e))
        fail_flag.write_text("fail")
        return EvalResult(0, -1.0, -1.0, time.time() - t0, 0, str(work_dir), str(e))
    finally:
        if running_flag.exists(): running_flag.unlink(missing_ok=True)
        if lock_path.exists(): lock_path.unlink(missing_ok=True)