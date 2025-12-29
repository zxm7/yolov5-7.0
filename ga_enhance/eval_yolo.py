# ga_enhance/eval_yolo.py
# =========================================================
# ✅ evaluate_params：评估“一个个体”的 mAP50（支持并行）
#
# 核心流程：
# 1) chrom -> params（decode_params 来自 enhance_ops，唯一权威）
# 2) 生成增强后的 val_ga 图片到 work_dir/images
# 3) 复制 labels 到 work_dir/labels
# 4) 生成 data_tmp.yaml 指向 work_dir/images
# 5) 在 yolov5/ 下执行 val.py 得到 mAP50
# 6) 成功：写 DONE + map50.txt（可缓存复用）
#    失败：写 FAIL + error.txt + stdout/stderr（定位原因）
#
# 并行安全关键点（必看）：
# - 同一个 chrom（同一个 work_dir）只能允许一个进程“生成增强图 + 跑 val”
# - 用 LOCK（原子创建）实现互斥：其他进程等待 DONE（或 FAIL）
#
# 缓存清理（避免爆盘）：
# - 提供 prune_cache()：保留最近 N 个 work_dir，且总容量不超过 MAX_CACHE_GB
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
from typing import Optional, Dict, Any

import yaml
import numpy as np

# ✅ 唯一权威：只从 enhance_ops 来解码 / 增强
from .enhance_ops import decode_params, enhance_val_images


# -------------------------------
# 路径配置（按你的项目结构）
# -------------------------------
# 1. 项目根目录：你 YOLOv5 代码存放的绝对路径
ROOT = Path("/home/zhangxu/yolov5-7.0")
# 2. YOLOv5 执行目录：你的根目录本身就是 YOLOv5 项目
YOLO_DIR = ROOT
# 3. 数据集根目录：指向包含 images/ 和 labels/ 的数据集文件夹
# 注意：这里要指向你 8:1:1 划分后的那个文件夹
SPLIT_DIR = ROOT / "datasets" / "UTDAC2020"
# 4. 验证集文件夹名称：对应你划分出的那 10% 验证集文件夹名
VAL_NAME = "val"
# 5. 基础配置文件：指向你训练时用的那个 utdac.yaml
BASE_DATA_YAML = ROOT / "data" / "utdac.yaml"
# 6. 权重文件路径：指向你刚跑出来的那个 0.797 分的最好权重 (YOLOv5n)
WEIGHTS = ROOT / "runs" / "train" / "yolov5n_scratch_baseline" / "weights" / "best.pt"
# 7. 缓存目录：用于存放 GA 过程中产生的临时图像
CACHE_DIR = ROOT / "ga_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# YOLO 验证参数（建议保持标准）
# -------------------------------
IMG_SIZE = 640
CONF_THRES = 0.001      # ✅ mAP 评估通常使用 0.001（避免 invalid results）
IOU_THRES = 0.6
DEVICE = "0"            # 强制用 GPU：0
YOLO_WORKERS = 0        # val.py 的 dataloader workers
YOLO_BATCH_SIZE = 16     # 控制显存，防止 OOM
YOLO_HALF = True
# -------------------------------
# 缓存控制：避免爆盘
# -------------------------------
KEEP_RECENT_N = 80       # 至少保留最近 80 个 work_dir（你可以按硬盘改）
MAX_CACHE_GB = 25        # ga_cache 总容量上限（GB），超过则按旧的先删


# -------------------------------
# 日志控制
# -------------------------------
DEFAULT_QUIET_ENHANCE = True     # 并行时建议 True，避免终端刷爆
DEFAULT_VERBOSE = True           # True: 打印阶段信息


@dataclass
class EvalResult:
    ok: int                 # 1成功 / 0失败
    map50: float            # 成功为 [0,1]，失败为 -1
    time_sec: float
    cache_hit: int          # 1=缓存命中 / 0=实际执行
    work_dir: str
    msg: str


# -------------------------------
# 小工具
# -------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, flush=True)


def _write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s if s is not None else "", encoding="utf-8", errors="ignore")


def _stable_hash(chrom: np.ndarray) -> str:
    """
    生成稳定 hash：避免浮点抖动 & 缓存误命中。
    hash 只跟：
      - VAL_NAME（换验证集后不误命中旧缓存）
      - chrom（量化后）
    """
    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)
    q = np.round(chrom, 6)
    key = f"{VAL_NAME}|" + ",".join([f"{x:.6f}" for x in q.tolist()])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _read_map50_from_results_csv(save_dir: Path) -> Optional[float]:
    """
    优先从 yolov5 val 输出的 results.csv 读取 mAP@0.5
    比解析 stdout 稳定。
    """
    # 兼容不同版本目录结构：尝试多个候选路径
    candidates = [
        save_dir / "results.csv",
        save_dir.parent / "results.csv",
        save_dir / "results.txt",        # 有些版本只有 txt
        save_dir.parent / "results.txt",
    ]

    csv_path = next((p for p in candidates if p.exists() and p.suffix == ".csv"), None)
    if csv_path is None:
        return None

    import csv as _csv
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        row = rows[-1]

    # 常见列名（yolov5 版本可能不同）
    for k in ["metrics/mAP_0.5", "mAP@0.5", "map50", "metrics/mAP50"]:
        if k in row:
            try:
                return float(row[k])
            except Exception:
                return None
    return None


def _read_map50_from_stdout(text: Optional[str]) -> Optional[float]:
    if text is None: return None # ✅ 如果没读到文字，直接返回空，不再报错
    text = _ANSI_RE.sub("", text)


    m = re.search(
        r"^\s*all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except Exception:
                pass
    return total


def prune_cache(cache_dir: Path = CACHE_DIR,
                keep_recent_n: int = KEEP_RECENT_N,
                max_cache_gb: float = MAX_CACHE_GB,
                verbose: bool = False):
    """
    清理 ga_cache：避免长期跑 GA 把硬盘吃爆。
    策略：
      1) work_* 按修改时间排序，保留最近 keep_recent_n 个
      2) 剩余的按旧到新删除，直到总容量 <= max_cache_gb
    注意：
      - 正在运行的 work_dir 里有 RUNNING_* 或 LOCK，则跳过
    """
    if not cache_dir.exists():
        return

    works = sorted(cache_dir.glob("work_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not works:
        return

    # 先算总容量
    total_bytes = sum(_dir_size_bytes(w) for w in works)
    max_bytes = int(max_cache_gb * (1024**3))

    # 先保护最近 N 个
    protected = set(works[:keep_recent_n])

    # 如果已达标就不删
    if total_bytes <= max_bytes:
        return

    # 从最旧开始删
    for w in reversed(works):
        if w in protected:
            continue

        # 跳过正在运行
        if any(w.glob("RUNNING_*")) or (w / "LOCK").exists():
            continue

        sz = _dir_size_bytes(w)
        try:
            shutil.rmtree(w, ignore_errors=True)
            total_bytes -= sz
            if verbose:
                _log(f"[缓存清理] 删除 {w.name} 释放 {sz/1024**3:.2f} GB", True)
        except Exception:
            pass

        if total_bytes <= max_bytes:
            break


# =========================================================
# ✅ 评估函数（并行安全）
# =========================================================
def evaluate_params(
    chrom,
    eval_tag: str,
    force_rebuild: bool = False,
    quiet_enhance: bool = DEFAULT_QUIET_ENHANCE,
    verbose: bool = DEFAULT_VERBOSE,
    wait_lock_sec: int = 3600,
) -> EvalResult:
    """
    评估一个个体（可被多进程并行调用）。

    参数：
      chrom: np.ndarray，长度=DIM，值∈[0,1]
      eval_tag: 唯一标签（建议包含 run_id + pid + eval_idx）
      force_rebuild: True 无视缓存强制重建
      quiet_enhance: True 增强过程不刷屏
      verbose: True 打印阶段信息
      wait_lock_sec: 等锁最长时间（秒）

    返回：
      EvalResult(ok/map50/time/cache_hit/work_dir/msg)
    """
    t0 = time.time()

    chrom = np.asarray(chrom, dtype=np.float32).reshape(-1)

    # ---------- 1) 解码参数 ----------
    try:
        params: Dict[str, Any] = decode_params(chrom)
    except Exception as e:
        return EvalResult(0, -1.0, time.time() - t0, 0, "", f"[解码失败] {e}")

    # ---------- 2) work_dir ----------
    h = _stable_hash(chrom)
    work_dir = CACHE_DIR / f"work_{h}"
    done_flag = work_dir / "DONE"
    fail_flag = work_dir / "FAIL"
    lock_path = work_dir / "LOCK"

    # ---------- 3) 缓存命中 ----------
    if done_flag.exists() and (not force_rebuild):
        try:
            map50 = float((work_dir / "map50.txt").read_text(encoding="utf-8").strip())
            return EvalResult(1, map50, time.time() - t0, 1, str(work_dir), "cache_hit")
        except Exception:
            # 缓存损坏：走重建
            pass

    # force_rebuild：清空旧目录（注意：并行情况下，只有拿到锁的进程才会做）
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 4) 原子锁：并行互斥 ----------
    got_lock = False
    if force_rebuild:
        # 强制重建也得抢锁，否则会和其他进程打架
        pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        got_lock = True
    except FileExistsError:
        got_lock = False

    if not got_lock and (not force_rebuild):
        # 说明别的进程正在处理同一个 work_dir：等待 DONE/FAIL
        _log(f"[等待] {work_dir.name} 正在被其他进程计算，等待结果...", verbose)
        wait_t0 = time.time()
        while time.time() - wait_t0 < wait_lock_sec:
            if done_flag.exists():
                try:
                    map50 = float((work_dir / "map50.txt").read_text(encoding="utf-8").strip())
                    return EvalResult(1, map50, time.time() - t0, 1, str(work_dir), "cache_wait_hit")
                except Exception:
                    break
            if fail_flag.exists():
                # 别人失败了，你可以选择：直接失败 or 自己重试
                err = (work_dir / "error.txt").read_text(encoding="utf-8", errors="ignore") if (work_dir / "error.txt").exists() else ""
                return EvalResult(0, -1.0, time.time() - t0, 0, str(work_dir), f"[等待后发现失败] {err[:200]}")
            time.sleep(1.0)

        # 超时：你可以选择自己强制重建（这里直接返回失败，避免无限耗时）
        return EvalResult(0, -1.0, time.time() - t0, 0, str(work_dir), "[等待锁超时] 可能有卡死进程未释放 LOCK")

    # 能走到这里，说明：
    # - got_lock=True（我来干活）
    # - 或 force_rebuild=True（下面也会干活，但也建议 got_lock=True 才安全）

    running_flag = work_dir / f"RUNNING_{os.getpid()}"
    running_flag.write_text(eval_tag, encoding="utf-8")

    try:
        # force_rebuild：拿到锁后再清空，避免并行互删
        if force_rebuild and work_dir.exists():
            # 注意：别把 LOCK 删了，所以先记住 lock 已在
            for child in work_dir.iterdir():
                if child.name in ("LOCK",):
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except Exception:
                        pass

        # ---------- 5) 检查数据路径 (已根据你的 images/val 结构修正) ----------
        src_img_dir = SPLIT_DIR / "images" / VAL_NAME    # 拼接结果：.../UTDAC2020/images/val
        src_label_dir = SPLIT_DIR / "labels" / VAL_NAME  # 拼接结果：.../UTDAC2020/labels/val
        if not src_img_dir.exists():
            raise FileNotFoundError(f"[数据错误] 找不到 {src_img_dir}")
        if not src_label_dir.exists():
            raise FileNotFoundError(f"[数据错误] 找不到 {src_label_dir}")
        if not BASE_DATA_YAML.exists():
            raise FileNotFoundError(f"[配置错误] 找不到 base data.yaml: {BASE_DATA_YAML}")
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"[权重错误] 找不到 weights: {WEIGHTS}")
        if not YOLO_DIR.exists():
            raise FileNotFoundError(f"[配置错误] 找不到 yolov5 目录: {YOLO_DIR}")

        dst_img_dir = work_dir / "images"
        dst_label_dir = work_dir / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        # 写 params.yaml（方便追溯）
        with (work_dir / "params.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"eval_tag": eval_tag, "chrom": chrom.tolist(), "params": params},
                f, allow_unicode=True
            )

        # ---------- 6) 增强图片 ----------
        # ✅ 只有在没图/或 force_rebuild 时才增强（否则直接复用）
        need_enhance = force_rebuild or (not any(dst_img_dir.glob("*.jpg")) and not any(dst_img_dir.glob("*.png")))
        if need_enhance:
            _log(f"[增强] {eval_tag} | work={work_dir.name} | 开始增强 val({VAL_NAME}) ...", verbose)
            n = enhance_val_images(src_img_dir, dst_img_dir, params, quiet=quiet_enhance)
            if n <= 0:
                raise RuntimeError("[增强失败] 输出 0 张图片")
            _log(f"[增强] {eval_tag} | 完成：{n} 张", verbose)
        else:
            _log(f"[增强] {eval_tag} | 命中增强缓存（跳过增强）", verbose)

        # ---------- 7) labels 复制 ----------
        # ✅ 只在必要时复制，避免每次都拷贝 905 个 txt
        need_copy_labels = force_rebuild or (not any(dst_label_dir.glob("*.txt")))
        if need_copy_labels:
            _log(f"[标签] {eval_tag} | 复制 labels ...", verbose)
            for p in src_label_dir.glob("*.txt"):
                shutil.copy2(p, dst_label_dir / p.name)
            _log(f"[标签] {eval_tag} | labels 复制完成", verbose)
        else:
            _log(f"[标签] {eval_tag} | 命中 labels 缓存（跳过复制）", verbose)

        # ---------- 8) 写临时 data yaml ----------
        data = yaml.safe_load(BASE_DATA_YAML.read_text(encoding="utf-8"))

        # 关键：val 指向增强后的 images 目录
        data["val"] = str(dst_img_dir)

        # 保险：保证 nc/names 不丢（你的 base_data_yaml 应该已有）
        tmp_yaml = work_dir / f"data_tmp_{eval_tag}.yaml"
        tmp_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

        # ---------- 9) 运行 YOLOv5 val.py ----------
        yolo_out_root = work_dir / "yolo_val"
        yolo_out_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "val.py",
            "--data", str(tmp_yaml),
            "--weights", str(WEIGHTS),
            "--imgsz", str(IMG_SIZE),
            "--conf-thres", str(CONF_THRES),
            "--iou-thres", str(IOU_THRES),
            "--device", DEVICE,
            "--workers", str(YOLO_WORKERS),
            "--batch-size", str(YOLO_BATCH_SIZE),
            "--project", str(yolo_out_root),
            "--name", eval_tag,
            "--exist-ok",
        ]
        if YOLO_HALF:
            cmd.append("--half")

        _log(f"[YOLO] {eval_tag} | 开始验证（可能需要几十秒~几分钟）...", verbose)
        proc = subprocess.run(
            cmd,
            cwd=str(YOLO_DIR),
            capture_output=True,
            text=True
        )

        # 无论成功失败，都落盘，方便定位
        _write_text(work_dir / "val_stdout.txt", proc.stdout)
        _write_text(work_dir / "val_stderr.txt", proc.stderr)

        if proc.returncode != 0:
            fail_flag.write_text(f"val.py returncode={proc.returncode}", encoding="utf-8")
            raise RuntimeError(f"[YOLO验证失败] returncode={proc.returncode}，看 {work_dir/'val_stderr.txt'}")

        # ---------- 10) 解析 mAP50 ----------
        # yolov5 默认输出在：yolo_out_root/eval_tag/
        save_dir = yolo_out_root / eval_tag

        map50 = _read_map50_from_results_csv(save_dir)
        if map50 is None:
            map50 = _read_map50_from_stdout(proc.stdout) or _read_map50_from_stdout(proc.stderr)

        if map50 is None:
            fail_flag.write_text("map50 parse failed", encoding="utf-8")
            raise RuntimeError(
                f"[解析失败] 没拿到 mAP50。请看：\n"
                f"  {work_dir/'val_stdout.txt'}\n"
                f"  {work_dir/'val_stderr.txt'}"
            )

        # ---------- 11) 成功：写 DONE ----------
        (work_dir / "map50.txt").write_text(f"{float(map50):.6f}", encoding="utf-8")
        done_flag.write_text("ok", encoding="utf-8")
        if fail_flag.exists():
            try:
                fail_flag.unlink()
            except Exception:
                pass

        _log(f"[完成] {eval_tag} | mAP50={map50:.6f} | work={work_dir.name}", verbose)

        # ---------- 12) 顺手清缓存（避免爆盘） ----------
        # 你也可以选择只在 ga_main 每代结束后再调用
        prune_cache(verbose=False)

        return EvalResult(1, float(map50), time.time() - t0, 0, str(work_dir), "ok")

    except Exception as e:
        _write_text(work_dir / "error.txt", str(e))
        fail_flag.write_text("fail", encoding="utf-8")
        _log(f"[失败] {eval_tag} | {e}", verbose)
        return EvalResult(0, -1.0, time.time() - t0, 0, str(work_dir), str(e))

    finally:
        # 清理 RUNNING
        if running_flag.exists():
            try:
                running_flag.unlink()
            except Exception:
                pass

        # 释放 LOCK（非常关键）
        if lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass