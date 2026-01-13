# ga_enhance/woa_main.py
# =========================================================
# ✅ WOA 主循环（与 ga_main 控盘/日志逻辑完全一致）
# =========================================================

from __future__ import annotations
import os
import re
import csv
import json
import time
import shutil
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from multiprocessing import Pool

# 保持与框架一致的导入
from .enhance_ops import DIM, decode_params
from .eval_yolo import evaluate_params, EvalResult, CACHE_DIR

# -------------------------------
# 0) WOA 超参数 (鲸鱼算法核心)
# -------------------------------
N_WHALES = 50  # 对应原 POP_SIZE
MAX_ITER = 100  # 最大迭代次数
B_CONSTANT = 1.0  # 螺旋形状常数
PATIENCE = 15  # 早停机制：连续 15 代不提升则停止

# -------------------------------
# 1) 并行与日志设置 (同步 ga_main)
# -------------------------------
N_WORKERS = 12
LOG_ROOT = Path(__file__).resolve().parents[1] / "woa_logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 2) 缓存控盘设置 (同步 ga_main)
# -------------------------------
KEEP_RECENT_SUCCESS = 12
DELETE_FAIL_CACHE = True  # 🟢 设为 False 以保留错误日志供排查


def _next_run_dir(start_idx: int = 1) -> Path:
    existing = []
    for p in LOG_ROOT.glob("run*"):
        m = re.match(r"run(\d+)$", p.name)
        if m: existing.append(int(m.group(1)))
    run_id = max(existing, default=start_idx - 1) + 1
    run_dir = LOG_ROOT / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _format_params(params: dict) -> str:
    return ", ".join([f"{k}={float(v):.4f}" for k, v in params.items()])


def _cleanup_cache(keep_success_dirs: List[Path], keep_best_dir: Path | None):
    """
    清理缓存：同步 ga_main 的控盘逻辑
    """
    keep = set([p.resolve() for p in keep_success_dirs])
    if keep_best_dir is not None:
        keep.add(keep_best_dir.resolve())

    removed, failed_removed = 0, 0
    for wd in CACHE_DIR.glob("work_*"):
        if not wd.is_dir(): continue
        if wd.resolve() in keep: continue

        # 失败缓存：如果开启了删除则删，否则保留日志
        if (wd / "FAIL").exists() and DELETE_FAIL_CACHE:
            shutil.rmtree(wd, ignore_errors=True)
            removed += 1
            failed_removed += 1
            continue

        # 成功缓存：不在保留名单里的旧成功记录，直接删（控盘）
        if (wd / "DONE").exists():
            shutil.rmtree(wd, ignore_errors=True)
            removed += 1

    print(f"[缓存清理] 删除 {removed} 个目录（其中 FAIL={failed_removed}）")


def main():
    run_dir = _next_run_dir()
    run_name = run_dir.name
    hist_csv = run_dir / "woa_history.csv"
    best_json = run_dir / "best.json"

    # 同步 ga_main 的 CSV 表头逻辑
    sample_params = decode_params(np.full((DIM,), 0.5, dtype=np.float32))
    param_cols = list(sample_params.keys())
    gene_cols = [f"g{i}" for i in range(DIM)]
    header = ["iter", "whale_idx", "global_eval_idx", "eval_tag"] + gene_cols + param_cols + \
             ["map50", "eval_time_sec", "ok", "cache_hit", "work_dir"]

    if not hist_csv.exists():
        with hist_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    # 1. 初始化种群
    whales = np.random.rand(N_WHALES, DIM).astype(np.float32)

    global_best_map = -1.0
    global_best_pos = None
    global_best_params = {}
    global_best_work_dir: Path | None = None

    global_eval_idx = 0
    recent_success_work_dirs: List[Path] = []
    no_improve_count = 0

    print("=" * 60)
    print(f"[WOA 启动] run_dir={run_dir}")
    print(f"[WOA 启动] N_WHALES={N_WHALES}, MAX_ITER={MAX_ITER}, DIM={DIM}, N_WORKERS={N_WORKERS}")
    print("=" * 60)

    with Pool(processes=N_WORKERS) as pool:
        for t in range(MAX_ITER):
            prev_best_map = global_best_map
            a = 2 - t * (2 / MAX_ITER)

            print("\n" + "-" * 60)
            print(f"[第 {t + 1}/{MAX_ITER} 轮迭代] 开始评估 {N_WHALES} 头鲸鱼...")
            print("-" * 60)

            # --- A. 并行评估 ---
            tasks = []
            for i in range(N_WHALES):
                global_eval_idx += 1
                eval_tag = f"{run_name}_e{global_eval_idx}"
                tasks.append((i, global_eval_idx, eval_tag, whales[i].copy()))

            async_results = []
            for (idx, gei, tag, chrom) in tasks:
                async_results.append((idx, gei, tag, chrom, pool.apply_async(
                    evaluate_params, kwds=dict(chrom=chrom, eval_tag=tag, force_rebuild=False, quiet_enhance=True)
                )))

            # 收集结果 (同步 ga_main 的输出格式)
            gen_maps = []
            for k, (idx, gei, tag, chrom, ar) in enumerate(async_results, start=1):
                res: EvalResult = ar.get()
                params = decode_params(chrom)

                # 写 CSV (每条记录立即落盘)
                row = [t, idx, gei, tag] + chrom.tolist() + [params[c] for c in param_cols] + \
                      [res.map50, res.time_sec, res.ok, res.cache_hit, res.work_dir]
                with hist_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

                status = "OK" if res.ok == 1 else "FAIL"
                hit = "HIT" if res.cache_hit == 1 else "MISS"
                print(f"[进度] iter={t + 1}/{MAX_ITER} 鲸鱼={idx + 1}/{N_WHALES} "
                      f"({k}/{N_WHALES}) 状态={status} cache={hit} "
                      f"mAP50={res.map50:.4f} time={res.time_sec:.1f}s")

                if res.ok == 1:
                    gen_maps.append(res.map50)
                    wd = Path(res.work_dir)
                    recent_success_work_dirs.append(wd)
                    if len(recent_success_work_dirs) > KEEP_RECENT_SUCCESS:
                        recent_success_work_dirs = recent_success_work_dirs[-KEEP_RECENT_SUCCESS:]

                    # 更新全局最优 (同步 ga_main 的星星打印)
                    if res.map50 > global_best_map:
                        global_best_map = res.map50
                        global_best_pos = chrom.copy()
                        global_best_params = params
                        global_best_work_dir = wd
                        print("⭐" * 60)
                        print(f"[全局最优更新] NEW BEST! mAP50={global_best_map:.4f}")
                        print(f"[全局最优更新] params=({_format_params(global_best_params)})")
                        print("⭐" * 60)

            # --- B. 统计与早停 ---
            if global_best_map > prev_best_map:
                no_improve_count = 0
            else:
                no_improve_count += 1

            if gen_maps:
                best_gen, mean_gen, std_gen = np.max(gen_maps), np.mean(gen_maps), np.std(gen_maps)
                print(
                    f"[本代统计] best={best_gen:.4f} mean={mean_gen:.4f} std={std_gen:.4f} | global_best={global_best_map:.4f}")

            if no_improve_count >= PATIENCE:
                print(f"\n[早停触发] 连续 {PATIENCE} 代无提升，自动结束。")
                break

            # --- C. 更新鲸鱼位置 (WOA 核心) ---
            if global_best_pos is not None:
                new_whales = []
                for i in range(N_WHALES):
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    l = np.random.uniform(-1, 1)
                    p = np.random.rand()

                    if p < 0.5:
                        if abs(A) < 1:
                            D = abs(C * global_best_pos - whales[i])
                            new_pos = global_best_pos - A * D
                        else:
                            rand_idx = np.random.randint(0, N_WHALES)
                            new_pos = whales[rand_idx] - A * abs(C * whales[rand_idx] - whales[i])
                    else:
                        new_pos = abs(global_best_pos - whales[i]) * np.exp(B_CONSTANT * l) * np.cos(
                            2 * np.pi * l) + global_best_pos
                    new_whales.append(_clip01(new_pos))
                whales = np.stack(new_whales).astype(np.float32)
            else:
                whales = np.random.rand(N_WHALES, DIM).astype(np.float32)

            _cleanup_cache(recent_success_work_dirs, global_best_work_dir)

    # 3. 保存最终结果 (同步 ga_main 的 best.json 结构)
    best_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algo": "WOA",
        "best": {
            "map50": float(global_best_map),
            "params": global_best_params,
            "chrom": global_best_pos.tolist() if global_best_pos is not None else [],
            "work_dir": str(global_best_work_dir) if global_best_work_dir else ""
        }
    }
    best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(f"\n[WOA 完成] run_dir={run_dir} | Best mAP50: {global_best_map:.6f}")


if __name__ == "__main__":
    main()