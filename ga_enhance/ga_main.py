# ga_enhance/ga_main.py
# =========================================================
# ✅ GA 主循环（支持并行评估）
#
# 你要的终端过程输出包括：
# - 当前第几代 / 总代数
# - 第几个个体 / 总个体数
# - 当前个体 mAP、耗时、是否命中缓存
# - 每代结束：best / mean / std / global_best
# - 全局最优更新提示（打印 best params）
#
# 你要的“以后只改一处参数”也满足：
# - DIM / decode_params 都来自 enhance_ops.py
#
# 控盘策略：
# - 每代结束清理缓存：
#   只保留最近 KEEP_RECENT_SUCCESS 个成功 work_dir + 全局 best work_dir
#   失败 work_dir 默认直接删除（DELETE_FAIL_CACHE）
# =========================================================

from __future__ import annotations

import os
import re
import csv
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
from multiprocessing import Pool

from .enhance_ops import DIM, decode_params
from .eval_yolo import evaluate_params, EvalResult, CACHE_DIR

# -------------------------------
# 0) GA 超参数（你可以调）
# -------------------------------
POP_SIZE = 8
N_GEN = 5
ELITE = 2               # 精英保留数量
TOURNAMENT_K = 3        # 锦标赛选择规模
CX_PROB = 0.8           # 交叉概率
MUT_PROB = 0.3          # 变异概率
MUT_SIGMA = 0.08        # 变异强度（对 [0,1] 空间加高斯噪声）
SEED = 0

# -------------------------------
# 1) 并行设置（固定 2）
# -------------------------------
N_WORKERS = 1
QUIET_ENHANCE = True    # 并行时建议 True：避免 2 个进程一起刷屏

# -------------------------------
# 2) 缓存控盘设置
# -------------------------------
KEEP_RECENT_SUCCESS = 12
DELETE_FAIL_CACHE = True

# -------------------------------
# 3) 日志输出目录
# -------------------------------
LOG_ROOT = Path(__file__).resolve().parents[1] / "ga_logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

def _next_run_dir(start_idx: int = 4) -> Path:
    """
    run 编号自动递增（重启/关机不影响）
    """
    existing = []
    for p in LOG_ROOT.glob("run*"):
        m = re.match(r"run(\d+)$", p.name)
        if m:
            existing.append(int(m.group(1)))
    run_id = max(existing, default=start_idx - 1) + 1
    run_dir = LOG_ROOT / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _init_pop(rng: np.random.Generator) -> np.ndarray:
    """
    初始化种群：POP_SIZE × DIM，均匀随机 [0,1]
    """
    return rng.random((POP_SIZE, DIM), dtype=np.float32)

def _tournament_select(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    锦标赛选择：随机抽 K 个，选适应度最高者
    """
    idxs = rng.integers(0, len(pop), size=TOURNAMENT_K)
    best = idxs[np.argmax(fit[idxs])]
    return pop[best].copy()

def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    uniform crossover（均匀交叉）
    """
    if rng.random() > CX_PROB:
        return a.copy(), b.copy()

    mask = rng.random(DIM) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2

def _mutate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    高斯变异：对染色体加噪声，然后裁剪回 [0,1]
    """
    if rng.random() > MUT_PROB:
        return x
    noise = rng.normal(0.0, MUT_SIGMA, size=DIM).astype(np.float32)
    y = x + noise
    return _clip01(y)

def _format_params(params: dict) -> str:
    """
    把 params dict 格式化成一行可读文本
    """
    return ", ".join([f"{k}={float(v):.4f}" for k, v in params.items()])

def _cleanup_cache(keep_success_dirs: List[Path], keep_best_dir: Path | None):
    """
    清理 ga_cache：只保留指定成功目录 + best 目录
    """
    keep = set([p.resolve() for p in keep_success_dirs])
    if keep_best_dir is not None:
        keep.add(keep_best_dir.resolve())

    removed = 0
    failed_removed = 0

    for wd in CACHE_DIR.glob("work_*"):
        if not wd.is_dir():
            continue
        # 保留名单
        if wd.resolve() in keep:
            continue

        # 失败缓存：可选择直接删
        if (wd / "FAIL").exists() and DELETE_FAIL_CACHE:
            shutil.rmtree(wd, ignore_errors=True)
            removed += 1
            failed_removed += 1
            continue

        # 成功缓存：如果不在保留名单里也删（控盘）
        if (wd / "DONE").exists():
            shutil.rmtree(wd, ignore_errors=True)
            removed += 1
            continue

        # RUNNING 或其它异常目录：保守起见不删（你也可改成删）
        # 这里选择跳过

    print(f"[缓存清理] 删除 {removed} 个目录（其中 FAIL={failed_removed}）")

def main():
    rng = np.random.default_rng(SEED)

    run_dir = _next_run_dir(start_idx=4)
    run_name = run_dir.name

    # ga_history.csv
    hist_csv = run_dir / "ga_history.csv"
    best_json = run_dir / "best.json"

    # 记录表头：基因列 + 解码参数列 + 结果列
    # 参数列名来自 decode_params(0.5...0.5)，确保与你增强一致
    sample_params = decode_params(np.full((DIM,), 0.5, dtype=np.float32))
    param_cols = list(sample_params.keys())
    gene_cols = [f"g{i}" for i in range(DIM)]

    header = [
        "gen", "idx_in_gen", "global_eval_idx", "eval_tag",
        *gene_cols,
        *param_cols,
        "map50", "eval_time_sec", "ok", "cache_hit", "work_dir"
    ]

    # 初始化 CSV
    if not hist_csv.exists():
        with hist_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    # 初始化种群
    pop = _init_pop(rng)

    # 全局 best
    global_best_map = -1.0
    global_best_chrom: List[float] = []
    global_best_params: dict = {}
    global_best_work_dir: Path | None = None

    global_eval_idx = 0
    recent_success_work_dirs: List[Path] = []  # 控盘：记录最近成功缓存

    print("=" * 60)
    print(f"[GA启动] run_dir={run_dir}")
    print(f"[GA启动] POP_SIZE={POP_SIZE}, N_GEN={N_GEN}, DIM={DIM}, N_WORKERS={N_WORKERS}")
    print("=" * 60)

    # 进程池（并行评估个体）
    with Pool(processes=N_WORKERS) as pool:
        for gen in range(N_GEN):
            print("\n" + "-" * 60)
            print(f"[第 {gen+1}/{N_GEN} 代] 开始评估 {POP_SIZE} 个体...")
            print("-" * 60)

            # 这一代要评估的任务列表
            tasks = []
            for i in range(POP_SIZE):
                global_eval_idx += 1
                # eval_tag：建议包含 run + pid（主进程 pid）+ eval_idx
                # 注意：这里的 pid 是主进程的 pid，仅用于 tag 唯一，不影响并行安全
                eval_tag = f"{run_name}_p{os.getpid()}_e{global_eval_idx}"
                chrom = pop[i].copy()
                tasks.append((i, global_eval_idx, eval_tag, chrom))

            # 提交并行评估
            async_results = []
            for (idx_in_gen, gei, tag, chrom) in tasks:
                async_results.append((
                    idx_in_gen, gei, tag, chrom,
                    pool.apply_async(
                        evaluate_params,
                        kwds=dict(chrom=chrom, eval_tag=tag, force_rebuild=False, quiet_enhance=False)
                    )
                ))

            # 收集结果（按提交顺序收集，终端输出更稳定）
            fitness = np.full((POP_SIZE,), -1.0, dtype=np.float32)
            gen_maps = []
            gen_ok = 0
            gen_cache_hit = 0

            for k, (idx_in_gen, gei, tag, chrom, ar) in enumerate(async_results, start=1):
                # 等待单个结果返回（并行必须“等”，否则这一代无法做选择/交叉/变异）
                res: EvalResult = ar.get()

                params = decode_params(chrom)  # 记录用（与增强一致）
                row = [
                    gen, idx_in_gen, gei, tag,
                    *[float(x) for x in chrom.tolist()],
                    *[float(params[c]) for c in param_cols],
                    float(res.map50), float(res.time_sec),
                    int(res.ok), int(res.cache_hit), res.work_dir
                ]

                # 写 CSV（每个个体立即落盘，防止中途崩溃丢数据）
                with hist_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

                # 终端过程性输出（你要的重点！）
                status = "OK" if res.ok == 1 else "FAIL"
                hit = "HIT" if res.cache_hit == 1 else "MISS"
                print(f"[进度] gen={gen+1}/{N_GEN} 个体={idx_in_gen+1}/{POP_SIZE} "
                      f"({k}/{POP_SIZE}) 状态={status} cache={hit} "
                      f"mAP50={res.map50:.4f} time={res.time_sec:.1f}s")

                if res.ok == 1:
                    fitness[idx_in_gen] = res.map50
                    gen_maps.append(res.map50)
                    gen_ok += 1
                    gen_cache_hit += res.cache_hit

                    # 控盘记录：成功的 work_dir
                    wd = Path(res.work_dir)
                    recent_success_work_dirs.append(wd)
                    # 只保留最近 KEEP_RECENT_SUCCESS 个
                    if len(recent_success_work_dirs) > KEEP_RECENT_SUCCESS:
                        recent_success_work_dirs = recent_success_work_dirs[-KEEP_RECENT_SUCCESS:]

                    # 全局 best 更新
                    if res.map50 > global_best_map:
                        global_best_map = res.map50
                        global_best_chrom = chrom.tolist()
                        global_best_params = params
                        global_best_work_dir = wd

                        print("⭐" * 60)
                        print(f"[全局最优更新] NEW BEST! mAP50={global_best_map:.4f}")
                        print(f"[全局最优更新] chrom={global_best_chrom}")
                        print(f"[全局最优更新] params=({_format_params(global_best_params)})")
                        print("⭐" * 60)

            # 一代汇总
            if gen_maps:
                best_gen = float(np.max(gen_maps))
                mean_gen = float(np.mean(gen_maps))
                std_gen = float(np.std(gen_maps))
            else:
                best_gen, mean_gen, std_gen = -1.0, -1.0, -1.0

            print("-" * 60)
            print(f"[第 {gen+1} 代结束] OK个体={gen_ok}/{POP_SIZE}, cache命中={gen_cache_hit}/{gen_ok if gen_ok else 1}")
            print(f"[第 {gen+1} 代统计] best={best_gen:.4f} mean={mean_gen:.4f} std={std_gen:.4f} | global_best={global_best_map:.4f}")
            print("-" * 60)

            # 控盘清理（每代结束做一次）
            _cleanup_cache(recent_success_work_dirs, global_best_work_dir)

            # 如果这一代全失败：直接重新随机一代，避免 GA 断掉
            if np.all(fitness < 0):
                print("[警告] 本代全部评估失败（全是 -1）。将重新随机种群继续跑。")
                pop = _init_pop(rng)
                continue

            # 生成下一代：精英保留 + 选择/交叉/变异
            next_pop = []

            # 精英
            elite_idx = np.argsort(fitness)[::-1][:ELITE]
            for ei in elite_idx:
                next_pop.append(pop[ei].copy())

            # 其余通过遗传操作产生
            while len(next_pop) < POP_SIZE:
                p1 = _tournament_select(pop, fitness, rng)
                p2 = _tournament_select(pop, fitness, rng)
                c1, c2 = _crossover(p1, p2, rng)
                c1 = _mutate(c1, rng)
                c2 = _mutate(c2, rng)
                next_pop.append(c1)
                if len(next_pop) < POP_SIZE:
                    next_pop.append(c2)

            pop = np.stack(next_pop, axis=0).astype(np.float32)

    # 保存 best.json（包含 GA 重要超参）
    best_payload = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DIM": DIM,
        "GA_hyperparams": {
            "POP_SIZE": POP_SIZE,
            "N_GEN": N_GEN,
            "ELITE": ELITE,
            "TOURNAMENT_K": TOURNAMENT_K,
            "CX_PROB": CX_PROB,
            "MUT_PROB": MUT_PROB,
            "MUT_SIGMA": MUT_SIGMA,
            "SEED": SEED,
            "N_WORKERS": N_WORKERS,
            "KEEP_RECENT_SUCCESS": KEEP_RECENT_SUCCESS,
            "DELETE_FAIL_CACHE": DELETE_FAIL_CACHE,
        },
        "best": {
            "map50": float(global_best_map),
            "chrom": global_best_chrom,
            "params": global_best_params,
            "work_dir": str(global_best_work_dir) if global_best_work_dir else "",
        }
    }

    best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[GA完成] run_dir={run_dir}")
    print(f"[GA完成] best mAP50 = {global_best_map:.6f}")
    print(f"[GA完成] best chrom = {global_best_chrom}")
    print(f"[GA完成] best params = {global_best_params}")
    print("=" * 60)

if __name__ == "__main__":
    main()