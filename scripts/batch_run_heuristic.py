#!/usr/bin/env python3
"""
Batch-run AIDE with the default HeuristicPolicy over CTU tasks and random seeds.

Writes one log bundle per (task, seed) under --out_logs_dir/<exp_name>/ and appends
rows to --runs_index (CSV): task, seed, log_dir, status, final_best_metric, n_nodes, n_buggy.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.agent import Agent
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.policy import (
    ControllerPolicy,
    HeuristicPolicy,
    HeuristicPlusControllerPolicy,
    SearchPolicy,
    UnifiedControllerPolicy,
)
from aide.rlhf.ctu_dataset import (
    build_aide_inputs,
    is_kaggle_index,
    load_task_index,
    materialize_workspace,
)
from data.hf_utils import HF_DATA_PREFIX, HF_KAGGLE_PREFIX
from aide.utils.config import _load_cfg, load_task_desc, prep_agent_workspace, prep_cfg, save_run

from dotenv import load_dotenv

load_dotenv()

SeedArg = int | str  # int or "auto"


def _safe_dirname(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="data/ctu_datasets_info.csv")
    p.add_argument("--max_tasks", type=int, default=5, help="First N tasks from the CSV.")
    p.add_argument("--task_offset", type=int, default=0, help="Skip first N tasks.")
    p.add_argument("--seeds_per_task", type=int, default=2)
    p.add_argument("--steps", type=int, default=35, help="AIDE steps per run (passed to exp loop).")
    p.add_argument("--out_logs_dir", type=str, default="data/heuristic_runs/logs")
    p.add_argument("--out_workspace_dir", type=str, default="data/heuristic_runs/workspaces")
    p.add_argument(
        "--runs_index",
        type=str,
        default="data/heuristic_runs_index.csv",
        help="Append CSV with one row per (task, seed) run.",
    )
    p.add_argument("--materialize_root", type=str, default="data/heuristic_runs/ctu_materialized")
    p.add_argument(
        "--data_source",
        type=str,
        default="hf",
        choices=["hf", "local", "relbench"],
        help="Where to load parquet tables from (default: Hugging Face).",
    )
    p.add_argument(
        "--local_data_root",
        type=str,
        default=None,
        help="Root dir for --data_source local (e.g. data/kaggle_materialized/kaggle).",
    )
    p.add_argument("--hf_repo", type=str, default="guilhermedrud/ctu_datasets")
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Run a single task by row_name (from CSV). Use with --seed.",
    )
    p.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Run a single task by index into the CSV slice (after --task_offset).",
    )
    p.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Seed for a run: integer, or 'auto' (next unused seed on HF for the task).",
    )
    p.add_argument(
        "--upload_hf",
        action="store_true",
        help="Upload experiment logs to Hugging Face after each run (data/hf_utils.py).",
    )
    p.add_argument(
        "--upload_gcs",
        action="store_true",
        help="[standby] Upload experiment logs to GCS after each run.",
    )
    p.add_argument("--gcs_bucket", type=str, default="benchmark-public-data")
    p.add_argument("--gcs_prefix", type=str, default="aide-runs")
    p.add_argument("--gcs_project", type=str, default="numi-platform")
    p.add_argument(
        "--policy_kind",
        type=str,
        default="heuristic",
        choices=["heuristic", "controller", "llm"],
        help="Search policy kind.",
    )
    p.add_argument(
        "--controller_kind",
        type=str,
        default="none",
        choices=["none", "llm", "random", "unified"],
        help="Hint controller kind (use with heuristic for heuristic_plus_controller).",
    )
    p.add_argument("--controller_model", type=str, default=None)
    p.add_argument("--controller_temp", type=float, default=0.7)
    p.add_argument("--hint_max_chars", type=int, default=600)
    p.add_argument("--hint_pool_path", type=str, default=None)
    return p.parse_args()


def _build_policy(policy_kind: str, controller_kind: str) -> SearchPolicy:
    if controller_kind == "unified":
        return UnifiedControllerPolicy()
    if policy_kind == "controller":
        return ControllerPolicy()
    if policy_kind == "heuristic" and controller_kind != "none":
        return HeuristicPlusControllerPolicy()
    return HeuristicPolicy()


def _best_metric_float(journal: Journal) -> float | None:
    best = journal.get_best_node(only_good=True)
    if best is None or best.metric is None or best.metric.value is None:
        return None
    return float(best.metric.value)


def run_one(task_row_name: str, seed: int, args: argparse.Namespace) -> dict:
    random.seed(seed)
    tasks = load_task_index(args.csv_path)
    task = next((t for t in tasks if t.row_name == task_row_name), None)
    if task is None:
        raise ValueError(f"Unknown task row: {task_row_name}")

    flat_tabular = is_kaggle_index(args.csv_path)
    hf_data_prefix = HF_KAGGLE_PREFIX if flat_tabular else HF_DATA_PREFIX

    safe = _safe_dirname(task.row_name)
    # HF data is identical across seeds; cache per task to avoid re-downloading.
    mat_dir = Path(args.materialize_root) / (safe if args.data_source == "hf" else f"{safe}__seed{seed}")
    materialize_workspace(
        task,
        mat_dir,
        source=args.data_source,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
        hf_data_prefix=hf_data_prefix,
        local_root=args.local_data_root,
    )
    aide_inputs = build_aide_inputs(task, flat_tabular=flat_tabular)

    _cfg = _load_cfg(use_cli_args=False)
    _cfg.data_dir = str(mat_dir / "input")
    _cfg.goal = aide_inputs["goal"]
    _cfg.eval = aide_inputs["eval"]
    _cfg.log_dir = str(Path(args.out_logs_dir).resolve())
    _cfg.workspace_dir = str(Path(args.out_workspace_dir).resolve())
    _cfg.exp_name = f"{safe}__seed{seed}"
    _cfg.agent.search.policy_kind = args.policy_kind
    _cfg.agent.search.controller_kind = args.controller_kind
    if args.controller_model:
        _cfg.agent.search.controller_model = args.controller_model
    _cfg.agent.search.controller_temp = args.controller_temp
    _cfg.agent.search.hint_max_chars = args.hint_max_chars
    if args.hint_pool_path:
        _cfg.agent.search.hint_pool_path = args.hint_pool_path
    _cfg.generate_report = False

    cfg = prep_cfg(_cfg)
    task_desc = load_task_desc(cfg)

    prep_agent_workspace(cfg)
    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        policy=_build_policy(args.policy_kind, args.controller_kind),
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec, resolve=True),  # type: ignore[arg-type]
    )

    status = "ok"
    error_message = ""
    failed_step = 0
    print(
        f"[AIDE] starting {task.row_name} seed={seed} steps={args.steps} log_dir={cfg.log_dir}",
        flush=True,
    )
    try:
        pbar = tqdm(
            range(1, args.steps + 1),
            desc=f"{task.row_name} s{seed}",
            file=sys.stdout,
            mininterval=2.0,
            dynamic_ncols=True,
        )
        for step in pbar:
            failed_step = step
            agent.step(exec_callback=interpreter.run)
            save_run(cfg, journal)
            n_nodes = len(journal.nodes)
            n_buggy = sum(1 for n in journal.nodes if n.is_buggy)
            best = _best_metric_float(journal)
            pbar.set_postfix(nodes=n_nodes, buggy=n_buggy, best=best, refresh=True)
            print(
                f"[AIDE] step {step}/{args.steps} nodes={n_nodes} buggy={n_buggy} best={best}",
                flush=True,
            )
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        error_message = str(exc)
        print(
            f"\n[AIDE ERROR] {task.row_name} seed={seed} failed at step {failed_step}/{args.steps} "
            f"after {len(journal.nodes)} node(s): {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.stderr.flush()
        # Best-effort partial checkpoint for debugging / HF upload
        if journal.nodes:
            try:
                save_run(cfg, journal)
            except Exception as save_exc:
                print(f"[AIDE WARN] could not save partial run: {save_exc}", file=sys.stderr, flush=True)
    finally:
        interpreter.cleanup_session()

    best = _best_metric_float(journal)
    n_buggy = sum(1 for n in journal.nodes if n.is_buggy)
    row = {
        "task": task.row_name,
        "seed": seed,
        "log_dir": str(cfg.log_dir),
        "status": status,
        "error_message": error_message,
        "failed_step": failed_step if status != "ok" else "",
        "final_best_metric": best,
        "n_nodes": len(journal.nodes),
        "n_buggy": n_buggy,
        "hf_uri": "",
        "gcs_uri": "",
    }
    if args.upload_hf and journal.nodes:
        try:
            from data.hf_utils import upload_aide_log_dir as upload_aide_log_dir_hf

            row["hf_uri"] = upload_aide_log_dir_hf(
                cfg.log_dir,
                task_name=task.row_name,
                seed=seed,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
        except Exception as exc:
            row["status"] = f"{status}+upload_failed"
            row["error_message"] = f"{error_message}; upload: {exc}".strip("; ")
            print(f"[AIDE ERROR] HF upload failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
    if args.upload_gcs and journal.nodes:
        try:
            from data.data_utils import upload_aide_log_dir

            row["gcs_uri"] = upload_aide_log_dir(
                cfg.log_dir,
                task_name=task.row_name,
                seed=seed,
                bucket_name=args.gcs_bucket,
                gcs_prefix=args.gcs_prefix,
                project=args.gcs_project,
            )
        except Exception as exc:
            row["status"] = f"{status}+upload_failed"
            row["error_message"] = f"{error_message}; upload: {exc}".strip("; ")
            print(f"[AIDE ERROR] GCS upload failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
    print(f"[AIDE] finished {task.row_name} seed={seed} status={row['status']} row={row}", flush=True)
    return row


def _parse_seed_arg(seed: str | None) -> SeedArg | None:
    if seed is None:
        return None
    if seed == "auto":
        return "auto"
    try:
        return int(seed)
    except ValueError as exc:
        raise SystemExit(f"Invalid --seed {seed!r}: use an integer or 'auto'.") from exc


def _resolve_single_task(args: argparse.Namespace) -> tuple[str | None, SeedArg | None]:
    tasks = load_task_index(args.csv_path)
    seed = _parse_seed_arg(args.seed)
    if args.task_name is not None:
        return args.task_name, seed
    if args.task_index is not None:
        idx = args.task_offset + args.task_index
        if idx < 0 or idx >= len(tasks):
            raise ValueError(f"task_index {args.task_index} (csv index {idx}) out of range")
        return tasks[idx].row_name, seed
    return None, None


def _resolve_seed_for_task(
    task_name: str,
    seed: SeedArg,
    repo_files: frozenset[str] | None,
    args: argparse.Namespace,
) -> int:
    if seed != "auto":
        return seed
    from data.hf_utils import list_aide_run_seeds_on_hf, next_available_seed

    resolved = next_available_seed(
        task_name,
        repo_files,
        repo_id=args.hf_repo,
        revision=args.hf_revision,
    )
    taken = list_aide_run_seeds_on_hf(task_name, repo_files)
    print(f"seed auto -> {resolved} for {task_name} (HF seeds: {sorted(taken)})")
    return resolved


def main() -> None:
    args = parse_args()
    Path(args.out_logs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_workspace_dir).mkdir(parents=True, exist_ok=True)
    Path(args.materialize_root).mkdir(parents=True, exist_ok=True)
    Path(args.runs_index).parent.mkdir(parents=True, exist_ok=True)

    print(f"data_source={args.data_source}  hf_repo={args.hf_repo}")
    print(f"upload_hf={args.upload_hf}  upload_gcs={args.upload_gcs}")
    if args.data_source == "local" and not args.local_data_root:
        raise SystemExit("--local_data_root is required when --data_source local")

    single_task, single_seed = _resolve_single_task(args)
    if (single_task is None) ^ (single_seed is None):
        raise SystemExit("For a single run, pass both --task_name/--task_index and --seed.")

    tasks = load_task_index(args.csv_path)
    use_auto_seed = single_seed == "auto" or (
        single_task is None and _parse_seed_arg(args.seed) == "auto"
    )
    repo_files = None
    if use_auto_seed:
        from data.hf_utils import list_repo_files_cached

        print(f"Listing HF runs on {args.hf_repo} ({args.hf_revision})...")
        repo_files = list_repo_files_cached(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        )

    if single_task is not None:
        slice_tasks = [next(t for t in tasks if t.row_name == single_task)]
        seed_plan: list[SeedArg] = [single_seed]
    elif _parse_seed_arg(args.seed) == "auto":
        slice_tasks = tasks[args.task_offset : args.task_offset + args.max_tasks]
        seed_plan = ["auto"]  # one run per task, next HF seed each
    else:
        slice_tasks = tasks[args.task_offset : args.task_offset + args.max_tasks]
        seed_plan = list(range(args.seeds_per_task))

    index_path = Path(args.runs_index)
    write_header = not index_path.is_file()
    fieldnames = [
        "task",
        "seed",
        "log_dir",
        "status",
        "error_message",
        "failed_step",
        "final_best_metric",
        "n_nodes",
        "n_buggy",
        "hf_uri",
        "gcs_uri",
    ]
    failures = 0
    with index_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for task in slice_tasks:
            for seed_arg in seed_plan:
                seed = _resolve_seed_for_task(task.row_name, seed_arg, repo_files, args)
                row = run_one(task.row_name, seed, args)
                w.writerow(row)
                f.flush()
                if not str(row["status"]).startswith("ok"):
                    failures += 1

    if failures:
        print(f"[AIDE] {failures} run(s) failed — see error tracebacks above.", flush=True)
        raise SystemExit(1)
    print("[AIDE] all runs completed successfully.", flush=True)


if __name__ == "__main__":
    main()
