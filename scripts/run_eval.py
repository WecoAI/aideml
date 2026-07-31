#!/usr/bin/env python3
"""
Run AIDE on the RelBench + MLE-bench evaluation manifest.

Example:
  python scripts/run_eval.py --benchmark relbench --max_tasks 2 --steps 10
  python scripts/run_eval.py --task_id rel-f1__driver-dnf --seed 0
  python scripts/run_eval.py --task_index 1 --seed 0 --benchmark relbench \\
      --model_tag aide_baseline --upload_hf --relbench_download

MLE-bench setup (one-time):
  git clone https://github.com/openai/mle-bench
  cd mle-bench && pip install -e .
  mlebench prepare -c nomad2018-predict-transparent-conductors

RelBench setup:
  pip install relbench
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.agent import Agent
from aide.backend import get_usage, reset_usage
from aide.eval.grade import grade_mlebench_submission, grade_relbench_submission
from aide.eval.manifest import EvalTask, filter_tasks, load_eval_manifest
from aide.eval.mlebench_task import (
    build_aide_inputs as build_mlebench_inputs,
    load_task_info as load_mlebench_task_info,
    materialize_mlebench,
)
from aide.controller_trace import configure as configure_controller_trace
from aide.eval.relbench_native import (
    build_aide_inputs as build_relbench_inputs,
    load_task_info as load_relbench_task_info,
    materialize_relbench_native,
)
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.policy import (
    ControllerPolicy,
    HeuristicPolicy,
    HeuristicPlusControllerPolicy,
    SearchPolicy,
    UnifiedControllerPolicy,
)
from aide.utils.config import _load_cfg, load_task_desc, prep_agent_workspace, prep_cfg, save_run

from dotenv import load_dotenv

load_dotenv()


def _safe_dirname(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AIDE eval harness on RelBench + MLE-bench tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--manifest", type=str, default="data/eval/eval_tasks.jsonl")
    p.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "relbench", "mlebench"],
    )
    p.add_argument("--task_id", type=str, default=None, help="Run a single manifest task id.")
    p.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Run task at this index into the filtered manifest slice.",
    )
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed (for Polyaxon matrix runs). Mutually exclusive with --seeds.",
    )
    p.add_argument("--seeds", type=int, default=1, help="Number of seeds per task (0..seeds-1).")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out_logs_dir", type=str, default="data/eval/logs")
    p.add_argument("--out_workspace_dir", type=str, default="data/eval/workspaces")
    p.add_argument("--materialize_root", type=str, default="data/eval/materialized")
    p.add_argument("--results_csv", type=str, default="data/eval/eval_results.csv")
    p.add_argument(
        "--aggregate_path",
        type=str,
        default=None,
        help="Aggregate metrics JSON path (default: data/eval/metrics_<model_tag>.json). "
        "Set per-worker for parallel runs to avoid clobbering.",
    )
    p.add_argument("--relbench_download", action="store_true", help="Download RelBench data if missing.")
    p.add_argument("--mlebench_data_dir", type=str, default=None, help="MLE-bench cache data dir.")
    p.add_argument(
        "--model_tag",
        type=str,
        default="aide_baseline",
        help="Eval variant tag (aide_baseline, dpo_full, dpo_hint_only).",
    )
    p.add_argument("--upload_hf", action="store_true", help="Upload logs + metrics to Hugging Face.")
    p.add_argument("--hf_repo", type=str, default="guilhermedrud/ctu_datasets")
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument(
        "--controller_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for controller (default: CONTROLLER_OPENAI_BASE_URL env).",
    )
    p.add_argument("--policy_kind", type=str, default="heuristic", choices=["heuristic", "controller", "llm"])
    p.add_argument(
        "--controller_kind",
        type=str,
        default="none",
        choices=["none", "llm", "random", "unified"],
        help="'unified': trained model replaces the GPT review and drives tree expansion.",
    )
    p.add_argument("--controller_model", type=str, default=None)
    p.add_argument("--controller_temp", type=float, default=0.7)
    p.add_argument("--hint_max_chars", type=int, default=600)
    p.add_argument("--hint_pool_path", type=str, default=None)
    p.add_argument(
        "--num_drafts",
        type=int,
        default=None,
        help="Drafts before controller-driven search (default: config value, usually 5).",
    )
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


def _resolve_task_list(args: argparse.Namespace) -> list[EvalTask]:
    tasks = load_eval_manifest(args.manifest)
    tasks = filter_tasks(tasks, benchmark=args.benchmark, task_id=args.task_id)
    if args.task_index is not None:
        if args.task_id is not None:
            raise SystemExit("Use only one of --task_id or --task_index.")
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise SystemExit(f"--task_index {args.task_index} out of range (0..{len(tasks) - 1})")
        return [tasks[args.task_index]]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    return tasks


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed is not None and args.seeds != 1:
        raise SystemExit("Use only one of --seed or --seeds.")
    if args.seed is not None:
        return [args.seed]
    return list(range(args.seeds))


def _materialize_task(
    task: EvalTask, mat_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (aide_inputs, task_metadata for the hint controller prompt)."""
    if task.benchmark == "relbench":
        assert task.relbench_dataset and task.relbench_task
        input_dir = materialize_relbench_native(
            task.relbench_dataset,
            task.relbench_task,
            mat_dir,
            download=args.relbench_download,
        )
        task_info = load_relbench_task_info(input_dir)
        metadata = {
            "task_type": str(task_info.get("task_type", "")),
            "target_column": str(task_info.get("target_col", "")),
            "target_table": str(task_info.get("entity_table", "")),
            "dataset_name": str(task_info.get("dataset", "")),
            "task_name": str(task_info.get("task", "")),
        }
        return build_relbench_inputs(task_info, primary_metric=task.metric), metadata
    assert task.mlebench_competition_id
    input_dir = materialize_mlebench(
        task.mlebench_competition_id,
        mat_dir,
        data_dir=args.mlebench_data_dir,
    )
    task_info = load_mlebench_task_info(input_dir)
    description = (input_dir / "description.md").read_text()
    metadata = {
        "dataset_name": task.mlebench_competition_id,
        "task_name": task.mlebench_competition_id,
    }
    return build_mlebench_inputs(task_info, description), metadata


def _grade_task(
    task: EvalTask,
    workspace_dir: Path,
    args: argparse.Namespace,
) -> dict:
    submission = workspace_dir / "submission.csv"
    if task.benchmark == "relbench":
        assert task.relbench_dataset and task.relbench_task
        return grade_relbench_submission(
            submission,
            task.relbench_dataset,
            task.relbench_task,
            primary_metric=task.metric,
            download=args.relbench_download,
        )
    assert task.mlebench_competition_id
    return grade_mlebench_submission(
        submission,
        task.mlebench_competition_id,
        data_dir=args.mlebench_data_dir,
    )


def _metrics_row(task: EvalTask, seed: int, args: argparse.Namespace, row: dict) -> dict:
    """Compact metrics dict for JSON export and plotting."""
    return {
        "model_tag": args.model_tag,
        "benchmark": task.benchmark,
        "task": task.id,
        "seed": seed,
        "status": row["status"],
        "primary_metric": task.metric,
        "higher_is_better": task.higher_is_better,
        "best_val_metric": row["best_val_metric"],
        "official_metric": row["official_metric"],
        "grade_status": row["grade_status"],
        "tokens_in": row["tokens_in"],
        "tokens_out": row["tokens_out"],
        "total_tokens": row["total_tokens"],
        "n_nodes": row["n_nodes"],
        "n_buggy": row["n_buggy"],
        "n_llm_calls": row["n_llm_calls"],
        "policy_kind": args.policy_kind,
        "controller_kind": args.controller_kind,
        "controller_model": args.controller_model,
        "log_dir": row["log_dir"],
        "hf_uri": row.get("hf_uri", ""),
    }


def _load_aggregate_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _save_aggregate_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def run_one(task: EvalTask, seed: int, args: argparse.Namespace) -> dict:
    random.seed(seed)
    safe = _safe_dirname(task.id)
    mat_dir = Path(args.materialize_root) / safe
    aide_inputs, task_metadata = _materialize_task(task, mat_dir, args)

    controller_base_url = args.controller_base_url or os.getenv("CONTROLLER_OPENAI_BASE_URL")

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
    _cfg.agent.search.controller_base_url = controller_base_url
    _cfg.agent.search.hint_max_chars = args.hint_max_chars
    _cfg.agent.search.task_metadata = {k: v for k, v in task_metadata.items() if v}
    if args.hint_pool_path:
        _cfg.agent.search.hint_pool_path = args.hint_pool_path
    if args.num_drafts is not None:
        _cfg.agent.search.num_drafts = args.num_drafts
    _cfg.generate_report = False

    cfg = prep_cfg(_cfg)
    if cfg.workspace_dir.exists():
        shutil.rmtree(cfg.workspace_dir)
    uses_controller_llm = bool(
        args.controller_model
        and (
            args.policy_kind == "controller"
            or args.controller_kind in ("llm", "random", "unified")
        )
    )
    if uses_controller_llm:
        configure_controller_trace(
            log_path=cfg.log_dir / "controller.jsonl",
            terminal=os.getenv("AIDE_CONTROLLER_LOG_TERMINAL", "1") not in ("0", "false", "False"),
        )
        print(
            f"[eval] controller trace -> {cfg.log_dir / 'controller.jsonl'} "
            f"(num_drafts={cfg.agent.search.num_drafts})",
            flush=True,
        )
    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)

    reset_usage()
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
        f"[eval] starting {task.id} seed={seed} benchmark={task.benchmark} "
        f"model_tag={args.model_tag}",
        flush=True,
    )

    try:
        pbar = tqdm(range(1, args.steps + 1), desc=f"{task.id} s{seed}", file=sys.stdout)
        for step in pbar:
            failed_step = step
            agent.step(exec_callback=interpreter.run)
            save_run(cfg, journal)
            pbar.set_postfix(
                nodes=len(journal.nodes),
                best=_best_metric_float(journal),
                tokens=journal.total_tokens,
            )
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        error_message = str(exc)
        traceback.print_exc()
        if journal.nodes:
            try:
                save_run(cfg, journal)
            except Exception:
                pass
    finally:
        interpreter.cleanup_session()

    usage = get_usage()
    grade_result = _grade_task(task, Path(cfg.workspace_dir), args)
    medal_or_threshold = ""
    if task.benchmark == "mlebench":
        medal_or_threshold = json.dumps(
            {
                "any_medal": grade_result.get("any_medal"),
                "gold": grade_result.get("gold_medal"),
                "silver": grade_result.get("silver_medal"),
                "bronze": grade_result.get("bronze_medal"),
            }
        )
    elif grade_result.get("all_metrics"):
        medal_or_threshold = json.dumps(grade_result["all_metrics"])

    row = {
        "model_tag": args.model_tag,
        "benchmark": task.benchmark,
        "task": task.id,
        "seed": seed,
        "status": status,
        "error_message": error_message,
        "failed_step": failed_step if status != "ok" else "",
        "n_nodes": len(journal.nodes),
        "n_buggy": sum(1 for n in journal.nodes if n.is_buggy),
        "best_val_metric": _best_metric_float(journal),
        "official_metric": grade_result.get("official_metric"),
        "grade_status": grade_result.get("status"),
        "grade_error": grade_result.get("error", ""),
        "primary_metric": task.metric,
        "higher_is_better": task.higher_is_better,
        "medal_or_all_metrics": medal_or_threshold,
        "tokens_in": usage.get("in", 0),
        "tokens_out": usage.get("out", 0),
        "total_tokens": usage.get("total", 0),
        "n_llm_calls": usage.get("n_calls", 0),
        "policy_kind": args.policy_kind,
        "controller_kind": args.controller_kind,
        "controller_model": args.controller_model or "",
        "log_dir": str(cfg.log_dir),
        "workspace_dir": str(cfg.workspace_dir),
        "hf_uri": "",
    }

    metrics_json = _metrics_row(task, seed, args, row)
    metrics_path = Path(cfg.log_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_json, indent=2))

    if args.upload_hf:
        try:
            from data.hf_utils import upload_eval_run_dir

            row["hf_uri"] = upload_eval_run_dir(
                cfg.log_dir,
                model_tag=args.model_tag,
                task_name=task.id,
                seed=seed,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
            metrics_json["hf_uri"] = row["hf_uri"]
            metrics_path.write_text(json.dumps(metrics_json, indent=2))
        except Exception as exc:
            row["status"] = f"{status}+upload_failed"
            row["error_message"] = f"{error_message}; upload: {exc}".strip("; ")
            print(f"[eval ERROR] HF upload failed: {exc}", file=sys.stderr, flush=True)

    print(f"[eval] finished {task.id} seed={seed} row={json.dumps(row, default=str)}", flush=True)
    return row


def main() -> None:
    args = parse_args()
    for d in (args.out_logs_dir, args.out_workspace_dir, args.materialize_root):
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    tasks = _resolve_task_list(args)
    seeds = _resolve_seeds(args)
    if not tasks:
        raise SystemExit("No eval tasks matched the filters.")

    aggregate_path = (
        Path(args.aggregate_path)
        if args.aggregate_path
        else Path(f"data/eval/metrics_{args.model_tag}.json")
    )
    aggregate_rows = _load_aggregate_metrics(aggregate_path)

    fieldnames = [
        "model_tag",
        "benchmark",
        "task",
        "seed",
        "status",
        "error_message",
        "failed_step",
        "n_nodes",
        "n_buggy",
        "best_val_metric",
        "official_metric",
        "grade_status",
        "grade_error",
        "primary_metric",
        "higher_is_better",
        "medal_or_all_metrics",
        "tokens_in",
        "tokens_out",
        "total_tokens",
        "n_llm_calls",
        "policy_kind",
        "controller_kind",
        "controller_model",
        "log_dir",
        "workspace_dir",
        "hf_uri",
    ]
    results_path = Path(args.results_csv)
    write_header = not results_path.is_file()
    failures = 0
    session_rows: list[dict] = []

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for task in tasks:
            for seed in seeds:
                row = run_one(task, seed, args)
                writer.writerow(row)
                f.flush()
                session_rows.append(_metrics_row(task, seed, args, row))
                if not str(row["status"]).startswith("ok"):
                    failures += 1

    # Merge session rows into aggregate metrics (replace same task+seed).
    for new_row in session_rows:
        aggregate_rows = [
            r
            for r in aggregate_rows
            if not (r.get("task") == new_row["task"] and r.get("seed") == new_row["seed"])
        ]
        aggregate_rows.append(new_row)
    _save_aggregate_metrics(aggregate_path, aggregate_rows)

    if args.upload_hf:
        try:
            from data.hf_utils import upload_eval_metrics

            upload_eval_metrics(
                aggregate_path,
                model_tag=args.model_tag,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
        except Exception as exc:
            print(f"[eval WARN] aggregate metrics upload failed: {exc}", file=sys.stderr)

    if failures:
        raise SystemExit(f"{failures} eval run(s) failed.")
    print(f"[eval] all runs completed. Results -> {results_path}")
    print(f"[eval] aggregate metrics -> {aggregate_path}")


if __name__ == "__main__":
    main()
