#!/usr/bin/env python3
"""Smoke test: AIDE with the unified analyzer served by a local vLLM model.

Picks a random CTU task from the HF dataset repo, downloads its parquet tables,
and runs a short AIDE search where the local model (e.g. base Qwen3.5-9B, no
fine-tuning) replaces the GPT review and drives tree expansion
(controller_kind=unified). The coding LLM is still agent.code.model (GPT).

Exits non-zero if the unified analyzer never produced a usable review
(every call failed to parse -> every node fell back to the GPT review).

Example (vLLM already serving on :8100):
  CONTROLLER_OPENAI_API_KEY=dummy python scripts/run_unified_smoke.py \
      --controller_model aide-unified-base \
      --controller_base_url http://127.0.0.1:8100/v1 --steps 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.agent import Agent
from aide.controller_trace import configure as configure_controller_trace
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.policy import UnifiedControllerPolicy
from aide.rlhf.ctu_dataset import (
    build_aide_inputs,
    is_kaggle_index,
    load_task_index,
    materialize_workspace,
)
from aide.utils.config import _load_cfg, load_task_desc, prep_agent_workspace, prep_cfg, save_run
from data.hf_utils import HF_DATA_PREFIX, HF_KAGGLE_PREFIX

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv_path", type=str, default="data/ctu_datasets_info.csv")
    p.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Row index into the task CSV; omit to pick a random task (seeded by --seed).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num_drafts", type=int, default=2)
    p.add_argument("--exec_timeout", type=int, default=900, help="Per-step exec timeout (s).")
    p.add_argument("--controller_model", type=str, required=True)
    p.add_argument("--controller_base_url", type=str, default="http://127.0.0.1:8100/v1")
    p.add_argument("--controller_temp", type=float, default=0.0)
    p.add_argument("--hf_repo", type=str, default="guilhermedrud/ctu_datasets")
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument("--out_dir", type=str, default="data/unified_smoke")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    tasks = load_task_index(args.csv_path)
    if args.task_index is not None:
        if not (0 <= args.task_index < len(tasks)):
            raise SystemExit(f"--task_index {args.task_index} out of range (0..{len(tasks) - 1})")
        task = tasks[args.task_index]
    else:
        task = random.choice(tasks)
    print(f"[smoke] task: {task.row_name} (type={task.task_type}, target={task.target_column})", flush=True)

    out_dir = Path(args.out_dir)
    flat_tabular = is_kaggle_index(args.csv_path)
    hf_data_prefix = HF_KAGGLE_PREFIX if flat_tabular else HF_DATA_PREFIX

    safe = task.row_name.replace("/", "_").replace(" ", "_")
    mat_dir = out_dir / "materialized" / safe
    print(f"[smoke] downloading data from HF ({args.hf_repo}) -> {mat_dir}", flush=True)
    materialize_workspace(
        task,
        mat_dir,
        source="hf",
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
        hf_data_prefix=hf_data_prefix,
    )
    aide_inputs = build_aide_inputs(task, flat_tabular=flat_tabular)

    _cfg = _load_cfg(use_cli_args=False)
    _cfg.data_dir = str(mat_dir / "input")
    _cfg.goal = aide_inputs["goal"]
    _cfg.eval = aide_inputs["eval"]
    _cfg.log_dir = str((out_dir / "logs").resolve())
    _cfg.workspace_dir = str((out_dir / "workspaces").resolve())
    _cfg.exp_name = f"unified_smoke_{safe}__seed{args.seed}"
    _cfg.exec.timeout = args.exec_timeout
    _cfg.agent.steps = args.steps
    _cfg.agent.search.num_drafts = args.num_drafts
    _cfg.agent.search.controller_kind = "unified"
    _cfg.agent.search.controller_model = args.controller_model
    _cfg.agent.search.controller_temp = args.controller_temp
    _cfg.agent.search.controller_base_url = args.controller_base_url
    _cfg.agent.search.task_metadata = {
        "task_type": task.task_type,
        "target_column": task.target_column,
        "target_table": task.target_table,
        "dataset_name": task.dataset_name,
        "task_name": task.task_name,
    }
    _cfg.generate_report = False

    cfg = prep_cfg(_cfg)
    configure_controller_trace(log_path=cfg.log_dir / "controller.jsonl", terminal=True)
    print(f"[smoke] logs -> {cfg.log_dir}", flush=True)

    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        policy=UnifiedControllerPolicy(),
    )
    assert agent._unified_analyzer is not None, "unified analyzer not enabled"
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec, resolve=True),  # type: ignore[arg-type]
    )

    try:
        for step in range(1, args.steps + 1):
            print(f"[smoke] step {step}/{args.steps}", flush=True)
            agent.step(exec_callback=interpreter.run)
            save_run(cfg, journal)
    finally:
        interpreter.cleanup_session()

    # ---- Report ---------------------------------------------------------
    print("\n[smoke] node summary:")
    for n in journal.nodes:
        metric = n.metric.value if (n.metric is not None and n.metric.value is not None) else None
        print(
            f"  step={n.step} stage={n.stage_name} buggy={n.is_buggy} metric={metric} "
            f"next_action={n.next_action} conf={n.next_confidence} "
            f"hint={'yes' if n.next_hint else 'no'}"
        )

    trace_path = cfg.log_dir / "controller.jsonl"
    llm_calls, parse_failures = [], []
    if trace_path.is_file():
        with trace_path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") == "llm_call":
                    llm_calls.append(e)
                    if e.get("parse_error"):
                        parse_failures.append(e)

    n_unified = sum(1 for n in journal.nodes if n.next_action is not None)
    print(
        f"\n[smoke] analyzer calls={len(llm_calls)} parse_failures={len(parse_failures)} "
        f"nodes_reviewed_by_analyzer={n_unified}/{len(journal.nodes)}"
    )
    for e in parse_failures[:3]:
        print(f"[smoke] parse failure ({e['parse_error']}), raw output head:")
        print("  " + (e.get("raw_output") or "")[:300].replace("\n", "\n  "))

    if not llm_calls:
        print("[smoke] FAIL: analyzer was never called")
        return 1
    if n_unified == 0:
        print("[smoke] FAIL: no node was successfully reviewed by the unified analyzer "
              "(all calls failed or unparsable; GPT fallback reviewed everything)")
        return 1
    print("[smoke] OK: unified analyzer reviewed nodes and drove the search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
