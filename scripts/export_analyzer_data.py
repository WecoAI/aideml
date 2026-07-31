#!/usr/bin/env python3
"""Export unified analyzer (review + action + hint) SFT and preference data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.rlhf.analyzer_exporter import export_logs_dir_unified
from aide.rlhf.hint_exporter import ExportConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export unified analyzer training data.")
    p.add_argument("--logs_dir", type=str, default="data/heuristic_runs/logs")
    p.add_argument(
        "--out",
        type=str,
        default="data/heuristic_runs/analyzer/sft.jsonl",
    )
    p.add_argument(
        "--preferences_out",
        type=str,
        default="data/heuristic_runs/analyzer/preferences.jsonl",
    )
    p.add_argument("--ctu_csv", type=str, default="data/ctu_datasets_info.csv")
    p.add_argument(
        "--future_strategy",
        type=str,
        default="best_child_by_subtree",
        choices=["best_child_by_subtree", "best_descendant_k", "best_leaf"],
    )
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument(
        "--target_source",
        type=str,
        default="plan",
        choices=["plan", "analysis", "teacher"],
    )
    p.add_argument("--teacher_model", type=str, default=None)
    p.add_argument("--max_hint_chars", type=int, default=600)
    p.add_argument("--min_preference_gap", type=float, default=0.01)
    p.add_argument("--min_preference_gap_frac", type=float, default=0.005)
    p.add_argument("--max_pairs_per_node", type=int, default=3)
    p.add_argument(
        "--holdout_datasets",
        type=str,
        nargs="*",
        default=[],
        help="Dataset name prefixes routed to sft_val.jsonl.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExportConfig(
        future_strategy=args.future_strategy,  # type: ignore[arg-type]
        horizon=args.horizon,
        min_delta=args.min_delta,
        target_source=args.target_source,  # type: ignore[arg-type]
        teacher_model=args.teacher_model,
        max_hint_chars=args.max_hint_chars,
        min_preference_gap=args.min_preference_gap,
        min_preference_gap_frac=args.min_preference_gap_frac,
        max_pairs_per_node=args.max_pairs_per_node,
        holdout_datasets=set(args.holdout_datasets),
    )
    counts = export_logs_dir_unified(
        args.logs_dir,
        args.out,
        args.preferences_out,
        args.ctu_csv,
        cfg=cfg,
    )
    print(json.dumps(counts, indent=2))
    print(f"SFT train -> {args.out}")
    if cfg.holdout_datasets:
        print(f"SFT val   -> {Path(args.out).parent / 'sft_val.jsonl'}")
    print(f"Prefs     -> {args.preferences_out}")


if __name__ == "__main__":
    main()
