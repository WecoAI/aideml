"""
Export unified analyzer training data (review + action + hint) from AIDE journals.

Reuses the hindsight logic in hint_exporter (future-node selection, hint targets,
preference pairs) and re-renders every row in the unified analyzer format:

- input:  node state *before* review (no status/metric/analysis of the current
  node, since the model must produce them), via format_analyzer_input.
- target: review labels logged in the journal (is_buggy, metric, analysis from
  the GPT review — distilled into the trained model) plus the hindsight
  action/hint/confidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..journal import Journal, Node
from ..utils import serialize
from .analyzer_prompt import (
    ANALYZER_PROMPT_VERSION,
    ANALYZER_SYSTEM_PROMPT,
    format_analyzer_input,
    format_analyzer_target,
)
from .ctu_dataset import CTUTask, load_ctu_index
from .hint_exporter import (
    ExportConfig,
    _resolve_task_and_metrics,
    _task_metadata,
    export_journal_file as export_hint_journal_file,
)
from .observation import task_desc_to_string
from .offline_extractor import _parse_run_dirname


def _unified_target(
    node: Node,
    controller_target_json: str,
    *,
    maximize: bool,
    cfg: ExportConfig,
) -> str:
    """Wrap a hint-exporter target ({action,hint,confidence}) with review labels."""
    d = json.loads(controller_target_json)

    metric_value: float | None = None
    if not node.is_buggy and node.metric is not None and node.metric.value is not None:
        metric_value = float(node.metric.value)

    lower_is_better = not maximize
    if node.metric is not None and node.metric.maximize is not None:
        lower_is_better = not node.metric.maximize

    return format_analyzer_target(
        is_bug=bool(node.is_buggy),
        metric=metric_value,
        lower_is_better=lower_is_better,
        analysis=node.analysis or "",
        action=d["action"],
        hint=d.get("hint", ""),
        confidence=float(d.get("confidence", 0.5)),
        max_hint_chars=cfg.max_hint_chars,
    )


def export_journal_file_unified(
    journal_path: Path,
    ctu_tasks_by_name: dict[str, CTUTask],
    *,
    cfg: ExportConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hint_sft, hint_prefs = export_hint_journal_file(journal_path, ctu_tasks_by_name, cfg=cfg)

    journal = serialize.load_json(journal_path, Journal)
    id2node = {n.id: n for n in journal.nodes}
    dirname = journal_path.parent.name
    task_guess, _seed = _parse_run_dirname(dirname)
    task, task_desc, _baseline, maximize = _resolve_task_and_metrics(
        journal_path, dirname, task_guess, ctu_tasks_by_name, journal
    )
    td_str = task_desc_to_string(task_desc)
    dataset_metadata = _task_metadata(task)

    def _input_for(node: Node) -> str:
        return format_analyzer_input(
            td_str,
            node,
            dataset_metadata=dataset_metadata,
            max_code_chars=cfg.max_code_chars,
            max_output_chars=cfg.max_output_chars,
        )

    sft_rows: list[dict[str, Any]] = []
    for row in hint_sft:
        node = id2node.get(row["node_id"])
        if node is None:
            continue
        user_input = _input_for(node)
        target = _unified_target(node, row["target"], maximize=maximize, cfg=cfg)
        sft_rows.append(
            {
                **row,
                "input": user_input,
                "target": target,
                "messages": [
                    {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {**row["metadata"], "prompt_version": ANALYZER_PROMPT_VERSION},
            }
        )

    pref_rows: list[dict[str, Any]] = []
    for row in hint_prefs:
        node = id2node.get(row["node_id"])
        if node is None:
            continue
        user_input = _input_for(node)
        chosen = _unified_target(node, row["chosen"], maximize=maximize, cfg=cfg)
        rejected = _unified_target(node, row["rejected"], maximize=maximize, cfg=cfg)
        # Review fields are identical for both sides (same node); if the
        # action/hint halves also match, the pair carries no signal.
        if chosen == rejected:
            continue
        pref_rows.append(
            {
                **row,
                "prompt": user_input,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {**row["metadata"], "prompt_version": ANALYZER_PROMPT_VERSION},
            }
        )

    return sft_rows, pref_rows


def _chatify_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["prompt"] = [
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]
    row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
    row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
    return row


def export_logs_dir_unified(
    logs_root: str | Path,
    out_sft: str | Path,
    out_prefs: str | Path,
    ctu_csv: str | Path,
    *,
    cfg: ExportConfig | None = None,
) -> dict[str, int]:
    cfg = cfg or ExportConfig()
    logs_root = Path(logs_root)
    out_sft = Path(out_sft)
    out_prefs = Path(out_prefs)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    out_prefs.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_ctu_index(ctu_csv)
    by_name = {t.row_name: t for t in tasks}

    holdout = cfg.holdout_datasets
    val_sft = out_sft.parent / "sft_val.jsonl" if holdout else None

    counts = {"sft": 0, "sft_val": 0, "preferences": 0, "errors": 0}

    def _is_holdout(task_id: str) -> bool:
        if not holdout:
            return False
        return any(h in task_id or task_id.startswith(h) for h in holdout)

    f_val_ctx = val_sft.open("w") if val_sft else open(os.devnull, "w")
    with out_sft.open("w") as f_train, out_prefs.open("w") as f_pref, f_val_ctx as f_val:
        for journal_path in sorted(logs_root.rglob("journal.json")):
            try:
                sft_rows, pref_rows = export_journal_file_unified(
                    journal_path, by_name, cfg=cfg
                )
            except Exception as exc:
                f_train.write(
                    json.dumps({"error": str(exc), "journal_path": str(journal_path)}) + "\n"
                )
                counts["errors"] += 1
                continue

            for row in sft_rows:
                if _is_holdout(row["task_id"]):
                    f_val.write(json.dumps(row) + "\n")
                    counts["sft_val"] += 1
                else:
                    f_train.write(json.dumps(row) + "\n")
                    counts["sft"] += 1

            for row in pref_rows:
                f_pref.write(json.dumps(_chatify_preference_row(row)) + "\n")
                counts["preferences"] += 1

    return counts
