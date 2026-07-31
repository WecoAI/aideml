"""Prompt formatting and parsing for the unified analyzer + controller.

One trained model call per AIDE step, at review time: it analyzes the executed
code (replacing the GPT feedback/review call) and, in the same output, decides
how the search tree should expand from this node plus a hint for the coding LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..journal import Node
from .hint_prompt import (
    DEFAULT_MAX_ANALYSIS_CHARS,
    DEFAULT_MAX_CODE_CHARS,
    DEFAULT_MAX_HINT_CHARS,
    DEFAULT_MAX_OUTPUT_CHARS,
    VALID_ACTIONS,
    ControllerAction,
    _extract_json_object,
    _node_depth,
    _truncate,
    build_history_summary,
    format_dataset_metadata,
)

ANALYZER_PROMPT_VERSION = "analyzer_prompt_v1"

ANALYZER_SYSTEM_PROMPT = (
    "You are the analyzer and strategic controller for AIDE, a tree-search ML engineering agent.\n"
    "You are given the current solution code and its execution output.\n"
    "First analyze the execution: decide whether it failed or has a bug, extract the value of "
    "the validation metric printed by the code (null if it did not run successfully), and "
    "summarize the empirical findings (or the bug and a proposed fix) in 2-3 sentences.\n"
    "Then decide how the search tree should expand from this node and write a concise hint "
    "that will guide the coding LLM in the next step. Do not write full code.\n\n"
    "Reply with exactly one JSON object (no markdown):\n"
    '{"is_bug":true|false,"metric":<number or null>,"lower_is_better":true|false,'
    '"analysis":"<2-3 sentence analysis>",'
    '"action":"debug|improve|abandon","hint":"short strategic guidance","confidence":0.0}'
)


@dataclass
class AnalyzerOutput:
    is_bug: bool
    metric: float | None
    lower_is_better: bool
    analysis: str
    action: ControllerAction
    hint: str
    confidence: float


def _ancestor_history(node: Node) -> str:
    """Lineage summary excluding the current node.

    At inference the current node has no review labels yet (we are producing
    them), so including it in history would diverge from training prompts.
    """
    if node.parent is None:
        return "(root)"
    return build_history_summary(node.parent)


def format_analyzer_input(
    task_desc: str,
    node: Node,
    *,
    history_summary: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Deterministic user prompt for the unified analyzer (pre-review node state)."""
    history = history_summary if history_summary is not None else _ancestor_history(node)
    depth = _node_depth(node)
    code = _truncate(node.code, max_code_chars)
    term_out = _truncate(node.term_out, max_output_chars)
    dataset_meta = format_dataset_metadata(dataset_metadata)

    return (
        f"Task:\n{task_desc.strip()}\n\n"
        f"Dataset metadata:\n{dataset_meta}\n\n"
        f"Current node:\n"
        f"- depth: {depth}\n"
        f"- stage: {node.stage_name}\n\n"
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Execution output:\n```text\n{term_out}\n```\n\n"
        f"Ancestor history:\n```text\n{history}\n```\n\n"
        "Analyze the execution output, then decide the next tree-expansion action and write "
        "one strategic hint for the next code-generation step."
    )


def format_analyzer_target(
    *,
    is_bug: bool,
    metric: float | None,
    lower_is_better: bool,
    analysis: str,
    action: ControllerAction,
    hint: str,
    confidence: float,
    max_hint_chars: int = DEFAULT_MAX_HINT_CHARS,
    max_analysis_chars: int = DEFAULT_MAX_ANALYSIS_CHARS,
) -> str:
    """Serialize the training / inference target as compact JSON.

    Key order matters: analysis fields come first so the model reasons about
    the execution before committing to an action.
    """
    return json.dumps(
        {
            "is_bug": bool(is_bug),
            "metric": None if metric is None else float(metric),
            "lower_is_better": bool(lower_is_better),
            "analysis": _truncate(analysis, max_analysis_chars),
            "action": action,
            "hint": _truncate(hint, max_hint_chars),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        },
        separators=(",", ":"),
    )


def _coerce_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return default


def parse_analyzer_output(
    text: str,
    *,
    max_hint_chars: int = DEFAULT_MAX_HINT_CHARS,
    max_analysis_chars: int = DEFAULT_MAX_ANALYSIS_CHARS,
) -> AnalyzerOutput | None:
    """Parse model output into a validated AnalyzerOutput (None on failure)."""
    obj = _extract_json_object(text)
    if obj is None:
        return None

    action = obj.get("action")
    if action not in VALID_ACTIONS:
        return None

    is_bug = _coerce_bool(obj.get("is_bug"))
    if is_bug is None:
        return None

    metric = obj.get("metric")
    if isinstance(metric, bool):
        metric = None
    if metric is not None:
        try:
            metric = float(metric)
        except (TypeError, ValueError):
            metric = None

    lower_is_better = _coerce_bool(obj.get("lower_is_better"), default=False)

    analysis = obj.get("analysis")
    analysis = _truncate(analysis.strip(), max_analysis_chars) if isinstance(analysis, str) else ""

    hint = obj.get("hint")
    hint = _truncate(hint.strip(), max_hint_chars) if isinstance(hint, str) else ""
    if not hint and action != "abandon":
        return None

    confidence = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return AnalyzerOutput(
        is_bug=is_bug,
        metric=metric,
        lower_is_better=bool(lower_is_better),
        analysis=analysis,
        action=action,  # type: ignore[arg-type]
        hint=hint,
        confidence=confidence,
    )
