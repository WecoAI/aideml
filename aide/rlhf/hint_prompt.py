"""Shared prompt formatting and output parsing for the hindsight controller."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from ..journal import Node

HINT_PROMPT_VERSION = "hint_prompt_v1"

ControllerAction = Literal["debug", "improve", "abandon"]

VALID_ACTIONS: frozenset[str] = frozenset({"debug", "improve", "abandon"})

HINT_SYSTEM_PROMPT = (
    "You are a strategic controller for AIDE.\n"
    "Your job is to decide how the search tree should expand and provide a concise hint "
    "that will help the coding LLM improve the current solution.\n"
    "Do not write full code. Give the most useful next insight.\n\n"
    "Reply with exactly one JSON object (no markdown):\n"
    '{"action":"debug|improve|abandon","hint":"short strategic guidance","confidence":0.0}'
)

DEFAULT_MAX_CODE_CHARS = 8000
DEFAULT_MAX_OUTPUT_CHARS = 4000
DEFAULT_MAX_ANALYSIS_CHARS = 2000
DEFAULT_MAX_HINT_CHARS = 600


@dataclass
class ControllerOutput:
    action: ControllerAction
    hint: str
    confidence: float


def _truncate(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _node_depth(node: Node) -> int:
    depth = 0
    cur = node.parent
    while cur is not None:
        depth += 1
        cur = cur.parent
    return depth


def _node_status(node: Node) -> str:
    if node.is_buggy:
        return "buggy"
    if node.metric is not None and node.metric.value is not None:
        return "valid"
    return "unknown"


def _node_metric_str(node: Node) -> str:
    if node.metric is None or node.metric.value is None:
        return "N/A"
    return f"{node.metric.value:.6g}"


def build_history_summary(node: Node, max_ancestors: int = 8) -> str:
    """Short lineage from root to current node."""
    path: list[Node] = []
    cur: Node | None = node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    if len(path) > max_ancestors:
        path = path[-max_ancestors:]

    lines: list[str] = []
    for n in path:
        metric = _node_metric_str(n)
        lines.append(
            f"- step={n.step} stage={n.stage_name} status={'buggy' if n.is_buggy else 'ok'} metric={metric}"
        )
    return "\n".join(lines) if lines else "(root)"


def format_dataset_metadata(task_metadata: dict[str, Any] | None) -> str:
    if not task_metadata:
        return "(none)"
    parts: list[str] = []
    for key in ("task_type", "target_column", "target_table", "dataset_name", "task_name"):
        if key in task_metadata and task_metadata[key]:
            parts.append(f"{key}: {task_metadata[key]}")
    if not parts:
        return json.dumps(task_metadata, separators=(",", ":"))
    return "\n".join(parts)


def format_controller_input(
    task_desc: str,
    node: Node,
    *,
    history_summary: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    max_analysis_chars: int = DEFAULT_MAX_ANALYSIS_CHARS,
) -> str:
    """Deterministic user prompt for the controller (current node state only)."""
    history = history_summary if history_summary is not None else build_history_summary(node)
    status = _node_status(node)
    depth = _node_depth(node)
    code = _truncate(node.code, max_code_chars)
    term_out = _truncate(node.term_out, max_output_chars)
    analysis = _truncate(node.analysis, max_analysis_chars)
    dataset_meta = format_dataset_metadata(dataset_metadata)

    return (
        f"Task:\n{task_desc.strip()}\n\n"
        f"Dataset metadata:\n{dataset_meta}\n\n"
        f"Current node:\n"
        f"- depth: {depth}\n"
        f"- status: {status}\n"
        f"- metric: {_node_metric_str(node)}\n\n"
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Execution output:\n```text\n{term_out}\n```\n\n"
        f"Current analysis:\n```text\n{analysis}\n```\n\n"
        f"Recent history:\n```text\n{history}\n```\n\n"
        "Decide the next tree-expansion action and write one strategic hint for the next code-generation step."
    )


def format_controller_target(
    action: ControllerAction,
    hint: str,
    confidence: float,
    *,
    max_hint_chars: int = DEFAULT_MAX_HINT_CHARS,
) -> str:
    """Serialize the training / inference target as compact JSON."""
    hint = _truncate(hint, max_hint_chars)
    confidence = max(0.0, min(1.0, float(confidence)))
    return json.dumps(
        {"action": action, "hint": hint, "confidence": round(confidence, 3)},
        separators=(",", ":"),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Try to find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_controller_output(
    text: str,
    *,
    max_hint_chars: int = DEFAULT_MAX_HINT_CHARS,
) -> ControllerOutput | None:
    """Parse model output into a validated ControllerOutput."""
    obj = _extract_json_object(text)
    if obj is None:
        return None

    action = obj.get("action")
    if action not in VALID_ACTIONS:
        return None

    hint = obj.get("hint")
    if not isinstance(hint, str):
        hint = ""
    hint = _truncate(hint.strip(), max_hint_chars)
    # abandon is a valid decision even without hint text
    if not hint and action != "abandon":
        return None

    confidence = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return ControllerOutput(
        action=action,  # type: ignore[arg-type]
        hint=hint,
        confidence=confidence,
    )


def abandon_hint_template(node: Node) -> str:
    """Templated hint for abandon-labeled training rows."""
    if node.is_buggy:
        return (
            "This branch has repeated execution failures without a viable fix path. "
            "Abandon this direction and start a fresh approach with a simpler baseline."
        )
    return (
        "This branch shows limited improvement potential. "
        "Abandon further refinement here and explore a different modeling strategy."
    )
