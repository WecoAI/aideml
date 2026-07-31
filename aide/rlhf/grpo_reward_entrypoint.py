"""
OpenRLHF custom reward for offline GRPO (single-turn).

Pass this file's path via `--reward.remote_url /path/to/grpo_reward_entrypoint.py`.
OpenRLHF (>=0.10) loads any reward endpoint ending in `.py` with importlib and
calls `reward_func(queries, prompts, labels)` where each query is the decoded
prompt+response; it expects a dict with "rewards"/"scores" tensors back.

NOTE: OpenRLHF loads this file standalone (not as part of the `aide` package),
so imports must be absolute — `aide` must be pip-installed in the environment.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from aide.rlhf.grpo_verifier import reward_one


def _response_from_query(query: str, prompt: str) -> str:
    """Strip the prompt prefix from the decoded query, leaving the generation."""
    if prompt:
        if query.startswith(prompt):
            return query[len(prompt) :]
        idx = query.find(prompt)
        if idx >= 0:
            return query[idx + len(prompt) :]
    return query


def _coerce_label(label: Any) -> dict[str, Any]:
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            return {}
    return label if isinstance(label, dict) else {}


def reward_func(
    queries: list[str],
    prompts: list[str] | None = None,
    labels: list[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    n = len(queries)
    prompts = prompts if prompts is not None else [""] * n
    labels = labels if labels is not None else [{}] * n

    values: list[float] = []
    for i, query in enumerate(queries):
        prompt = prompts[i] if i < len(prompts) else ""
        label = _coerce_label(labels[i] if i < len(labels) else {})
        if not label:
            values.append(-1.0)
            continue
        response = _response_from_query(query or "", prompt or "")
        values.append(reward_one(response, label))

    rewards = torch.tensor(values, dtype=torch.float32)
    return {"rewards": rewards, "scores": rewards, "extra_logs": {}}
