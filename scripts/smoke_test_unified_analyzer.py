#!/usr/bin/env python3
"""Smoke-test the unified analyzer wiring via a mock OpenAI-compatible server.

Covers: UnifiedAnalyzer query/parse -> Agent.parse_exec_result field application
-> UnifiedControllerPolicy selection from cached suggestions.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.agent import Agent
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.policy import SearchAction, UnifiedControllerPolicy
from aide.utils.config import _load_cfg

MOCK_RESPONSE = json.dumps(
    {
        "is_bug": False,
        "metric": 0.71,
        "lower_is_better": False,
        "analysis": "The script trained a gradient boosting model and printed a validation F1 of 0.71.",
        "action": "improve",
        "hint": "Add temporal aggregation features from the driver history tables.",
        "confidence": 0.8,
    }
)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        body = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MOCK_RESPONSE},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _build_cfg(base_url: str):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = tempfile.mkdtemp()
    cfg.goal = "Predict driver DNF"
    cfg.log_dir = tempfile.mkdtemp()
    cfg.workspace_dir = tempfile.mkdtemp()
    cfg.agent.search.controller_kind = "unified"
    cfg.agent.search.controller_model = "aide-analyzer-mock"
    cfg.agent.search.controller_temp = 0.0
    cfg.agent.search.controller_base_url = base_url
    cfg.agent.search.num_drafts = 1
    cfg.agent.search.task_metadata = {
        "task_type": "binary_classification",
        "target_column": "did_not_finish",
    }
    return cfg


def main() -> None:
    port = 8766
    server = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cfg = _build_cfg(f"http://127.0.0.1:{port}/v1")
    journal = Journal()
    agent = Agent(
        task_desc={"Task goal": "Predict driver DNF"},
        cfg=cfg,
        journal=journal,
        policy=UnifiedControllerPolicy(),
    )
    assert agent._unified_analyzer is not None, "Unified analyzer not enabled"

    node = Node(plan="baseline", code="print('f1 0.71')")
    journal.append(node)
    agent.parse_exec_result(
        node=node,
        exec_result=ExecutionResult(
            term_out=["f1 0.71\n"], exec_time=0.1, exc_type=None, exc_info=None, exc_stack=None
        ),
    )
    server.shutdown()

    assert node.is_buggy is False, node.is_buggy
    assert node.metric is not None and abs(float(node.metric.value) - 0.71) < 1e-9
    assert node.metric.maximize is True
    assert node.analysis and "0.71" in node.analysis
    assert node.next_action == "improve" and node.next_hint and node.next_confidence == 0.8
    print(f"OK analyzer review: metric={node.metric.value} next_action={node.next_action}")

    action: SearchAction = agent.policy.select(
        journal=journal,
        task_desc="Predict driver DNF",
        search_cfg=cfg.agent.search,
        step_idx=1,
        total_steps=10,
    )
    assert action.kind == "improve" and action.parent_id == node.id, action
    assert action.hint == node.next_hint
    print(f"OK policy select: {action.kind} parent={action.parent_id[:8]} hint={action.hint[:40]}...")

    # All-abandoned -> draft
    node.next_action = "abandon"
    action = agent.policy.select(
        journal=journal,
        task_desc="Predict driver DNF",
        search_cfg=cfg.agent.search,
        step_idx=2,
        total_steps=10,
    )
    assert action.kind == "draft", action
    print("OK policy select: all abandoned -> draft")


if __name__ == "__main__":
    main()
