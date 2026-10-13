"""Opt-in five-step integration test using real provider APIs and bundled data."""

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from aide import backend, run
from aide.backend import backend_anthropic, backend_openai, utils
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.serialize import load_json

pytestmark = pytest.mark.skipif(
    os.getenv("AIDE_RUN_LIVE_E2E") != "1",
    reason="Set AIDE_RUN_LIVE_E2E=1 and provider/model settings to use real APIs",
)

REPO = Path(__file__).resolve().parents[2]


def check_predictions(directory, expected_ids):
    predictions = pd.read_csv(directory)
    assert list(predictions.columns) == ["Id", "SalePrice"]
    assert predictions.Id.tolist() == expected_ids.tolist()
    assert np.isfinite(predictions.SalePrice).all()
    assert (predictions.SalePrice > 0).all()
    return predictions.SalePrice.to_numpy()


def test_house_prices_five_steps(monkeypatch, tmp_path):
    """Exercise the actual run loop, APIs, execution, journal and report."""
    cfg = _load_cfg(use_cli_args=False)
    root = Path(os.getenv("AIDE_E2E_OUTPUT_DIR", str(tmp_path))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg.data_dir = str(REPO / "aide/example_tasks/house_prices")
    description = root / "task.md"
    description.write_text(
        (REPO / "aide/example_tasks/house_prices.md").read_text()
        + """

## Integration run constraints
Use only numpy, pandas, scipy and scikit-learn. Do not install packages or make
network calls. Keep each solution under 120 seconds on two CPU cores. Start with
a simple baseline, then make one small improvement per iteration; no grid search.
Evaluate every solution with KFold(n_splits=2, shuffle=True, random_state=42).
Fit preprocessing inside each fold using only that fold's training rows. Exclude
Id from the features. Train on log(SalePrice), and report the single overall RMSE
between log(SalePrice) and the log of the out-of-fold price predictions (not the
mean of fold RMSE values, and not log1p). Use fixed random states in estimators.
Save all out-of-fold predictions in ./working/validation_predictions.csv with
exactly Id,SalePrice columns, in original train.csv row order, with positive
finite prices. Fit on all training data for ./working/submission.csv, with the
same columns in original test.csv row order. Print the overall validation metric
as VALIDATION_LOG_RMSE=<number> with at least eight decimal places.
"""
    )
    cfg.desc_file = str(description)
    cfg.log_dir = str(root / "logs")
    cfg.workspace_dir = str(root / "workspaces")
    cfg.exp_name = "house-prices-live"
    cfg.agent.steps = 5
    cfg.agent.search.num_drafts = 1
    cfg.agent.k_fold_validation = 2
    cfg.exec.timeout = 120
    cfg.generate_report = True
    stages = {
        "CODE": cfg.agent.code,
        "FEEDBACK": cfg.agent.feedback,
        "REPORT": cfg.report,
    }
    for name, stage in stages.items():
        prefix = "CODE" if name == "REPORT" else name
        stage.model = os.environ[f"AIDE_E2E_{prefix}_MODEL"]
        stage.provider = os.environ[f"AIDE_E2E_{prefix}_PROVIDER"]
        stage.temp = None
        stage.max_tokens = 8192 if name == "CODE" else 4096
        stage.reasoning_effort = os.getenv(f"AIDE_E2E_{prefix}_EFFORT") or None
    cfg = prep_cfg(cfg)
    monkeypatch.setattr(run, "load_cfg", lambda: cfg)

    # Keep real API calls bounded. Missing credentials fail an explicitly enabled run.
    providers = {stage.provider for stage in stages.values()}
    assert providers <= {"openai", "anthropic"}, providers
    for provider in providers:
        assert os.getenv(f"{provider.upper()}_API_KEY"), f"Missing {provider} key"
        module = backend_openai if provider == "openai" else backend_anthropic
        getattr(module, f"_setup_{provider}_client")()
        client = module._client.with_options(timeout=90, max_retries=0)
        monkeypatch.setattr(module, "_client", client)
    monkeypatch.setattr(utils, "BACKOFF_MAX_TRIES", 1)
    calls = []
    for provider in providers:
        real_query = backend.provider_to_query_func[provider]

        def record_call(_query=real_query, _provider=provider, **kwargs):
            assert len(calls) < 24, "Live API call budget exhausted"
            spec = kwargs.get("func_spec")
            record = {
                "provider": _provider,
                "model": kwargs["model"],
                "function": spec.name if spec else None,
            }
            calls.append(record)
            result = _query(**kwargs)
            record.update(
                seconds=result[1], input_tokens=result[2], output_tokens=result[3]
            )
            return result

        monkeypatch.setitem(backend.provider_to_query_func, provider, record_call)

    train = pd.read_csv(Path(cfg.data_dir) / "train.csv")
    test = pd.read_csv(Path(cfg.data_dir) / "test.csv")
    working = cfg.workspace_dir / "working"
    executions = []
    interpreters = set()
    real_execute = Interpreter.run

    def execute(self, *args, **kwargs):
        interpreters.add(self)
        # Avoid stale output and keep provider keys out of the training subprocess.
        for filename in ("submission.csv", "validation_predictions.csv"):
            (working / filename).unlink(missing_ok=True)
        with monkeypatch.context() as child_env:
            for key in list(os.environ):
                if key.endswith("API_KEY"):
                    child_env.delenv(key)
            result = real_execute(self, *args, **kwargs)
        executions.append(result)
        return result

    monkeypatch.setattr(Interpreter, "run", execute)
    checked = []
    real_save = run.save_run

    def save_and_check(config, journal):
        real_save(config, journal)
        node = journal.nodes[-1]
        print(
            f"Step {len(journal)}: {node.stage_name}, metric={node.metric.value}",
            flush=True,
        )
        if node.is_buggy:
            return  # AIDE may recover in a later debugging step.
        prices = check_predictions(working / "validation_predictions.csv", train.Id)
        score = float(np.sqrt(np.mean((np.log(prices) - np.log(train.SalePrice)) ** 2)))
        assert math.isclose(score, node.metric.value, rel_tol=1e-5, abs_tol=1e-7)
        assert node.metric.maximize is False
        check_predictions(working / "submission.csv", test.Id)
        destination = cfg.log_dir / f"step-{len(journal)}"
        destination.mkdir()
        for filename in ("submission.csv", "validation_predictions.csv"):
            shutil.copyfile(working / filename, destination / filename)
        checked.append(
            {"step": len(journal), "stage": node.stage_name, "log_rmse": score}
        )

    monkeypatch.setattr(run, "save_run", save_and_check)
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (cfg.log_dir / "run.txt").open("w") as output:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                run.run()
    finally:
        for interpreter in interpreters:
            interpreter.cleanup_session()

    journal = load_json(cfg.log_dir / "journal.json", Journal)
    assert len(journal) == len(executions) == 5
    assert len(checked) >= 2 and any(item["stage"] == "improve" for item in checked)
    assert sum(call["function"] == "submit_task_metric" for call in calls) == 1
    assert sum(call["function"] == "submit_review" for call in calls) == 5
    best = journal.get_best_node()
    assert best is not None and best.metric.maximize is False
    assert best.metric.value == min(node.metric.value for node in journal.good_nodes)
    assert (cfg.log_dir / "best_solution.py").read_text() == best.code
    assert len((cfg.log_dir / "report.md").read_text().strip()) > 100
    assert (cfg.log_dir / "tree_plot.html").stat().st_size > 0
    receipt = {
        "status": "passed",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        ).hexdigest(),
        "training_rows": len(train),
        "test_rows": len(test),
        "steps": checked,
        "total_steps": len(journal),
        "best_step": best.step + 1,
        "best_log_rmse": best.metric.value,
        "calls": calls,
        "log_dir": str(cfg.log_dir),
    }
    (cfg.log_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
