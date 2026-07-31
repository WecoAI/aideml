#!/usr/bin/env python3
"""
Materialize native RelBench eval tasks and upload parquet bundles to Hugging Face.

Reads the eval manifest (data/eval/eval_tasks.jsonl), keeps the RelBench tasks,
materializes each with snap-stanford relbench, and uploads to a new relbench/
folder on the HF dataset repo.

Destination layout:
  guilhermedrud/ctu_datasets/relbench/<task_id>/{train,val,test}.parquet
  guilhermedrud/ctu_datasets/relbench/<task_id>/db_tables/*.parquet
  guilhermedrud/ctu_datasets/relbench/<task_id>/task_info.json

<task_id> is the manifest id, e.g. rel-f1__driver-position.

Examples:
  # All RelBench tasks in the manifest
  python scripts/upload_relbench_to_hf.py

  # One task, skip if already on HF
  python scripts/upload_relbench_to_hf.py --task_id rel-f1__driver-dnf --skip_uploaded
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.eval.manifest import filter_tasks, load_eval_manifest
from aide.eval.relbench_native import materialize_relbench_native
from data.hf_utils import (
    HF_REPO,
    list_repo_files_cached,
    relbench_task_exists_on_hf,
    upload_relbench_task_data,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=str, default="data/eval/eval_tasks.jsonl")
    p.add_argument("--hf_repo", type=str, default=HF_REPO)
    p.add_argument("--hf_revision", type=str, default="main")
    p.add_argument("--task_id", type=str, default=None, help="Upload a single manifest task id.")
    p.add_argument("--task_offset", type=int, default=0, help="Skip first N RelBench tasks.")
    p.add_argument("--max_tasks", type=int, default=None, help="Upload first N tasks (default: all).")
    p.add_argument("--skip_uploaded", action="store_true", help="Skip tasks already present on HF.")
    p.add_argument(
        "--no_download",
        action="store_true",
        help="Do not download RelBench data (expects a local relbench cache).",
    )
    p.add_argument("--work_dir", type=str, default=None, help="Reuse a local staging dir.")
    p.add_argument(
        "--skip_errors",
        action="store_true",
        help="Log and continue when a task fails to materialize.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = load_eval_manifest(args.manifest)
    tasks = filter_tasks(tasks, benchmark="relbench", task_id=args.task_id)
    if not tasks:
        raise SystemExit(
            f"No RelBench tasks matched (task_id={args.task_id!r}) in {args.manifest}"
        )

    if args.task_id is None:
        end = None if args.max_tasks is None else args.task_offset + args.max_tasks
        tasks = tasks[args.task_offset : end]

    work_root = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="relbench_hf_upload_"))
    )
    work_root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    repo_files = None
    if args.skip_uploaded:
        print(f"Listing files on hf://{args.hf_repo} ({args.hf_revision})...")
        repo_files = list_repo_files_cached(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        )
        print(f"Found {len(repo_files)} file(s) in repo.")

    for task in tqdm(tasks, desc="upload relbench->hf"):
        if args.skip_uploaded and relbench_task_exists_on_hf(
            task.id,
            repo_files,
            repo_id=args.hf_repo,
            revision=args.hf_revision,
        ):
            tqdm.write(f"skip (already on HF): {task.id}")
            continue

        assert task.relbench_dataset and task.relbench_task
        staging = work_root / task.id
        if staging.exists():
            shutil.rmtree(staging)
        try:
            input_dir = materialize_relbench_native(
                task.relbench_dataset,
                task.relbench_task,
                staging,
                download=not args.no_download,
            )
            upload_relbench_task_data(
                input_dir,
                task.id,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
        except Exception as exc:
            if not args.skip_errors:
                raise
            print(f"FAILED {task.id}: {type(exc).__name__}: {exc}")
            failed.append(task.id)

    if failed:
        print(f"Failed tasks ({len(failed)}): {', '.join(failed)}")
    print(
        f"Done. Data repo: "
        f"https://huggingface.co/datasets/{args.hf_repo}/tree/{args.hf_revision}/relbench"
    )


if __name__ == "__main__":
    main()
