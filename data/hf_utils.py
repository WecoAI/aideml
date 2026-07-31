"""Hugging Face Hub helpers for CTU data and AIDE run logs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HF_REPO = "guilhermedrud/ctu_datasets"
HF_DATA_PREFIX = "data"
HF_RUNS_PREFIX = "runs"
HF_KAGGLE_PREFIX = "kaggle"
HF_EVAL_PREFIX = "eval"
HF_RELBENCH_PREFIX = "relbench"


def _token(token: str | None = None) -> str | None:
    return token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=_token(token))


def _repo_path(prefix: str, *parts: str) -> str:
    return "/".join([prefix.strip("/"), *[p.strip("/") for p in parts if p]])


def upload_file_to_hf(
    local_path: str | Path,
    path_in_repo: str,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    path_in_repo = path_in_repo.strip("/")
    _api(token).upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    dest = f"{repo_id}/{path_in_repo}"
    print(f"Uploaded {local_path} -> hf://{dest}")
    return dest


def upload_folder_to_hf(
    local_folder: str | Path,
    repo_id: str = HF_REPO,
    path_in_repo: str = "",
    *,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload a local folder into a HF dataset repo. Returns hub path prefix."""
    local_folder = Path(local_folder)
    if not local_folder.is_dir():
        raise NotADirectoryError(local_folder)

    path_in_repo = path_in_repo.strip("/")
    _api(token).upload_folder(
        folder_path=str(local_folder),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo or None,
        revision=revision,
    )
    dest = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
    print(f"Uploaded {local_folder} -> hf://{dest}")
    return dest


def upload_ctu_task_data(
    input_dir: str | Path,
    task_name: str,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload materialized CTU tables for one task to data/<task_name>/."""
    return upload_folder_to_hf(
        input_dir,
        repo_id=repo_id,
        path_in_repo=_repo_path(HF_DATA_PREFIX, task_name),
        revision=revision,
        token=token,
    )


def upload_kaggle_task_data(
    input_dir: str | Path,
    dataset_name: str,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload materialized Kaggle tables for one dataset to kaggle/<dataset_name>/."""
    return upload_folder_to_hf(
        input_dir,
        repo_id=repo_id,
        path_in_repo=_repo_path(HF_KAGGLE_PREFIX, dataset_name),
        revision=revision,
        token=token,
    )


def upload_relbench_task_data(
    input_dir: str | Path,
    task_id: str,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload materialized RelBench tables for one task to relbench/<task_id>/."""
    return upload_folder_to_hf(
        input_dir,
        repo_id=repo_id,
        path_in_repo=_repo_path(HF_RELBENCH_PREFIX, task_id),
        revision=revision,
        token=token,
    )


def relbench_task_exists_on_hf(
    task_id: str,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """True if relbench/<task_id>/train.parquet exists on the HF dataset repo."""
    marker = _repo_path(HF_RELBENCH_PREFIX, task_id, "train.parquet")
    if repo_files is not None:
        return marker in repo_files
    api = _api(token)
    return api.file_exists(
        repo_id=repo_id,
        filename=marker,
        repo_type="dataset",
        revision=revision,
    )


def list_uploaded_relbench_tasks(
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> set[str]:
    """Return RelBench task ids that already have train.parquet on HF."""
    if repo_files is None:
        repo_files = list_repo_files_cached(
            repo_id=repo_id, revision=revision, token=token
        )
    prefix = f"{HF_RELBENCH_PREFIX}/"
    suffix = "/train.parquet"
    out: set[str] = set()
    for path in repo_files:
        if path.startswith(prefix) and path.endswith(suffix):
            task_id = path[len(prefix) : -len(suffix)]
            if task_id and "/" not in task_id:
                out.add(task_id)
    return out


def kaggle_task_exists_on_hf(
    dataset_name: str,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """True if kaggle/<dataset_name>/train.parquet exists on the HF dataset repo."""
    marker = _repo_path(HF_KAGGLE_PREFIX, dataset_name, "train.parquet")
    if repo_files is not None:
        return marker in repo_files
    api = _api(token)
    return api.file_exists(
        repo_id=repo_id,
        filename=marker,
        repo_type="dataset",
        revision=revision,
    )


def list_uploaded_kaggle_tasks(
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> set[str]:
    """Return Kaggle dataset names that already have train.parquet on HF."""
    if repo_files is None:
        repo_files = list_repo_files_cached(
            repo_id=repo_id, revision=revision, token=token
        )
    prefix = f"{HF_KAGGLE_PREFIX}/"
    suffix = "/train.parquet"
    out: set[str] = set()
    for path in repo_files:
        if path.startswith(prefix) and path.endswith(suffix):
            dataset_name = path[len(prefix) : -len(suffix)]
            if dataset_name and "/" not in dataset_name:
                out.add(dataset_name)
    return out


def upload_aide_log_dir(
    local_log_dir: str | Path,
    *,
    task_name: str,
    seed: int,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload one AIDE log bundle to runs/<task_name>/seed<seed>/."""
    path_in_repo = _repo_path(HF_RUNS_PREFIX, task_name, f"seed{seed}")
    upload_folder_to_hf(
        local_log_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        revision=revision,
        token=token,
    )
    return f"hf://{repo_id}/{path_in_repo}/"


def upload_model_dir(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload a local model checkpoint directory to a HF model repo."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)

    api = _api(token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    print(f"Uploaded model {local_dir} -> hf://{repo_id}")
    return f"hf://{repo_id}"


def upload_eval_run_dir(
    local_log_dir: str | Path,
    *,
    model_tag: str,
    task_name: str,
    seed: int,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload one eval run log bundle to eval/<model_tag>/<task_name>/seed<seed>/."""
    path_in_repo = _repo_path(HF_EVAL_PREFIX, model_tag, task_name, f"seed{seed}")
    upload_folder_to_hf(
        local_log_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        revision=revision,
        token=token,
    )
    return f"hf://{repo_id}/{path_in_repo}/"


def upload_eval_metrics(
    local_json: str | Path,
    *,
    model_tag: str,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Upload aggregate eval metrics JSON to eval/<model_tag>/metrics.json."""
    local_json = Path(local_json)
    if not local_json.is_file():
        raise FileNotFoundError(local_json)
    path_in_repo = _repo_path(HF_EVAL_PREFIX, model_tag, "metrics.json")
    _api(token).upload_file(
        path_or_fileobj=str(local_json),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    dest = f"{repo_id}/{path_in_repo}"
    print(f"Uploaded metrics {local_json} -> hf://{dest}")
    return f"hf://{dest}"


def download_ctu_task_data(
    task_name: str,
    dest_input_dir: str | Path,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """
    Download data/<task_name>/ from HF into dest_input_dir (train/val/test + db_tables).
    Skips download if train.parquet already exists.
    """
    from huggingface_hub import hf_hub_download

    dest_input_dir = Path(dest_input_dir)
    dest_input_dir.mkdir(parents=True, exist_ok=True)
    if (dest_input_dir / "train.parquet").is_file():
        return dest_input_dir

    remote_prefix = _repo_path(HF_DATA_PREFIX, task_name)
    api = _api(token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    task_files = [f for f in repo_files if f.startswith(f"{remote_prefix}/")]
    if not task_files:
        raise FileNotFoundError(
            f"No files under {remote_prefix}/ in {repo_id} (revision={revision})"
        )

    for remote_path in task_files:
        rel = remote_path[len(remote_prefix) + 1 :]
        if not rel:
            continue
        local_path = dest_input_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=_token(token),
        )
        shutil.copy2(cached, local_path)

    print(f"Downloaded hf://{repo_id}/{remote_prefix}/ -> {dest_input_dir}")
    return dest_input_dir


def download_kaggle_task_data(
    dataset_name: str,
    dest_input_dir: str | Path,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """
    Download kaggle/<dataset_name>/ from HF into dest_input_dir (train/val/test).
    Skips download if train.parquet already exists.
    """
    from huggingface_hub import hf_hub_download

    dest_input_dir = Path(dest_input_dir)
    dest_input_dir.mkdir(parents=True, exist_ok=True)
    if (dest_input_dir / "train.parquet").is_file():
        return dest_input_dir

    remote_prefix = _repo_path(HF_KAGGLE_PREFIX, dataset_name)
    api = _api(token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    task_files = [f for f in repo_files if f.startswith(f"{remote_prefix}/")]
    if not task_files:
        raise FileNotFoundError(
            f"No files under {remote_prefix}/ in {repo_id} (revision={revision})"
        )

    for remote_path in task_files:
        rel = remote_path[len(remote_prefix) + 1 :]
        if not rel:
            continue
        local_path = dest_input_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=_token(token),
        )
        shutil.copy2(cached, local_path)

    print(f"Downloaded hf://{repo_id}/{remote_prefix}/ -> {dest_input_dir}")
    return dest_input_dir


def download_relbench_task_data(
    task_id: str,
    dest_input_dir: str | Path,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    """
    Download relbench/<task_id>/ from HF into dest_input_dir
    (train/val/test + db_tables + task_info.json).
    Skips download if train.parquet already exists.
    """
    from huggingface_hub import hf_hub_download

    dest_input_dir = Path(dest_input_dir)
    dest_input_dir.mkdir(parents=True, exist_ok=True)
    if (dest_input_dir / "train.parquet").is_file():
        return dest_input_dir

    remote_prefix = _repo_path(HF_RELBENCH_PREFIX, task_id)
    api = _api(token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    task_files = [f for f in repo_files if f.startswith(f"{remote_prefix}/")]
    if not task_files:
        raise FileNotFoundError(
            f"No files under {remote_prefix}/ in {repo_id} (revision={revision})"
        )

    for remote_path in task_files:
        rel = remote_path[len(remote_prefix) + 1 :]
        if not rel:
            continue
        local_path = dest_input_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=_token(token),
        )
        shutil.copy2(cached, local_path)

    print(f"Downloaded hf://{repo_id}/{remote_prefix}/ -> {dest_input_dir}")
    return dest_input_dir


def list_hf_paths(
    prefix: str,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
    limit: int | None = 20,
) -> list[str]:
    api = _api(token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    prefix = prefix.strip("/")
    matched = [f for f in files if f.startswith(f"{prefix}/") or f == prefix]
    if limit is None:
        return matched
    return matched[:limit]


def list_repo_files_cached(
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> frozenset[str]:
    """List all files in a HF dataset repo (call once, reuse for many lookups)."""
    api = _api(token)
    return frozenset(
        api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    )


def ctu_task_exists_on_hf(
    task_name: str,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """True if data/<task_name>/train.parquet exists on the HF dataset repo."""
    marker = _repo_path(HF_DATA_PREFIX, task_name, "train.parquet")
    if repo_files is not None:
        return marker in repo_files
    api = _api(token)
    return api.file_exists(
        repo_id=repo_id,
        filename=marker,
        repo_type="dataset",
        revision=revision,
    )


def aide_run_exists_on_hf(
    task_name: str,
    seed: int,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """True if runs/<task_name>/seed<seed>/journal.json exists on HF."""
    marker = _repo_path(HF_RUNS_PREFIX, task_name, f"seed{seed}", "journal.json")
    if repo_files is not None:
        return marker in repo_files
    api = _api(token)
    return api.file_exists(
        repo_id=repo_id,
        filename=marker,
        repo_type="dataset",
        revision=revision,
    )


def list_aide_run_seeds_on_hf(
    task_name: str,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> set[int]:
    """Return seed indices that already have journal.json on HF for this task."""
    if repo_files is None:
        repo_files = list_repo_files_cached(
            repo_id=repo_id, revision=revision, token=token
        )
    prefix = _repo_path(HF_RUNS_PREFIX, task_name, "seed")
    suffix = "/journal.json"
    seeds: set[int] = set()
    for path in repo_files:
        if not path.startswith(f"{prefix}") or not path.endswith(suffix):
            continue
        # runs/<task>/seed<N>/journal.json
        middle = path[len(f"{HF_RUNS_PREFIX}/{task_name}/") : -len(suffix)]
        if middle.startswith("seed") and middle[4:].isdigit():
            seeds.add(int(middle[4:]))
    return seeds


def next_available_seed(
    task_name: str,
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> int:
    """Smallest non-negative seed with no completed run on HF for this task."""
    taken = list_aide_run_seeds_on_hf(
        task_name, repo_files, repo_id=repo_id, revision=revision, token=token
    )
    seed = 0
    while seed in taken:
        seed += 1
    return seed


def list_uploaded_ctu_tasks(
    repo_files: frozenset[str] | None = None,
    *,
    repo_id: str = HF_REPO,
    revision: str = "main",
    token: str | None = None,
) -> set[str]:
    """Return task row_names that already have train.parquet on HF."""
    if repo_files is None:
        repo_files = list_repo_files_cached(
            repo_id=repo_id, revision=revision, token=token
        )
    prefix = f"{HF_DATA_PREFIX}/"
    suffix = "/train.parquet"
    out: set[str] = set()
    for path in repo_files:
        if path.startswith(prefix) and path.endswith(suffix):
            # data/<task_name>/train.parquet
            task_name = path[len(prefix) : -len(suffix)]
            if task_name and "/" not in task_name:
                out.add(task_name)
    return out
