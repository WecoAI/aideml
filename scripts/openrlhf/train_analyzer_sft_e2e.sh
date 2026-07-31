#!/usr/bin/env bash
# Unified-analyzer SFT on Polyaxon.
#
# Modes (DATA_SOURCE):
#   scratch  (default) — read prebuilt sft.jsonl from /scratch/aide_rl (axia)
#   download — download HF journals + export SFT every run, then train (nebius)
#
# After data is ready:
#   fine-tune with OpenRLHF / DeepSpeed, then upload HF checkpoint.
#
# Required env:
#   HF_TOKEN          - download journals/base model + write finetuned model
# Optional:
#   WANDB_API_KEY     - if unset, wandb logging is stripped
#   DATA_SOURCE       - scratch | download (default: scratch)
#   MODEL_ID, NUM_GPUS, HF_MODEL_REPO, HF_DATASET_REPO, WORK_DIR,
#   SCRATCH_ROOT, SFT_JSONL, MAX_EPOCHS, MAX_LEN, MBS, BS, HOLDOUT_DATASETS

set -euo pipefail
cd "$(dirname "$0")/../.."

_ok() { echo "[OK]   $*"; }
_info() { echo "[INFO] $*"; }
_fail() { echo "[FAIL] $*"; exit 1; }

DATA_SOURCE="${DATA_SOURCE:-scratch}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
NUM_GPUS="${NUM_GPUS:-2}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/aide_rl}"
HF_DATASET_REPO="${HF_DATASET_REPO:-guilhermedrud/ctu_datasets}"
HF_MODEL_REPO="${HF_MODEL_REPO:-guilhermedrud/aide-analyzer-sft-qwen3.5-9b}"
WORK_DIR="${WORK_DIR:-$(pwd)/outputs/analyzer_sft}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_LEN="${MAX_LEN:-2048}"
HOLDOUT_DATASETS="${HOLDOUT_DATASETS:-}"
CONFIG_SRC="${CONFIG:-configs/openrlhf/analyzer_sft.yaml}"
CTU_CSV="${CTU_CSV:-data/ctu_datasets_info.csv}"

case "${DATA_SOURCE}" in
  scratch)
    SFT_JSONL="${SFT_JSONL:-${SCRATCH_ROOT}/analyzer/sft.jsonl}"
    HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
    ;;
  download)
    SFT_JSONL="${SFT_JSONL:-${WORK_DIR}/analyzer/sft.jsonl}"
    HF_HOME="${HF_HOME:-${WORK_DIR}/hf_cache}"
    ;;
  *)
    _fail "DATA_SOURCE must be 'scratch' or 'download' (got: ${DATA_SOURCE})"
    ;;
esac

export HF_HOME
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${WORK_DIR}/ckpt"
mkdir -p "${WORK_DIR}" "${CKPT_DIR}" "${HF_HOME}"

SYS_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "${SYS_PYTHON}" ]] || _fail "no python3/python in PATH"

_info "data_source=${DATA_SOURCE} model=${MODEL_ID} gpus=${NUM_GPUS}"
_info "sft_jsonl=${SFT_JSONL}"
_info "model_repo=${HF_MODEL_REPO} work_dir=${WORK_DIR}"
_info "python=${SYS_PYTHON} ($("${SYS_PYTHON}" -V 2>&1))"

# ---------------------------------------------------------------------------
# 1) Install aide
# ---------------------------------------------------------------------------
_info "=== install aide ==="
"${SYS_PYTHON}" -m pip install -e .
"${SYS_PYTHON}" -m pip uninstall -y nvtx >/dev/null 2>&1 || true
_ok "aide installed"

# ---------------------------------------------------------------------------
# 2) Resolve SFT dataset
# ---------------------------------------------------------------------------
if [[ "${DATA_SOURCE}" == "download" ]]; then
  LOGS_DIR="${WORK_DIR}/logs"
  ANALYZER_DIR="${WORK_DIR}/analyzer"
  mkdir -p "${LOGS_DIR}" "${ANALYZER_DIR}"

  _info "=== download journals from hf://${HF_DATASET_REPO}/runs/ ==="
  "${SYS_PYTHON}" scripts/download_runs_hf.py \
    --hf_repo "${HF_DATASET_REPO}" \
    --dest_dir "${LOGS_DIR}" \
    --skip_existing
  N_JOURNALS="$(find "${LOGS_DIR}" -name journal.json | wc -l | tr -d ' ')"
  [[ "${N_JOURNALS}" -gt 0 ]] || _fail "no journal.json under ${LOGS_DIR}"
  _ok "found ${N_JOURNALS} journal(s)"

  _info "=== export analyzer SFT dataset ==="
  EXPORT_ARGS=(
    --logs_dir "${LOGS_DIR}"
    --out "${SFT_JSONL}"
    --preferences_out "${ANALYZER_DIR}/preferences.jsonl"
    --ctu_csv "${CTU_CSV}"
  )
  if [[ -n "${HOLDOUT_DATASETS}" ]]; then
    # shellcheck disable=SC2086
    EXPORT_ARGS+=(--holdout_datasets ${HOLDOUT_DATASETS})
  fi
  "${SYS_PYTHON}" scripts/export_analyzer_data.py "${EXPORT_ARGS[@]}"
else
  _info "=== locate scratch SFT dataset ==="
  [[ -f "${SFT_JSONL}" ]] || _fail \
    "missing ${SFT_JSONL}. Run the data-prep job first: polyaxon/plx_analyzer_data_prep_axia.yaml"
  if [[ -f "${SCRATCH_ROOT}/analyzer/MANIFEST.txt" ]]; then
    _info "scratch manifest:"
    cat "${SCRATCH_ROOT}/analyzer/MANIFEST.txt"
  fi
fi

[[ -f "${SFT_JSONL}" ]] || _fail "missing ${SFT_JSONL}"
N_ROWS="$("${SYS_PYTHON}" -c "print(sum(1 for _ in open('${SFT_JSONL}')))")"
[[ "${N_ROWS}" -gt 0 ]] || _fail "SFT dataset is empty: ${SFT_JSONL}"
_ok "SFT rows=${N_ROWS} -> ${SFT_JSONL}"

# ---------------------------------------------------------------------------
# 3) Train (optionally strip wandb when no API key)
# ---------------------------------------------------------------------------
TRAIN_CONFIG="${CONFIG_SRC}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  _info "WANDB_API_KEY unset; stripping logger.wandb from train config"
  TRAIN_CONFIG="${WORK_DIR}/analyzer_sft.nowandb.yaml"
  "${SYS_PYTHON}" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CONFIG_SRC}").read_text()) or {}
logger = cfg.get("logger") or {}
logger.pop("wandb", None)
cfg["logger"] = logger
Path("${TRAIN_CONFIG}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "${TRAIN_CONFIG}")
PY
fi

_info "=== OpenRLHF SFT (${NUM_GPUS} GPUs) ==="
export CONFIG="${TRAIN_CONFIG}"
export PRETRAIN="${MODEL_ID}"
export DATASET="${SFT_JSONL}"
export OUTPUT_DIR="${CKPT_DIR}"
export NUM_GPUS
export MAX_EPOCHS
export MAX_LEN
bash scripts/openrlhf/train_hint_sft.sh
_ok "training finished -> ${CKPT_DIR}"

HF_CKPT="${CKPT_DIR}"
if [[ ! -f "${HF_CKPT}/config.json" ]]; then
  CANDIDATE="$(find "${CKPT_DIR}" -name config.json -type f | head -n 1 || true)"
  if [[ -n "${CANDIDATE}" ]]; then
    HF_CKPT="$(dirname "${CANDIDATE}")"
  fi
fi
[[ -f "${HF_CKPT}/config.json" ]] || _fail "no HF checkpoint (config.json) under ${CKPT_DIR}"
_ok "HF checkpoint at ${HF_CKPT}"

# ---------------------------------------------------------------------------
# 4) Upload model
# ---------------------------------------------------------------------------
_info "=== upload model to hf://${HF_MODEL_REPO} (public) ==="
[[ -n "${HF_TOKEN}" ]] || _fail "HF_TOKEN required to upload the model"
"${SYS_PYTHON}" scripts/upload_model_to_hf.py \
  --local_dir "${HF_CKPT}" \
  --repo_id "${HF_MODEL_REPO}"
_ok "uploaded hf://${HF_MODEL_REPO}"
_ok "analyzer SFT e2e complete"
