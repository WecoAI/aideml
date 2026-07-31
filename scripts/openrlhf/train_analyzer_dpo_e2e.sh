#!/usr/bin/env bash
# Unified-analyzer DPO on Polyaxon (axia), using scratch preferences + SFT model:
#   1) install aide on the prebaked OpenRLHF image
#   2) read preferences.jsonl from /scratch/aide_rl/analyzer/
#   3) DPO-fine-tune starting from the analyzer SFT HF model
#   4) upload the HF-format checkpoint to a model repo
#
# Prerequisite: run polyaxon/plx_analyzer_data_prep_axia.yaml so that
#   ${SCRATCH_ROOT}/analyzer/preferences.jsonl exists.
#
# Required env:
#   HF_TOKEN          - download SFT base + write DPO model (write scope)
# Optional:
#   WANDB_API_KEY     - if unset, wandb logging is stripped
#   MODEL_ID          - SFT checkpoint (default: guilhermedrud/aide-analyzer-sft-qwen3.5-9b)
#   NUM_GPUS          - default 4
#   SCRATCH_ROOT, PREFS_JSONL, HF_MODEL_REPO, WORK_DIR,
#   MAX_EPOCHS, MAX_LEN, MBS, BS, BETA, LR

set -euo pipefail
cd "$(dirname "$0")/../.."

_ok() { echo "[OK]   $*"; }
_info() { echo "[INFO] $*"; }
_fail() { echo "[FAIL] $*"; exit 1; }

MODEL_ID="${MODEL_ID:-guilhermedrud/aide-analyzer-sft-qwen3.5-9b}"
NUM_GPUS="${NUM_GPUS:-4}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/aide_rl}"
PREFS_JSONL="${PREFS_JSONL:-${SCRATCH_ROOT}/analyzer/preferences.jsonl}"
HF_MODEL_REPO="${HF_MODEL_REPO:-guilhermedrud/aide-analyzer-dpo-qwen3.5-9b}"
WORK_DIR="${WORK_DIR:-$(pwd)/outputs/analyzer_dpo}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_LEN="${MAX_LEN:-2048}"
CONFIG_SRC="${CONFIG:-configs/openrlhf/analyzer_dpo.yaml}"

export HF_HOME
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${WORK_DIR}/ckpt"
mkdir -p "${WORK_DIR}" "${CKPT_DIR}" "${HF_HOME}"

SYS_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "${SYS_PYTHON}" ]] || _fail "no python3/python in PATH"

_info "model=${MODEL_ID} gpus=${NUM_GPUS}"
_info "prefs_jsonl=${PREFS_JSONL}"
_info "model_repo=${HF_MODEL_REPO}"
_info "scratch_root=${SCRATCH_ROOT} work_dir=${WORK_DIR}"
_info "python=${SYS_PYTHON} ($("${SYS_PYTHON}" -V 2>&1))"

# ---------------------------------------------------------------------------
# 1) Install aide
# ---------------------------------------------------------------------------
_info "=== install aide ==="
"${SYS_PYTHON}" -m pip install -e .
"${SYS_PYTHON}" -m pip uninstall -y nvtx >/dev/null 2>&1 || true
_ok "aide installed"

# ---------------------------------------------------------------------------
# 2) Locate scratch preference dataset
# ---------------------------------------------------------------------------
_info "=== locate scratch preferences ==="
[[ -f "${PREFS_JSONL}" ]] || _fail \
  "missing ${PREFS_JSONL}. Run the data-prep job first: polyaxon/plx_analyzer_data_prep_axia.yaml"
if [[ -f "${SCRATCH_ROOT}/analyzer/MANIFEST.txt" ]]; then
  _info "scratch manifest:"
  cat "${SCRATCH_ROOT}/analyzer/MANIFEST.txt"
fi
N_ROWS="$("${SYS_PYTHON}" -c "print(sum(1 for _ in open('${PREFS_JSONL}')))")"
[[ "${N_ROWS}" -gt 0 ]] || _fail "preferences dataset is empty: ${PREFS_JSONL}"
_ok "preference pairs=${N_ROWS} -> ${PREFS_JSONL}"

# ---------------------------------------------------------------------------
# 3) Train DPO (optionally strip wandb when no API key)
# ---------------------------------------------------------------------------
TRAIN_CONFIG="${CONFIG_SRC}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  _info "WANDB_API_KEY unset; stripping logger.wandb from train config"
  TRAIN_CONFIG="${WORK_DIR}/analyzer_dpo.nowandb.yaml"
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

_info "=== OpenRLHF DPO (${NUM_GPUS} GPUs) ==="
export CONFIG="${TRAIN_CONFIG}"
export PRETRAIN="${MODEL_ID}"
export DATASET="${PREFS_JSONL}"
export OUTPUT_DIR="${CKPT_DIR}"
export NUM_GPUS
export MAX_EPOCHS
export MAX_LEN
bash scripts/openrlhf/train_hint_dpo.sh
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
_ok "analyzer DPO e2e complete"
