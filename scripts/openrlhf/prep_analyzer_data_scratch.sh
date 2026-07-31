#!/usr/bin/env bash
# Download AIDE journals from HF and export the unified-analyzer SFT dataset
# onto the axia node local scratch under /scratch/aide_rl/.
#
# Intended to run as a CPU-only Polyaxon job (no GPU). Training jobs then
# read the prebuilt jsonl from scratch instead of re-downloading every run.
#
# Layout written:
#   ${SCRATCH_ROOT}/journals/              # HF runs/ bundles (journal.json, ...)
#   ${SCRATCH_ROOT}/analyzer/sft.jsonl
#   ${SCRATCH_ROOT}/analyzer/preferences.jsonl
#   ${SCRATCH_ROOT}/analyzer/MANIFEST.txt
#
# Env:
#   HF_TOKEN           - recommended (higher rate limits); set on Polyaxon project
#   HF_DATASET_REPO    - default guilhermedrud/ctu_datasets
#   SCRATCH_ROOT       - default /scratch/aide_rl
#   HOLDOUT_DATASETS   - optional space-separated prefixes for sft_val.jsonl
#   CTU_CSV            - default data/ctu_datasets_info.csv
#   FORCE_REDOWNLOAD   - if 1, re-download journals even when present

set -euo pipefail
cd "$(dirname "$0")/../.."

_ok() { echo "[OK]   $*"; }
_info() { echo "[INFO] $*"; }
_fail() { echo "[FAIL] $*"; exit 1; }

HF_DATASET_REPO="${HF_DATASET_REPO:-guilhermedrud/ctu_datasets}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/aide_rl}"
HOLDOUT_DATASETS="${HOLDOUT_DATASETS:-}"
CTU_CSV="${CTU_CSV:-data/ctu_datasets_info.csv}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"

export HF_HOME
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

JOURNALS_DIR="${SCRATCH_ROOT}/journals"
ANALYZER_DIR="${SCRATCH_ROOT}/analyzer"
SFT_JSONL="${ANALYZER_DIR}/sft.jsonl"
PREFS_JSONL="${ANALYZER_DIR}/preferences.jsonl"
MANIFEST="${ANALYZER_DIR}/MANIFEST.txt"

mkdir -p "${JOURNALS_DIR}" "${ANALYZER_DIR}" "${HF_HOME}"

SYS_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "${SYS_PYTHON}" ]] || _fail "no python3/python in PATH"

_info "scratch_root=${SCRATCH_ROOT}"
_info "dataset_repo=${HF_DATASET_REPO}"
_info "python=${SYS_PYTHON} ($("${SYS_PYTHON}" -V 2>&1))"

# ---------------------------------------------------------------------------
# 1) Install aide (CPU job; only need export deps)
# ---------------------------------------------------------------------------
_info "=== install aide ==="
"${SYS_PYTHON}" -m pip install -e .
_ok "aide installed"

# ---------------------------------------------------------------------------
# 2) Download journals onto scratch
# ---------------------------------------------------------------------------
_info "=== download journals -> ${JOURNALS_DIR} ==="
DL_ARGS=(
  --hf_repo "${HF_DATASET_REPO}"
  --dest_dir "${JOURNALS_DIR}"
)
if [[ "${FORCE_REDOWNLOAD}" != "1" ]]; then
  DL_ARGS+=(--skip_existing)
fi
"${SYS_PYTHON}" scripts/download_runs_hf.py "${DL_ARGS[@]}"
N_JOURNALS="$(find "${JOURNALS_DIR}" -name journal.json | wc -l | tr -d ' ')"
[[ "${N_JOURNALS}" -gt 0 ]] || _fail "no journal.json under ${JOURNALS_DIR}"
_ok "found ${N_JOURNALS} journal(s)"

# ---------------------------------------------------------------------------
# 3) Export analyzer SFT dataset onto scratch
# ---------------------------------------------------------------------------
_info "=== export analyzer SFT -> ${ANALYZER_DIR} ==="
EXPORT_ARGS=(
  --logs_dir "${JOURNALS_DIR}"
  --out "${SFT_JSONL}"
  --preferences_out "${PREFS_JSONL}"
  --ctu_csv "${CTU_CSV}"
)
if [[ -n "${HOLDOUT_DATASETS}" ]]; then
  # shellcheck disable=SC2086
  EXPORT_ARGS+=(--holdout_datasets ${HOLDOUT_DATASETS})
fi
"${SYS_PYTHON}" scripts/export_analyzer_data.py "${EXPORT_ARGS[@]}"
[[ -f "${SFT_JSONL}" ]] || _fail "missing ${SFT_JSONL}"
N_ROWS="$("${SYS_PYTHON}" -c "print(sum(1 for _ in open('${SFT_JSONL}')))")"
[[ "${N_ROWS}" -gt 0 ]] || _fail "SFT dataset is empty: ${SFT_JSONL}"
N_PREFS=0
if [[ -f "${PREFS_JSONL}" ]]; then
  N_PREFS="$("${SYS_PYTHON}" -c "print(sum(1 for _ in open('${PREFS_JSONL}')))")"
fi
_ok "SFT rows=${N_ROWS} prefs=${N_PREFS}"

{
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hf_dataset_repo=${HF_DATASET_REPO}"
  echo "journals_dir=${JOURNALS_DIR}"
  echo "n_journals=${N_JOURNALS}"
  echo "sft_jsonl=${SFT_JSONL}"
  echo "sft_rows=${N_ROWS}"
  echo "preferences_jsonl=${PREFS_JSONL}"
  echo "preferences_rows=${N_PREFS}"
  echo "holdout_datasets=${HOLDOUT_DATASETS}"
} > "${MANIFEST}"
_ok "wrote ${MANIFEST}"
cat "${MANIFEST}"
_ok "analyzer data prep complete -> ${SCRATCH_ROOT}"
