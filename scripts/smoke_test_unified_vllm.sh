#!/usr/bin/env bash
# Serve a base (non-finetuned) model with vLLM and run the unified-analyzer
# AIDE smoke test on one random CTU task from the HF dataset repo.
#
# Designed to run on Polyaxon (polyaxon/plx_unified_smoke_axia.yaml) with the
# prebaked guilhermedrud/aide-openrlhf image, but works anywhere with a GPU:
#
#   MODEL_ID=Qwen/Qwen3.5-9B STEPS=5 bash scripts/smoke_test_unified_vllm.sh
#
# Requires: OPENAI_API_KEY (coding LLM + review fallback), HF_TOKEN (dataset).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-aide-unified-base}"
PORT="${PORT:-8100}"
STEPS="${STEPS:-5}"
NUM_DRAFTS="${NUM_DRAFTS:-2}"
SEED="${SEED:-0}"
TASK_INDEX="${TASK_INDEX:--1}"          # -1 = random task from the CSV
CSV_PATH="${CSV_PATH:-data/ctu_datasets_info.csv}"
HF_REPO="${HF_REPO:-guilhermedrud/ctu_datasets}"
OUT_DIR="${OUT_DIR:-data/unified_smoke}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-900}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
# Same serving flags the DPO eval scripts use for this model family; override if needed.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gdn-prefill-backend triton --language-model-only}"
VLLM_WAIT_SECS="${VLLM_WAIT_SECS:-1800}"  # includes model download on cold cache
SKIP_INSTALL="${SKIP_INSTALL:-0}"

mkdir -p "${OUT_DIR}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

echo "[smoke] model=${MODEL_ID} served_as=${SERVED_NAME} port=${PORT} steps=${STEPS}"
echo "[smoke] python=${PYTHON}"
"${PYTHON}" -c "import torch; print('[smoke] cuda available:', torch.cuda.is_available())" || true

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  echo "[smoke] installing aide + libs for generated code"
  # No --no-deps: pyproject deps are unpinned and pip's default only-if-needed
  # policy won't touch the baked torch/vllm stack, but it will pull small
  # pure-python deps missing from the image (e.g. backoff).
  "${PYTHON}" -m pip install -q -e .
  # The generated solutions typically import these; the prebaked image lacks them.
  "${PYTHON}" -m pip install -q scikit-learn lightgbm || true
fi

run_vllm() {
  local vllm_bin="${PYTHON%/*}/vllm"
  if [[ -x "${vllm_bin}" ]]; then "${vllm_bin}" "$@"; return; fi
  if command -v vllm >/dev/null 2>&1; then vllm "$@"; return; fi
  "${PYTHON}" -m vllm "$@"
}

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid ${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

MODELS_URL="http://127.0.0.1:${PORT}/v1/models"
if curl -sf "${MODELS_URL}" 2>/dev/null | grep -q "\"${SERVED_NAME}\""; then
  echo "[vllm] reusing existing server on port ${PORT}"
else
  echo "[vllm] serving ${MODEL_ID} as ${SERVED_NAME} on port ${PORT} (log: ${OUT_DIR}/vllm.log)"
  # shellcheck disable=SC2086
  run_vllm serve "${MODEL_ID}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    ${VLLM_EXTRA_ARGS} \
    >"${OUT_DIR}/vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "[vllm] waiting up to ${VLLM_WAIT_SECS}s for readiness (includes model download)"
  ready=0
  for s in $(seq 1 $((VLLM_WAIT_SECS / 5))); do
    if curl -sf "${MODELS_URL}" 2>/dev/null | grep -q "\"${SERVED_NAME}\""; then
      echo "[vllm] ready after $((s * 5))s"
      ready=1
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[vllm] server exited before ready; last log lines:" >&2
      tail -n 40 "${OUT_DIR}/vllm.log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[vllm] timed out waiting for ${SERVED_NAME}; last log lines:" >&2
    tail -n 40 "${OUT_DIR}/vllm.log" >&2
    exit 1
  fi
fi

DRIVER_ARGS=(
  --csv_path "${CSV_PATH}"
  --seed "${SEED}"
  --steps "${STEPS}"
  --num_drafts "${NUM_DRAFTS}"
  --exec_timeout "${EXEC_TIMEOUT}"
  --controller_model "${SERVED_NAME}"
  --controller_base_url "http://127.0.0.1:${PORT}/v1"
  --hf_repo "${HF_REPO}"
  --out_dir "${OUT_DIR}"
)
if [[ "${TASK_INDEX}" != "-1" ]]; then
  DRIVER_ARGS+=(--task_index "${TASK_INDEX}")
fi

set +e
CONTROLLER_OPENAI_API_KEY="${CONTROLLER_OPENAI_API_KEY:-dummy}" \
  "${PYTHON}" scripts/run_unified_smoke.py "${DRIVER_ARGS[@]}"
rc=$?
set -e
echo "[smoke] driver exit code: ${rc}"
exit "${rc}"
