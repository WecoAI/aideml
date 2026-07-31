#!/usr/bin/env bash
# Run AIDE eval with a local vLLM-served DPO controller model.
#
# Modes:
#   MODE=full       — policy_kind=controller (DPO drives tree actions + hints)
#   MODE=hint_only  — policy_kind=heuristic + controller_kind=llm (hints only)
#   MODE=unified    — controller_kind=unified (trained model replaces GPT review
#                     and drives tree expansion; requires an analyzer checkpoint)
#
# Examples:
#   MODE=full SMOKE=1 BENCHMARK=relbench TASK_ID=rel-f1__driver-dnf ./scripts/eval_dpo_vllm.sh
#   MODE=hint_only STEPS=20 SEEDS=1 ./scripts/eval_dpo_vllm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Prefer project venv (Python 3.11+). Bare `vllm` on PATH often points at ~/.local (Python 3.9).
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

run_vllm() {
  local vllm_bin="${PYTHON%/*}/vllm"
  if [[ -x "${vllm_bin}" ]]; then
    "${vllm_bin}" "$@"
    return
  fi
  if "${PYTHON}" -m vllm --help >/dev/null 2>&1; then
    "${PYTHON}" -m vllm "$@"
    return
  fi
  if command -v vllm >/dev/null 2>&1; then
    vllm "$@"
    return
  fi
  echo "vllm not found for ${PYTHON}. Install into the project venv:" >&2
  echo "  ${PYTHON} -m pip install vllm" >&2
  exit 1
}

MODE="${MODE:-full}"
MODEL_PATH="${MODEL_PATH:-checkpoints/aide_hint_controller_dpo}"
SERVED_NAME="${SERVED_NAME:-aide-dpo}"
PORT="${PORT:-8100}"
BENCHMARK="${BENCHMARK:-relbench}"
STEPS="${STEPS:-20}"
SEEDS="${SEEDS:-1}"
TASK_ID="${TASK_ID:-}"
SMOKE="${SMOKE:-0}"
UPLOAD_HF="${UPLOAD_HF:-1}"

if [[ "${SMOKE}" == "1" ]]; then
  STEPS=3
  SEEDS=1
  NUM_DRAFTS=1
  if [[ -z "${TASK_ID}" ]]; then
    TASK_ID="rel-f1__driver-dnf"
  fi
fi
NUM_DRAFTS="${NUM_DRAFTS:-}"

case "${MODE}" in
  full)
    POLICY_KIND="controller"
    CONTROLLER_KIND="none"
    MODEL_TAG="dpo_full"
    ;;
  hint_only)
    POLICY_KIND="heuristic"
    CONTROLLER_KIND="llm"
    MODEL_TAG="dpo_hint_only"
    ;;
  unified)
    POLICY_KIND="heuristic"
    CONTROLLER_KIND="unified"
    MODEL_TAG="dpo_unified"
    ;;
  *)
    echo "Unknown MODE=${MODE} (use full, hint_only or unified)" >&2
    exit 1
    ;;
esac

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid ${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! [[ -x "${PYTHON%/*}/vllm" ]] && ! command -v vllm >/dev/null 2>&1 && ! "${PYTHON}" -m vllm --help >/dev/null 2>&1; then
  echo "vllm not found. Install with: ${PYTHON} -m pip install vllm" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model not found at ${MODEL_PATH} (missing config.json)" >&2
  exit 1
fi

MODELS_URL="http://127.0.0.1:${PORT}/v1/models"
vllm_serving_our_model() {
  curl -sf "${MODELS_URL}" 2>/dev/null | grep -q "\"${SERVED_NAME}\""
}

port_in_use() {
  ss -tln 2>/dev/null | grep -qE ":${PORT}[[:space:]]" \
    || netstat -tln 2>/dev/null | grep -qE ":${PORT}[[:space:]]"
}

if vllm_serving_our_model; then
  echo "[vllm] reusing existing server for ${SERVED_NAME} on port ${PORT}"
  VLLM_PID=""
elif port_in_use; then
  echo "[vllm] port ${PORT} is in use by a non-vLLM service (likely another /health responder)." >&2
  echo "[vllm] /v1/models does not list ${SERVED_NAME}. Try PORT=8100 or free the port." >&2
  exit 1
else
  echo "[vllm] serving ${MODEL_PATH} as ${SERVED_NAME} on port ${PORT} (python=${PYTHON})"
  # Qwen3.5 SFT/DPO checkpoints keep the VLM config but only text weights — skip vision stack.
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  VLLM_ARGS=(
    --served-model-name "${SERVED_NAME}"
    --port "${PORT}"
    --host 127.0.0.1
    --gdn-prefill-backend triton
    --language-model-only
    --trust-remote-code
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
  )
  if [[ "${VLLM_LOG_REQUESTS:-0}" == "1" ]]; then
    VLLM_ARGS+=(--enable-log-requests)
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
    echo "[vllm] request logging enabled (set VLLM_LOGGING_LEVEL=DEBUG for prompts)"
  fi
  run_vllm serve "${MODEL_PATH}" "${VLLM_ARGS[@]}" &
  VLLM_PID=$!
fi

echo "[vllm] waiting for ${MODELS_URL} to list ${SERVED_NAME}"
for i in $(seq 1 180); do
  if vllm_serving_our_model; then
    echo "[vllm] ready after ${i}s"
    break
  fi
  if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] server exited before becoming ready" >&2
    exit 1
  fi
  sleep 2
  if [[ "${i}" -eq 180 ]]; then
    echo "[vllm] timed out waiting for ${SERVED_NAME} on port ${PORT}" >&2
    exit 1
  fi
done

export CONTROLLER_OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export CONTROLLER_OPENAI_API_KEY="${CONTROLLER_OPENAI_API_KEY:-dummy}"

EVAL_ARGS=(
  --benchmark "${BENCHMARK}"
  --steps "${STEPS}"
  --seeds "${SEEDS}"
  --policy_kind "${POLICY_KIND}"
  --controller_kind "${CONTROLLER_KIND}"
  --controller_model "${SERVED_NAME}"
  --controller_base_url "${CONTROLLER_OPENAI_BASE_URL}"
  --model_tag "${MODEL_TAG}"
  --relbench_download
)

if [[ "${UPLOAD_HF}" == "1" ]]; then
  EVAL_ARGS+=(--upload_hf)
fi

if [[ -n "${TASK_ID}" ]]; then
  EVAL_ARGS+=(--task_id "${TASK_ID}")
fi

if [[ -n "${NUM_DRAFTS}" ]]; then
  EVAL_ARGS+=(--num_drafts "${NUM_DRAFTS}")
fi

echo "[eval] MODE=${MODE} MODEL_TAG=${MODEL_TAG} STEPS=${STEPS} SEEDS=${SEEDS}"
"${PYTHON}" scripts/run_eval.py "${EVAL_ARGS[@]}"
