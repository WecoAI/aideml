#!/usr/bin/env bash
# Run the AIDE DPO eval across ALL tasks in the eval manifest, in parallel.
#
# A single vLLM controller server can serve many concurrent eval clients, so by
# default we launch ONE vLLM server and fan out the manifest tasks against it
# (JOBS controls how many eval processes run at once). If you have multiple GPUs
# you can launch several vLLM servers and round-robin tasks across them.
#
# Modes (same as scripts/eval_dpo_vllm.sh):
#   MODE=full       — policy_kind=controller (DPO drives tree actions + hints)
#   MODE=hint_only  — policy_kind=heuristic + controller_kind=llm (hints only)
#   MODE=unified    — controller_kind=unified (trained model replaces GPT review
#                     and drives tree expansion; requires an analyzer checkpoint)
#
# Examples:
#   # All tasks, one vLLM, 4 parallel eval clients:
#   JOBS=4 ./scripts/eval_dpo_all.sh
#
#   # Smoke test (3 steps) across every task, no HF upload:
#   SMOKE=1 UPLOAD_HF=0 ./scripts/eval_dpo_all.sh
#
#   # Two vLLM servers pinned to GPUs 0 and 1, tasks split across them:
#   NUM_VLLM=2 GPUS=0,1 ./scripts/eval_dpo_all.sh
#
#   # Only a subset of tasks:
#   TASKS="rel-f1__driver-dnf leaf-classification" ./scripts/eval_dpo_all.sh
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
MANIFEST="${MANIFEST:-data/eval/eval_tasks.jsonl}"
BENCHMARK="${BENCHMARK:-all}"
BASE_PORT="${BASE_PORT:-8100}"
STEPS="${STEPS:-35}"
SEEDS="${SEEDS:-1}"
SMOKE="${SMOKE:-0}"
UPLOAD_HF="${UPLOAD_HF:-1}"
NUM_VLLM="${NUM_VLLM:-1}"
GPUS="${GPUS:-}"                       # optional comma list, e.g. "0,1"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# How many eval clients to run at once. One vLLM happily serves several.
JOBS="${JOBS:-$(( NUM_VLLM > 1 ? NUM_VLLM : 2 ))}"

if [[ "${SMOKE}" == "1" ]]; then
  STEPS=3
  SEEDS=1
  NUM_DRAFTS="${NUM_DRAFTS:-1}"
fi
NUM_DRAFTS="${NUM_DRAFTS:-}"

case "${MODE}" in
  full)
    POLICY_KIND="controller"
    CONTROLLER_KIND="none"
    MODEL_TAG="${MODEL_TAG:-dpo_full}"
    ;;
  hint_only)
    POLICY_KIND="heuristic"
    CONTROLLER_KIND="llm"
    MODEL_TAG="${MODEL_TAG:-dpo_hint_only}"
    ;;
  unified)
    POLICY_KIND="heuristic"
    CONTROLLER_KIND="unified"
    MODEL_TAG="${MODEL_TAG:-dpo_unified}"
    ;;
  *)
    echo "Unknown MODE=${MODE} (use full, hint_only or unified)" >&2
    exit 1
    ;;
esac

if ! [[ -x "${PYTHON%/*}/vllm" ]] && ! command -v vllm >/dev/null 2>&1 && ! "${PYTHON}" -m vllm --help >/dev/null 2>&1; then
  echo "vllm not found. Install with: ${PYTHON} -m pip install vllm" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model not found at ${MODEL_PATH} (missing config.json)" >&2
  exit 1
fi

# ---- Resolve task list ------------------------------------------------------
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_IDS <<< "${TASKS}"
else
  mapfile -t TASK_IDS < <("${PYTHON}" - "${MANIFEST}" "${BENCHMARK}" <<'PY'
import json, sys
manifest, benchmark = sys.argv[1], sys.argv[2]
with open(manifest) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if benchmark in ("all", "", row.get("benchmark")):
            print(row["id"])
PY
)
fi

if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
  echo "No tasks resolved from ${MANIFEST} (BENCHMARK=${BENCHMARK})" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-data/eval/parallel/${MODEL_TAG}}"
mkdir -p "${OUT_DIR}"

echo "[plan] MODE=${MODE} MODEL_TAG=${MODEL_TAG} STEPS=${STEPS} SEEDS=${SEEDS}"
echo "[plan] tasks=${#TASK_IDS[@]} vllm_servers=${NUM_VLLM} parallel_jobs=${JOBS}"
echo "[plan] outputs -> ${OUT_DIR}"

# ---- Start vLLM server(s) ---------------------------------------------------
declare -a VLLM_PIDS=()
declare -a PORTS=()
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

cleanup() {
  for pid in "${VLLM_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[vllm] stopping pid ${pid}"
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

serving_our_model() {
  curl -sf "http://127.0.0.1:${1}/v1/models" 2>/dev/null | grep -q "\"${SERVED_NAME}\""
}
port_in_use() {
  ss -tln 2>/dev/null | grep -qE ":${1}[[:space:]]" \
    || netstat -tln 2>/dev/null | grep -qE ":${1}[[:space:]]"
}

for i in $(seq 0 $((NUM_VLLM - 1))); do
  port=$((BASE_PORT + i))
  PORTS[$i]="${port}"
  if serving_our_model "${port}"; then
    echo "[vllm] reusing existing server for ${SERVED_NAME} on port ${port}"
    VLLM_PIDS[$i]=""
    continue
  fi
  if port_in_use "${port}"; then
    echo "[vllm] port ${port} is in use by a non-vLLM service; free it or change BASE_PORT." >&2
    exit 1
  fi
  vllm_args=(
    --served-model-name "${SERVED_NAME}"
    --port "${port}"
    --host 127.0.0.1
    --gdn-prefill-backend triton
    --language-model-only
    --trust-remote-code
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
  )
  if [[ -n "${GPUS}" ]]; then
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    echo "[vllm] serving ${MODEL_PATH} as ${SERVED_NAME} on port ${port} (CUDA_VISIBLE_DEVICES=${gpu})"
    CUDA_VISIBLE_DEVICES="${gpu}" run_vllm serve "${MODEL_PATH}" "${vllm_args[@]}" \
      >"${OUT_DIR}/vllm_${port}.log" 2>&1 &
  else
    echo "[vllm] serving ${MODEL_PATH} as ${SERVED_NAME} on port ${port}"
    run_vllm serve "${MODEL_PATH}" "${vllm_args[@]}" \
      >"${OUT_DIR}/vllm_${port}.log" 2>&1 &
  fi
  VLLM_PIDS[$i]=$!
  disown "${VLLM_PIDS[$i]}"   # keep it out of the `jobs` table used for throttling
done

# ---- Wait for all servers to be ready ---------------------------------------
for i in $(seq 0 $((NUM_VLLM - 1))); do
  port="${PORTS[$i]}"
  echo "[vllm] waiting for ${SERVED_NAME} on port ${port}"
  for s in $(seq 1 180); do
    if serving_our_model "${port}"; then
      echo "[vllm] port ${port} ready after ${s}s"
      break
    fi
    pid="${VLLM_PIDS[$i]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "[vllm] server on port ${port} exited before ready (see ${OUT_DIR}/vllm_${port}.log)" >&2
      exit 1
    fi
    sleep 2
    if [[ "${s}" -eq 180 ]]; then
      echo "[vllm] timed out waiting for ${SERVED_NAME} on port ${port}" >&2
      exit 1
    fi
  done
done

# ---- Run one eval task ------------------------------------------------------
run_one_task() {
  local task="$1" port="$2"
  local safe base_url results_csv agg log rc
  safe="$(printf '%s' "${task}" | tr '/ ' '__')"
  base_url="http://127.0.0.1:${port}/v1"
  results_csv="${OUT_DIR}/eval_results_${MODEL_TAG}_${safe}.csv"
  agg="${OUT_DIR}/metrics_${MODEL_TAG}_${safe}.json"
  log="${OUT_DIR}/run_${safe}.log"

  local args=(
    --manifest "${MANIFEST}"
    --benchmark all
    --task_id "${task}"
    --steps "${STEPS}"
    --seeds "${SEEDS}"
    --policy_kind "${POLICY_KIND}"
    --controller_kind "${CONTROLLER_KIND}"
    --controller_model "${SERVED_NAME}"
    --controller_base_url "${base_url}"
    --model_tag "${MODEL_TAG}"
    --relbench_download
    --results_csv "${results_csv}"
    --aggregate_path "${agg}"
  )
  [[ "${UPLOAD_HF}" == "1" ]] && args+=(--upload_hf)
  [[ -n "${NUM_DRAFTS}" ]] && args+=(--num_drafts "${NUM_DRAFTS}")

  echo "[start] ${task} -> port ${port} (log: ${log})"
  set +e
  CONTROLLER_OPENAI_BASE_URL="${base_url}" \
  CONTROLLER_OPENAI_API_KEY="${CONTROLLER_OPENAI_API_KEY:-dummy}" \
    "${PYTHON}" scripts/run_eval.py "${args[@]}" >"${log}" 2>&1
  rc=$?
  set -e
  printf '%s' "${rc}" > "${OUT_DIR}/status_${safe}.txt"
  if [[ "${rc}" -eq 0 ]]; then
    echo "[ok]   ${task} (port ${port})"
  else
    echo "[FAIL] ${task} (port ${port}) rc=${rc} -> ${log}"
  fi
}

# ---- Fan out tasks with a JOBS-sized pool -----------------------------------
idx=0
for task in "${TASK_IDS[@]}"; do
  srv=$(( idx % NUM_VLLM ))
  port="${PORTS[$srv]}"
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    sleep 1
  done
  run_one_task "${task}" "${port}" &
  idx=$((idx + 1))
done
wait

# ---- Merge per-task results + (optionally) upload one aggregate --------------
"${PYTHON}" - "${OUT_DIR}" "${MODEL_TAG}" "${UPLOAD_HF}" "${TASK_IDS[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
model_tag = sys.argv[2]
upload = sys.argv[3] == "1"
task_ids = sys.argv[4:]

# Merge aggregate metrics JSON.
rows = []
for agg in sorted(out_dir.glob(f"metrics_{model_tag}_*.json")):
    try:
        data = json.loads(agg.read_text())
        if isinstance(data, list):
            rows.extend(data)
    except json.JSONDecodeError:
        pass
merged_json = Path("data/eval") / f"metrics_{model_tag}.json"
merged_json.parent.mkdir(parents=True, exist_ok=True)
merged_json.write_text(json.dumps(rows, indent=2))
print(f"[merge] {len(rows)} metric rows -> {merged_json}")

# Merge per-task result CSVs into one.
csv_paths = sorted(out_dir.glob(f"eval_results_{model_tag}_*.csv"))
all_rows, fieldnames = [], None
for p in csv_paths:
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames and fieldnames is None:
            fieldnames = r.fieldnames
        all_rows.extend(list(r))
if fieldnames:
    merged_csv = out_dir / f"eval_results_{model_tag}_ALL.csv"
    with merged_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[merge] {len(all_rows)} result rows -> {merged_csv}")

if upload and rows:
    try:
        from data.hf_utils import upload_eval_metrics
        upload_eval_metrics(merged_json, model_tag=model_tag)
    except Exception as exc:  # noqa: BLE001
        print(f"[merge WARN] aggregate metrics upload failed: {exc}", file=sys.stderr)
PY

# ---- Summary ----------------------------------------------------------------
failures=0
echo "===================== eval summary ====================="
for task in "${TASK_IDS[@]}"; do
  safe="$(printf '%s' "${task}" | tr '/ ' '__')"
  status_file="${OUT_DIR}/status_${safe}.txt"
  if [[ -f "${status_file}" ]] && [[ "$(cat "${status_file}")" == "0" ]]; then
    printf '  OK    %s\n' "${task}"
  else
    printf '  FAIL  %s\n' "${task}"
    failures=$((failures + 1))
  fi
done
echo "========================================================"
echo "[done] ${#TASK_IDS[@]} task(s), ${failures} failure(s). Logs -> ${OUT_DIR}"

if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
