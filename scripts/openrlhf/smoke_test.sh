#!/usr/bin/env bash
# Smoke-test OpenRLHF + Ray on GPU for the AIDE RLHF pipeline.
#
# Modes (env MODE):
#   gpu    - nvidia-smi + system torch only (no venv, no installs)
#   venv   - create .venv and verify torch inside it
#   check  - gpu + venv + install + imports + ray
#   sft    - minimal supervised fine-tuning
#   grpo   - minimal GRPO via Ray + train_ppo_ray
#   all    - check + sft + grpo (default)
#
# Usage:
#   MODE=gpu bash scripts/openrlhf/smoke_test.sh
#   MODE=check bash scripts/openrlhf/smoke_test.sh

set -e
cd "$(dirname "$0")/../.."

MODE="${MODE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/openrlhf_smoke}"
SMOKE_ROOT="${OUTPUT_DIR}"
VENV_DIR="${VENV_DIR:-/tmp/aide-openrlhf-venv}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
RAY_IP="${RAY_IP:-127.0.0.1}"
INSTALL_OPENRLHF="${INSTALL_OPENRLHF:-1}"
# PREBAKED=1 -> image already has torch/vllm/openrlhf in system python:
# skip venv + heavy installs, just `pip install -e . --no-deps` for aide.
# PREBAKED=auto (default) detects this by trying to import openrlhf+vllm.
PREBAKED="${PREBAKED:-auto}"
HF_HOME="${HF_HOME:-${SMOKE_ROOT}/hf_cache}"
export HF_HOME

mkdir -p "${SMOKE_ROOT}"

_ok() { echo "[OK]   $*"; }
_fail() { echo "[FAIL] $*"; }
_info() { echo "[INFO] $*"; }
_warn() { echo "[WARN] $*"; }

_should_run() {
  case "${MODE}" in
    all)
      case "$1" in
        check|sft|grpo) return 0 ;;
      esac
      ;;
    "$1") return 0 ;;
  esac
  return 1
}

_stop_ray() {
  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
}

trap _stop_ray EXIT

# Resolve a usable system python once; bare images (e.g. nvidia/cuda:*-base)
# ship neither `python` nor `python3`.
SYS_PYTHON="$(command -v python3 || command -v python || true)"

_detect_prebaked() {
  if [[ "${PREBAKED}" != "auto" ]]; then
    return 0
  fi
  if [[ -n "${SYS_PYTHON}" ]] && "${SYS_PYTHON}" -c "import openrlhf, vllm, torch" >/dev/null 2>&1; then
    PREBAKED=1
    _info "Detected prebaked image (openrlhf+vllm importable in system python)"
  else
    PREBAKED=0
  fi
}
_detect_prebaked

check_gpu_system() {
  _info "=== GPU / CUDA (system python) ==="
  command -v nvidia-smi >/dev/null 2>&1 || { _fail "nvidia-smi not found"; return 1; }
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  if [[ -z "${SYS_PYTHON}" ]]; then
    _fail "no python/python3 found in the image; use a python-enabled image (e.g. the prebaked one from docker/openrlhf/Dockerfile)"
    return 1
  fi
  "${SYS_PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  _ok "GPU visible to system PyTorch"
}

_bootstrap_venv_pip() {
  local py="${VENV_DIR}/bin/python"
  if "${py}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  _info "bootstrapping pip into venv (get-pip.py; apt mirrors blocked on cluster)"
  local getter="/tmp/get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${getter}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${getter}" https://bootstrap.pypa.io/get-pip.py
  else
    _fail "need curl or wget to bootstrap pip"
    return 1
  fi
  "${py}" "${getter}"
  _ok "pip bootstrapped in venv"
}

setup_venv() {
  if [[ -n "${_VENV_READY:-}" ]]; then
    return 0
  fi
  if [[ "${PREBAKED}" == "1" ]]; then
    _VENV_READY=1
    _info "=== Prebaked image: using system python (no venv) ==="
    # Some images only ship python3; make sure `python` resolves for the
    # remaining steps.
    if ! command -v python >/dev/null 2>&1; then
      mkdir -p /tmp/aide-pybin
      ln -sf "${SYS_PYTHON}" /tmp/aide-pybin/python
      export PATH="/tmp/aide-pybin:${PATH}"
    fi
    _ok "python: ${SYS_PYTHON} ($("${SYS_PYTHON}" -V 2>&1))"
    return 0
  fi
  _info "=== Python venv (${VENV_DIR}) ==="
  if [[ -z "${SYS_PYTHON}" ]]; then
    _fail "no python/python3 in the image; cannot create venv"
    return 1
  fi
  _info "system python: ${SYS_PYTHON} ($("${SYS_PYTHON}" -V 2>&1))"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    rm -rf "${VENV_DIR}"
    # ensurepip needs python3.12-venv via apt, but cluster blocks archive.ubuntu.com.
    "${SYS_PYTHON}" -m venv --system-site-packages --without-pip "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  _bootstrap_venv_pip

  _VENV_READY=1
  _ok "venv: $(python -c 'import sys; print(sys.executable)')"
}

check_gpu_venv() {
  setup_venv
  _info "=== GPU / CUDA (venv python) ==="
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  _ok "GPU visible inside venv"
}

venv_python() {
  if [[ "${PREBAKED}" == "1" ]]; then
    echo "${SYS_PYTHON}"
  else
    echo "${VENV_DIR}/bin/python"
  fi
}

install_openrlhf() {
  if [[ "${INSTALL_OPENRLHF}" != "1" ]]; then
    _info "Skipping OpenRLHF install (INSTALL_OPENRLHF=${INSTALL_OPENRLHF})"
    return 0
  fi
  setup_venv
  if [[ "${PREBAKED}" == "1" ]]; then
    _info "=== Prebaked image: installing aide package ==="
    # No --no-deps: pyproject deps are unpinned and pip's default
    # only-if-needed policy won't touch the baked torch/vllm stack, but it
    # will pull small pure-python deps missing from the image (e.g. backoff).
    "${SYS_PYTHON}" -m pip install -e .
    # The base NVIDIA image ships an `nvtx` package whose DummyDomain API is
    # incompatible with deepspeed's profiling wrapper (push_range TypeError).
    # Without it deepspeed falls back to torch.cuda.nvtx, which works.
    "${SYS_PYTHON}" -m pip uninstall -y nvtx >/dev/null 2>&1 || true
    _ok "aide installed on prebaked stack"
    return 0
  fi
  _info "=== Install OpenRLHF stack ==="
  python -m pip install -U pip wheel
  python -m pip install "vllm==0.19.1"
  python -m pip install "flash-attn" --no-build-isolation
  python -m pip install "openrlhf[vllm]" --no-build-isolation
  python -m pip install -e .
  _ok "OpenRLHF installed in venv"
}

check_imports() {
  setup_venv
  _info "=== Import check ==="
  python - <<'PY'
import openrlhf
import ray
import vllm
import torch
print("openrlhf:", getattr(openrlhf, "__version__", "unknown"))
print("ray:", ray.__version__)
print("vllm:", vllm.__version__)
print("torch cuda:", torch.cuda.is_available())
PY
  python -c "from aide.rlhf.grpo_reward_entrypoint import reward_func; print('reward_func ok')"
  _ok "Imports"
}

check_ray() {
  setup_venv
  _info "=== Ray head (ip=${RAY_IP}) ==="
  _stop_ray
  ray start \
    --head \
    --node-ip-address="${RAY_IP}" \
    --dashboard-host="${RAY_IP}" \
    --num-gpus="${NUM_GPUS}" \
    --disable-usage-stats
  ray status
  python - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
resources = ray.cluster_resources()
print("ray cluster resources:", resources)
assert resources.get("GPU", 0) >= 1, "Ray sees 0 GPUs"
PY
  _ok "Ray head running with GPU resources"
}

# Args below target OpenRLHF 0.10.x (namespaced --data.*/--train.*/... CLI;
# the old flat flags like --pretrain/--dataset were removed upstream).
_run_sft() {
  setup_venv
  _info "=== SFT smoke (${SMOKE_MODEL}) ==="
  local out="${SMOKE_ROOT}/sft"
  mkdir -p "${out}"

  deepspeed --module openrlhf.cli.train_sft \
    --model.model_name_or_path "${SMOKE_MODEL}" \
    --data.dataset data/openrlhf_smoke/sft.jsonl \
    --data.input_key messages \
    --data.apply_chat_template \
    --data.max_len 512 \
    --train.max_epochs 1 \
    --train.batch_size 1 \
    --train.micro_batch_size 1 \
    --adam.lr 2e-5 \
    --ds.zero_stage 2 \
    --ds.param_dtype bf16 \
    --ckpt.output_dir "${out}" \
    --ckpt.save_hf
  _ok "SFT smoke finished -> ${out}"
}

_run_grpo() {
  setup_venv
  _info "=== GRPO smoke via Ray (${SMOKE_MODEL}) ==="
  local out="${SMOKE_ROOT}/grpo"
  local py
  py="$(venv_python)"
  mkdir -p "${out}"

  if ! ray status >/dev/null 2>&1; then
    check_ray
  fi

  # GRPO = PPO trainer with group_norm advantage estimator + KL-in-loss.
  # A reward endpoint ending in .py is importlib-loaded and its
  # reward_func(queries, prompts, labels) is called (see aide/rlhf/grpo_reward_entrypoint.py).
  #
  # Ray's --working-dir packaging honors .gitignore, which ignores
  # data/openrlhf_smoke/ — without this flag the dataset is silently
  # dropped from the job sandbox (FileNotFoundError in PPOTrainer).
  RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 \
  ray job submit --address="http://${RAY_IP}:8265" \
    --working-dir "$(pwd)" \
    -- "${py}" -m openrlhf.cli.train_ppo_ray \
    --actor.model_name_or_path "${SMOKE_MODEL}" \
    --data.prompt_dataset data/openrlhf_smoke/grpo_prompts.jsonl \
    --data.input_key messages \
    --data.label_key label \
    --data.apply_chat_template \
    --reward.remote_url aide/rlhf/grpo_reward_entrypoint.py \
    --algo.advantage.estimator group_norm \
    --algo.kl.use_loss \
    --algo.kl.init_coef 0.001 \
    --train.max_epochs 1 \
    --train.num_episodes 1 \
    --train.batch_size 4 \
    --train.micro_batch_size 1 \
    --data.max_len 512 \
    --rollout.batch_size 2 \
    --rollout.micro_batch_size 1 \
    --rollout.n_samples_per_prompt 2 \
    --rollout.max_new_tokens 128 \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --train.colocate_all \
    --vllm.gpu_memory_utilization 0.5 \
    --vllm.enforce_eager \
    --vllm.enable_sleep \
    --ds.enable_sleep \
    --ds.zero_stage 2 \
    --ds.param_dtype bf16 \
    --actor.adam.lr 1e-6 \
    --ckpt.output_dir "${out}" \
    --ckpt.save_hf
  _ok "GRPO smoke finished -> ${out}"
}

main() {
  _info "mode=${MODE} model=${SMOKE_MODEL} gpus=${NUM_GPUS}"
  _info "pwd=$(pwd) venv=${VENV_DIR} root=${SMOKE_ROOT}"

  if _should_run gpu; then
    check_gpu_system
  fi

  if _should_run venv; then
    check_gpu_system
    check_gpu_venv
  fi

  if _should_run check; then
    check_gpu_system
    install_openrlhf
    check_imports
    check_ray
  fi

  if _should_run sft; then
    install_openrlhf
    check_gpu_venv
    _run_sft
  fi

  if _should_run grpo; then
    install_openrlhf
    check_gpu_venv
    check_ray
    _run_grpo
  fi

  _ok "OpenRLHF smoke test passed (mode=${MODE})"
}

main "$@"
