#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# YAML is the single source of truth; env vars below override individual keys.
CONFIG="${CONFIG:-configs/openrlhf/hint_dpo.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

# Unique wandb run name = YAML base name + timestamp, unless WANDB_RUN_NAME is set.
BASE_RUN_NAME="$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print((c.get('logger',{}).get('wandb',{}) or {}).get('run_name','hint_controller_dpo'))")"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${BASE_RUN_NAME}_$(date +%m%dT%H%M)}"

# Optional env overrides (only applied when set).
OVERRIDES=()
add_override() { if [[ -n "${2:-}" ]]; then OVERRIDES+=("$1=$2"); fi; }
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  add_override logger.wandb.run_name "${WANDB_RUN_NAME}"
  add_override logger.wandb.project "${WANDB_PROJECT:-}"
  add_override logger.wandb.key "${WANDB_API_KEY}"
fi
add_override model.model_name_or_path "${PRETRAIN:-}"
add_override model.beta "${BETA:-}"
add_override ckpt.output_dir "${OUTPUT_DIR:-}"
add_override data.dataset "${DATASET:-}"
add_override train.max_epochs "${MAX_EPOCHS:-}"
add_override adam.lr "${LR:-}"
add_override data.max_len "${MAX_LEN:-}"
add_override ds.zero_stage "${ZERO_STAGE:-}"
add_override model.gradient_checkpointing_enable "${GRADIENT_CHECKPOINTING_ENABLE:-}"

# DeepSpeed needs grad_acc = global_batch / (micro_batch * num_gpus) to be a
# positive integer, so the global batch must be a multiple of micro * num_gpus.
# Resolve effective values (env override > YAML) and round the global batch up.
read -r CFG_BS CFG_MBS < <(python3 -c "import yaml; t=(yaml.safe_load(open('${CONFIG}')) or {}).get('train',{}) or {}; print(t.get('batch_size',1), t.get('micro_batch_size',1))")
EFF_MBS="${MBS:-$CFG_MBS}"
EFF_BS="${BS:-$CFG_BS}"
UNIT=$(( EFF_MBS * NUM_GPUS ))
if (( EFF_BS < UNIT || EFF_BS % UNIT != 0 )); then
  NEW_BS=$(( ((EFF_BS + UNIT - 1) / UNIT) * UNIT ))
  (( NEW_BS < UNIT )) && NEW_BS=$UNIT
  echo "WARN: global batch_size ${EFF_BS} is not a positive multiple of micro_batch_size(${EFF_MBS}) * num_gpus(${NUM_GPUS}); using ${NEW_BS}." >&2
  EFF_BS=$NEW_BS
fi
add_override train.batch_size "$EFF_BS"
add_override train.micro_batch_size "$EFF_MBS"
add_override model.gradient_checkpointing_enable true

if ((${#OVERRIDES[@]})); then
  mapfile -t CFG_ARGS < <(python3 scripts/openrlhf/_yaml_to_args.py "${CONFIG}" "${OVERRIDES[@]}")
else
  mapfile -t CFG_ARGS < <(python3 scripts/openrlhf/_yaml_to_args.py "${CONFIG}")
fi

echo "config: ${CONFIG}"
echo "num GPUs: ${NUM_GPUS}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "wandb run name: ${WANDB_RUN_NAME}"
else
  echo "wandb: disabled (WANDB_API_KEY unset)"
fi

deepspeed --num_gpus "${NUM_GPUS}" --module openrlhf.cli.train_dpo "${CFG_ARGS[@]}"
