#!/usr/bin/env bash
set -euo pipefail

# D0 matched reproduction: A3 12.5K initialisation -> C1 three prompts, 2.5K
# steps, effective batch 4.  Usage:
#   bash server/run_goal13_d0_matched.sh [config] [steps]

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
readonly R="$W/vla_workspace/repo/starvla_55_20261030/repo"
readonly C=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124
readonly G13="${GOAL13_ROOT:-$W/results/star105_goal13_20261107}"
readonly E=/root/starvla_cu124/bin/python
readonly TOOLS_DIR="${GOAL13_TOOLS:-/tmp/goal13_tools}"
readonly SRC_CONFIG="${1:-$TOOLS_DIR/goal13_d0_matched.yaml}"
readonly STEPS="${2:-2500}"
readonly RUN_ID="star13_d0_matched_${STEPS}"
readonly SNAPSHOT="$G13/configs/goal13_d0_matched_snapshot.yaml"
readonly CONTRACT="$G13/contracts/D0_matched_contract.json"
readonly LOG="$G13/logs/${RUN_ID}.log"
readonly RUN="$G13/checkpoints/$RUN_ID"
readonly CKPT="$RUN/final_model/model.safetensors"
readonly A3="$W/results/star105_goal10p5_20261105/checkpoints/star105_A3_qwengr00t_12500/final_model/model.safetensors"
readonly A3_SHA=66597a306f69839be85f186d0f7a1c1637b3396d66896d382443cf9b8a75613c
readonly AUDIT="$G13/audit/goal13_audit.json"
readonly SPLIT="$W/vla_workspace/data/jgate_c1/c1_split_manifest_v1.json"
readonly OBSERVED="$G13/audit/observed_d0.json"
readonly PD_BATCH=4
readonly ACCUM=1
readonly TOTAL_BATCH=$((PD_BATCH * ACCUM))

for path in "$SRC_CONFIG" "$A3" "$AUDIT" "$SPLIT" "$OBSERVED"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$A3" | cut -d' ' -f1)" == "$A3_SHA" ]] || { echo "A3 checkpoint SHA mismatch" >&2; exit 2; }
if [[ -f "$CKPT" ]]; then
  echo "checkpoint already exists; refusing to overwrite: $CKPT" >&2
  exit 3
fi
if [[ -e "$LOG" ]]; then
  echo "log already exists; refusing to overwrite: $LOG" >&2
  exit 3
fi

mkdir -p "$G13/configs" "$G13/contracts" "$G13/logs" "$G13/checkpoints"
if [[ -f "$SNAPSHOT" ]]; then
  cmp -s "$SRC_CONFIG" "$SNAPSHOT" || { echo "existing config snapshot differs: $SNAPSHOT" >&2; exit 3; }
else
  cp -p "$SRC_CONFIG" "$SNAPSHOT"
  chmod 444 "$SNAPSHOT"
fi
readonly CONFIG_SHA="$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)"

if [[ -f "$CONTRACT" ]]; then
  "$E" - "$CONTRACT" "$CONFIG_SHA" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("contract_status") != "PASS":
    raise SystemExit(f"contract is not PASS: {payload.get('contract_status')}")
if payload.get("config_sha256") != sys.argv[2]:
    raise SystemExit("contract config SHA mismatch")
PY
else
  PYTHONPATH=. "$E" "$TOOLS_DIR/build_goal13_contract.py" \
    --audit "$AUDIT" \
    --split-manifest "$SPLIT" \
    --data-arm D0 \
    --output "$CONTRACT" \
    --code-commit "$(git -C "$R" rev-parse HEAD)" \
    --config "$SNAPSHOT" \
    --observed-dataset-contract "$OBSERVED" \
    --per-device-batch "$PD_BATCH" \
    --grad-accum "$ACCUM" \
    --num-processes 1 \
    --seed 17
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 2>/dev/null || true)
if (( free_mib < 30000 )); then
  echo "GPU1 free=${free_mib}MiB < 30000MiB; refusing to start" >&2
  exit 4
fi
if [[ -n "$compute" ]]; then
  echo "GPU1 has compute processes: $compute" >&2
  exit 5
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$C"
set -u
cd "$R"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$R:/root/code/LIBERO"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export STARVLA_DISABLE_DEEPSPEED=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/root/code/LIBERO/libero
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="$ACCUM"
export STAR105_EXPECTED_GRAD_ACCUM="$ACCUM"
export STAR105_EXPECTED_TOTAL_BATCH="$TOTAL_BATCH"
export STAR105_EXPECTED_C1_LENGTH=11630
export STAR105_EXPECTED_C1_SHARE=1.0
export STAR105_EXPECTED_CONFIG_SHA="$CONFIG_SHA"
export STAR105_CONFIG_PATH="$SNAPSHOT"

echo "== D0 matched run_id=$RUN_ID steps=$STEPS config_sha=$CONFIG_SHA =="
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps "$ACCUM" \
  starVLA/training/train_starvla.py \
  --config_yaml "$SNAPSHOT" \
  --run_root_dir "$G13/checkpoints" \
  --run_id "$RUN_ID" \
  --trainer.max_train_steps "$STEPS" \
  --trainer.save_interval "$STEPS" 2>&1 | tee "$LOG"

[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 6; }
grep -q "STAR105_RUNTIME_BATCH_CONTRACT accum=${ACCUM} total_batch=${TOTAL_BATCH}" "$LOG" || { echo "runtime batch marker missing" >&2; exit 7; }
grep -q "STAR105_RUNTIME_DATASET_CONTRACT libero_90_gate_lerobot:length=11630:weight=1.000000" "$LOG" || { echo "runtime dataset marker missing" >&2; exit 8; }
grep -q "STAR105_RUNTIME_CONFIG_SHA ${CONFIG_SHA}" "$LOG" || { echo "runtime config marker missing" >&2; exit 9; }
tr '\n' ' ' < "$LOG" | grep -q "$A3" || { echo "A3 pretrained checkpoint path missing from log" >&2; exit 10; }

"$E" "$TOOLS_DIR/check_goal13_checkpoint_health.py" \
  --checkpoint "$CKPT" \
  --output "$G13/contracts/D0_matched_checkpoint_health.json"

"$E" "$TOOLS_DIR/verify_goal13_postrun.py" \
  --contract "$CONTRACT" \
  --run-dir "$RUN" \
  --train-log "$LOG" \
  --config-snapshot "$SNAPSHOT" \
  --checkpoint-health "$G13/contracts/D0_matched_checkpoint_health.json" \
  --output "$G13/contracts/D0_matched_postrun.json" \
  --expected-steps "$STEPS" \
  --repo "$R" \
  --require-runtime-markers

echo "D0_MATCHED_OK run=$RUN checkpoint=$CKPT postrun=$G13/contracts/D0_matched_postrun.json"
