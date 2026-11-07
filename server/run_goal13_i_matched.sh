#!/usr/bin/env bash
set -euo pipefail

# Matched interface-layer launcher for Goal13.
# Usage:
#   bash server/run_goal13_i_matched.sh I0 <config> [steps]
#   bash server/run_goal13_i_matched.sh I2 <config> [steps]
#
# I0 and I2 both start from the D0 matched checkpoint and use the same C1
# data, steps, batch and seed. I1/I3 are evaluation-only diagnostics on I2 and
# are not launched here.

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
readonly R="$W/vla_workspace/repo/starvla_55_20261030/repo"
readonly C=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124
readonly G13="${GOAL13_ROOT:-$W/results/star105_goal13_20261107}"
readonly E=/root/starvla_cu124/bin/python
readonly TOOLS_DIR="${GOAL13_TOOLS:-/tmp/goal13_tools}"

readonly ARM="${1:?usage: run_goal13_i_matched.sh <I0|I2> <config> [steps]}"
readonly SRC_CONFIG="${2:?usage: run_goal13_i_matched.sh <I0|I2> <config> [steps]}"
readonly STEPS="${3:-500}"
case "$ARM" in
  I0) readonly MARKER_MODE=none; readonly MARKER_SOURCE=none; readonly TRAINABLE_MODULES=action_model; readonly ARM_LOWER=i0 ;;
  I2) readonly MARKER_MODE=predicted; readonly MARKER_SOURCE=point_head; readonly TRAINABLE_MODULES=action_model,point_evidence_projector; readonly ARM_LOWER=i2 ;;
  *) echo "unsupported training arm: $ARM (only I0/I2 train)" >&2; exit 2 ;;
esac

readonly RUN_ID="star13_${ARM_LOWER}_matched_${STEPS}"
readonly SNAPSHOT="$G13/configs/goal13_${ARM_LOWER}_matched_${STEPS}_snapshot.yaml"
readonly CONTRACT="$G13/contracts/${ARM}_matched_${STEPS}_contract.json"
readonly PREFLIGHT="$G13/contracts/${ARM}_matched_${STEPS}_preflight.json"
readonly LOG="$G13/logs/${RUN_ID}.log"
readonly RUN="$G13/checkpoints/$RUN_ID"
readonly CKPT="$RUN/final_model/model.safetensors"

readonly D0="$G13/checkpoints/star13_d0_matched_2500/final_model/model.safetensors"
readonly D0_SHA=6df931aeaed2f870affc775b6728a3eb99e5e59cebc21b7c66ef6906f3e7493e
readonly POINT="$W/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt"
readonly POINT_SHA=693f0dd8086a4d64cd4f1064a7f4d5c1f934f0914123df5478cdc1f86815fce7
readonly AUDIT="$G13/audit/goal13_audit.json"
readonly SPLIT="$W/vla_workspace/data/jgate_c1/c1_split_manifest_v1.json"
readonly OBSERVED="$G13/audit/observed_d0.json"
readonly PD_BATCH=4
readonly ACCUM=1
readonly TOTAL_BATCH=$((PD_BATCH * ACCUM))

for path in "$SRC_CONFIG" "$D0" "$AUDIT" "$SPLIT" "$OBSERVED"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
if [[ "$ARM" == "I2" ]]; then
  [[ -f "$POINT" ]] || { echo "missing point-head checkpoint: $POINT" >&2; exit 2; }
  [[ "$(sha256sum "$POINT" | cut -d' ' -f1)" == "$POINT_SHA" ]] || { echo "point-head checkpoint SHA mismatch" >&2; exit 2; }
fi
[[ "$(sha256sum "$D0" | cut -d' ' -f1)" == "$D0_SHA" ]] || { echo "D0 checkpoint SHA mismatch" >&2; exit 2; }
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

POINT_ARGS=()
if [[ "$ARM" == "I2" ]]; then
  POINT_ARGS=(--point-checkpoint "$POINT" --point-checkpoint-sha256 "$POINT_SHA")
fi

"$E" "$TOOLS_DIR/preflight_goal13_i_config.py" \
  --config "$SNAPSHOT" \
  --arm "$ARM" \
  --init-checkpoint "$D0" \
  "${POINT_ARGS[@]}" \
  --trainable-modules "$TRAINABLE_MODULES" \
  --output "$PREFLIGHT"

if [[ -f "$CONTRACT" ]]; then
  "$E" - "$CONTRACT" "$CONFIG_SHA" "$ARM" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("contract_status") != "PASS":
    raise SystemExit(f"contract is not PASS: {payload.get('contract_status')}")
if payload.get("config_sha256") != sys.argv[2]:
    raise SystemExit("contract config SHA mismatch")
if payload.get("arm") != sys.argv[3]:
    raise SystemExit("contract arm mismatch")
PY
else
  PYTHONPATH=. "$E" "$TOOLS_DIR/build_goal13_contract.py" \
    --audit "$AUDIT" \
    --split-manifest "$SPLIT" \
    --data-arm D0 \
    --arm "$ARM" \
    --parent-arm D0 \
    --marker-mode "$MARKER_MODE" \
    --marker-source "$MARKER_SOURCE" \
    --init-checkpoint "$D0" \
    --init-checkpoint-sha256 "$D0_SHA" \
    "${POINT_ARGS[@]}" \
    --trainable-modules "$TRAINABLE_MODULES" \
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

echo "== ${ARM} matched run_id=$RUN_ID steps=$STEPS marker=$MARKER_MODE config_sha=$CONFIG_SHA =="
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

"$E" "$TOOLS_DIR/check_goal13_checkpoint_health.py" \
  --checkpoint "$CKPT" \
  --output "$G13/contracts/${ARM}_matched_${STEPS}_checkpoint_health.json"

"$E" "$TOOLS_DIR/verify_goal13_postrun.py" \
  --contract "$CONTRACT" \
  --run-dir "$RUN" \
  --train-log "$LOG" \
  --config-snapshot "$SNAPSHOT" \
  --checkpoint-health "$G13/contracts/${ARM}_matched_${STEPS}_checkpoint_health.json" \
  --output "$G13/contracts/${ARM}_matched_${STEPS}_postrun.json" \
  --expected-steps "$STEPS" \
  --expected-arm "$ARM" \
  --expected-marker-mode "$MARKER_MODE" \
  --repo "$R" \
  --require-runtime-markers

echo "${ARM}_MATCHED_OK run=$RUN checkpoint=$CKPT postrun=$G13/contracts/${ARM}_matched_${STEPS}_postrun.json"
