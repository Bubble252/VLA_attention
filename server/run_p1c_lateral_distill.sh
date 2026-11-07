#!/usr/bin/env bash
set -euo pipefail

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
readonly R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
readonly PY="$W/envs/oft/bin/python"
readonly OUT_ROOT="$W/results/K4_lite/T4b/p1_lateral_distill_20261107"
readonly CONFIG="$R/configs/vla/p1_lateral_distill_v3.json"

run_name="${1:?usage: run_p1c_lateral_distill.sh RUN_NAME ARM STEPS [SEED]}"
arm="${2:?usage: run_p1c_lateral_distill.sh RUN_NAME ARM STEPS [SEED]}"
steps="${3:?usage: run_p1c_lateral_distill.sh RUN_NAME ARM STEPS [SEED]}"
seed="${4:-17}"

case "$arm" in
  R1-CAP|R2-ROUTE|R3-GEO) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac

test -x "$PY"
test -f "$CONFIG"
test -d "$OUT_ROOT"

export PYTHONPATH="$R:$R/src:$W/repos/openvla-oft:$W/repos/libero"
export LIBERO_CONFIG_PATH="$W/libero_config"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

cd "$R"
"$PY" scripts/train_p1_lateral_distill.py \
  --config "$CONFIG" \
  --train-cache "$W/results/K4_lite/T4_residual/cache_train_v2" \
  --train-teacher "$OUT_ROOT/p1b_teacher_full" \
  --validation-cache "$W/results/K4_lite/T4_residual/cache_validation_v2" \
  --validation-teacher "$OUT_ROOT/p1b_teacher_validation" \
  --arm "$arm" \
  --output "$OUT_ROOT/$run_name/$arm" \
  --max-steps "$steps" \
  --seed "$seed" \
  --log-every 100 \
  2>&1 | tee "$OUT_ROOT/${run_name}_${arm}.log"
