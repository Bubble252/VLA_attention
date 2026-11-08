#!/usr/bin/env bash
set -euo pipefail

# Goal18: destination precision + residual correction (precision-first).
# Usage: bash server/run_goal18_precision.sh <mode>
# Modes: contract train eval-features eval-grounding eval-execution seeds all

readonly MODE="${1:-contract}"
readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
readonly R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
readonly P="$W/repo/starvla_55_20261030/repo"
readonly OUT="$W/results/K4_lite/T4b/goal18_destination_precision_20261108"
readonly PY_OFT="$W/envs/oft/bin/python"
readonly PY_STAR=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124/bin/python
readonly QWEN_CFG="$P/examples/simBenchmarks/LIBERO/train_files/star105/starvla_qwengr00t_M3_point.yaml"
readonly TRAIN_INDEX="$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/features/train/index.jsonl"
readonly VAL_INDEX="$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/features/val/index.jsonl"
readonly POINT_HEAD="$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/train/G4-MARGIN/point_head.pt"
readonly BASE_ROWS="$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/eval/grounding/G4-MARGIN/grounding_rows.jsonl"
readonly EVAL_MANIFEST="$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl"
readonly EVAL_SIDECAR="$W/results/K4_lite/T0c_audit/sidecar_v2/manifest.jsonl"
readonly EVAL_SUMMARY="$W/results/K4_lite/T0c_audit/sidecar_v2/summary.json"
readonly CAMERA_JSON="$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/grounding/summary.json"
readonly SPLIT_AUDIT="$R/tmp_goal16_stage/p1a_evidence_audit.json"
readonly MODEL="$W/models/openvla--openvla-7b"
readonly CHECKPOINT="$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt"
readonly STATS="$CHECKPOINT/dataset_statistics.json"
readonly EXEC_CONFIG="$R/configs/vla/goal16_reguide_rebinding.json"
readonly GOAL_CONFIG="$R/configs/vla/goal18_precision.json"
readonly TRAIN_OUT="$OUT/train"
readonly EVAL_FEATURES="$OUT/features/eval"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export STARVLA_DISABLE_DEEPSPEED=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$R:$R/src:$P:$P/scripts:$W/repos/libero"

mkdir -p "$OUT/logs"

check_gpu() {
  local free_mib compute
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null || true)
  if (( free_mib < 30000 )); then
    echo "GPU${CUDA_VISIBLE_DEVICES} free=${free_mib}MiB < 30000MiB; refusing to start" >&2
    exit 4
  fi
  if [[ -n "$compute" && "${GOAL18_ALLOW_SHARED_GPU:-0}" != "1" ]]; then
    echo "GPU${CUDA_VISIBLE_DEVICES} has compute processes: $compute" >&2
    exit 5
  fi
  if [[ -n "$compute" ]]; then
    echo "[gpu] sharing GPU${CUDA_VISIBLE_DEVICES}; free=${free_mib}MiB, processes=${compute}" >&2
  fi
}

contract_status() {
  "$PY_OFT" - "$OUT/run_contract.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("missing run contract")
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("status") == "PASS" else 1)
PY
}

run_contract() {
  check_gpu
  "$PY_OFT" "$R/scripts/build_goal18_contract.py" \
    --config "$GOAL_CONFIG" \
    --train-index "$TRAIN_INDEX" \
    --val-index "$VAL_INDEX" \
    --point-head "$POINT_HEAD" \
    --eval-sidecar "$EVAL_SIDECAR" \
    --eval-summary "$EVAL_SUMMARY" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$OUT/run_contract.json" \
    2>&1 | tee "$OUT/logs/contract.log"
  contract_status || { echo "Goal18 contract is INVALID" >&2; exit 3; }
}

run_train() {
  check_gpu
  contract_status || { echo "run contract is not PASS; refusing training" >&2; exit 3; }
  if [[ -f "$TRAIN_OUT/calibration.json" ]]; then
    echo "[train] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/train_goal18_precision.py" \
    --config "$GOAL_CONFIG" \
    --train-index "$TRAIN_INDEX" \
    --val-index "$VAL_INDEX" \
    --point-head "$POINT_HEAD" \
    --output "$TRAIN_OUT" \
    2>&1 | tee "$OUT/logs/train.log"
}

run_eval_features() {
  check_gpu
  if [[ -s "$EVAL_FEATURES/index.jsonl" && -f "$EVAL_FEATURES/summary.json" ]]; then
    echo "[eval-features] exists; skip"
    return 0
  fi
  if [[ -d "$EVAL_FEATURES" ]]; then
    echo "[eval-features] removing incomplete output: $EVAL_FEATURES"
    rm -rf "$EVAL_FEATURES"
  fi
  "$PY_STAR" "$R/scripts/extract_goal18_eval_features.py" \
    --config "$QWEN_CFG" \
    --sidecar "$EVAL_SIDECAR" \
    --output "$EVAL_FEATURES" \
    2>&1 | tee "$OUT/logs/eval_features.log"
}

run_grounding_arm() {
  local arm="$1"
  local output="$OUT/eval/grounding/$arm"
  if [[ -f "$output/summary.json" ]]; then
    echo "[grounding:$arm] exists; skip"
    return 0
  fi
  local extra=()
  if [[ "$arm" == "C2-RESIDUAL" ]]; then
    [[ -f "$TRAIN_OUT/residual_head.pt" ]] || { echo "[grounding:$arm] missing residual head" >&2; return 1; }
    [[ -f "$EVAL_FEATURES/index.jsonl" ]] || { echo "[grounding:$arm] missing eval features" >&2; return 1; }
    extra+=(--residual-head "$TRAIN_OUT/residual_head.pt" --eval-feature-index "$EVAL_FEATURES/index.jsonl")
  fi
  "$PY_OFT" "$R/scripts/evaluate_goal18_grounding.py" \
    --config "$GOAL_CONFIG" \
    --base-rows "$BASE_ROWS" \
    --calibration "$TRAIN_OUT/calibration.json" \
    --sidecar "$EVAL_SIDECAR" \
    --arm "$arm" \
    --output "$output" \
    "${extra[@]}" \
    2>&1 | tee "$OUT/logs/grounding_${arm}.log" || true
}

run_eval_grounding() {
  check_gpu
  [[ -f "$TRAIN_OUT/calibration.json" ]] || { echo "missing calibration; run train first" >&2; exit 3; }
  for arm in C0 C1-GLOBAL C1-PROMPT C1-RELATION; do
    run_grounding_arm "$arm"
  done
  if [[ -f "$TRAIN_OUT/residual_head.pt" && -f "$EVAL_FEATURES/index.jsonl" ]]; then
    run_grounding_arm C2-RESIDUAL
  else
    echo "[grounding:C2-RESIDUAL] skipped (missing head/features)"
  fi
}

run_execution_arm() {
  local arm="$1"
  local grounding="$OUT/eval/grounding/$arm/predicted_points.jsonl"
  local output="$OUT/eval/execution/$arm"
  [[ -f "$grounding" ]] || { echo "[execution:$arm] missing grounding sidecar" >&2; return 1; }
  if [[ -f "$output/summary.json" ]]; then
    echo "[execution:$arm] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/audit_goal18_execution.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --statistics "$STATS" \
    --manifest "$EVAL_MANIFEST" \
    --goal-config "$GOAL_CONFIG" \
    --execution-config "$EXEC_CONFIG" \
    --grounding-sidecar "$grounding" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$output" \
    --arms B0,E1-PRED \
    --settle-steps 32 \
    2>&1 | tee "$OUT/logs/execution_${arm}.log" || true
}

run_eval_execution() {
  check_gpu
  if [[ ! -f "$OUT/eval/grounding/C1-PROMPT/summary.json" ]]; then
    echo "missing C1-PROMPT grounding; run eval-grounding first" >&2
    exit 3
  fi
  run_execution_arm C1-PROMPT
  if [[ -f "$OUT/eval/grounding/C2-RESIDUAL/summary.json" ]]; then
    run_execution_arm C2-RESIDUAL
  fi
}

run_seeds() {
  check_gpu
  for seed in 7 17 27; do
    local output="$OUT/eval/execution_seed_${seed}/C1-PROMPT"
    if [[ -f "$output/summary.json" ]]; then
      echo "[seed:$seed] exists; skip"
      continue
    fi
    "$PY_OFT" "$R/scripts/audit_goal18_execution.py" \
      --model "$MODEL" \
      --checkpoint "$CHECKPOINT" \
      --statistics "$STATS" \
      --manifest "$EVAL_MANIFEST" \
      --goal-config "$GOAL_CONFIG" \
      --execution-config "$EXEC_CONFIG" \
      --grounding-sidecar "$OUT/eval/grounding/C1-PROMPT/predicted_points.jsonl" \
      --split-audit "$SPLIT_AUDIT" \
      --output "$output" \
      --arms B0,E1-PRED \
      --settle-steps 32 \
      --seed "$seed" \
      2>&1 | tee "$OUT/logs/execution_seed_${seed}.log" || true
  done
}

case "$MODE" in
  contract) run_contract ;;
  train) run_train ;;
  eval-features) run_eval_features ;;
  eval-grounding) run_eval_grounding ;;
  eval-execution) run_eval_execution ;;
  seeds) run_seeds ;;
  all)
    run_contract
    run_train
    run_eval_features
    run_eval_grounding
    run_eval_execution
    ;;
  *)
    echo "usage: $0 <contract|train|eval-features|eval-grounding|eval-execution|seeds|all>" >&2
    exit 2
    ;;
esac
