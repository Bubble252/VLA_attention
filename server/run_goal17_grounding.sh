#!/usr/bin/env bash
set -euo pipefail

# Goal17: compositional grounding repair + simplified R-PRED.
# Usage: bash server/run_goal17_grounding.sh <mode>
# Modes: preflight manifest contract features merge contract-final
#        train-no-margin train-margin train-c eval-grounding eval-e1 eval-e2
#        calib-settle eval-e1-settle all

readonly MODE="${1:-preflight}"
readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
readonly R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
readonly P="$W/repo/starvla_55_20261030/repo"
readonly OUT="$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108"
readonly PY_OFT="$W/envs/oft/bin/python"
readonly PY_STAR=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124/bin/python
readonly QWEN_CFG="$P/examples/simBenchmarks/LIBERO/train_files/star105/starvla_qwengr00t_M3_point.yaml"
readonly TRAIN_MANIFEST="/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_manifests/train_sidecar_v1e/point_manifest.jsonl"
readonly VAL_MANIFEST="/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_manifests/validation_sidecar_v1/point_manifest.jsonl"
readonly BASE_TRAIN_INDEX="/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_head/features/train/index.jsonl"
readonly BASE_VAL_INDEX="/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_head/features/val/index.jsonl"
readonly GOAL_CONFIG="$R/configs/vla/goal17_grounding.json"
readonly EXEC_CONFIG="$R/configs/vla/goal16_reguide_rebinding.json"
readonly EVAL_MANIFEST="$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl"
readonly EVAL_SIDECAR="$W/results/K4_lite/T0c_audit/sidecar_v2/manifest.jsonl"
readonly EVAL_SUMMARY="$W/results/K4_lite/T0c_audit/sidecar_v2/summary.json"
readonly CAMERA_JSON="$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/grounding/summary.json"
readonly SPLIT_AUDIT="$R/tmp_goal16_stage/p1a_evidence_audit.json"
readonly CHECKPOINT="$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt"
readonly STATS="$CHECKPOINT/dataset_statistics.json"
readonly MODEL="$W/models/openvla--openvla-7b"
readonly BASELINE_HEAD="/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
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
}

run_manifest() {
  if [[ -f "$OUT/manifests/4prompt/summary.json" ]]; then
    echo "[manifest] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/build_goal17_counterfactual_manifest.py" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --output-dir "$OUT/manifests/4prompt" \
    2>&1 | tee "$OUT/logs/manifest.log"
}

run_contract_preliminary() {
  if [[ -f "$OUT/run_contract_preliminary.json" ]]; then
    echo "[contract] preliminary exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/build_goal17_contract.py" \
    --config "$GOAL_CONFIG" \
    --train-manifest "$OUT/manifests/4prompt/train_4prompt.jsonl" \
    --val-manifest "$OUT/manifests/4prompt/val_4prompt.jsonl" \
    --eval-sidecar "$EVAL_SIDECAR" \
    --eval-summary "$EVAL_SUMMARY" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$OUT/run_contract_preliminary.json" \
    2>&1 | tee "$OUT/logs/contract_preliminary.log"
}

extract_one() {
  local split="$1" manifest="$2" output="$3"
  if [[ -f "$output/summary.json" ]]; then
    echo "[features:$split] exists; skip"
    return 0
  fi
  "$PY_STAR" "$P/scripts/extract_c1_point_features.py" \
    --config "$QWEN_CFG" \
    --manifest "$manifest" \
    --output "$output" \
    2>&1 | tee "$OUT/logs/features_${split}.log"
}

run_features() {
  extract_one train_white_right "$OUT/manifests/4prompt/train_white_right.jsonl" \
    "$OUT/features/train_white_right"
  extract_one val_white_right "$OUT/manifests/4prompt/val_white_right.jsonl" \
    "$OUT/features/val_white_right"
}

run_merge() {
  if [[ -f "$OUT/features/train/index.jsonl" && -f "$OUT/features/val/index.jsonl" ]]; then
    echo "[merge] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/merge_goal17_feature_index.py" \
    --base-index "$BASE_TRAIN_INDEX" \
    --white-right-index "$OUT/features/train_white_right/index.jsonl" \
    --output "$OUT/features/train/index.jsonl" \
    --expected-images 764 \
    2>&1 | tee "$OUT/logs/merge_train.log"
  "$PY_OFT" "$R/scripts/merge_goal17_feature_index.py" \
    --base-index "$BASE_VAL_INDEX" \
    --white-right-index "$OUT/features/val_white_right/index.jsonl" \
    --output "$OUT/features/val/index.jsonl" \
    --expected-images 118 \
    2>&1 | tee "$OUT/logs/merge_val.log"
}

run_contract_final() {
  "$PY_OFT" "$R/scripts/build_goal17_contract.py" \
    --config "$GOAL_CONFIG" \
    --train-manifest "$OUT/manifests/4prompt/train_4prompt.jsonl" \
    --val-manifest "$OUT/manifests/4prompt/val_4prompt.jsonl" \
    --train-index "$OUT/features/train/index.jsonl" \
    --val-index "$OUT/features/val/index.jsonl" \
    --eval-sidecar "$EVAL_SIDECAR" \
    --eval-summary "$EVAL_SUMMARY" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$OUT/run_contract.json" \
    2>&1 | tee "$OUT/logs/contract_final.log"
}

train_arm() {
  local arm="$1" prompts="$2" lambda_dest="$3" lambda_obj="$4" val_prompts="$5"
  local output="$OUT/train/$arm"
  if [[ -f "$output/summary.json" ]]; then
    echo "[train:$arm] exists; skip"
    return 0
  fi
  "$PY_STAR" "$R/scripts/train_goal17_grounding.py" \
    --train-index "$OUT/features/train/index.jsonl" \
    --val-index "$OUT/features/val/index.jsonl" \
    --output "$output" \
    --prompts "$prompts" \
    --val-prompts "$val_prompts" \
    --steps 5000 \
    --group-batch-size 4 \
    --hidden-dim 3584 \
    --lambda-dest "$lambda_dest" \
    --lambda-obj "$lambda_obj" \
    2>&1 | tee "$OUT/logs/train_${arm}.log"
}

eval_grounding_one() {
  local arm="$1" head="$2"
  local output="$OUT/eval/grounding/$arm"
  if [[ -f "$output/summary.json" ]]; then
    echo "[eval-grounding:$arm] exists; skip"
    return 0
  fi
  "$PY_STAR" "$R/scripts/evaluate_goal17_grounding.py" \
    --config "$GOAL_CONFIG" \
    --sidecar "$EVAL_SIDECAR" \
    --point-head "$head" \
    --camera-json "$CAMERA_JSON" \
    --output "$output" \
    2>&1 | tee "$OUT/logs/eval_grounding_${arm}.log" || true
}

run_eval_grounding() {
  eval_grounding_one G3-BASE "$BASELINE_HEAD"
  eval_grounding_one G4-NO-MARGIN "$OUT/train/G4-NO-MARGIN/point_head.pt"
  eval_grounding_one G4-MARGIN "$OUT/train/G4-MARGIN/point_head.pt"
  eval_grounding_one G3-MARGIN "$OUT/train/G3-MARGIN/point_head.pt"
}

run_eval_e1() {
  local output="$OUT/eval/e1"
  if [[ -f "$output/summary.json" ]]; then
    echo "[eval-e1] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/audit_goal17_rpred.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --statistics "$STATS" \
    --manifest "$EVAL_MANIFEST" \
    --goal-config "$GOAL_CONFIG" \
    --execution-config "$EXEC_CONFIG" \
    --grounding-sidecar "$OUT/eval/grounding/G4-MARGIN/predicted_points.jsonl" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$output" \
    --arms B0,E1-ORACLE,E1-PRED \
    2>&1 | tee "$OUT/logs/eval_e1.log" || true
}

run_calib_settle() {
  local settle="${SETTLE_STEPS:?set SETTLE_STEPS, e.g. SETTLE_STEPS=16}"
  local max_inits="${MAX_INITS:-3}"
  local arm="${CALIB_ARM:-E1-PRED}"
  local output="$OUT/eval/calib/${arm}_settle_${settle}"
  if [[ -f "$output/summary.json" ]]; then
    echo "[calib-settle:$settle] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/audit_goal17_rpred.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --statistics "$STATS" \
    --manifest "$EVAL_MANIFEST" \
    --goal-config "$GOAL_CONFIG" \
    --execution-config "$EXEC_CONFIG" \
    --grounding-sidecar "$OUT/eval/grounding/G4-MARGIN/predicted_points.jsonl" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$output" \
    --arms "$arm" \
    --allow-partial \
    --max-inits "$max_inits" \
    --settle-steps "$settle" \
    2>&1 | tee "$OUT/logs/calib_${arm}_settle_${settle}.log" || true
}

run_eval_e1_settle() {
  local settle="${SETTLE_STEPS:?set SETTLE_STEPS, e.g. SETTLE_STEPS=16}"
  local output="$OUT/eval/e1_settle_${settle}"
  if [[ -f "$output/summary.json" ]]; then
    echo "[eval-e1-settle:$settle] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/audit_goal17_rpred.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --statistics "$STATS" \
    --manifest "$EVAL_MANIFEST" \
    --goal-config "$GOAL_CONFIG" \
    --execution-config "$EXEC_CONFIG" \
    --grounding-sidecar "$OUT/eval/grounding/G4-MARGIN/predicted_points.jsonl" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$output" \
    --arms B0,E1-ORACLE,E1-PRED \
    --settle-steps "$settle" \
    2>&1 | tee "$OUT/logs/eval_e1_settle_${settle}.log" || true
}

run_eval_e2() {
  local grounding="$OUT/eval/grounding/G4-MARGIN/summary.json"
  if [[ ! -f "$grounding" ]]; then
    echo "[eval-e2] grounding summary missing; skip" >&2
    return 1
  fi
  "$PY_OFT" - "$grounding" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("passed") is not True:
    raise SystemExit("grounding gate did not pass; refusing E2")
PY
  local output="$OUT/eval/e2"
  if [[ -f "$output/summary.json" ]]; then
    echo "[eval-e2] exists; skip"
    return 0
  fi
  "$PY_OFT" "$R/scripts/audit_goal17_rpred.py" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --statistics "$STATS" \
    --manifest "$EVAL_MANIFEST" \
    --goal-config "$GOAL_CONFIG" \
    --execution-config "$EXEC_CONFIG" \
    --grounding-sidecar "$OUT/eval/grounding/G4-MARGIN/predicted_points.jsonl" \
    --split-audit "$SPLIT_AUDIT" \
    --output "$output" \
    --arms E2-GEO,E2-PRED \
    2>&1 | tee "$OUT/logs/eval_e2.log" || true
}

case "$MODE" in
  preflight)
    check_gpu
    for path in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$BASE_TRAIN_INDEX" "$BASE_VAL_INDEX" \
                "$EVAL_SIDECAR" "$EVAL_SUMMARY" "$CAMERA_JSON" "$SPLIT_AUDIT" \
                "$CHECKPOINT" "$STATS" "$BASELINE_HEAD"; do
      [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 3; }
    done
    echo "[preflight] PASS"
    ;;
  manifest) run_manifest ;;
  contract) run_contract_preliminary ;;
  features) check_gpu; run_features ;;
  merge) run_merge ;;
  contract-final) run_contract_final ;;
  train-no-margin)
    check_gpu
    train_arm G4-NO-MARGIN "red_left,red_right,white_left,white_right" 0.0 0.0 "red_left,red_right,white_left,white_right"
    ;;
  train-margin)
    check_gpu
    train_arm G4-MARGIN "red_left,red_right,white_left,white_right" 0.5 0.5 "red_left,red_right,white_left,white_right"
    ;;
  train-c)
    check_gpu
    train_arm G3-MARGIN "red_left,red_right,white_left" 0.5 0.5 "red_left,red_right,white_left"
    ;;
  eval-grounding) check_gpu; run_eval_grounding ;;
  eval-e1) check_gpu; run_eval_e1 ;;
  eval-e2) check_gpu; run_eval_e2 ;;
  calib-settle) check_gpu; run_calib_settle ;;
  eval-e1-settle) check_gpu; run_eval_e1_settle ;;
  all)
    check_gpu
    run_manifest
    run_contract_preliminary
    run_features
    run_merge
    run_contract_final
    train_arm G4-NO-MARGIN "red_left,red_right,white_left,white_right" 0.0 0.0 "red_left,red_right,white_left,white_right"
    train_arm G4-MARGIN "red_left,red_right,white_left,white_right" 0.5 0.5 "red_left,red_right,white_left,white_right"
    train_arm G3-MARGIN "red_left,red_right,white_left" 0.5 0.5 "red_left,red_right,white_left"
    run_eval_grounding
    run_eval_e1
    run_eval_e2 || true
    ;;
  *)
    echo "usage: $0 <preflight|manifest|contract|features|merge|contract-final|train-no-margin|train-margin|train-c|eval-grounding|eval-e1|eval-e2|calib-settle|eval-e1-settle|all>" >&2
    exit 2
    ;;
esac
