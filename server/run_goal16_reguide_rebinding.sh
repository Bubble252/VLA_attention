#!/usr/bin/env bash
set -euo pipefail

readonly MODE="${1:-contract}"
readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
readonly R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
readonly C="$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt"
readonly OUT="$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108"
readonly GROUNDING="$OUT/grounding"
readonly PY="$W/envs/oft/bin/python"
readonly MANIFEST="$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export TF_CPP_MIN_LOG_LEVEL=3

case "$MODE" in
  contract)
    PYTHONPATH="$R/src:$R" "$PY" "$R/scripts/build_goal16_contract.py" \
      --config "$R/configs/vla/goal16_reguide_rebinding.json" \
      --manifest "$MANIFEST" \
      --checkpoint "$C" \
      --statistics "$C/dataset_statistics.json" \
      --point-head /vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt \
      --grounding-sidecar "$GROUNDING/predicted_points.jsonl" \
      --split-audit "$R/tmp_goal16_stage/p1a_evidence_audit.json" \
      --output "$OUT/run_contract.json"
    ;;
  pilot)
    PYTHONPATH="$R/src:$R:$W/repos/libero" "$PY" "$R/scripts/audit_goal16_reguide_rebinding.py" \
      --model "$W/models/openvla--openvla-7b" \
      --checkpoint "$C" \
      --statistics "$C/dataset_statistics.json" \
      --manifest "$MANIFEST" \
      --gate-config "$R/configs/vla/goal16_reguide_rebinding.json" \
      --grounding-sidecar "$GROUNDING/predicted_points.jsonl" \
      --split-audit "$R/tmp_goal16_stage/p1a_evidence_audit.json" \
      --output "$OUT/pilot_15init"
    ;;
  *)
    echo "usage: $0 [contract|pilot]" >&2
    exit 2
    ;;
esac
