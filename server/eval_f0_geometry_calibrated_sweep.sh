#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"
MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CAL="$P/results/F0v2_geometry_calibration_V3_val/calibration.json"
OUT="$P/results/F0v2_geometry16_heldout"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DATA" "$MANIFEST" "$CAL"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
resolution=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_resolution"])' "$CAL")
quantile=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_threshold_quantile"])' "$CAL")
for seed in 17 29 41; do
  if test "$seed" = 17; then ck="$P/checkpoints/F0v2_idea_validation_100/V1"; else ck="$P/checkpoints/F0v2_geometry/matched_v1_seed${seed}_s100"; fi
  report="$OUT/v1_seed${seed}_heldout"
  if ! test -s "$report/report.json"; then
    test -s "$ck/adapter_model.safetensors" || { echo "MISSING_CHECKPOINT=$ck" >&2; exit 3; }
    "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --output "$report" --checkpoint "$ck" --model-id "v1_seed${seed}" --seed 23 --common-resolution "$resolution" --selected-threshold-quantile "$quantile" >/dev/null
  fi
done
for mode in kl kl_rank kl_moment js; do
  for seed in 17 29 41; do
    tag="geom16_${mode}_seed${seed}_s100"; ck="$P/checkpoints/F0v2_geometry/$tag"; report="$OUT/${tag}_heldout"
    test -s "$ck/adapter_model.safetensors" || { echo "MISSING_CHECKPOINT=$ck" >&2; exit 3; }
    if ! test -s "$report/report.json"; then
      "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --output "$report" --checkpoint "$ck" --model-id "$tag" --seed 23 --common-resolution "$resolution" --selected-threshold-quantile "$quantile" >/dev/null
    fi
  done
done
for mode in kl kl_rank kl_moment js; do
  "$PY" scripts/paired_geometry_bootstrap.py --baseline-template "$OUT/v1_seed{seed}_heldout/report.json" --candidate-template "$OUT/geom16_${mode}_seed{seed}_s100_heldout/report.json" --losses "$mode" --seeds 17,29,41 --output "$OUT/paired_${mode}_vs_v1.json"
done
echo CALIBRATED_GEOMETRY_SWEEP_EVAL_OK
REMOTE
