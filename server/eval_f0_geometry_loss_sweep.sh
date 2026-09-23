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
TEACHER="$P/teacher_maps/F0v2_heldout_sd1_5_nulltext_phrase_64_seed23"
CALIBRATION="$P/results/F0v2_geometry_calibration_V3_val/calibration.json"
OUT="$P/results/F0v2_geometry_heldout"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DATA" "$MANIFEST" "$TEACHER" "$CALIBRATION"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
resolution=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_resolution"])' "$CALIBRATION")
quantile=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_threshold_quantile"])' "$CALIBRATION")
for mode in kl kl_rank kl_moment js; do
  for seed in 17 29 41; do
    tag="geom16_${mode}_seed${seed}"
    checkpoint="$P/checkpoints/F0v2_geometry/${tag}_s100"
    report="$OUT/${tag}_heldout"
    test -s "$checkpoint/adapter_model.safetensors" || { echo "MISSING_CHECKPOINT=$checkpoint" >&2; exit 3; }
    if test -s "$report/report.json"; then echo "REUSE_REPORT=$tag"; else
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --output "$report" --checkpoint "$checkpoint" --model-id "$tag" --seed 23 --common-resolution "$resolution" --selected-threshold-quantile "$quantile"
    fi
  done
done
"$PY" scripts/summarize_geometry_sweep.py --input-dir "$OUT" --output "$P/results/F0v2_geometry_heldout/summary.json"
echo GEOMETRY_LOSS_SWEEP_EVAL_OK
REMOTE
