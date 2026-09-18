#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly STEPS="${SWEEP_STEPS:-100}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU, steps=$STEPS"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' SWEEP_STEPS='$STEPS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"; MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CKROOT="$P/checkpoints/F0v2_lambda_sweep_${SWEEP_STEPS}"; OUTROOT="$P/results/F0v2_lambda_sweep_heldout_${SWEEP_STEPS}"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DATA" "$MANIFEST" "$CKROOT"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
test ! -e "$OUTROOT" || { echo "SWEEP_EVAL_ROOT_EXISTS=$OUTROOT" >&2; exit 4; }
mkdir -p "$P/jobs"
for lambda in 0.00 0.01 0.03 0.10 0.30; do
  tag="l${lambda/./p}"
  for variant in V3 V4; do
    ck="$CKROOT/${variant}_${tag}"; out="$OUTROOT/${variant}_${tag}"
    test -s "$ck/adapter_model.safetensors" || { echo "MISSING_CHECKPOINT=$variant/$tag" >&2; exit 5; }
    "$PY" "$R/scripts/eval_qwen_heldout.py" --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --output "$out" --model-id "${variant}_${tag}" --checkpoint "$ck" --seed 23 2>&1 | tee "$P/jobs/F0v2_lambda_eval_${variant}_${tag}.log"
    test -s "$out/report.json" || { echo "MISSING_REPORT=$variant/$tag" >&2; exit 6; }
    echo "SWEEP_EVAL_OK=$variant/$tag"
  done
done
echo LAMBDA_SWEEP_EVAL_OK
REMOTE
