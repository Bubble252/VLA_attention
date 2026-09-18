#!/usr/bin/env bash
set -euo pipefail

# Run after V3/V4 checkpoint exports exist.  All five models use the same
# image-disjoint 64-row manifest and evaluator; no weights are modified.
MODEL=${MODEL:?path to Qwen checkpoint}
ROOT=${ROOT:?Flickr30k Entities data root}
MANIFEST=${MANIFEST:?held-out jsonl}
OUT=${OUT:?held-out result root}
EVAL=${EVAL:-scripts/eval_qwen_heldout.py}
PYTHON=${PYTHON:-python3}

run_one() {
  local id=$1 ckpt=${2:-}
  local args=(--model "$MODEL" --dataset-root "$ROOT" --manifest "$MANIFEST" \
    --output "$OUT/$id" --model-id "$id" --seed 23)
  if [[ -n "$ckpt" ]]; then args+=(--checkpoint "$ckpt"); fi
  echo "[$(date -u +%FT%TZ)] evaluating $id"
  "$PYTHON" "$EVAL" "${args[@]}"
}

run_one V0
run_one V1 "$OUT/../checkpoints/F0v2_idea_validation/V1"
run_one V2 "$OUT/../checkpoints/F0v2_idea_validation/V2"
run_one V3 "$OUT/../checkpoints/F0v2_idea_validation/V3"
run_one V4 "$OUT/../checkpoints/F0v2_idea_validation/V4"
