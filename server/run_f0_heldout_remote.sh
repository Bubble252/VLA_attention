#!/usr/bin/env bash
set -euo pipefail

# Run locally after validation scripts have been uploaded. It executes the
# held-out evaluator inside the 101 P1 environment, never on the laptop.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"
PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"
ROOT="$P/data/flickr30k_entities"
MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CHECKPOINT_ROOT="$P/checkpoints/F0v2_idea_validation"
OUT="$P/results/F0v2_heldout_64"
export PYTHONPATH="$R/src:$R"
cd "$R"

for path in "$PY" "$MODEL" "$ROOT" "$MANIFEST" "$CHECKPOINT_ROOT/V1" "$CHECKPOINT_ROOT/V2" "$CHECKPOINT_ROOT/V3" "$CHECKPOINT_ROOT/V4"; do
  test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }
done
for id in V0 V1 V2 V3 V4; do
  test ! -e "$OUT/$id/report.json" || { echo "HELDOUT_ARTIFACT_EXISTS=$OUT/$id/report.json" >&2; exit 4; }
done

MODEL="$MODEL" ROOT="$ROOT" MANIFEST="$MANIFEST" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  OUT="$OUT" PYTHON="$PY" EVAL="$R/scripts/eval_qwen_heldout.py" \
  bash "$R/scripts/run_f0_heldout_all.sh" 2>&1 | tee "$P/jobs/F0v2_heldout_V0_V4.log"

for id in V0 V1 V2 V3 V4; do test -s "$OUT/$id/report.json"; done
echo HELDOUT_MODELS_OK
REMOTE
