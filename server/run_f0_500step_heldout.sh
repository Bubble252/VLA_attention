#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; ROOT="$P/data/flickr30k_entities"; MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CKPT="$P/checkpoints/F0v2_idea_validation_500"; OUT="$P/results/F0v2_heldout_64_500"; export PYTHONPATH="$R/src:$R"; cd "$R"
for id in V1 V2 V3 V4; do test -s "$CKPT/$id/adapter_model.safetensors" || { echo "MISSING=$CKPT/$id" >&2; exit 2; }; done
for id in V1 V2 V3 V4; do test ! -e "$OUT/$id/report.json" || { echo "EXISTS=$OUT/$id/report.json" >&2; exit 4; }; done
run() { local id=$1; "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$ROOT" --manifest "$MANIFEST" --output "$OUT/$id" --model-id "${id}_500step" --checkpoint "$CKPT/$id" --seed 23; }
for id in V1 V2 V3 V4; do run "$id"; done
echo HELDOUT_500_MODELS_OK
REMOTE
