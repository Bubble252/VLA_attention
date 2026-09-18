#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; ROOT="$P/data/flickr30k_entities"; MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CKPT="$P/checkpoints/F0v2_idea_validation_100"; OUT="$P/results/F0v2_heldout_64_100"; export PYTHONPATH="$R/src:$R"; cd "$R"
for id in V1 V2 V3 V4; do test -s "$CKPT/$id/adapter_model.safetensors" || { echo "MISSING=$CKPT/$id" >&2; exit 2; }; done
for id in V1 V2 V3 V4; do test ! -e "$OUT/$id/report.json" || { echo "EXISTS=$OUT/$id/report.json" >&2; exit 4; }; done
run() { local id=$1 ckpt=$2; "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$ROOT" --manifest "$MANIFEST" --output "$OUT/$id" --model-id "${id}_100step" --checkpoint "$ckpt" --seed 23; }
run V1 "$CKPT/V1"; run V2 "$CKPT/V2"; run V3 "$CKPT/V3"; run V4 "$CKPT/V4"
for id in V1 V2 V3 V4; do test -s "$OUT/$id/report.json"; done
echo HELDOUT_100_MODELS_OK
REMOTE
