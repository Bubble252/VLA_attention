#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"
MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
LORA="$R/configs/qwen_lora_v1.json"
export PYTHONPATH="$R/src:$R"; cd "$R"
for seed in 29 41; do
  tag="matched_v1_seed${seed}_s100"
  ck="$P/checkpoints/F0v2_geometry/$tag"; out="$P/results/F0v2_geometry/$tag.json"
  test ! -e "$ck" && test ! -e "$out" || { echo "ARTIFACT_EXISTS=$tag" >&2; exit 4; }
  "$PY" scripts/run_qwen_caption_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --lora-config "$LORA" --output "$out" --checkpoint "$ck" --max-steps 100 --seed "$seed" 2>&1 | tee "$P/jobs/${tag}.log"
  test -s "$ck/adapter_model.safetensors"
done
echo MATCHED_V1_SEEDS_OK
REMOTE
