#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly STEPS="${F0_STEPS:-100}"
readonly TAG="${F0_TAG:-100}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU, steps=$STEPS"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' F0_STEPS='$STEPS' F0_TAG='$TAG' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DINO="$P/models/DINOv2-ViT-L-14"
DATA="$P/data/flickr30k_entities"; MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
ROOT="$P/checkpoints/F0v2_idea_validation_${F0_TAG}"; OUT="$P/results/F0v2_idea_validation_${F0_TAG}"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DINO" "$DATA" "$MANIFEST" "$CACHE" "$LORA"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
test ! -e "$ROOT" && test ! -e "$OUT" || { echo "LONG_ARTIFACT_EXISTS" >&2; exit 4; }
mkdir -p "$P/jobs"
run() { local id=$1; shift; "$PY" "$@" 2>&1 | tee "$P/jobs/F0v2_${id}_100step.log"; }
run V1_${F0_TAG} scripts/run_qwen_caption_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --lora-config "$LORA" --output "$OUT/V1_${F0_TAG}.json" --checkpoint "$ROOT/V1" --max-steps "$F0_STEPS" --seed 17
run V2_${F0_TAG} scripts/run_qwen_v2_smoke.py --model "$MODEL" --dino "$DINO" --dataset-root "$DATA" --manifest "$MANIFEST" --lora-config "$LORA" --output "$OUT/V2_${F0_TAG}.json" --checkpoint "$ROOT/V2" --max-steps "$F0_STEPS" --seed 17
run V3_${F0_TAG} scripts/run_qwen_v3_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" --lora-config "$LORA" --output "$OUT/V3_${F0_TAG}.json" --checkpoint "$ROOT/V3" --max-steps "$F0_STEPS" --seed 17
run V4_${F0_TAG} scripts/run_qwen_v4_smoke.py --model "$MODEL" --dino "$DINO" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" --lora-config "$LORA" --output "$OUT/V4_${F0_TAG}.json" --checkpoint "$ROOT/V4" --max-steps "$F0_STEPS" --seed 17
for id in V1 V2 V3 V4; do test -s "$ROOT/$id/adapter_model.safetensors"; done
echo LONG_TRAIN_100_OK
REMOTE
