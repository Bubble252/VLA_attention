#!/usr/bin/env bash
set -euo pipefail

# Run from a local machine with working `ssh vla101`. This starts only the
# missing 20-step V3/V4 checkpoint smokes and refuses to overwrite artifacts.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"
PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"
DINO="$P/models/DINOv2-ViT-L-14"
DATA="$P/data/flickr30k_entities"
MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"
LORA="$R/configs/qwen_lora_v1.json"
RESULT="$P/results/F0v2_idea_validation"
CKPT="$P/checkpoints/F0v2_idea_validation"
export PYTHONPATH="$R/src:$R"
echo "Remote CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for path in "$PY" "$MODEL" "$DATA" "$MANIFEST" "$CACHE" "$LORA"; do
  test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }
done
test "$(find "$CACHE" -maxdepth 1 -name '*.npy' | wc -l)" -ge 256 || { echo CACHE_INCOMPLETE >&2; exit 3; }

run_v3() {
  test ! -e "$CKPT/V3" && test ! -e "$RESULT/V3_checkpoint_smoke.json" || { echo V3_ARTIFACT_EXISTS >&2; exit 4; }
  mkdir -p "$P/jobs"
  "$PY" "$R/scripts/run_qwen_v3_smoke.py" \
    --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" \
    --lora-config "$LORA" --output "$RESULT/V3_checkpoint_smoke.json" \
    --checkpoint "$CKPT/V3" --max-steps 20 --seed 17 \
    2>&1 | tee "$P/jobs/F0v2_V3_checkpoint_smoke.log"
  test -s "$RESULT/V3_checkpoint_smoke.json" && test -s "$CKPT/V3/adapter_model.safetensors"
  echo V3_CHECKPOINT_OK
}
run_v3

run_v4() {
  test ! -e "$CKPT/V4" && test ! -e "$RESULT/V4_checkpoint_smoke.json" || { echo V4_ARTIFACT_EXISTS >&2; exit 5; }
  "$PY" "$R/scripts/run_qwen_v4_smoke.py" \
    --model "$MODEL" --dino "$DINO" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" \
    --lora-config "$LORA" --output "$RESULT/V4_checkpoint_smoke.json" \
    --checkpoint "$CKPT/V4" --max-steps 20 --seed 17 \
    2>&1 | tee "$P/jobs/F0v2_V4_checkpoint_smoke.log"
  test -s "$RESULT/V4_checkpoint_smoke.json" && test -s "$CKPT/V4/adapter_model.safetensors" && test -s "$CKPT/V4/dino_projector.pt"
  echo V4_CHECKPOINT_OK
}
run_v4
echo CHECKPOINT_SMOKES_OK
REMOTE
