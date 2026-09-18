#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly STEPS="${SWEEP_STEPS:-100}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU, steps=$STEPS"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' SWEEP_STEPS='$STEPS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DINO="$P/models/DINOv2-ViT-L-14"
DATA="$P/data/flickr30k_entities"; MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
CKROOT="$P/checkpoints/F0v2_lambda_sweep_${SWEEP_STEPS}"; OUTROOT="$P/results/F0v2_lambda_sweep_${SWEEP_STEPS}"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DINO" "$DATA" "$MANIFEST" "$CACHE" "$LORA"; do
  test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }
done
test ! -e "$CKROOT" && test ! -e "$OUTROOT" || { echo "SWEEP_ARTIFACT_ROOT_EXISTS" >&2; exit 4; }
mkdir -p "$P/jobs"
for lambda in 0.00 0.01 0.03 0.10 0.30; do
  tag="l${lambda/./p}"
  for variant in V3 V4; do
    ck="$CKROOT/${variant}_${tag}"; out="$OUTROOT/${variant}_${tag}.json"
    test ! -e "$ck" && test ! -e "$out" || { echo "ARTIFACT_EXISTS=$variant/$tag" >&2; exit 5; }
    if test "$variant" = V3; then
      "$PY" "$R/scripts/run_qwen_v3_smoke.py" --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" --lora-config "$LORA" --output "$out" --checkpoint "$ck" --max-steps "$SWEEP_STEPS" --lambda-sem "$lambda" --seed 17 2>&1 | tee "$P/jobs/F0v2_lambda_${variant}_${tag}.log"
    else
      "$PY" "$R/scripts/run_qwen_v4_smoke.py" --model "$MODEL" --dino "$DINO" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" --lora-config "$LORA" --output "$out" --checkpoint "$ck" --max-steps "$SWEEP_STEPS" --lambda-sem "$lambda" --lambda-ret 0.1 --seed 17 2>&1 | tee "$P/jobs/F0v2_lambda_${variant}_${tag}.log"
    fi
    test -s "$out" && test -s "$ck/adapter_model.safetensors" || { echo "SWEEP_RUN_INCOMPLETE=$variant/$tag" >&2; exit 6; }
    echo "SWEEP_TRAIN_OK=$variant/$tag"
  done
done
echo LAMBDA_SWEEP_TRAIN_OK
REMOTE
