#!/usr/bin/env bash
set -euo pipefail

# Fine lambda and teacher-temperature sweep after the coarse lambda grid.
# V3 is isolated first because it supplied the only reproducible directional
# signal; V4 is reserved for confirming an accepted V3 candidate.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly STEPS="${SWEEP_STEPS:-100}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU, steps=$STEPS"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' SWEEP_STEPS='$STEPS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"; TRAIN="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
TEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"; CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
CKROOT="$P/checkpoints/F0v2_v3_conditioning_sweep_${SWEEP_STEPS}"; TRAINOUT="$P/results/F0v2_v3_conditioning_sweep_${SWEEP_STEPS}"; EVALOUT="$P/results/F0v2_v3_conditioning_heldout_${SWEEP_STEPS}"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DATA" "$TRAIN" "$TEST" "$CACHE" "$LORA"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
test ! -e "$CKROOT" && test ! -e "$TRAINOUT" && test ! -e "$EVALOUT" || { echo "CONDITIONING_SWEEP_ARTIFACT_EXISTS" >&2; exit 4; }
mkdir -p "$P/jobs"

# tag lambda temperature: refine lambda near 0.1 at T=1, then vary teacher
# temperature at lambda=0.1. Existing lambda=0.1,T=1 run is the shared anchor.
for spec in "l0p06 0.06 t1p00 1.00" "l0p08 0.08 t1p00 1.00" "l0p12 0.12 t1p00 1.00" "l0p15 0.15 t1p00 1.00" "l0p10 0.10 t0p50 0.50" "l0p10 0.10 t0p75 0.75" "l0p10 0.10 t1p25 1.25" "l0p10 0.10 t1p50 1.50"; do
  set -- $spec; ltag=$1; lambda=$2; ttag=$3; temp=$4; tag="${ltag}_${ttag}"
  ck="$CKROOT/V3_${tag}"; trainout="$TRAINOUT/V3_${tag}.json"; evalout="$EVALOUT/V3_${tag}"
  "$PY" scripts/run_qwen_v3_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TRAIN" --cache "$CACHE" --lora-config "$LORA" --output "$trainout" --checkpoint "$ck" --max-steps "$SWEEP_STEPS" --lambda-sem "$lambda" --teacher-temperature "$temp" --seed 17 2>&1 | tee "$P/jobs/F0v2_conditioning_train_${tag}.log"
  test -s "$trainout" && test -s "$ck/adapter_model.safetensors" || { echo "TRAIN_INCOMPLETE=$tag" >&2; exit 5; }
  "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TEST" --output "$evalout" --model-id "V3_${tag}" --checkpoint "$ck" --seed 23 2>&1 | tee "$P/jobs/F0v2_conditioning_eval_${tag}.log"
  test -s "$evalout/report.json" || { echo "EVAL_INCOMPLETE=$tag" >&2; exit 6; }
  echo CONDITIONING_SWEEP_OK="$tag"
done
echo V3_CONDITIONING_SWEEP_OK
REMOTE
