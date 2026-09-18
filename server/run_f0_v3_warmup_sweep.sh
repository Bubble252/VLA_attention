#!/usr/bin/env bash
set -euo pipefail
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly SEED="${SWEEP_SEED:-29}"
readonly STEPS="${SWEEP_STEPS:-100}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' SWEEP_SEED='$SEED' SWEEP_STEPS='$STEPS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"; TRAIN="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
TEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"; CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
ROOT="$P/checkpoints/F0v2_v3_warmup_sweep_seed${SWEEP_SEED}"; RES="$P/results/F0v2_v3_warmup_sweep_seed${SWEEP_SEED}"; EVAL="$P/results/F0v2_v3_warmup_heldout_seed${SWEEP_SEED}"
export PYTHONPATH="$R/src:$R"; cd "$R"; test ! -e "$ROOT" && test ! -e "$RES" && test ! -e "$EVAL" || { echo WARMUP_ARTIFACT_EXISTS >&2; exit 4; }
for warmup in 20 50; do
 tag="w${warmup}"
 "$PY" scripts/run_qwen_v3_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TRAIN" --cache "$CACHE" --lora-config "$LORA" --output "$RES/$tag.json" --checkpoint "$ROOT/$tag" --max-steps "$SWEEP_STEPS" --lambda-sem 0.10 --teacher-temperature 1.25 --semantic-warmup-steps "$warmup" --seed "$SWEEP_SEED" 2>&1 | tee "$P/jobs/F0v2_warmup_train_${tag}_seed${SWEEP_SEED}.log"
 "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TEST" --output "$EVAL/$tag" --model-id "V3_${tag}_seed${SWEEP_SEED}" --checkpoint "$ROOT/$tag" --seed 23 2>&1 | tee "$P/jobs/F0v2_warmup_eval_${tag}_seed${SWEEP_SEED}.log"
 test -s "$EVAL/$tag/report.json" || { echo WARMUP_EVAL_MISSING >&2; exit 5; }
done
echo V3_WARMUP_SWEEP_OK
REMOTE
