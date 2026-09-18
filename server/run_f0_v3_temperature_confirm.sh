#!/usr/bin/env bash
set -euo pipefail

# Independent-seed confirmation: candidate (lambda=.1, T=1.25) versus the
# matched semantic-off baseline. No other hyperparameters change.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly SEED="${CONFIRM_SEED:-29}"
readonly STEPS="${SWEEP_STEPS:-100}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' CONFIRM_SEED='$SEED' SWEEP_STEPS='$STEPS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention; R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"; TRAIN="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
TEST="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"; CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
ROOT="$P/checkpoints/F0v2_v3_temperature_confirm_seed${CONFIRM_SEED}"; RES="$P/results/F0v2_v3_temperature_confirm_seed${CONFIRM_SEED}"; EVAL="$P/results/F0v2_v3_temperature_confirm_heldout_seed${CONFIRM_SEED}"
export PYTHONPATH="$R/src:$R"; cd "$R"; test ! -e "$ROOT" && test ! -e "$RES" && test ! -e "$EVAL" || { echo CONFIRM_ARTIFACT_EXISTS >&2; exit 4; }
for spec in "baseline 0.00 1.00" "candidate 0.10 1.25"; do
 set -- $spec; name=$1; lambda=$2; temp=$3
 "$PY" scripts/run_qwen_v3_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TRAIN" --cache "$CACHE" --lora-config "$LORA" --output "$RES/$name.json" --checkpoint "$ROOT/$name" --max-steps "$SWEEP_STEPS" --lambda-sem "$lambda" --teacher-temperature "$temp" --seed "$CONFIRM_SEED" 2>&1 | tee "$P/jobs/F0v2_confirm_${name}_seed${CONFIRM_SEED}.log"
 "$PY" scripts/eval_qwen_heldout.py --model "$MODEL" --dataset-root "$DATA" --manifest "$TEST" --output "$EVAL/$name" --model-id "V3_${name}_seed${CONFIRM_SEED}" --checkpoint "$ROOT/$name" --seed 23 2>&1 | tee "$P/jobs/F0v2_confirm_eval_${name}_seed${CONFIRM_SEED}.log"
 test -s "$EVAL/$name/report.json" || { echo CONFIRM_EVAL_MISSING >&2; exit 5; }
done
echo V3_TEMPERATURE_CONFIRM_OK
REMOTE
