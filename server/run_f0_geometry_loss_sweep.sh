#!/usr/bin/env bash
set -euo pipefail

# Controlled geometry-loss sweep for V3. Each mode uses the same data, cache,
# lambda, temperature, common 32x32 grid and 100 steps; only loss mode/seed vary.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
readonly STEPS="${F0_GEOM_STEPS:-100}"
readonly SEEDS="${F0_GEOM_SEEDS:-17,29,41}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU, steps=$STEPS, seeds=$SEEDS"

ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' F0_GEOM_STEPS='$STEPS' F0_GEOM_SEEDS='$SEEDS' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/flickr30k_entities"
MANIFEST="$R/experiment_workspace/manifests/flickr30k_entities/F0_aligned_caption_phrase_256_seed19_v2.jsonl"
CACHE="$P/teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17_driver"; LORA="$R/configs/qwen_lora_v1.json"
export PYTHONPATH="$R/src:$R"; cd "$R"
for path in "$PY" "$MODEL" "$DATA" "$MANIFEST" "$CACHE" "$LORA"; do test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }; done
IFS=',' read -r -a seeds <<< "$F0_GEOM_SEEDS"
for mode in kl kl_rank kl_moment js; do
  for seed in "${seeds[@]}"; do
    tag="geom_${mode}_seed${seed}_s${F0_GEOM_STEPS}"
    root="$P/checkpoints/F0v2_geometry/${tag}"; out="$P/results/F0v2_geometry/${tag}.json"
    test ! -e "$root" && test ! -e "$out" || { echo "ARTIFACT_EXISTS=$tag" >&2; exit 4; }
    mkdir -p "$P/jobs"
    "$PY" scripts/run_qwen_v3_smoke.py --model "$MODEL" --dataset-root "$DATA" --manifest "$MANIFEST" --cache "$CACHE" --lora-config "$LORA" --output "$out" --checkpoint "$root" --max-steps "$F0_GEOM_STEPS" --seed "$seed" --lambda-sem .1 --teacher-temperature 1.25 --semantic-loss-mode "$mode" --common-resolution 32 --rank-quantile .25 --rank-margin .5 --rank-weight 1.0 --moment-weight 1.0 2>&1 | tee "$P/jobs/${tag}.log"
    test -s "$root/adapter_model.safetensors"
  done
done
echo GEOMETRY_LOSS_SWEEP_TRAIN_OK
REMOTE
