#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"; PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"; DATA="$P/data/capability"
MAN="$R/experiment_workspace/manifests/capability"
OUT="$P/results/P4_capability_pilot_20260923_v2"; LOG="$P/jobs/P4_capability_pilot_20260923_v2.log"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$R/src:$R"
export PYCOCOEVALCAP_PATH="$P/envs/coco_caption_scorer"
export HF_HOME="$P/hf_cache"
cd "$R"
mkdir -p "$OUT" "$P/jobs"
for path in "$PY" "$MODEL" "$MAN/coco_captions_val_pilot128.jsonl" "$MAN/vqav2_val_pilot128.jsonl" "$MAN/textvqa_val_pilot128.jsonl" "$MAN/pope_pilot384.jsonl" "$MAN/mme_test_pilot128.jsonl" "$MAN/worldmedqa_v_pilot256.jsonl"; do
  test -e "$path" || { echo "MISSING=$path" >&2; exit 2; }
done

run_one() {
  local model_id="$1" adapter="$2" benchmark="$3" manifest="$4" caption_metrics="${5:-0}"
  local dest="$OUT/$model_id/$benchmark/report.json"
  local dataset_name="${benchmark%%_*}"
  case "$dataset_name" in coco) dataset_name="coco" ;; esac
  if test -s "$dest"; then echo "SKIP_COMPLETE $model_id $benchmark"; return; fi
  local args=(scripts/eval_capability_suite.py --model "$MODEL" --model-id "$model_id" --model-revision "Qwen2.5-VL-7B-Instruct@local" --dataset-root "$DATA/$dataset_name" --manifest "$manifest" --output "$dest" --max-new-tokens 64 --max-image-pixels 1003520 --checkpoint-every 10 --resume)
  if test -n "$adapter"; then args+=(--adapter "$adapter"); fi
  if test "$caption_metrics" = 1; then args+=(--caption-metrics); fi
  "$PY" "${args[@]}"
}

for model_id in V0 V1 V2 V3_KLrank V4_KLrank; do
  case "$model_id" in
    V0) adapter="" ;;
    V1) adapter="$P/checkpoints/F0v2_idea_validation_100/V1" ;;
    V2) adapter="$P/checkpoints/F0v2_idea_validation_100/V2" ;;
    V3_KLrank) adapter="$P/checkpoints/F0v2_geometry/geom16_kl_rank_seed17_s100" ;;
    V4_KLrank) adapter="$P/checkpoints/F0v2_geometry/geom16_kl_rank_v4_seed17_s100" ;;
  esac
  if test -n "$adapter"; then test -s "$adapter/adapter_model.safetensors" || { echo "MISSING_ADAPTER=$adapter" >&2; exit 3; }; fi
  run_one "$model_id" "$adapter" coco_captions "$MAN/coco_captions_val_pilot128.jsonl" 1
  run_one "$model_id" "$adapter" vqa "$MAN/vqav2_val_pilot128.jsonl"
  run_one "$model_id" "$adapter" textvqa "$MAN/textvqa_val_pilot128.jsonl"
  run_one "$model_id" "$adapter" pope "$MAN/pope_pilot384.jsonl"
  run_one "$model_id" "$adapter" mme "$MAN/mme_test_pilot128.jsonl"
  run_one "$model_id" "$adapter" worldmedqa "$MAN/worldmedqa_v_pilot256.jsonl"
done
echo P4_CAPABILITY_PILOT_OK
REMOTE
