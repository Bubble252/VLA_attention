#!/usr/bin/env bash
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
PY=$W/envs/oft/bin/python
MODEL=$W/models/openvla--openvla-7b
DATA=$W/artifacts/rlds_inventory_v2_20260928
RESULTS=$P/results/P6_vla_b0_b4/formal_offline_20261007
OUT=$P/results/P6_vla_b0_b4/posttrain_eval_20261008
LOG=$W/logs/p6_posttrain_eval_20261008.log
mkdir -p "$W/logs" "$P/results/P6_vla_b0_b4"
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH="$R/src:$R"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
fail() { echo "P6_POSTTRAIN_EVAL_FAILED stage=$1 reason=$2"; exit 1; }
for path in "$MODEL" "$DATA/offline_eval.jsonl" "$DATA/train_statistics.json" "$RESULTS"; do
  test -e "$path" || fail preflight "missing $path"
done
test ! -e "$OUT" || fail preflight "output exists: $OUT"
select_idle_gpu() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read -r idx uuid; do
    idx=$(echo "$idx" | xargs); uuid=$(echo "$uuid" | xargs)
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sed 's/[[:space:]]//g' | grep -Fxq "$uuid"; then
      echo "$idx"; return 0
    fi
  done
}
idx=$(select_idle_gpu || true)
test -n "$idx" || fail gpu no_idle_gpu
echo "START_POSTTRAIN_ACTION_EVAL gpu=$idx $(date -Is)"
CUDA_VISIBLE_DEVICES="$idx" "$PY" "$R/scripts/eval_oft_formal_offline.py" \
  --model "$MODEL" \
  --results-root "$RESULTS" \
  --eval-manifest "$DATA/offline_eval.jsonl" \
  --statistics "$DATA/train_statistics.json" \
  --output "$OUT/action_eval" \
  --groups B0,B1,B2,B3,B4 \
  --checkpoint-step 10080 \
  --seed 17

echo "POSTTRAIN_ACTION_EVAL_COMPLETE $(date -Is)"
