#!/usr/bin/env bash
# Remote-only helper. It waits for an actually idle GPU and runs the
# isolated semantic shared-pair pilot. It never stops or edits production jobs.
set -euo pipefail

P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
PY=$P/envs/p1/bin/python
export PYTHONPATH="$R:$R/src"
export OMP_NUM_THREADS=2

SNAP=$W/experiment_workspace/results/cache_safety_snapshots/20260929T155338Z
PILOT=$SNAP/semantic_shared_pilot_v4_pair_parity_seed17_ready
MANIFEST=$PILOT/pilot_shared_groups.jsonl
MODEL=$P/models/teachers/stable-diffusion-v1-5
DATASET=$W/cache_inputs/libero_spatial_task0_v1
REFERENCE=$W/teacher_cache/semantic_train_task0_v3
LOG=$PILOT/pilot_waiter.log
LOCK=$W/semantic_shared_pilot_waiter.lock
PID_FILE=$SNAP/semantic_shared_pilot_waiter.pid
REPORT=$PILOT/semantic_pair_parity.json
PILOT_SCRIPT=$R/scripts/run_sd_nulltext_shared_pair_pilot.py
COMPARE_SCRIPT=$R/scripts/compare_semantic_shared_pilot.py

mkdir -p "$PILOT"
exec >>"$LOG" 2>&1
exec 9>"$LOCK"
flock -n 9 || { echo "PILOT_WAITER_ALREADY_RUNNING"; exit 3; }
printf '%s\n' "$$" >"$PID_FILE"

test -s "$MANIFEST"
test -s "$PILOT_SCRIPT"
test -s "$COMPARE_SCRIPT"
test ! -e "$REPORT"

select_idle_gpu() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    while IFS=, read -r index uuid; do
      index=$(printf '%s' "$index" | xargs)
      uuid=$(printf '%s' "$uuid" | xargs)
      if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
          sed 's/[[:space:]]//g' | grep -Fxq "$uuid"; then
        printf '%s %s\n' "$index" "$uuid"
        return 0
      fi
    done
  return 1
}

echo "SHARED_PILOT_WAITER_STARTED $(date -Is) pid=$$"
while true; do
  pair=$(select_idle_gpu || true)
  if [ -n "$pair" ]; then
    read -r GPU_INDEX GPU_UUID <<<"$pair"
    echo "SHARED_PILOT_IDLE_GPU index=$GPU_INDEX uuid=$GPU_UUID $(date -Is)"
    break
  fi
  echo "SHARED_PILOT_WAIT_IDLE_GPU $(date -Is)"
  sleep 60
done

cat >"$PILOT/gpu_window.json.tmp" <<EOF
{
  "pilot_only": true,
  "exclusive_gpu": true,
  "gpu_index": $GPU_INDEX,
  "gpu_uuid": "$GPU_UUID",
  "checked_at_unix": $(date +%s)
}
EOF
mv "$PILOT/gpu_window.json.tmp" "$PILOT/gpu_window.json"

CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$PILOT_SCRIPT" \
  --model "$MODEL" \
  --dataset-root "$DATASET" \
  --manifest "$MANIFEST" \
  --cache "$PILOT" \
  --steps 20 \
  --inner-steps 10 \
  --guidance-scale 7.5 \
  --resolution 16 \
  --seed 17

CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$COMPARE_SCRIPT" \
  --pilot-dir "$PILOT" \
  --reference-cache "$REFERENCE" \
  --output "$REPORT" \
  --atol 1e-6 \
  --rtol 1e-4

echo "SHARED_PILOT_PARITY_COMPLETE $(date -Is)"
