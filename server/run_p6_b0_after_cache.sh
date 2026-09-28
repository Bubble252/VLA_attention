#!/usr/bin/env bash
# Wait for the audited cache queue and one idle physical GPU, then run the
# authorized P1 -> B0 smoke -> fresh-process restore -> two official rollouts.
set -euo pipefail

P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
PY=$W/envs/oft/bin/python
QUEUE_PID="${QUEUE_PID:-}"
QUEUE_PID_FILE=$W/artifacts/corrected_cache_queue.pid
QUEUE_LOG=$W/logs/corrected_cache_queue.log
QUEUE_SUCCESS=$W/artifacts/cache_queue_success.json
MODEL=$W/models/openvla--openvla-7b
LOCK=$W/source_models_20260928.json
DATA=$W/artifacts/rlds_inventory_v2_20260928
OUT=$P/results/P6_vla_b0_b4
P1_ROOT=$R/experiment_workspace/results/P6_vla_b0_b4
P1=$P1_ROOT/p1_interface/p1_interface.json
TRAIN=$OUT/B0_smoke_seed17
RESTORE=$OUT/B0_restore_seed17
ROLLOUT=$OUT/B0_rollout_seed17
GATE=$W/artifacts/gpu_window_b0.json
DRIVER_LOG=$W/logs/p6_b0_after_cache.log

mkdir -p "$W/logs" "$W/artifacts"
exec > >(tee -a "$DRIVER_LOG") 2>&1
export PYTHONPATH="$R/src:$W/repos/openvla-oft:$W/repos/libero"
export LIBERO_CONFIG_PATH="$W/libero_config"
export OMP_NUM_THREADS=2

fail() {
  printf 'P6_B0_DRIVER_FAILED stage=%s reason=%s\n' "$1" "$2"
  exit 1
}

echo "P6_B0_DRIVER_STARTED $(date -Is) queue_pid=$QUEUE_PID"

verify_cache_audits() {
  "$PY" - "$P" "$W" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
w = Path(sys.argv[2])
parity = json.loads((p / "results/cache_v3_gate/deterministic_parity.json").read_text())
assert len(parity) == 2
assert all(x.get("passed") is True and x.get("max_abs_error") == 0
           and x.get("attention_tensors") == 100 for x in parity)
audits = [
    json.loads((w / "artifacts/semantic_train_v3_audit.json").read_text()),
    json.loads((w / "artifacts/radio_train_v1_audit.json").read_text()),
]
root = p / "teacher_maps/F1_train_sd1_5_reconstruction_v3_9952_seed17"
audits.extend(json.loads((root / f"part{i}_audit.json").read_text()) for i in range(4))
assert all(a.get("passed") is True and not a.get("failures")
           and not a.get("extra_files") for a in audits)
assert audits[0].get("expected") == 10080 and audits[0].get("valid") == 10080
assert audits[1].get("expected") == 5040 and audits[1].get("valid") == 5040
assert sum(a.get("valid", 0) for a in audits[2:]) == 9952
print("ALL_CACHE_AUDITS_PASSED")
PY
}

write_cache_success() {
  "$PY" - "$QUEUE_SUCCESS" "$QUEUE_PID" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "queue_pid": int(sys.argv[2]),
    "completed_at_unix": time.time(),
    "audits_verified_by": "run_p6_b0_after_cache.sh",
    "marker": "CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED",
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
tmp.replace(path)
PY
}

# Prefer a durable success artifact.  The log marker is retained for backward
# compatibility with queue launches that redirect stdout; the process-exit
# path also verifies every audit so a foreground queue with an unredirected
# stdout pipe cannot be mistaken for a failure.
while true; do
  if [ -z "$QUEUE_PID" ] && [ -s "$QUEUE_PID_FILE" ]; then
    QUEUE_PID=$(cat "$QUEUE_PID_FILE")
  fi
  [ -n "$QUEUE_PID" ] || fail cache_queue "queue PID is not known yet"
  receipt_matches=false
  if [ -s "$QUEUE_SUCCESS" ]; then
    receipt_matches=$("$PY" - "$QUEUE_SUCCESS" "$QUEUE_PID" <<'PY'
import json
import sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text())
print("true" if receipt.get("queue_pid") == int(sys.argv[2])
      and receipt.get("marker") == "CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED"
      else "false")
PY
)
  fi
  if [ "$receipt_matches" = true ] || grep -q 'CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED' "$QUEUE_LOG" 2>/dev/null; then
    verify_cache_audits || fail cache_queue "success marker exists but cache audit verification failed"
    [ "$receipt_matches" = true ] || write_cache_success
    break
  fi
  queue_state=$(ps -o stat= -p "$QUEUE_PID" 2>/dev/null | tr -d ' ' || true)
  if ! kill -0 "$QUEUE_PID" 2>/dev/null || [[ "$queue_state" == Z* ]]; then
    echo "CACHE_QUEUE_EXITED; verifying durable audit outputs"
    verify_cache_audits || fail cache_queue "queue exited without a valid audited cache; inspect $QUEUE_LOG"
    write_cache_success
    break
  fi
  queue_cmd=$(tr '\0' ' ' <"/proc/$QUEUE_PID/cmdline" 2>/dev/null || true)
  case "$queue_cmd" in
    *run_corrected_cache_queue.sh*) ;;
    *) fail cache_queue "queue PID no longer identifies the expected script" ;;
  esac
  echo "WAIT_CACHE_QUEUE $(date -Is)"
  sleep 60
done

verify_cache_audits

select_idle_gpu() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    while IFS=, read -r index uuid; do
      index=$(printf '%s' "$index" | xargs)
      uuid=$(printf '%s' "$uuid" | xargs)
      if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
          sed 's/[[:space:]]//g' | grep -Fxq "$uuid"; then
        printf '%s %s\n' "$index" "$uuid"
        exit 0
      fi
    done
}

write_gpu_gate() {
  local index=$1 uuid=$2
  "$PY" - "$GATE" "$index" "$uuid" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "cache_audit_passed": True,
    "exclusive_b0_window": True,
    "checked_at_unix": time.time(),
    "gpu_index": int(sys.argv[2]),
    "gpu_uuid": sys.argv[3],
    "evidence": "cache queue success marker plus all parity/cache audit reports",
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
}

wait_for_idle_gpu() {
  while true; do
    pair=$(select_idle_gpu || true)
    if [ -n "$pair" ]; then
      read -r GPU_INDEX GPU_UUID <<<"$pair"
      echo "IDLE_GPU_FOUND index=$GPU_INDEX uuid=$GPU_UUID $(date -Is)"
      return
    fi
    echo "WAIT_IDLE_GPU $(date -Is)"
    sleep 60
  done
}

wait_for_idle_gpu
write_gpu_gate "$GPU_INDEX" "$GPU_UUID"
CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$R/scripts/run_oft_p1_interface.py" \
  --model "$MODEL" --source-lock "$LOCK" \
  --train-manifest "$DATA/train.jsonl" \
  --statistics "$DATA/train_statistics.json" \
  --gpu-window "$GATE" --output "$P1_ROOT/p1_interface"
test -s "$P1" || fail p1 "report missing"
grep -q '"status": "passed"' "$P1" || fail p1 "report did not pass"

wait_for_idle_gpu
write_gpu_gate "$GPU_INDEX" "$GPU_UUID"
CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$R/scripts/run_oft_b0_smoke.py" \
  --model "$MODEL" --source-lock "$LOCK" \
  --train-manifest "$DATA/train.jsonl" \
  --eval-manifest "$DATA/offline_eval.jsonl" \
  --statistics "$DATA/train_statistics.json" \
  --gpu-window "$GATE" --p1-report "$P1" \
  --output "$TRAIN" --steps 30 --seed 17
test -s "$TRAIN/checkpoint_manifest.json" || fail b0_train "sealed checkpoint missing"

wait_for_idle_gpu
write_gpu_gate "$GPU_INDEX" "$GPU_UUID"
CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$R/scripts/run_oft_b0_smoke.py" \
  --model "$MODEL" --source-lock "$LOCK" \
  --train-manifest "$DATA/train.jsonl" \
  --eval-manifest "$DATA/offline_eval.jsonl" \
  --statistics "$DATA/train_statistics.json" \
  --gpu-window "$GATE" --p1-report "$P1" \
  --output "$RESTORE" --steps 30 --seed 17 --restore "$TRAIN"
test -s "$RESTORE/restore_report.json" || fail restore "restore report missing"
test -s "$RESTORE/continuation_report.json" || fail restore "continuation report missing"

wait_for_idle_gpu
write_gpu_gate "$GPU_INDEX" "$GPU_UUID"
CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PY" "$R/scripts/eval_oft_b0_rollout.py" \
  --model "$MODEL" --source-lock "$LOCK" \
  --checkpoint "$TRAIN" --restore-report-dir "$RESTORE" \
  --statistics "$DATA/train_statistics.json" \
  --init-inventory "$W/artifacts/rollout_init_inventory_20260928.json" \
  --gpu-window "$GATE" --output "$ROLLOUT"
test -s "$ROLLOUT/summary.json" || fail rollout "summary missing"

echo "P6_B0_GPU_GATE_COMPLETE $(date -Is)"
echo "P1=$P1"
echo "B0_TRAIN=$TRAIN"
echo "B0_RESTORE=$RESTORE"
echo "B0_ROLLOUT=$ROLLOUT"
echo "No B1-B4 or later-stage experiment was launched."
