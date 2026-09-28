#!/usr/bin/env bash
# Run on 101. Continuation queue only: no student training or B0 launch.
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
export PYTHONPATH="$R:$R/src" OMP_NUM_THREADS=2
PY=$P/envs/p1/bin/python
exec 9>"$W/corrected_cache_queue.lock"
flock -n 9 || exit 3
"$PY" - <<'PY'
import json
from pathlib import Path
p=Path('/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/cache_v3_gate/deterministic_parity.json')
rows=json.loads(p.read_text())
assert len(rows)==2 and all(r['passed'] and r['max_abs_error']==0 and r['attention_tensors']==100 for r in rows)
PY
# Wait only for the explicitly identified smoke process; never infer success
# from a PID disappearing. Actual maps and hashes are validated below.
while [ -r /proc/648822/cmdline ] && tr '\0' ' ' < /proc/648822/cmdline | grep -q 'semantic_smoke_v3'; do sleep 60; done
"$PY" "$W/validate_typed_teacher_cache.py" --manifest "$W/cache_inputs/libero_spatial_task0_v1/semantic_smoke.jsonl" --cache "$W/teacher_cache/semantic_smoke_v3" --kind semantic --output "$W/artifacts/semantic_smoke_v3_audit.json"
"$PY" - <<'PY'
import json
from pathlib import Path
w=Path('/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace')
assert json.loads((w/'artifacts/radio_train_v1_audit.json').read_text())['passed']
PY
CUDA_VISIBLE_DEVICES=0 "$PY" "$R/scripts/run_sd_nulltext_batch.py" --model "$P/models/teachers/stable-diffusion-v1-5" --dataset-root "$W/cache_inputs/libero_spatial_task0_v1" --manifest "$W/cache_inputs/libero_spatial_task0_v1/semantic.jsonl" --cache "$W/teacher_cache/semantic_train_task0_v3" --steps 20 --inner-steps 10 --seed 17 > "$W/logs/semantic_train_task0_v3.log" 2>&1 &
semantic_pid=$!
printf '%s\n' "$semantic_pid" > "$W/artifacts/semantic_train_task0_v3.pid"
# Four disjoint Flickr shards, two workers per GPU, new versioned outputs.
root=$P/teacher_maps/F1_train_sd1_5_reconstruction_v3_9952_seed17
mkdir -p "$root"
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i%2)) "$PY" "$R/scripts/run_sd_nulltext_batch.py" --model "$P/models/teachers/stable-diffusion-v1-5" --dataset-root "$P/data/flickr30k_entities" --manifest "$P/tmp/f1_train_cache_splits4_aligned/part$i.jsonl" --cache "$root/part$i" --steps 20 --inner-steps 10 --seed 17 > "$P/jobs/flickr_v3_part$i.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$root/worker_pids.txt"
rc=0
for pid in "${pids[@]}" "$semantic_pid"; do wait "$pid" || rc=1; done
[ "$rc" -eq 0 ] || { echo CACHE_WORKER_FAILED; exit 2; }
"$PY" "$W/validate_typed_teacher_cache.py" --manifest "$W/cache_inputs/libero_spatial_task0_v1/semantic.jsonl" --cache "$W/teacher_cache/semantic_train_task0_v3" --kind semantic --output "$W/artifacts/semantic_train_v3_audit.json"
for i in 0 1 2 3; do
  "$PY" "$W/validate_typed_teacher_cache.py" --manifest "$P/tmp/f1_train_cache_splits4_aligned/part$i.jsonl" --cache "$root/part$i" --kind semantic --output "$root/part${i}_audit.json"
done
echo CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED
