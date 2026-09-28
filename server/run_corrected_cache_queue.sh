#!/usr/bin/env bash
# Run on 101. Continuation queue only: no student training or B0 launch.
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
export PYTHONPATH="$R:$R/src" OMP_NUM_THREADS=2
PY=$P/envs/p1/bin/python
mkdir -p "$W/logs" "$W/artifacts"
exec >> "$W/logs/corrected_cache_queue.log" 2>&1
exec 9>"$W/corrected_cache_queue.lock"
flock -n 9 || exit 3
printf '%s\n' "$$" > "$W/artifacts/corrected_cache_queue.pid"
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
semantic_root=$W/teacher_cache/semantic_train_task0_v3
semantic_manifest=$W/cache_inputs/libero_spatial_task0_v1/semantic.jsonl
mkdir -p "$semantic_root"
semantic_pids=()
semantic_rows=$(grep -c . "$semantic_manifest")
semantic_shards=4
semantic_chunk=$(( (semantic_rows + semantic_shards - 1) / semantic_shards ))
for i in 0 1 2 3; do
  start=$((i * semantic_chunk))
  end=$(( (i + 1) * semantic_chunk ))
  [ "$end" -le "$semantic_rows" ] || end="$semantic_rows"
  [ "$start" -lt "$end" ] || continue
  CUDA_VISIBLE_DEVICES=$((i%2)) "$PY" "$R/scripts/run_sd_nulltext_batch.py" \
    --model "$P/models/teachers/stable-diffusion-v1-5" \
    --dataset-root "$W/cache_inputs/libero_spatial_task0_v1" \
    --manifest "$semantic_manifest" --cache "$semantic_root" \
    --steps 20 --inner-steps 10 --seed 17 \
    --row-start "$start" --row-end "$end" --worker-id "semantic_part$i" \
    > "$W/logs/semantic_train_task0_v3_part$i.log" 2>&1 &
  semantic_pids+=("$!")
done
printf '%s\n' "${semantic_pids[@]}" > "$W/artifacts/semantic_train_task0_v3.pids"
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
for pid in "${pids[@]}" "${semantic_pids[@]}"; do wait "$pid" || rc=1; done
[ "$rc" -eq 0 ] || { echo CACHE_WORKER_FAILED; exit 2; }
"$PY" - "$semantic_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
parts = [json.loads((root / f"progress_semantic_part{i}.json").read_text()) for i in range(4)]
assert all(p["completed"] == p["total"] and p["failed"] == 0 for p in parts)
summary = {
    "total": sum(p["total"] for p in parts),
    "completed": sum(p["completed"] for p in parts),
    "failed": sum(p["failed"] for p in parts),
    "manifest_sha256": parts[0]["manifest_sha256"],
    "extractor_sha256s": sorted({p["extractor_sha256"] for p in parts}),
    "workers": parts,
}
tmp = root / "progress.json.tmp"
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(root / "progress.json")
failures = []
for i in range(4):
    failures.extend(json.loads((root / f"failures_semantic_part{i}.json").read_text()))
tmp = root / "failures.json.tmp"
tmp.write_text(json.dumps(failures, indent=2) + "\n")
tmp.replace(root / "failures.json")
PY
"$PY" "$W/validate_typed_teacher_cache.py" --manifest "$semantic_manifest" --cache "$semantic_root" --kind semantic --output "$W/artifacts/semantic_train_v3_audit.json"
for i in 0 1 2 3; do
  "$PY" "$W/validate_typed_teacher_cache.py" --manifest "$P/tmp/f1_train_cache_splits4_aligned/part$i.jsonl" --cache "$root/part$i" --kind semantic --output "$root/part${i}_audit.json"
done
 "$PY" - "$W/artifacts/cache_queue_success.json" "$$" <<'PY'
import json
import sys
import time
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "queue_pid": int(sys.argv[2]),
    "completed_at_unix": time.time(),
    "marker": "CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED",
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
tmp.replace(path)
PY
echo CACHE_GENERATION_AND_AUDIT_COMPLETE_NO_TRAINING_LAUNCHED
