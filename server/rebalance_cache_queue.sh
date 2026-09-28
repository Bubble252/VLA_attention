#!/usr/bin/env bash
# Rebalance the corrected SD cache queue after saving a progress snapshot.
set -euo pipefail

P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace

mkdir -p "$W/artifacts" "$W/logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot=$W/artifacts/cache_rebalance_stop_${stamp}.json

python3 - "$P" "$W" "$snapshot" <<'PY'
import json, subprocess, sys, time
from pathlib import Path

p, w, out = map(Path, sys.argv[1:])
ps = subprocess.check_output(["ps", "-eo", "pid=,ppid=,etime=,stat=,args="], text=True)
rows = [line.strip() for line in ps.splitlines()
        if any(x in line for x in ["run_corrected_cache_queue.sh",
                                   "run_sd_nulltext_batch.py",
                                   "run_p6_b0_after_cache.sh"])]
paths = [w / "teacher_cache/semantic_train_task0_v3/progress.json"]
paths.extend(p / f"teacher_maps/F1_train_sd1_5_reconstruction_v3_9952_seed17/part{i}/progress.json"
             for i in range(4))
progress = {}
for path in paths:
    if path.exists():
        progress[str(path)] = json.loads(path.read_text())
report = {
    "created_at_unix": time.time(),
    "process_lines": rows,
    "progress": progress,
    "note": "saved before throughput rebalance; cache files are preserved",
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
PY

list_target_pids() {
  python3 - <<'PY'
import os
targets = {
    "run_corrected_cache_queue.sh",
    "run_sd_nulltext_batch.py",
    "run_p6_b0_after_cache.sh",
}
excluded = {os.getpid()}
pid = os.getppid()
while pid > 1:
    excluded.add(pid)
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().split()
        pid = int(fields[3])
    except (OSError, ValueError, IndexError):
        break
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    candidate = int(entry.name)
    if candidate in excluded:
        continue
    try:
        argv = open(f"/proc/{candidate}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    names = {os.path.basename(arg.decode(errors="replace")) for arg in argv if arg}
    if names & targets:
        print(candidate)
PY
}

mapfile -t pids < <(list_target_pids)
if [ "${#pids[@]}" -gt 0 ]; then
  kill -TERM "${pids[@]}"
fi
for _ in $(seq 1 30); do
  remaining=$(list_target_pids | wc -l)
  [ "$remaining" -eq 0 ] && break
  sleep 1
done
remaining_pids=$(list_target_pids)
if [ -n "$remaining_pids" ]; then
  ps -p "$remaining_pids" -o pid,ppid,etime,stat,args
  exit 2
fi

setsid nohup bash "$R/server/run_corrected_cache_queue.sh" </dev/null > "$W/logs/corrected_cache_queue.rebalance.nohup.log" 2>&1 &
queue_pid=$!
printf '%s\n' "$queue_pid" > "$W/artifacts/corrected_cache_queue.pid"

QUEUE_PID=$queue_pid setsid nohup bash "$R/server/run_p6_b0_after_cache.sh" </dev/null > "$W/logs/p6_b0_after_cache.nohup.log" 2>&1 &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$W/artifacts/p6_b0_after_cache.pid"

echo "CACHE_QUEUE_REBALANCED snapshot=$snapshot queue_pid=$queue_pid runner_pid=$runner_pid"
