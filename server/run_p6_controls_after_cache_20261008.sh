#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-vla101}"
ssh -o ConnectTimeout=20 "$HOST" 'bash -s' <<'REMOTE_SETUP'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
GUARD=$W/artifacts/p6_controls_after_cache_20261008.pid
SCRIPT=$W/artifacts/p6_controls_after_cache_20261008.sh
LOG=$W/logs/p6_controls_after_cache_20261008.log
if [ -f "$GUARD" ] && ps -p "$(cat "$GUARD")" >/dev/null 2>&1; then
  echo "P6_CONTROLS_GUARD_ALREADY_RUNNING pid=$(cat "$GUARD") log=$LOG"
  exit 0
fi
cat > "$SCRIPT" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
CACHE=$W/teacher_cache/semantic_eval_task0_v1
OUT=$P/results/P6_vla_b0_b4/posttrain_eval_20261008/controls_eval_full_seed17
LOG_PREFIX='[p6_controls_after_cache]'
echo "$LOG_PREFIX started $(date -Is)"
echo "$LOG_PREFIX cache=$CACHE"
echo "$LOG_PREFIX output=$OUT"
while true; do
  if python3 - <<'PY'
import json, sys
from pathlib import Path
root=Path('/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/teacher_cache/semantic_eval_task0_v1')
names=['eval_gpu1a','eval_gpu1b','eval_gpu0a','eval_gpu0b']
progress=[root/f'progress_{name}.json' for name in names]
missing=[str(p) for p in progress if not p.exists()]
if missing:
    print('not_ready missing_progress', missing)
    sys.exit(1)
completed=total=failed=0
for p in progress:
    d=json.load(open(p))
    completed += int(d.get('completed',0)); total += int(d.get('total',0)); failed += int(d.get('failed',0))
map_count=len(list(root.glob('*.npy')))
print({'completed':completed,'total':total,'failed':failed,'map_count':map_count})
if total != 3076 or completed != total or failed != 0 or map_count != 3076:
    sys.exit(1)
PY
  then
    break
  fi
  sleep 300
done
if [ -e "$OUT" ]; then
  echo "$LOG_PREFIX output exists; refusing overwrite: $OUT"
  exit 0
fi
cd "$R"
echo "$LOG_PREFIX launching controls $(date -Is)"
PYTHONPATH=$R/src:$R CUDA_VISIBLE_DEVICES=0 $W/envs/oft/bin/python scripts/eval_oft_posttrain_controls.py \
  --model $W/models/openvla--openvla-7b \
  --results-root $P/results/P6_vla_b0_b4/formal_offline_20261007 \
  --eval-manifest $W/artifacts/rlds_inventory_v2_20260928/offline_eval.jsonl \
  --statistics $W/artifacts/rlds_inventory_v2_20260928/train_statistics.json \
  --semantic-root $CACHE \
  --semantic-manifest $W/cache_inputs/libero_spatial_task0_eval_v1/semantic.jsonl \
  --output $OUT \
  --groups B0,B1,B2,B3,B4 \
  --checkpoint-step 10080 --seed 17 --max-transitions 0
echo "$LOG_PREFIX controls complete $(date -Is)"
REMOTE_SCRIPT
chmod +x "$SCRIPT"
nohup "$SCRIPT" > "$LOG" 2>&1 &
echo $! > "$GUARD"
echo "P6_CONTROLS_GUARD_STARTED pid=$(cat "$GUARD") log=$LOG script=$SCRIPT"
REMOTE_SETUP
