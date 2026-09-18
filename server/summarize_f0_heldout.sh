#!/usr/bin/env bash
set -euo pipefail

readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
ssh "$REMOTE_HOST" bash -s <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
OUT="$P/results/F0v2_heldout_64"
LOG="$P/jobs/F0v2_heldout_V0_V4.log"
echo "=== heldout process/log status ==="
if test -f "$LOG"; then tail -n 12 "$LOG"; else echo LOG_MISSING; fi
echo "=== report presence ==="
for id in V0 V1 V2 V3 V4; do
  if test -s "$OUT/$id/report.json"; then echo "$id=READY"; else echo "$id=MISSING"; fi
done
if ! for id in V0 V1 V2 V3 V4; do test -s "$OUT/$id/report.json"; done; then
  echo "HELDOUT_INCOMPLETE"
  exit 0
fi
OUT="$OUT" "$P/envs/p1/bin/python" - <<'PY'
import json, os
from pathlib import Path
out=Path(os.environ['OUT'])
data={i:json.loads((out/i/'report.json').read_text())['summary'] for i in ('V0','V1','V2','V3','V4')}
print(json.dumps(data, indent=2))
criteria={
 'V3_gt_V1_mass': data['V3']['mass_in_box'] > data['V1']['mass_in_box'],
 'V3_gt_V1_iou': data['V3']['top20_box_iou'] > data['V1']['top20_box_iou'],
 'V4_gt_V2_mass': data['V4']['mass_in_box'] > data['V2']['mass_in_box'],
 'V4_gt_V2_iou': data['V4']['top20_box_iou'] > data['V2']['top20_box_iou'],
}
print('CRITERIA=' + json.dumps(criteria, sort_keys=True))
print('MODEL_HELDOUT_READY')
PY
REMOTE
