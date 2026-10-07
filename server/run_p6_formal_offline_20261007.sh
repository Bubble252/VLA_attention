#!/usr/bin/env bash
set -euo pipefail

# Fail-closed, serial B0-B4 offline launcher. It never starts rollout.
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$P/repo/VLA_attention
W=$P/vla_workspace
PY=$W/envs/oft/bin/python
MODEL=$W/models/openvla--openvla-7b
LOCK=$W/source_models_20260928.json
DATA=$W/artifacts/rlds_inventory_v2_20260928
OUT=$P/results/P6_vla_b0_b4/formal_offline_20261007
P1=$R/experiment_workspace/results/P6_vla_b0_b4/p1_interface/p1_interface.json
SEMROOT=$W/teacher_cache/semantic_train_task0_v3
SEMMAN=$W/cache_inputs/libero_spatial_task0_v1/semantic.jsonl
RADROOT=$W/teacher_cache/radio_train_task0_v1
RADMAN=$RADROOT/cache_manifest.jsonl
BUNDLE_SHA=06e67101427cf375fcd1c42b90519bdab51d03eef721e531817ad8358927081d
GIT_SHA=9dbaa7178de32e353809b8c15d4f55438438812d
LOG=$W/logs/p6_formal_offline_20261007.log
mkdir -p "$W/logs" "$OUT"
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH="$R/src:$R"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export VLA_GIT_SHA="$GIT_SHA"

fail() { echo "FORMAL_OFFLINE_FAILED stage=$1 reason=$2"; exit 1; }
for path in "$MODEL" "$LOCK" "$DATA/train.jsonl" "$DATA/offline_eval.jsonl" "$DATA/train_statistics.json" "$P1" "$SEMROOT" "$SEMMAN" "$RADROOT" "$RADMAN"; do
  test -e "$path" || fail preflight "missing $path"
done
sha256sum "$R/vla_code_bundle.tar.gz" | awk -v expected="$BUNDLE_SHA" '$1!=expected{exit 1}' || fail preflight "bundle sha mismatch"
"$PY" - "$P1" <<'PY' || exit 1
import json,sys
p=json.load(open(sys.argv[1]))
assert p.get('status')=='passed'
PY

select_idle_gpu() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read -r idx uuid; do
    idx=$(echo "$idx" | xargs); uuid=$(echo "$uuid" | xargs)
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sed 's/[[:space:]]//g' | grep -Fxq "$uuid"; then
      echo "$idx $uuid"; return 0
    fi
  done
}
make_gate() {
  local index="$1" uuid="$2" gate="$3"
  "$PY" - "$gate" "$index" "$uuid" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'cache_audit_passed':True,'exclusive_b0_window':True,'checked_at_unix':time.time(),'gpu_index':int(sys.argv[2]),'gpu_uuid':sys.argv[3],'purpose':'formal offline B0-B4'},indent=2)+'\n')
PY
}
wait_idle() {
  while true; do
    pair=$(select_idle_gpu || true)
    if [ -n "$pair" ]; then echo "$pair"; return; fi
    echo "WAIT_IDLE_GPU $(date -Is)"; sleep 60
  done
}

for group in B0 B1 B2 B3 B4; do
  target="$OUT/${group}_seed17"
  test ! -e "$target" || fail "$group" "output exists: $target"
  read -r idx uuid <<<"$(wait_idle)"
  gate="$W/artifacts/gpu_window_formal_offline_20261007_${group}.json"
  make_gate "$idx" "$uuid" "$gate"
  echo "START_GROUP=$group GPU=$idx $(date -Is)"
  CUDA_VISIBLE_DEVICES="$idx" "$PY" "$R/scripts/run_oft_b0_b4_formal_offline.py" \
    --model "$MODEL" --source-lock "$LOCK" \
    --train-manifest "$DATA/train.jsonl" --eval-manifest "$DATA/offline_eval.jsonl" \
    --statistics "$DATA/train_statistics.json" --gpu-window "$gate" \
    --semantic-root "$SEMROOT" --semantic-manifest "$SEMMAN" \
    --radio-root "$RADROOT" --radio-manifest "$RADMAN" --p1-report "$P1" \
    --output "$target" --group "$group" --seed 17 --steps 10080 \
    --lambda-semantic 0.10 --lambda-retention 0.05 --lambda-containment 0.02 \
    --retention-warmup-steps 500 --containment-warmup-steps 1000 \
    --temperature 1.25 --code-bundle-sha256 "$BUNDLE_SHA"
  test -s "$target/final.json" || fail "$group" final_json_missing
  grep -q '"status": "passed"' "$target/final.json" || fail "$group" final_status_not_passed
  echo "FINISH_GROUP=$group $(date -Is)"
done

echo FORMAL_OFFLINE_B0_B4_SEED17_COMPLETE $(date -Is)
