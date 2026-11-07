#!/usr/bin/env bash
set -euo pipefail

# Run the frozen 15-init x 4-prompt C1 suite for a Goal13 postrun PASS.
# Usage: bash server/run_goal13_c1_pilot.sh <postrun.json> [max_inits] [port]

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
readonly R="$W/vla_workspace/repo/starvla_55_20261030/repo"
readonly C=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124
readonly G13="${GOAL13_ROOT:-$W/results/star105_goal13_20261107}"
readonly GATE_A="${GATE_A_ROOT:-$W/results/star105_goal10p5_20261105}/evals/gate_a.json"
readonly POSTRUN="${1:?usage: run_goal13_c1_pilot.sh <postrun.json> [max_inits] [port]}"
readonly MAX_INITS="${2:-15}"
readonly PORT="${3:-6799}"
readonly MANIFEST="${C1_MANIFEST:-$W/vla_workspace/data/jgate_c1/counterfactual_20261027/manifest.jsonl}"
readonly E=/root/starvla_cu124/bin/python
readonly TOOLS_DIR="${GOAL13_TOOLS:-/tmp/goal13_tools}"

for path in "$POSTRUN" "$GATE_A" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done

read -r RUN_ID CHECKPOINT ARM CONFIG_SHA < <(
  "$E" - "$POSTRUN" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(payload["checkpoint"])
run_id = checkpoint.parent.parent.name
print(run_id, checkpoint, payload["arm"], payload["config_snapshot_sha256"])
PY
)
readonly RUN_ID CHECKPOINT ARM CONFIG_SHA
readonly OUTPUT="$G13/c1/$RUN_ID"
readonly CONTRACT="$G13/contracts/${RUN_ID}_c1_pilot_contract.json"
readonly SERVER_LOG="$G13/logs/${RUN_ID}_c1_pilot_server.log"
readonly ROLLOUT_LOG="$G13/logs/${RUN_ID}_c1_pilot_rollout.log"
readonly SUMMARY_LOG="$G13/logs/${RUN_ID}_c1_pilot_summary.log"
readonly SUMMARY="$G13/c1/${RUN_ID}_summary.json"

"$E" - "$POSTRUN" "$GATE_A" "$CHECKPOINT" "$CONFIG_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

postrun_path, gate_path, checkpoint_path = map(Path, sys.argv[1:4])
expected_config_sha = sys.argv[4]
postrun = json.loads(postrun_path.read_text())
gate = json.loads(gate_path.read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if postrun.get("postrun_status") != "PASS":
    raise SystemExit(f"postrun is not PASS: {postrun.get('postrun_status')}")
if postrun.get("arm") != "D0":
    raise SystemExit(f"unexpected arm: {postrun.get('arm')}")
if postrun.get("config_snapshot_sha256") != expected_config_sha:
    raise SystemExit("config SHA drift between shell and postrun")
if postrun.get("checkpoint_sha256") != sha256(checkpoint_path):
    raise SystemExit("checkpoint SHA mismatch")
if gate.get("gate_status") != "complete" or gate.get("passed") is not True:
    raise SystemExit(f"Gate A is not complete/passed: {gate_path}")
PY

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 2>/dev/null || true)
if (( free_mib < 30000 )); then
  echo "GPU1 free=${free_mib}MiB < 30000MiB; refusing to start" >&2
  exit 4
fi
if [[ -n "$compute" ]]; then
  echo "GPU1 has compute processes: $compute" >&2
  exit 5
fi
if [[ -e "$OUTPUT" ]]; then
  echo "C1 pilot output already exists: $OUTPUT" >&2
  exit 6
fi
if [[ -e "$CONTRACT" ]]; then
  echo "C1 pilot contract already exists: $CONTRACT" >&2
  exit 6
fi

set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$C"
set -u
cd "$R"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$R:/root/code/LIBERO"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export STARVLA_DISABLE_DEEPSPEED=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/root/code/LIBERO/libero

mkdir -p "$G13/c1" "$G13/contracts" "$G13/logs"
nohup "$E" deployment/model_server/server_policy.py \
  --ckpt_path "$CHECKPOINT" --port "$PORT" --use_bf16 --idle_timeout -1 \
  > "$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "policy server died before listening" >&2
    tail -n 60 "$SERVER_LOG" >&2
    exit 7
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3<&- 3>&-
    break
  fi
  sleep 2
done
if ! (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "policy server did not open port $PORT" >&2
  tail -n 60 "$SERVER_LOG" >&2
  exit 8
fi

args=(
  scripts/star105_c1_rollout.py
  --manifest "$MANIFEST"
  --output "$OUTPUT"
  --arm "$RUN_ID"
  --port "$PORT"
  --seed 7
)
if (( MAX_INITS > 0 )); then
  args+=(--limit "$MAX_INITS")
fi
"$E" "${args[@]}" 2>&1 | tee "$ROLLOUT_LOG"

"$E" scripts/summarize_c1_base.py \
  --episodes "$OUTPUT/episodes.jsonl" \
  --output "$SUMMARY" 2>&1 | tee "$SUMMARY_LOG"

"$E" "$TOOLS_DIR/verify_goal13_c1_pilot.py" \
  --postrun "$POSTRUN" \
  --summary "$SUMMARY" \
  --output "$CONTRACT" \
  --expected-groups "$MAX_INITS" \
  --expected-arm "$ARM"

echo "D0_C1_PILOT_OK run=$RUN_ID output=$OUTPUT summary=$SUMMARY contract=$CONTRACT"
