#!/usr/bin/env bash
set -euo pipefail

# Goal13 C1 offline evaluator wrapper.  It consumes the postrun contract rather
# than guessing the checkpoint/config, and keeps validation/rollout data out of
# training.  Usage: bash server/run_goal13_offline_eval.sh <postrun.json> [limit]
# limit=0 evaluates the full validation manifest; limit=2 is only a smoke.

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
readonly R="$W/vla_workspace/repo/starvla_55_20261030/repo"
readonly C=/vepfs-mlp2/c20250405/400040/transfer/conda_envs/starvla_cu124
readonly G13="${GOAL13_ROOT:-$W/results/star105_goal13_20261107}"
readonly G12="${GOAL12_ROOT:-$W/results/star105_goal12_20261106}"
readonly POSTRUN="${1:?usage: run_goal13_offline_eval.sh <postrun_contract.json> [limit]}"
readonly LIMIT="${2:-0}"
read -r RUN_ID CHECKPOINT CONFIG ARM < <(
  /root/starvla_cu124/bin/python - "$POSTRUN" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(payload["checkpoint"])
run_id = checkpoint.parent.parent.name
print(run_id, checkpoint, payload["config_snapshot"], payload["arm"])
PY
)
readonly RUN="$G13/checkpoints/$RUN_ID"
readonly CHECKPOINT="$RUN/final_model/model.safetensors"
readonly CONFIG="$G13/configs/$(basename "$CONFIG")"
readonly SIDECAR="$W/vla_workspace/results/K4_lite/T2_future/validation_sidecar_v1/manifest.jsonl"
readonly TRAIN_ROOT="$W/results/star105_goal10p5_20261105/data/LEROBOT_LIBERO_DATA"
readonly VAL_ROOT="$G12/data/LEROBOT_LIBERO_DATA"
readonly VAL_EPISODES="$VAL_ROOT/libero_90_gate_validation_lerobot/meta/c1_source_episodes.jsonl"
readonly VAL_WHITELIST="$G12/manifests/c1_validation_episodes.json"
if (( LIMIT == 0 )); then
  readonly OUTPUT="$G13/evals/${RUN_ID}_offline_full.json"
  readonly LOG="$G13/evals/${RUN_ID}_offline_full.log"
else
  readonly OUTPUT="$G13/evals/${RUN_ID}_offline_limit${LIMIT}.json"
  readonly LOG="$G13/evals/${RUN_ID}_offline_limit${LIMIT}.log"
fi
readonly CONTRACT_OUTPUT="${OUTPUT%.json}_contract.json"
readonly E="/root/starvla_cu124/bin/python"
readonly TOOLS_DIR="${GOAL13_TOOLS:-/tmp/goal13_tools}"

for path in "$POSTRUN" "$CONFIG" "$CHECKPOINT" "$SIDECAR" "$VAL_EPISODES" "$VAL_WHITELIST"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done

"$E" - "$POSTRUN" "$CONFIG" "$CHECKPOINT" "$ARM" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

postrun, config, checkpoint = map(Path, sys.argv[1:4])
expected_arm = sys.argv[4]
payload = json.loads(postrun.read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if payload.get("postrun_status") != "PASS":
    raise SystemExit(f"postrun is not PASS: {payload.get('postrun_status')}")
if payload.get("arm") != expected_arm:
    raise SystemExit(f"unexpected arm: {payload.get('arm')} != {expected_arm}")
if payload.get("config_snapshot_sha256") != sha256(config):
    raise SystemExit("config snapshot SHA mismatch")
if payload.get("checkpoint_sha256") != sha256(checkpoint):
    raise SystemExit("checkpoint SHA mismatch")
PY

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 2>/dev/null || true)
if (( free_mib < 30000 )); then
  echo "GPU1 free=${free_mib}MiB < 30000MiB; refusing to evaluate" >&2
  exit 4
fi
if [[ -n "$compute" ]]; then
  echo "GPU1 has compute processes: $compute" >&2
  exit 5
fi

mkdir -p "$G13/evals"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite: $OUTPUT" >&2; exit 3; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 3; }

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

"$E" "$R/scripts/star12_counterfactual_eval.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --sidecar "$SIDECAR" \
  --validation-episodes "$VAL_EPISODES" \
  --validation-root "$VAL_ROOT" \
  --validation-whitelist "$VAL_WHITELIST" \
  --train-root "$TRAIN_ROOT" \
  --output "$OUTPUT" \
  --arm "$ARM" \
  --batch-size 2 \
  --limit "$LIMIT" \
  --seed 17 \
  --bootstrap-samples 200 \
  2>&1 | tee "$LOG"

if (( LIMIT == 2 )); then
  "$E" "$TOOLS_DIR/verify_goal13_offline_smoke.py" \
    --postrun "$POSTRUN" \
    --offline "$OUTPUT" \
    --output "$CONTRACT_OUTPUT" \
    --expected-limit-groups "$LIMIT" \
    --expected-arm "$ARM"
else
  "$E" "$TOOLS_DIR/verify_goal13_offline_eval.py" \
    --postrun "$POSTRUN" \
    --offline "$OUTPUT" \
    --output "$CONTRACT_OUTPUT" \
    --expected-groups 118 \
    --expected-destination-pairs 78 \
    --expected-object-pairs 78 \
    --expected-arm "$ARM"
fi

echo "D0_OFFLINE_EVAL_OK run=$RUN_ID output=$OUTPUT contract=$CONTRACT_OUTPUT log=$LOG"
