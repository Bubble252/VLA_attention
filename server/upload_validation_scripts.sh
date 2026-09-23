#!/usr/bin/env bash
set -euo pipefail

# Run this from the local VLA_attention repository on a machine that can SSH
# to 101. It uploads only the validation runners, then verifies every remote
# file against the local SHA256. It does not touch models, data, or checkpoints.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly REMOTE_ROOT="${REMOTE_ROOT:-/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention}"
readonly FILES=(
  src/vla_attention/losses.py
  scripts/run_qwen_v3_smoke.py
  scripts/run_qwen_v4_smoke.py
  scripts/calibrate_spatial_metrics.py
  scripts/summarize_geometry_sweep.py
  scripts/paired_geometry_bootstrap.py
  scripts/eval_qwen_heldout.py
  scripts/eval_teacher_map_controls.py
  scripts/run_f0_heldout_all.sh
  server/run_f0_geometry_loss_sweep.sh
  server/eval_f0_geometry_loss_sweep.sh
  server/eval_f0_geometry_calibrated_sweep.sh
  server/run_f0_geometry_matched_v1.sh
  tests/test_semantic_map_losses.py
  tests/test_spatial_calibration.py
  scripts/eval_capability_suite.py
  scripts/validate_capability_manifest.py
  tests/test_capability_eval.py
  tests/test_capability_manifest.py
  configs/experiments/P4_capability_eval_v0_v4.json
  experiments/P4-vlm-v0-v4/capability_manifest_schema.md
)

for file in "${FILES[@]}"; do
  test -f "$file" || { echo "MISSING_LOCAL=$file" >&2; exit 2; }
done

echo "[1/3] checking SSH: $REMOTE_HOST"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts' && echo REMOTE_READY"

echo "[2/3] uploading ${#FILES[@]} validation files"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/src/vla_attention' '$REMOTE_ROOT/server' '$REMOTE_ROOT/tests' '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/configs/experiments' '$REMOTE_ROOT/experiments/P4-vlm-v0-v4'"
scp src/vla_attention/losses.py "$REMOTE_HOST:$REMOTE_ROOT/src/vla_attention/losses.py"
scp scripts/run_qwen_v3_smoke.py scripts/run_qwen_v4_smoke.py scripts/calibrate_spatial_metrics.py scripts/summarize_geometry_sweep.py scripts/paired_geometry_bootstrap.py scripts/eval_qwen_heldout.py scripts/eval_teacher_map_controls.py scripts/run_f0_heldout_all.sh "$REMOTE_HOST:$REMOTE_ROOT/scripts/"
scp server/run_f0_geometry_loss_sweep.sh server/eval_f0_geometry_loss_sweep.sh server/eval_f0_geometry_calibrated_sweep.sh server/run_f0_geometry_matched_v1.sh "$REMOTE_HOST:$REMOTE_ROOT/server/"
scp tests/test_semantic_map_losses.py tests/test_spatial_calibration.py "$REMOTE_HOST:$REMOTE_ROOT/tests/"
scp scripts/eval_capability_suite.py scripts/validate_capability_manifest.py "$REMOTE_HOST:$REMOTE_ROOT/scripts/"
scp tests/test_capability_eval.py tests/test_capability_manifest.py "$REMOTE_HOST:$REMOTE_ROOT/tests/"
scp configs/experiments/P4_capability_eval_v0_v4.json "$REMOTE_HOST:$REMOTE_ROOT/configs/experiments/"
scp experiments/P4-vlm-v0-v4/capability_manifest_schema.md "$REMOTE_HOST:$REMOTE_ROOT/experiments/P4-vlm-v0-v4/"

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
sha256sum "${FILES[@]}" > "$manifest"
scp "$manifest" "$REMOTE_HOST:$REMOTE_ROOT/scripts/.validation_scripts.sha256"

echo "[3/3] verifying remote SHA256"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && sha256sum -c scripts/.validation_scripts.sha256 && chmod +x scripts/run_f0_heldout_all.sh server/run_f0_geometry_loss_sweep.sh server/eval_f0_geometry_loss_sweep.sh server/eval_f0_geometry_calibrated_sweep.sh server/run_f0_geometry_matched_v1.sh && echo UPLOAD_VERIFY_OK"
