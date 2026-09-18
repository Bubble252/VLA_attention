#!/usr/bin/env bash
set -euo pipefail

# Run this from the local VLA_attention repository on a machine that can SSH
# to 101. It uploads only the validation runners, then verifies every remote
# file against the local SHA256. It does not touch models, data, or checkpoints.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly REMOTE_ROOT="${REMOTE_ROOT:-/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention}"
readonly FILES=(
  scripts/run_qwen_v3_smoke.py
  scripts/run_qwen_v4_smoke.py
  scripts/eval_qwen_heldout.py
  scripts/eval_teacher_map_controls.py
  scripts/run_f0_heldout_all.sh
)

for file in "${FILES[@]}"; do
  test -f "$file" || { echo "MISSING_LOCAL=$file" >&2; exit 2; }
done

echo "[1/3] checking SSH: $REMOTE_HOST"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts' && echo REMOTE_READY"

echo "[2/3] uploading ${#FILES[@]} validation files"
scp "${FILES[@]}" "$REMOTE_HOST:$REMOTE_ROOT/scripts/"

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
sha256sum "${FILES[@]}" > "$manifest"
scp "$manifest" "$REMOTE_HOST:$REMOTE_ROOT/scripts/.validation_scripts.sha256"

echo "[3/3] verifying remote SHA256"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && sha256sum -c scripts/.validation_scripts.sha256 && chmod +x scripts/run_f0_heldout_all.sh && echo UPLOAD_VERIFY_OK"
