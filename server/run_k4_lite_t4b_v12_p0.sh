#!/usr/bin/env bash
set -euo pipefail

readonly W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
readonly R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
readonly OUT="$W/results/K4_lite/T4b/p0_authority_release_15init_20261107"
readonly LOG="$W/results/K4_lite/T4b/p0_authority_release_15init_20261107.log"
readonly CHECKPOINT="$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt"
readonly MANIFEST="$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl"
readonly CONFIG="configs/vla/k4_lite_t4b_v12_p0_authority_release.json"
readonly EXPECTED_FINGERPRINT=747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20
readonly EXPECTED_MANIFEST_SHA=e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e

test ! -e "$OUT" || { echo "output already exists: $OUT" >&2; exit 3; }
test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
test -d "$CHECKPOINT" || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA" || {
  echo "manifest SHA256 mismatch" >&2
  exit 4
}
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
if (( free_mib < 25000 )); then
  echo "GPU1 free=${free_mib}MiB < 25000MiB" >&2
  exit 4
fi

mkdir -p "$W/results/K4_lite/T4b"
{
  echo "started_at=$(date -Is)"
  echo "gpu=1"
  echo "gpu_free_mib=$free_mib"
  echo "checkpoint=$CHECKPOINT"
  echo "manifest=$MANIFEST"
  echo "manifest_sha256=$EXPECTED_MANIFEST_SHA"
  echo "config=$CONFIG"
  echo "output=$OUT"
} > "$LOG"

cd "$R"
export PYTHONPATH="$R:$R/src:$W/repos/openvla-oft:$W/repos/libero"
export LIBERO_CONFIG_PATH="$W/libero_config"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$W/envs/oft/bin/python" scripts/audit_k4_lite_t4b_v12_p0_authority_release.py \
  --model "$W/models/openvla--openvla-7b" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --gate-config "$CONFIG" \
  --expected-checkpoint-fingerprint "$EXPECTED_FINGERPRINT" \
  --output "$OUT" \
  --max-inits 15 \
  --seed 7 >> "$LOG" 2>&1
