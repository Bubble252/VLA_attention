#!/usr/bin/env bash
set -euo pipefail

# One guarded command for the complete F0 idea test. Run on a local machine
# with SSH access to 101. Existing reports/maps are reused; no checkpoints or
# reports are overwritten.
readonly REMOTE_HOST="${REMOTE_HOST:-vla101}"
readonly GPU="${CUDA_VISIBLE_DEVICES:-1}"
echo "Using remote CUDA_VISIBLE_DEVICES=$GPU"
ssh "$REMOTE_HOST" "CUDA_VISIBLE_DEVICES='$GPU' bash -s" <<'REMOTE'
set -euo pipefail
P=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R="$P/repo/VLA_attention"
PY="$P/envs/p1/bin/python"
MODEL="$P/models/Qwen2.5-VL-7B-Instruct"
ROOT="$P/data/flickr30k_entities"
HELD="$R/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl"
CKPT="$P/checkpoints/F0v2_idea_validation"
OUT="$P/results/F0v2_heldout_64"
TCACHE="$P/teacher_maps/F0v2_heldout_sd1_5_nulltext_phrase_64_seed23"
TLOG="$P/jobs/F0v2_heldout_sd_cache.log"
export PYTHONPATH="$R/src:$R"
cd "$R"

for path in "$PY" "$MODEL" "$ROOT" "$HELD" "$CKPT/V1" "$CKPT/V2" "$CKPT/V3" "$CKPT/V4"; do
  test -e "$path" || { echo "MISSING_REMOTE=$path" >&2; exit 2; }
done

all_reports=1
for id in V0 V1 V2 V3 V4; do test -s "$OUT/$id/report.json" || all_reports=0; done
if test "$all_reports" -eq 0; then
  MODEL="$MODEL" ROOT="$ROOT" MANIFEST="$HELD" CHECKPOINT_ROOT="$CKPT" \
    OUT="$OUT" PYTHON="$PY" EVAL="$R/scripts/eval_qwen_heldout.py" \
    bash "$R/scripts/run_f0_heldout_all.sh" 2>&1 | tee "$P/jobs/F0v2_heldout_V0_V4.log"
else
  echo "HELDOUT_REPORTS_ALREADY_COMPLETE"
fi
for id in V0 V1 V2 V3 V4; do test -s "$OUT/$id/report.json"; done

mkdir -p "$TCACHE"
maps=$(find "$TCACHE" -maxdepth 1 -name '*.npy' | wc -l)
if test "$maps" -lt 64; then
  "$PY" "$R/scripts/run_sd_cache.py" \
    --runner "$R/scripts/run_sd_nulltext_map_fp32_v2.py" --python "$PY" \
    --model "$P/models/teachers/stable-diffusion-v1-5" --dataset-root "$ROOT" \
    --manifest "$HELD" --cache "$TCACHE" --steps 20 --inner-steps 10 --seed 23 \
    2>&1 | tee "$TLOG"
else
  echo "HELDOUT_TEACHER_CACHE_ALREADY_COMPLETE maps=$maps"
fi
test "$(find "$TCACHE" -maxdepth 1 -name '*.npy' | wc -l)" -ge 64

"$PY" "$R/scripts/eval_teacher_map_controls.py" \
  --manifest "$HELD" --cache "$TCACHE" --dataset-root "$ROOT" \
  --output "$P/results/F0v2_heldout_teacher_controls.json" --seed 23

echo F0_FULL_VALIDATION_OK
REMOTE
