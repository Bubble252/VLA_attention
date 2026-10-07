#!/usr/bin/env bash
set -euo pipefail

# Build a small, reproducible code-only upload bundle.  Model/data/cache/
# checkpoint files are rejected by construction and are never copied.
readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly OUT="$(realpath -m "${1:-$ROOT/experiment_workspace/vla_upload}")"
readonly LIST="$OUT/files.txt"
mkdir -p "$OUT"
rm -f "$LIST" "$OUT/SHA256SUMS" "$OUT/manifest.json"

cd "$ROOT"
{
  find src/vla_attention -type f -name '*.py' -print
  find scripts -maxdepth 1 -type f -name 'validate_vla_config.py' -print
  printf '%s\n' scripts/run_oft_b3_retention_semantic_smoke.py
  find tests -maxdepth 1 -type f -name 'test_*contract*.py' -print
  find tests -maxdepth 1 -type f -name 'test_plugin_*.py' -print
  printf '%s\n' tests/test_vla_local_pipeline.py
  find configs/vla -maxdepth 1 -type f -name '*.json' -o -name 'README.md'
  printf '%s\n' experiments/P6-vla-b0-b4/protocol.md experiments/P6-vla-b0-b4/101_outage_readiness.md
  printf '%s\n' references/repos/starvla/INTERFACE_AUDIT.md
} | sort -u > "$LIST"

while IFS= read -r file; do
  test -f "$file" || { echo "MISSING=$file" >&2; exit 2; }
  case "$file" in
    */checkpoints/*|*/cache/*|*/caches/*|*.safetensors|*.bin|*.pt|*.pth|*.ckpt|data/*|datasets/*|artifacts/*)
      echo "FORBIDDEN_UPLOAD_FILE=$file" >&2; exit 3 ;;
  esac
done < "$LIST"

tar -czf "$OUT/vla_code_bundle.tar.gz" -T "$LIST"
sha256sum $(cat "$LIST") "${OUT#$ROOT/}/vla_code_bundle.tar.gz" > "$OUT/SHA256SUMS"
VLA_REPO_ROOT="$ROOT" VLA_UPLOAD_OUT="$OUT" python3 - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
root, out = Path(os.environ['VLA_REPO_ROOT']), Path(os.environ['VLA_UPLOAD_OUT'])
files = [x.strip() for x in (out/'files.txt').read_text().splitlines() if x.strip()]
payload = {
    'kind': 'vla_code_only_upload',
    'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
    'files': files,
    'sha256_manifest': str((out/'SHA256SUMS').relative_to(root)),
    'bundle': str((out/'vla_code_bundle.tar.gz').relative_to(root)),
    'large_artifacts_included': False,
}
(out/'manifest.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
print(out/'manifest.json')
PY
echo "VLA_UPLOAD_MANIFEST_OK files=$(wc -l < "$LIST") bundle=$(du -h "$OUT/vla_code_bundle.tar.gz" | awk '{print $1}')"
