#!/usr/bin/env bash
# Narrow operational entrypoint for this project only.  It never writes /root.
set -euo pipefail

readonly PROJECT_ROOT="/vepfs-mlp2/c20250405/400040/transfer/vla_attention"
readonly PROJECT_REPO="$PROJECT_ROOT/repo/VLA_attention"
readonly P1_ENV="$PROJECT_ROOT/envs/p1"
readonly QWEN_REPO="Qwen/Qwen2.5-VL-7B-Instruct"
readonly QWEN_DIR="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct"

usage() {
  cat <<'EOF'
Usage: bash server/vla101.sh <command>

Commands:
  status       Read only: host, project-path, disk, repo and P1-environment state.
  bootstrap    Create only this project's VEPFS subdirectories and P1 virtual environment.
  freeze-env   Write a package-version snapshot under this project's envs/p1/ directory.
  network-check
               Read only: check official Hugging Face and approved domestic
               mirror reachability from 101. No 7897 proxy is used on 101.
  download-qwen
               Download only the official Qwen2.5-VL-7B-Instruct checkpoint to VEPFS.
               It tries official Hugging Face directly. When official access is not
               reachable, it uses the explicit hf-mirror endpoint and writes the
               actual endpoint plus SHA256 manifest beside the checkpoint.
  inspect-flickr
               Read only: list the available files of the selected Flickr30k
               Hub dataset through the recorded mirror and probe the original
               Flickr30k Entities annotation archive endpoint.

This script requires a local SSH alias named `vla101`.  It deliberately has no
arbitrary remote-shell mode, no credential handling, no /root writes, no dataset
download, and no training command.
EOF
}

remote() {
  ssh vla101 "$@"
}

case "${1:-}" in
  status)
    remote "set -eu
      test -d '$PROJECT_ROOT'
      hostname
      df -h /vepfs-mlp2/c20250405/400040/transfer
      test -f '$PROJECT_REPO/pyproject.toml' && echo REPO_OK || echo REPO_MISSING
      test -x '$P1_ENV/bin/python' && echo P1_ENV_OK || echo P1_ENV_MISSING
      test -d '$QWEN_DIR' && echo QWEN_PRESENT || echo QWEN_MISSING"
    ;;
  bootstrap)
    remote "set -eu
      mkdir -p '$PROJECT_ROOT'/{repo,hf_cache,models,data,teacher_maps,runs,checkpoints,results,jobs,envs}
      if test ! -x '$P1_ENV/bin/python'; then
        /root/starvla_cu124/bin/python -m venv --system-site-packages '$P1_ENV'
      fi
      '$P1_ENV/bin/python' -c 'import torch, transformers, huggingface_hub, datasets, diffusers; print(\"P1_ENV_READY\")'"
    ;;
  freeze-env)
    remote "set -eu
      test -x '$P1_ENV/bin/python'
      '$P1_ENV/bin/python' - <<'PY'
from importlib.metadata import version
from pathlib import Path
names = ('torch', 'transformers', 'huggingface_hub', 'datasets', 'diffusers', 'accelerate', 'peft')
out = Path('$P1_ENV') / 'versions.txt'
out.write_text(''.join(f'{name}=={version(name)}\\n' for name in names))
print(out)
PY"
    ;;
  network-check)
    remote "set -eu
      curl --connect-timeout 15 --max-time 30 -sS -o /dev/null -w 'HF_OFFICIAL_HTTP=%{http_code}\\n' https://huggingface.co || echo HF_OFFICIAL_REQUEST_FAILED
      curl --connect-timeout 15 --max-time 30 -sS -o /dev/null -w 'HF_MIRROR_HTTP=%{http_code}\\n' https://hf-mirror.com || echo HF_MIRROR_REQUEST_FAILED"
    ;;
  download-qwen)
    remote "set -eu
      test -x '$P1_ENV/bin/python'
      mkdir -p '$PROJECT_ROOT/hf_cache' '$QWEN_DIR'
      export HF_HOME='$PROJECT_ROOT/hf_cache'
      command -v hf
      if curl --connect-timeout 15 --max-time 30 -fsS -o /dev/null https://huggingface.co; then
        export HF_ENDPOINT='https://huggingface.co'
      else
        export HF_ENDPOINT='https://hf-mirror.com'
      fi
      printf 'repository=%s\\nendpoint=%s\\ndownloaded_at_utc=%s\\n' '$QWEN_REPO' \"\$HF_ENDPOINT\" \"\$(date -u +%FT%TZ)\" > '$QWEN_DIR/SOURCE.txt'
      hf download '$QWEN_REPO' --local-dir '$QWEN_DIR'
      find '$QWEN_DIR' -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > '$QWEN_DIR/SHA256SUMS'
      du -sh '$QWEN_DIR'"
    ;;
  inspect-flickr)
    remote "set -eu
      export HF_ENDPOINT='https://hf-mirror.com'
      '$P1_ENV/bin/python' - <<'PY'
from huggingface_hub import HfApi
api = HfApi(endpoint='https://hf-mirror.com')
for name in api.list_repo_files('nlphuji/flickr30k', repo_type='dataset'):
    print('HF_FILE=' + name)
PY
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_PLUMMER_HTTP=%{http_code}\\n' https://bryanplummer.com/Flickr30kEntities/Annotations.zip || echo ENTITIES_PLUMMER_REQUEST_FAILED
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_VGG_ANNOTATIONS_HTTP=%{http_code}\\n' https://www.robots.ox.ac.uk/~vgg/data/flickr30k_entities/Annotations.zip || echo ENTITIES_VGG_ANNOTATIONS_REQUEST_FAILED
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_VGG_SENTENCES_HTTP=%{http_code}\\n' https://www.robots.ox.ac.uk/~vgg/data/flickr30k_entities/Sentences.zip || echo ENTITIES_VGG_SENTENCES_REQUEST_FAILED"
    ;;
  -h|--help|help|'') usage ;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
