#!/usr/bin/env bash
# Narrow operational entrypoint for this project only.  It never writes /root.
set -euo pipefail

readonly PROJECT_ROOT="/vepfs-mlp2/c20250405/400040/transfer/vla_attention"
readonly PROJECT_REPO="$PROJECT_ROOT/repo/VLA_attention"
readonly P1_ENV="$PROJECT_ROOT/envs/p1"
readonly QWEN_REPO="Qwen/Qwen2.5-VL-7B-Instruct"
readonly QWEN_DIR="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct"
readonly TEACHER_DIR="$PROJECT_ROOT/models/teachers"
readonly FLICKR_REPO="nlphuji/flickr30k"
readonly FLICKR_DIR="$PROJECT_ROOT/data/flickr30k_entities"
readonly ENTITIES_REPO="BryanPlummer/flickr30k_entities"

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
  inspect-entities-repos
               Read only: inspect file names, sizes and licenses in the original
               Flickr30k Entities repository and one public mirror candidate.
  probe-entities-blob
               Read only: test whether GitHub API raw media with an HTTP Range
               request returns ZIP bytes for the exact versioned annotations blob.
  probe-entities-proxies
               Read only: test candidate GitHub raw proxy endpoints with a
               16-byte Range request. A later full download must still verify
               the official Git blob SHA before being accepted.
  download-flickr
               Download full Flickr30k images/caption files and the original
               Flickr30k Entities annotation archive into VEPFS; extract only
               after archive integrity checks, and record source revisions,
               SHA256 and counts. The data is research/education only.
  download-teachers
               Download the four revision-pinned T_sem candidates sequentially
               through hf-mirror, recording source and SHA256 per teacher.

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
for item in api.list_datasets(search='flickr30k entities', limit=30):
    print('HF_CANDIDATE=' + item.id)
PY
      curl --connect-timeout 15 --max-time 30 -sS 'https://api.github.com/search/repositories?q=flickr30k%20entities&per_page=10' | '$P1_ENV/bin/python' -c \"import json,sys; [print('GITHUB_CANDIDATE=' + x['full_name'] + ' ' + x['html_url']) for x in json.load(sys.stdin).get('items', [])]\" || echo GITHUB_SEARCH_FAILED
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_PLUMMER_HTTP=%{http_code}\\n' https://bryanplummer.com/Flickr30kEntities/Annotations.zip || echo ENTITIES_PLUMMER_REQUEST_FAILED
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_VGG_ANNOTATIONS_HTTP=%{http_code}\\n' https://www.robots.ox.ac.uk/~vgg/data/flickr30k_entities/Annotations.zip || echo ENTITIES_VGG_ANNOTATIONS_REQUEST_FAILED
      curl --connect-timeout 15 --max-time 30 -sSIL -o /dev/null -w 'ENTITIES_VGG_SENTENCES_HTTP=%{http_code}\\n' https://www.robots.ox.ac.uk/~vgg/data/flickr30k_entities/Sentences.zip || echo ENTITIES_VGG_SENTENCES_REQUEST_FAILED"
    ;;
  inspect-entities-repos)
    remote "set -eu
      '$P1_ENV/bin/python' - <<'PY'
import json
from urllib.request import urlopen
for repo in ('BryanPlummer/flickr30k_entities', 'xmodal-multilang-retrieval/flickr30k_entities'):
    info = json.load(urlopen('https://api.github.com/repos/' + repo, timeout=30))
    branch = info['default_branch']
    tree = json.load(urlopen('https://api.github.com/repos/' + repo + '/git/trees/' + branch + '?recursive=1', timeout=30))['tree']
    print('REPO=' + repo)
    print('LICENSE=' + str((info.get('license') or {}).get('spdx_id')))
    for item in tree:
        name = item['path'].lower()
        if any(key in name for key in ('annotation', 'sentence', 'license', 'readme', '.zip', '.xml')):
            print('REPO_FILE=' + item['path'] + ' size=' + str(item.get('size', 'NA')))
    if repo == 'BryanPlummer/flickr30k_entities':
        readme = urlopen('https://raw.githubusercontent.com/' + repo + '/' + branch + '/README.md', timeout=30).read().decode()
        print('README_BEGIN')
        print(readme)
        print('README_END')
PY"
    ;;
  probe-entities-blob)
    remote "set -eu
      REV=\$(curl --connect-timeout 15 --max-time 30 -fsS 'https://api.github.com/repos/$ENTITIES_REPO/commits/master' | '$P1_ENV/bin/python' -c \"import json,sys; print(json.load(sys.stdin)['sha'])\")
      BLOB=\$(ENTITIES_REPO='$ENTITIES_REPO' ENTITIES_REV=\"\$REV\" '$P1_ENV/bin/python' - <<'PY'
import json, os
from urllib.request import urlopen
tree = json.load(urlopen(f\"https://api.github.com/repos/{os.environ['ENTITIES_REPO']}/git/trees/{os.environ['ENTITIES_REV']}?recursive=1\", timeout=60))['tree']
print(next(item['sha'] for item in tree if item['path'] == 'annotations.zip'))
PY
)
      echo ENTITIES_REV=\"\$REV\"
      echo ENTITIES_BLOB=\"\$BLOB\"
      curl --connect-timeout 15 --max-time 30 -fsS -H 'Accept: application/vnd.github.raw+json' --range 0-15 'https://api.github.com/repos/$ENTITIES_REPO/git/blobs/'\"\$BLOB\" | od -An -tx1"
    ;;
  probe-entities-proxies)
    remote "set -eu
      REV=\$(curl --connect-timeout 15 --max-time 30 -fsS 'https://api.github.com/repos/$ENTITIES_REPO/commits/master' | '$P1_ENV/bin/python' -c \"import json,sys; print(json.load(sys.stdin)['sha'])\")
      RAW='https://raw.githubusercontent.com/$ENTITIES_REPO/'\"\$REV\"'/annotations.zip'
      for proxy in 'https://ghproxy.net/' 'https://gh-proxy.com/' 'https://mirror.ghproxy.com/'; do
        printf 'PROXY=%s MAGIC=' \"\$proxy\"
        curl --http1.1 --connect-timeout 15 --max-time 30 -fsS --range 0-15 \"\$proxy\$RAW\" | od -An -tx1 | tr -d ' \\n' || echo REQUEST_FAILED
      done"
    ;;
  download-flickr)
    remote "set -eu
      test -x '$P1_ENV/bin/python'
      mkdir -p '$FLICKR_DIR/raw/hf' '$FLICKR_DIR/raw/entities' '$FLICKR_DIR/images' '$FLICKR_DIR/entities'
      export HF_HOME='$PROJECT_ROOT/hf_cache'
      export HF_ENDPOINT='https://hf-mirror.com'
      HF_REV=\$(curl --connect-timeout 15 --max-time 30 -fsS 'https://hf-mirror.com/api/datasets/$FLICKR_REPO' | '$P1_ENV/bin/python' -c \"import json,sys; print(json.load(sys.stdin)['sha'])\")
      ENTITIES_REV=\$(curl --connect-timeout 15 --max-time 30 -fsS 'https://api.github.com/repos/$ENTITIES_REPO/commits/master' | '$P1_ENV/bin/python' -c \"import json,sys; print(json.load(sys.stdin)['sha'])\")
      hf download '$FLICKR_REPO' flickr30k-images.zip flickr_annotations_30k.csv --repo-type dataset --revision \"\$HF_REV\" --local-dir '$FLICKR_DIR/raw/hf'
      # 101 reaches api.github.com but raw.githubusercontent.com and git HTTPS
      # are too slow. ghproxy.net passed a ZIP-magic probe; accept it only when
      # the downloaded bytes reproduce the official Git blob SHA.
      ENTITIES_BLOB=\$(ENTITIES_REPO='$ENTITIES_REPO' ENTITIES_REV=\"\$ENTITIES_REV\" '$P1_ENV/bin/python' - <<'PY'
import json, os
from urllib.request import urlopen
tree = json.load(urlopen(f\"https://api.github.com/repos/{os.environ['ENTITIES_REPO']}/git/trees/{os.environ['ENTITIES_REV']}?recursive=1\", timeout=60))['tree']
print(next(item['sha'] for item in tree if item['path'] == 'annotations.zip'))
PY
)
      ENTITIES_OUT='$FLICKR_DIR/raw/entities/annotations.zip'
      ENTITIES_CANDIDATE=\"\$ENTITIES_OUT.proxy\"
      # Continue-at mode preserves verified transport progress across a retry.
      # The final Git blob SHA remains the acceptance criterion.
      curl --http1.1 --continue-at - --connect-timeout 15 --max-time 900 --retry 5 --retry-all-errors --retry-delay 3 --fail --location 'https://ghproxy.net/https://raw.githubusercontent.com/$ENTITIES_REPO/'\"\$ENTITIES_REV\"'/annotations.zip' -o \"\$ENTITIES_CANDIDATE\"
      ENTITIES_BLOB=\"\$ENTITIES_BLOB\" ENTITIES_CANDIDATE=\"\$ENTITIES_CANDIDATE\" '$P1_ENV/bin/python' - <<'PY'
import hashlib, os
from pathlib import Path
path = Path(os.environ['ENTITIES_CANDIDATE'])
payload = path.read_bytes()
actual = hashlib.sha1(f'blob {len(payload)}\\0'.encode() + payload).hexdigest()
if actual != os.environ['ENTITIES_BLOB']:
    raise RuntimeError(f'Git blob SHA mismatch: expected {os.environ["ENTITIES_BLOB"]}, got {actual}')
print(f'entities_blob_verified={actual} bytes={len(payload)}')
PY
      unzip -t \"\$ENTITIES_CANDIDATE\" >/dev/null
      mv \"\$ENTITIES_CANDIDATE\" \"\$ENTITIES_OUT\"
      ENTITIES_TRANSPORT=ghproxy_net_raw_verified_blob
      printf 'image_dataset=%s\\nimage_endpoint=%s\\nimage_revision=%s\\nentities_repository=%s\\nentities_revision=%s\\nentities_blob=%s\\nentities_transport=%s\\ndownloaded_at_utc=%s\\nlicense_note=Flickr images: non-commercial research/education under Flickr Terms; cite Flickr30k and Flickr30k Entities.\\n' '$FLICKR_REPO' 'https://hf-mirror.com' \"\$HF_REV\" '$ENTITIES_REPO' \"\$ENTITIES_REV\" \"\$ENTITIES_BLOB\" \"\$ENTITIES_TRANSPORT\" \"\$(date -u +%FT%TZ)\" > '$FLICKR_DIR/SOURCE.txt'
      unzip -t '$FLICKR_DIR/raw/hf/flickr30k-images.zip' >/dev/null
      unzip -t '$FLICKR_DIR/raw/entities/annotations.zip' >/dev/null
      unzip -q -n '$FLICKR_DIR/raw/hf/flickr30k-images.zip' -d '$FLICKR_DIR/images'
      unzip -q -n '$FLICKR_DIR/raw/entities/annotations.zip' -d '$FLICKR_DIR/entities'
      find '$FLICKR_DIR/raw' -type f -print0 | sort -z | xargs -0 sha256sum > '$FLICKR_DIR/SHA256SUMS'
      printf 'jpeg_count='; find '$FLICKR_DIR/images' -type f -iname '*.jpg' | wc -l
      printf 'sentence_count='; find '$FLICKR_DIR/entities' -path '*/Sentences/*.txt' -type f | wc -l
      printf 'xml_count='; find '$FLICKR_DIR/entities' -path '*/Annotations/*.xml' -type f | wc -l
      du -sh '$FLICKR_DIR'"
    ;;
  download-teachers)
    remote "set -eu
      export HF_HOME='$PROJECT_ROOT/hf_cache'
      export HF_ENDPOINT='https://hf-mirror.com'
      mkdir -p '$TEACHER_DIR'
      while IFS='|' read -r id repo revision license; do
        dest='$TEACHER_DIR'/\"\$id\"
        mkdir -p \"\$dest\"
        printf 'repository=%s\\nrevision=%s\\nendpoint=%s\\nlicense=%s\\ndownloaded_at_utc=%s\\n' \"\$repo\" \"\$revision\" 'https://hf-mirror.com' \"\$license\" \"\$(date -u +%FT%TZ)\" > \"\$dest/SOURCE.txt\"
        hf download \"\$repo\" --revision \"\$revision\" --local-dir \"\$dest\"
        find \"\$dest\" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > \"\$dest/SHA256SUMS\"
        (cd \"\$dest\" && sha256sum -c SHA256SUMS >/dev/null)
        du -sh \"\$dest\"
      done <<'TEACHERS'
stable-diffusion-v1-5|runwayml/stable-diffusion-v1-5|451f4fe16113bff5a5d2269ed5ad43b0592e9a14|creativeml-openrail-m
pixart-alpha-xl|PixArt-alpha/PixArt-XL-2-1024-MS|b89adadeccd9ead2adcb9fa2825d3fabec48d404|openrail++
pixart-sigma-xl|PixArt-alpha/PixArt-Sigma-XL-2-1024-MS|e102b3591cc82e97071b8b4cb90d834d0c487207|openrail++
playground-v2-5|playgroundai/playground-v2.5-1024px-aesthetic|1e032f13f2fe6db2dc49947dbdbd196e753de573|playground-v2dot5-community
TEACHERS"
    ;;
  -h|--help|help|'') usage ;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
