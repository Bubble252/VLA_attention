#!/usr/bin/env bash
# Run on 101. Dedicated environment; never imports a model or starts CUDA work.
set -euo pipefail
VLA_PREP_ROOT=${VLA_PREP_ROOT:-/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace}
mkdir -p "$VLA_PREP_ROOT"/{repos,envs,models,data,logs,artifacts,tmp,pip_cache,hf_cache}
exec 9>"$VLA_PREP_ROOT/preparation.lock"
flock -n 9 || { echo 'PREPARATION_ALREADY_RUNNING'; exit 3; }
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export PIP_CACHE_DIR="$VLA_PREP_ROOT/pip_cache" TMPDIR="$VLA_PREP_ROOT/tmp"
export HF_HOME="$VLA_PREP_ROOT/hf_cache" TOKENIZERS_PARALLELISM=false
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=60
trap 'code=$?; printf "%s\n" "$code" > "$VLA_PREP_ROOT/artifacts/preparation.exit"' EXIT
checkout() {
  local name=$1 url=$2 revision=$3
  local destination="$VLA_PREP_ROOT/repos/$name"
  if [ ! -d "$destination/.git" ]; then
    mkdir -p "$destination"
    git -C "$destination" init
    git -C "$destination" remote add origin "$url"
  fi
  test -z "$(git -C "$destination" status --porcelain)" || {
    echo "DIRTY_SOURCE=$destination"; return 4;
  }
  if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$destination" -c http.version=HTTP/1.1 fetch --depth 1 origin "$revision"
  fi
  git -C "$destination" checkout --detach "$revision"
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
  printf '%s\t%s\t%s\n' "$name" "$url" "$revision" >> "$VLA_PREP_ROOT/artifacts/source_lock.tsv"
}
checkout openvla-oft https://github.com/moojink/openvla-oft.git e4287e94541f459edc4feabc4e181f537cd569a8
checkout libero https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e3132a39270c3a13ebe37270a43ece2a01
checkout transformers-oft https://github.com/moojink/transformers-openvla-oft.git bc339d9ad707454c0c115970db43c260067c61ab
checkout dlimp https://github.com/moojink/dlimp_openvla.git 040105d256bd28866cc6620621a3d5f7b6b91b46
checkout radio https://github.com/NVlabs/RADIO.git c0f37017930e9dda53f93424cf4bf39fc51f287e
VLA_PY="$VLA_PREP_ROOT/envs/oft/bin/python"
if [ ! -x "$VLA_PY" ]; then
  /usr/bin/python3.10 -m venv --without-pip "$VLA_PREP_ROOT/envs/oft"
fi
if ! "$VLA_PY" -m pip --version >/dev/null 2>&1; then
  # ensurepip is absent on this server. Install pip only inside the new venv.
  curl --fail --location --retry 5 --retry-all-errors https://bootstrap.pypa.io/get-pip.py -o "$VLA_PREP_ROOT/tmp/get-pip.py"
  sha256sum "$VLA_PREP_ROOT/tmp/get-pip.py" >> "$VLA_PREP_ROOT/artifacts/bootstrap_sha256.txt"
  "$VLA_PY" "$VLA_PREP_ROOT/tmp/get-pip.py" 'pip<26'
fi
"$VLA_PY" -m pip install 'pip<26' 'setuptools<81' wheel
# Pin NumPy for Torch 2.2 / TensorFlow 2.15 ABI compatibility. Direct VCS
# requirements are replaced with the frozen checkouts, without editing upstream.
"$VLA_PY" - "$VLA_PREP_ROOT" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1])
constraints=root/'artifacts/constraints.txt'
constraints.write_text('numpy==1.26.4\nhuggingface-hub<1\ntransformers==4.40.1\nwandb==0.17.0\naccelerate==0.30.1\nprotobuf==3.20.3\ntensorflow-metadata==1.15.0\n')
source=(root/'repos/openvla-oft/pyproject.toml').read_text()
import re
deps=re.search(r'dependencies = \[(.*?)\n\]',source,re.S).group(1)
requirements=[]
for line in deps.splitlines():
    line=line.strip()
    if line.startswith('#') or not line: continue
    dep=re.match(r'"([^"]+)"',line)
    if dep and ' @ git+' not in dep.group(1): requirements.append(dep.group(1))
(root/'artifacts/requirements-upstream.txt').write_text('\n'.join(requirements)+'\n')
PY
"$VLA_PY" -m pip install -c "$VLA_PREP_ROOT/artifacts/constraints.txt" -r "$VLA_PREP_ROOT/artifacts/requirements-upstream.txt"
"$VLA_PY" -m pip install -c "$VLA_PREP_ROOT/artifacts/constraints.txt" "$VLA_PREP_ROOT/repos/transformers-oft" "$VLA_PREP_ROOT/repos/dlimp"
"$VLA_PY" -m pip install --no-deps -e "$VLA_PREP_ROOT/repos/openvla-oft" -e "$VLA_PREP_ROOT/repos/libero"
"$VLA_PY" -m pip install -c "$VLA_PREP_ROOT/artifacts/constraints.txt" -r "$VLA_PREP_ROOT/repos/openvla-oft/experiments/robot/libero/libero_requirements.txt" pytest h5py pyyaml
"$VLA_PY" -m pip check
"$VLA_PY" -m pip freeze > "$VLA_PREP_ROOT/artifacts/pip_freeze.txt"
"$VLA_PY" - <<'PY'
import torch,transformers,peft,sys
print('CPU_IMPORT_OK',sys.version,torch.__version__,transformers.__version__,peft.__version__)
assert not torch.cuda.is_initialized()
PY
echo VLA_CPU_ENV_INSTALL_OK
# FlashAttention kernel validation, renderer and real P1 remain separate gates.
