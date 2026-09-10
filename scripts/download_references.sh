#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/references/papers"
REPO_DIR="$ROOT_DIR/references/repos"
LOG_FILE="$ROOT_DIR/references/download_failures.log"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7897}"
GIT_MIRROR_PREFIX="${GIT_MIRROR_PREFIX:-}"

mkdir -p "$PAPER_DIR" "$REPO_DIR"
: > "$LOG_FILE"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export GIT_TERMINAL_PROMPT=0

download_pdf() {
  local name="$1"
  local arxiv_id="$2"
  local out="$PAPER_DIR/${name}.pdf"
  if [[ -s "$out" ]]; then
    echo "exists: $out"
    return 0
  fi
  echo "download pdf: $name $arxiv_id"
  if ! curl -L --fail --retry 2 --connect-timeout 20 --proxy "$PROXY_URL" \
    "https://arxiv.org/pdf/${arxiv_id}" -o "$out"; then
    echo "PDF failed: $name https://arxiv.org/pdf/${arxiv_id}" >> "$LOG_FILE"
    rm -f "$out"
    return 1
  fi
}

clone_repo() {
  local name="$1"
  local url="$2"
  local out="$REPO_DIR/$name"
  if [[ -d "$out/.git" ]]; then
    echo "exists: $out"
    git -C "$out" rev-parse HEAD || true
    return 0
  fi
  clone_url="$url"
  if [[ -n "$GIT_MIRROR_PREFIX" ]]; then
    clone_url="${GIT_MIRROR_PREFIX%/}/${url#https://}"
  fi
  echo "clone repo: $name $clone_url"
  if ! git -c http.proxy="$PROXY_URL" -c https.proxy="$PROXY_URL" \
    clone --depth 1 "$clone_url" "$out"; then
    echo "REPO failed: $name $url" >> "$LOG_FILE"
    rm -rf "$out"
    return 1
  fi
  git -C "$out" rev-parse HEAD || true
}

download_pdf "lavender_2502.06814" "2502.06814" || true
download_pdf "spikingbrain_2509.05276" "2509.05276" || true
download_pdf "qwen25_vl_2502.13923" "2502.13923" || true
download_pdf "qwen2_vl_2409.12191" "2409.12191" || true
download_pdf "qwen3_vl_2511.21631" "2511.21631" || true
download_pdf "deepseek_vl2_2412.10302" "2412.10302" || true
download_pdf "openvla_2406.09246" "2406.09246" || true
download_pdf "pi0_2410.24164" "2410.24164" || true
download_pdf "lingbot_va_2601.21998" "2601.21998" || true
download_pdf "lingbot_vla_2601.18692" "2601.18692" || true
download_pdf "lingbot_vla_v2_2607.06403" "2607.06403" || true
download_pdf "libero_2306.03310" "2306.03310" || true
download_pdf "diffusion_policy_2303.04137" "2303.04137" || true
download_pdf "gla_2312.06635" "2312.06635" || true
download_pdf "attention_sinks_2309.17453" "2309.17453" || true
download_pdf "qwen_robotmanip_2606.17846" "2606.17846" || true

clone_repo "qwen2.5-vl" "https://github.com/QwenLM/Qwen2.5-VL.git" || true
clone_repo "qwen3-vl" "https://github.com/QwenLM/Qwen3-VL.git" || true
clone_repo "deepseek-vl2" "https://github.com/deepseek-ai/DeepSeek-VL2.git" || true
clone_repo "openvla" "https://github.com/openvla/openvla.git" || true
clone_repo "qwen-robotmanip" "https://github.com/QwenLM/Qwen-RobotManip.git" || true
clone_repo "openpi" "https://github.com/Physical-Intelligence/openpi.git" || true
clone_repo "libero" "https://github.com/Lifelong-Robot-Learning/LIBERO.git" || true
clone_repo "diffusion_policy" "https://github.com/real-stanford/diffusion_policy.git" || true
clone_repo "lingbot-va" "https://github.com/Robbyant/lingbot-va.git" || true
clone_repo "lingbot-vla" "https://github.com/Robbyant/lingbot-vla.git" || true
clone_repo "lingbot-vla-v2" "https://github.com/Robbyant/lingbot-vla-v2.git" || true

if compgen -G "$PAPER_DIR/*.pdf" > /dev/null; then
  (cd "$ROOT_DIR/references" && sha256sum papers/*.pdf > checksums.sha256)
fi

{
  echo "# Repository versions"
  echo
  for repo in "$REPO_DIR"/*; do
    if [[ -d "$repo/.git" ]]; then
      name="$(basename "$repo")"
      url="$(git -C "$repo" remote get-url origin 2>/dev/null || true)"
      head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
      echo "- $name: $head $url"
    fi
  done
} > "$ROOT_DIR/references/repo_versions.md"

echo "download log: $LOG_FILE"
if [[ -s "$LOG_FILE" ]]; then
  echo "some downloads failed; see $LOG_FILE"
  exit 1
fi
