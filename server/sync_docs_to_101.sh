#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly HOST="${REMOTE_HOST:-vla101}"
readonly REMOTE_ROOT="${REMOTE_ROOT:-/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention}"
readonly DRY_RUN="${DRY_RUN:-0}"

RSYNC_FLAGS=(-a -v --itemize-changes)
readonly MD_FILTERS=(
  --include='*/'
  --include='*.md'
  --exclude='*'
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
  echo 'DRY_RUN=1: no remote file will be changed'
fi

sync_markdown_tree() {
  local relative_path="$1"
  shift
  echo "syncing Markdown: $relative_path"
  rsync "${RSYNC_FLAGS[@]}" --delete "$@" \
    "$ROOT/$relative_path/" "$HOST:$REMOTE_ROOT/$relative_path/"
}

echo '[1/8] syncing doc/ (local is source of truth)'
rsync "${RSYNC_FLAGS[@]}" --delete "$ROOT/doc/" "$HOST:$REMOTE_ROOT/doc/"

echo '[2/8] syncing experiments/*.md'
sync_markdown_tree experiments "${MD_FILTERS[@]}"

echo '[3/8] syncing paper_draft/*.md'
sync_markdown_tree paper_draft "${MD_FILTERS[@]}"

echo '[4/8] syncing references/*.md (excluding cloned repos/)'
sync_markdown_tree references --exclude='/repos/' "${MD_FILTERS[@]}"

echo '[5/8] syncing server/*.md'
sync_markdown_tree server "${MD_FILTERS[@]}"

echo '[6/8] syncing tests/*.md'
sync_markdown_tree tests "${MD_FILTERS[@]}"

echo '[7/8] syncing PROGRESS.md, README.md, configs/vla/README.md'
rsync "${RSYNC_FLAGS[@]}" "$ROOT/experiment_workspace/PROGRESS.md" "$HOST:$REMOTE_ROOT/experiment_workspace/PROGRESS.md"
rsync "${RSYNC_FLAGS[@]}" "$ROOT/README.md" "$HOST:$REMOTE_ROOT/README.md"
rsync "${RSYNC_FLAGS[@]}" "$ROOT/configs/vla/README.md" "$HOST:$REMOTE_ROOT/configs/vla/README.md"

echo '[8/8] marking remote docs read-only snapshot'
if [[ "$DRY_RUN" == "1" ]]; then
  echo 'DRY_RUN=1: DEPLOYMENT.txt unchanged'
else
  ssh "$HOST" "printf 'source_commit=HEAD_snapshot\ndocs_source=local_git\ndocs_readonly=1\ndocs_scope=doc,experiments,paper_draft,references_without_repos,server,tests,README,PROGRESS,configs_vla_README\ndocs_synced_at=%s\n' \"\$(date -Is)\" > '$REMOTE_ROOT/DEPLOYMENT.txt'"
fi
echo SYNC_OK
