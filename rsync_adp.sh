#!/bin/bash
# rsync_adp.sh — sync this repo to the cluster over rsync/ssh
#
# Usage:
#   bash rsync_adp.sh              # sync (dry-run first by default)
#   bash rsync_adp.sh --go         # actual sync
#   bash rsync_adp.sh --pull       # pull results back from the cluster
#
# The .gitignore file drives the exclude list via --filter=':- .gitignore',
# so anything ignored locally (logs, outputs, caches) is never uploaded.

set -euo pipefail

REMOTE_USER="hyonghua"
REMOTE_HOST="login.lst.uni-saarland.de"
REMOTE_DIR="/nethome/hyonghua/adp"              # code dir (git-managed)
REMOTE_OUT="/scratch/${REMOTE_USER}/adp"        # checkpoints / logs / results (scratch)

MODE=${1:---dry}

case "$MODE" in
  --dry)
    echo "== DRY RUN (no files changed). Use --go to sync for real. =="
    rsync -avz --progress \
      --filter=":- .gitignore" \
      --exclude='.git/' \
      --exclude='_untracked/' \
      --exclude='condor_logs/' \
      --delete-during \
      ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    ;;
  --go)
    echo "== Syncing adp -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} =="
    rsync -avz --progress \
      --filter=":- .gitignore" \
      --exclude='.git/' \
      --exclude='_untracked/' \
      --exclude='condor_logs/' \
      --delete-during \
      ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    echo "== Sync complete =="
    ;;
  --pull)
    echo "== Pulling results back: ${REMOTE_OUT}/ -> ./results_out =="
    rsync -avz --progress \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUT}/" ./results_out/
    echo "== Pull complete =="
    ;;
  *)
    echo "Unknown mode: $MODE (use --dry, --go, or --pull)"; exit 1
    ;;
esac
