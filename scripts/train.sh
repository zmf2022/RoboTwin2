#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_SRC="${1:?config yaml}"; shift
case "$CFG_SRC" in /*) ;; *) CFG_SRC="$PWD/$CFG_SRC" ;; esac

CFG_RUN="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CFG_RUN"' EXIT
sed "s|~/|$HOME/|g" "$CFG_SRC" > "$CFG_RUN"

# DataLoader workers pass each tensor to the main process as an fd; the default soft limit (1024) is too low
ulimit -n "$(ulimit -Hn)"

cd "$ROOT/lingbot-vla-v2"
bash train.sh tasks/vla/train_lingbotvla.py "$CFG_RUN" "$@"
