#!/usr/bin/env bash
# Clone upstream Draft-OPD at the pinned commit and apply the launcher patch.
# Idempotent: an existing clone at the right commit is left alone.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PATCH="$HARNESS_ROOT/patches/001-launcher-fixes.patch"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  git clone "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi

# A clone owned by another uid (bind mount, root-built image) makes git abort
# with "dubious ownership", which later breaks the setuptools_scm build of sglang.
git -C "$UPSTREAM_DIR" rev-parse --git-dir >/dev/null 2>&1 \
  || git config --global --add safe.directory "$UPSTREAM_DIR"

head=$(git -C "$UPSTREAM_DIR" rev-parse HEAD)
if [[ "$head" != "$UPSTREAM_COMMIT" ]]; then
  if ! git -C "$UPSTREAM_DIR" diff --quiet; then
    echo "ERROR: $UPSTREAM_DIR is at $head with local changes; expected $UPSTREAM_COMMIT." >&2
    echo "       Move it aside or set UPSTREAM_DIR to a fresh path." >&2
    exit 1
  fi
  git -C "$UPSTREAM_DIR" fetch --quiet origin "$UPSTREAM_COMMIT" 2>/dev/null \
    || git -C "$UPSTREAM_DIR" fetch --quiet origin
  git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_COMMIT"
fi
echo "upstream: $UPSTREAM_DIR @ $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"

# The patch only touches the launcher: opt-in `pkill -9 python` (it would kill
# Jupyter, i.e. the container, on most GPU clouds), no GPU stress test pointing
# at the authors' cluster, overridable validation paths. No training math changes.
if git -C "$UPSTREAM_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch: already applied"
else
  git -C "$UPSTREAM_DIR" apply "$PATCH"
  echo "patch: applied $(basename "$PATCH")"
fi
