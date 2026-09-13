#!/usr/bin/env bash
# Convert a saved FSDP actor checkpoint into a standalone DFlash draft model
# that benchmark_sglang.py / repro/03_eval_table1.sh can load.
#
# The upstream verl/scripts/fsdp_to_dflash.sh hard-codes the authors' paths AND
# runs `rm -rf` on the actor dir afterwards. This wrapper does neither.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/verl"

: "${STEP_DIR:?Set STEP_DIR, e.g. checkpoints/verl-dflash-opd/3090-reduced/.../global_step_500}"
: "${REFERENCE_DRAFT_DIR:?Set REFERENCE_DRAFT_DIR to the initial drafter (z-lab/Qwen3-4B-DFlash-b16 snapshot)}"

ACTOR_DIR="${STEP_DIR}/actor"
TARGET_DIR="${TARGET_DIR:-${STEP_DIR}/draft_model}"

[[ -d "$ACTOR_DIR" ]] || { echo "ERROR: no actor dir at $ACTOR_DIR" >&2; exit 1; }

echo "actor     : $ACTOR_DIR"
echo "reference : $REFERENCE_DRAFT_DIR"
echo "output    : $TARGET_DIR"

python scripts/extract_dflash_draft_from_fsdp.py \
  --actor-dir "$ACTOR_DIR" \
  --reference-draft-dir "$REFERENCE_DRAFT_DIR" \
  --target-dir "$TARGET_DIR"

echo
echo "Done. Evaluate it against the DFlash baseline with:"
echo "  OPD_DRAFT=$TARGET_DIR bash repro/03_eval_table1.sh"
