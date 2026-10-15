#!/usr/bin/env bash
# FSDP actor checkpoint -> standalone DFlash drafter loadable by the bench.
#
#   bash h200/05_extract_draft.sh                       # latest step of the default run
#   STEP_DIR=/workspace/checkpoints/<exp>/global_step_6000 bash h200/05_extract_draft.sh
#
# Unlike upstream verl/scripts/fsdp_to_dflash.sh, this never deletes the actor dir.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

EXP_NAME=${EXP_NAME:-qwen3-4b-ins-draftopd-h200x1}
if [[ -z "${STEP_DIR:-}" ]]; then
  STEP_DIR=$(ls -d "$CKPT_ROOT/$EXP_NAME"/global_step_* 2>/dev/null | sort -V | tail -1 || true)
  [[ -n "$STEP_DIR" ]] || { echo "ERROR: no checkpoints under $CKPT_ROOT/$EXP_NAME" >&2; exit 1; }
fi
REFERENCE_DRAFT_DIR=${REFERENCE_DRAFT_DIR:-$(local_snapshot "$DFLASH_ID")}
OUT=${OUT:-$STEP_DIR/draft_model}

echo "actor     : $STEP_DIR/actor"
echo "reference : $REFERENCE_DRAFT_DIR"
echo "output    : $OUT"
python "$UPSTREAM_DIR/verl/scripts/extract_dflash_draft_from_fsdp.py" \
  --actor-dir "$STEP_DIR/actor" \
  --reference-draft-dir "$REFERENCE_DRAFT_DIR" \
  --target-dir "$OUT"

echo
echo "Evaluate against the released checkpoints (reuses their baseline):"
echo "  DRAFTS=\"mine=$OUT\" SKIP_BASELINE=1 bash h200/03_eval_table1.sh"
