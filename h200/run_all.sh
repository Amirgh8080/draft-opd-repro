#!/usr/bin/env bash
# One-shot: preflight -> code -> install -> assets -> Table 1 eval of the
# released checkpoints -> paper-recipe training -> extract -> eval of your drafter.
#
#   bash h200/run_all.sh                 # everything
#   bash h200/run_all.sh --eval-only     # Phase 1 only: released checkpoints on H200
#   bash h200/run_all.sh --train-only    # Phase 2 only (resumes from the last checkpoint)
#   bash h200/run_all.sh --thinking      # also evaluate the thinking-mode rows
#
# Completed stages leave markers in $WORKSPACE/.state; rerunning resumes.
# Training is never marker-gated: rerunning it resumes from its last checkpoint.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

DO_EVAL=1; DO_TRAIN=1; THINKING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-only)  DO_TRAIN=0; shift ;;
    --train-only) DO_EVAL=0; shift ;;
    --thinking)   THINKING=1; shift ;;
    -h|--help)    sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

LOG="$LOG_ROOT/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date +%H:%M:%S)] $*"; }

stage() {  # stage <name> <command...>
  local name="$1"; shift
  if [[ -f "$STATE_DIR/$name.done" ]]; then log "skip $name (rm $STATE_DIR/$name.done to redo)"; return; fi
  log "start $name"; "$@"; touch "$STATE_DIR/$name.done"; log "done  $name"
}

installed() {
  python - "$UPSTREAM_DIR" <<'PYEOF' 2>/dev/null
import sys, sglang, verl, flash_attn
assert sglang.__file__.startswith(sys.argv[1])
PYEOF
}

log "workspace $WORKSPACE | upstream $UPSTREAM_DIR | log $LOG"
bash "$H200_DIR/preflight.sh" || { log "preflight failed"; exit 1; }

bash "$H200_DIR/00_get_upstream.sh"
if installed; then log "python stack already installed (image build or earlier run)"
else log "installing python stack"; bash "$H200_DIR/01_install.sh"; fi

stage prefetch bash "$H200_DIR/02_prefetch.sh"

if (( DO_EVAL )); then
  stage eval_released_nothinking bash "$H200_DIR/03_eval_table1.sh"
  if (( THINKING )); then
    stage eval_released_thinking env MODE=thinking bash "$H200_DIR/03_eval_table1.sh"
  fi
fi

if (( DO_TRAIN )); then
  bash "$H200_DIR/04_train_paper.sh"
  bash "$H200_DIR/05_extract_draft.sh"
  step_dir=$(ls -d "$CKPT_ROOT/${EXP_NAME:-qwen3-4b-ins-draftopd-h200x1}"/global_step_* | sort -V | tail -1)
  if [[ -f "$LOG_ROOT/eval/nothinking_temp0.0/baseline.md" ]]; then
    DRAFTS="mine=$step_dir/draft_model" SKIP_BASELINE=1 bash "$H200_DIR/03_eval_table1.sh"
  else
    DRAFTS="mine=$step_dir/draft_model" bash "$H200_DIR/03_eval_table1.sh"
  fi
fi

log "finished. Table 1 summary: $LOG_ROOT/eval/nothinking_temp0.0/summary.md"
