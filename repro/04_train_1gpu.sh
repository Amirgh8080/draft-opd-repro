#!/usr/bin/env bash
# Draft-OPD training on a single RTX 3090 (24 GB, SM86).
#
# WHY ONE GPU IS ENOUGH (structurally):
#   distillation.teacher_logprob_source defaults to `auto`. With a composed
#   DFlash student whose teacher path == main model path, teacher_source.py:95
#   resolves it to `composed_main`: teacher log-probs come from the frozen main
#   model already inside the student module. requires_external_teacher() then
#   returns False and main_ppo.py:180 never allocates a teacher GPU pool.
#   Combined with rollout.free_cache_engine=True (the SGLang engine releases its
#   weights/KV during each training step), everything fits on one device.
#
# SCALE presets -- pick with SCALE=...
#   probe    64 prompts,  1 epoch,  512 tok   ~30 min   "does it run + not OOM"
#   reduced  2000 prompts, 2 epochs, 1024 tok ~1 day    thesis-scale experiment
#   paper    16000 prompts, 8 epochs, 4096 tok          see the ETA warning below
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCALE=${SCALE:-probe}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export RUN_GPU_STRESS_TEST_ON_EXIT=False

FULL_TRAIN_JSONL="$REPO_ROOT/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl"

case "$SCALE" in
  probe)   N_PROMPTS=64;    EPOCHS=1; RESP_LEN=512;  ANCHOR_STRIDE=4 ;;
  reduced) N_PROMPTS=2000;  EPOCHS=2; RESP_LEN=1024; ANCHOR_STRIDE=2 ;;
  paper)   N_PROMPTS=16000; EPOCHS=8; RESP_LEN=4096; ANCHOR_STRIDE=1 ;;
  *) echo "ERROR: SCALE must be probe | reduced | paper (got '$SCALE')" >&2; exit 1 ;;
esac
# Allow explicit overrides of any preset value.
N_PROMPTS=${N_PROMPTS_OVERRIDE:-$N_PROMPTS}
EPOCHS=${EPOCHS_OVERRIDE:-$EPOCHS}
RESP_LEN=${RESP_LEN_OVERRIDE:-$RESP_LEN}
ANCHOR_STRIDE=${ANCHOR_STRIDE_OVERRIDE:-$ANCHOR_STRIDE}

# ---------------------------------------------------------------- preflight --
[[ -f "$FULL_TRAIN_JSONL" ]] || { echo "ERROR: missing $FULL_TRAIN_JSONL" >&2; exit 1; }
for f in aime24_30 gsm8k_128 math500_128 mbpp_128; do
  [[ -f "$REPO_ROOT/repro/data/${f}_user_prompt.jsonl" ]] || {
    echo "ERROR: missing repro/data/${f}_user_prompt.jsonl" >&2
    echo "       Run: python repro/make_val_jsonl.py --out-dir repro/data" >&2
    exit 1; }
done
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH (e.g. the local snapshot of z-lab/Qwen3-4B-DFlash-b16)}"
: "${MAIN_MODEL_PATH:?Set MAIN_MODEL_PATH (e.g. the local snapshot of Qwen/Qwen3-4B)}"

if [[ "$SCALE" == "paper" ]]; then
  cat >&2 <<'WARN'
================================================================================
SCALE=paper on a single RTX 3090.

The paper's run is 16K prompts x 8 epochs = 128K speculative rollouts at up to
4096 tokens, plus replay forward passes from every draft anchor, on 8x H200.
One 3090 is roughly 1/15th of an H200 for this workload, and you have 1 instead
of 8 -- on the order of 100x the wall clock. Estimate: several months, and the
run must survive that long without an OOM, a driver reset, or a power blip.

trainer.resume_mode is 'disable' in the upstream script. For a run of this
length, override it: trainer.resume_mode=auto

This script will proceed -- it is your call. Ctrl-C within 15s to reconsider.
================================================================================
WARN
  sleep 15
fi

# ------------------------------------------------------- training subset ----
TRAIN_JSONL="$REPO_ROOT/repro/data/train_${SCALE}_${N_PROMPTS}.jsonl"
if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Building $N_PROMPTS-prompt training subset -> $TRAIN_JSONL"
  python - "$FULL_TRAIN_JSONL" "$TRAIN_JSONL" "$N_PROMPTS" <<'PYEOF'
import json, random, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l for l in open(src, encoding="utf-8") if l.strip()]
random.Random(42).shuffle(rows)          # stratifies across aops/code/math/gsm8k
rows = rows[:n]
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(rows)
from collections import Counter
c = Counter(json.loads(r)["data_source"] for r in rows)
print(f"  {len(rows)} prompts: {dict(c)}")
PYEOF
fi

VAL_FILES="['$REPO_ROOT/repro/data/aime24_30_user_prompt.jsonl','$REPO_ROOT/repro/data/gsm8k_128_user_prompt.jsonl','$REPO_ROOT/repro/data/math500_128_user_prompt.jsonl','$REPO_ROOT/repro/data/mbpp_128_user_prompt.jsonl']"

cat <<INFO

================================================================================
 Draft-OPD single-GPU training
   scale              : $SCALE  ($N_PROMPTS prompts, $EPOCHS epochs, ${RESP_LEN} resp tokens)
   target (frozen)    : $MAIN_MODEL_PATH
   draft  (trained)   : $DRAFT_MODEL_PATH
   teacher            : composed_main (no separate teacher GPU)
   anchor stride      : $ANCHOR_STRIDE  (1 = every draft block, as in the paper)
   checkpoints        : verl/checkpoints/verl-dflash-opd/
================================================================================

INFO

# ------------------------------------------------------------------ launch --
# Memory knobs for 24 GB, in the order to relax them if you OOM:
#   1. lower TEACHER_GPU_MEMORY_UTILIZATION (rollout engine weights + KV)
#   2. lower RESP_LEN
#   3. raise ANCHOR_STRIDE (fewer replay positions per sample)
MAIN_MODEL_PATH="$MAIN_MODEL_PATH" \
DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" \
TRAIN_JSONL="$TRAIN_JSONL" \
LR=${LR:-3e-4} \
train_epochs="$EPOCHS" \
STUDENT_WORLD_SIZE=1 \
TEACHER_WORLD_SIZE=1 \
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-4} \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT=512 \
MAX_RESPONSE_LENGTH="$RESP_LEN" \
ENABLE_THINKING=${ENABLE_THINKING:-False} \
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.55} \
REJECTED_DRAFT_POSITION_DECAY=0.8 \
TEST_FREQ=${TEST_FREQ:-50} \
SAVE_FREQ=${SAVE_FREQ:-100} \
SAVE_START_STEP=${SAVE_START_STEP:-0} \
EXP_NAME=${EXP_NAME:-"3090-${SCALE}"} \
bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh \
  "data.val_files=$VAL_FILES" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_stride="$ANCHOR_STRIDE" \
  trainer.resume_mode=${RESUME_MODE:-auto} \
  "$@"
