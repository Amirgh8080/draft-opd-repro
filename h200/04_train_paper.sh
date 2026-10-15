#!/usr/bin/env bash
# Draft-OPD training with the paper's recipe on ONE H200.
#
# Upstream run (run_qwen_gsm8k_forward-ins.sh defaults): 7 student GPUs, 21
# prompts per rollout step, 10 rollout steps accumulated per update phase, full
# 16K prompts, 8 epochs, 512 + 4096 tokens, LR 3e-4 cosine, warmup 0.05,
# anchor stride 1, gamma 0.8, fp32 model init, no offload, no gradient
# checkpointing, flash_attention_2 + packed sequences, fa3 rollout, seed 42.
#
# Here every one of those is kept. The only change is STUDENT_WORLD_SIZE=7 -> 1.
# That is sound because in this verl fork:
#   * the global batch is world-size independent: data.train_batch_size=21 and
#     trainer.update_accumulation_steps=10 are counted on the driver;
#   * the loss is normalised by global token count x dp_size before DP's
#     gradient average (workers/utils/losses.py), so a 21-sample mini-batch on
#     one rank gives the same gradient as 3 samples on each of 7 ranks;
#   * teacher log-probs come from the frozen Qwen3-4B inside the composed
#     student (teacher_logprob_source auto -> composed_main), so no separate
#     teacher GPU exists at any world size;
#   * the dataloader shuffle (seed 42) runs on the driver.
# What does differ, and is unavoidable with one GPU, is listed in h200/README.md.
#
#   bash h200/04_train_paper.sh                              # the full run
#   TIMING_PROBE=1 bash h200/04_train_paper.sh               # 20 steps, measure s/step
#   bash h200/04_train_paper.sh trainer.some_key=value       # extra hydra overrides
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RUN_GPU_STRESS_TEST_ON_EXIT=False KILL_STALE_PROCS=False

MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-$(local_snapshot "$TARGET_ID")}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-$(local_snapshot "$DFLASH_ID")}
TRAIN_JSONL=${TRAIN_JSONL:-$UPSTREAM_DIR/data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl}
VAL=("$DATA_DIR/aime24_30_user_prompt.jsonl" "$DATA_DIR/gsm8k_128_user_prompt.jsonl"
     "$DATA_DIR/math500_128_user_prompt.jsonl" "$DATA_DIR/mbpp_128_user_prompt.jsonl")
for f in "$TRAIN_JSONL" "${VAL[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run h200/02_prefetch.sh)" >&2; exit 1; }
done
VAL_FILES="['${VAL[0]}','${VAL[1]}','${VAL[2]}','${VAL[3]}']"

python -c "import flash_attn" 2>/dev/null || {
  echo "ERROR: flash_attn not importable. The paper path loads the target with" >&2
  echo "       flash_attention_2 and packed sequences. Rerun h200/01_install.sh." >&2
  exit 1; }

# A fixed name: the upstream default embeds today's date, so a run resumed the
# next day would start a fresh checkpoint dir instead of resuming.
EXP_NAME=${EXP_NAME:-qwen3-4b-ins-draftopd-h200x1}

EXTRA=()
if [[ "${TIMING_PROBE:-0}" == "1" ]]; then
  # Shortens the run AND the cosine schedule. Use only to measure s/step.
  EXP_NAME="${EXP_NAME}-timing"
  EXTRA+=(trainer.total_training_steps=20 trainer.val_before_train=False)
  SAVE_FREQ=-1
fi

CKPT_DIR=${CKPT_DIR:-$CKPT_ROOT/$EXP_NAME}
RUN_LOG_DIR="$LOG_ROOT/train/$EXP_NAME"
mkdir -p "$CKPT_DIR" "$RUN_LOG_DIR"
export WANDB_DIR=${WANDB_DIR:-$RUN_LOG_DIR/wandb}   # upstream logs wandb offline

# Checkpoint cadence is the one knob changed for practical reasons: upstream
# saves every 500 steps from step 5000, fine for a 1-2 day 8-GPU run, not for a
# single-GPU run measured in weeks. Saving does not touch the optimisation.
# Each checkpoint holds the full composed actor (fp32 target + drafter + Adam),
# roughly 25-30 GB, hence the retention limit. The final step is always saved.
SAVE_FREQ=${SAVE_FREQ:-250}
SAVE_START_STEP=${SAVE_START_STEP:-250}
KEEP_CKPTS=${KEEP_CKPTS:-4}

cat <<INFO | tee "$RUN_LOG_DIR/launch_info.txt"
================================================================================
 Draft-OPD, paper recipe, 1x H200            $(date -Is)
   upstream     : $UPSTREAM_DIR @ $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)
   target       : $MAIN_MODEL_PATH
   init drafter : $DRAFT_MODEL_PATH
   train data   : $TRAIN_JSONL ($(wc -l < "$TRAIN_JSONL") prompts)
   batch        : 21 prompts/step x 10 accumulated steps, 8 epochs, LR 3e-4
   lengths      : prompt 512, response 4096
   checkpoints  : $CKPT_DIR (every $SAVE_FREQ steps from $SAVE_START_STEP, keep $KEEP_CKPTS)
   logs         : $RUN_LOG_DIR
================================================================================
INFO

cd "$UPSTREAM_DIR"
# Values below equal the upstream -ins.sh defaults, set explicitly so the launch
# log records them. Exceptions: STUDENT_WORLD_SIZE (7 -> 1), the checkpoint
# cadence above, and the validation speed test, which upstream sizes as
# 7 workers x 2 samples = 14 per benchmark; 1 worker x 14 keeps the sample count.
# Validation never feeds back into training.
MAIN_MODEL_PATH="$MAIN_MODEL_PATH" \
DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" \
TRAIN_JSONL="$TRAIN_JSONL" \
LR=3e-4 \
train_epochs=8 \
STUDENT_WORLD_SIZE=1 \
TEACHER_WORLD_SIZE=1 \
TRAIN_PROMPT_BSZ=21 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT=512 \
MAX_RESPONSE_LENGTH=4096 \
ENABLE_THINKING=False \
DISTILLATION_LOSS_MODE=k3 \
FORWARD_KL_WEIGHT=1.0 \
REVERSE_KL_WEIGHT=0.0 \
REJECTED_DRAFT_USE_REVERSE_KL=True \
REJECTED_DRAFT_POSITION_DECAY_ENABLED=True \
REJECTED_DRAFT_POSITION_DECAY=0.8 \
stream_weight=1.0 \
rejected_draft_stream_weight=1.0 \
DFLASH_LM_HEAD_CHUNK_SIZE=512 \
RANDOM_RESPONSE_ANCHOR_ENABLED=False \
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.2} \
TEST_FREQ=${TEST_FREQ:-250} \
SAVE_FREQ="$SAVE_FREQ" \
SAVE_START_STEP="$SAVE_START_STEP" \
ROLLOUT_SPEED_TEST_WORKER_COUNT=1 \
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=14 \
EXP_NAME="$EXP_NAME" \
CKPT_DIR="$CKPT_DIR" \
bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh \
  "data.val_files=$VAL_FILES" \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep="$KEEP_CKPTS" \
  "${EXTRA[@]}" \
  "$@" 2>&1 | tee -a "$RUN_LOG_DIR/train.log"
