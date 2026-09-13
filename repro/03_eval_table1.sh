#!/usr/bin/env bash
# Reproduce the Qwen3-4B row of Table 1 on a single GPU.
#
# Default = "Thinking Mode Disabled, Temperature = 0, Q3-4B" (paper Table 1).
# Paper targets for that row:
#                GSM8K MATH500 AIME25  MBPP HumanEval SWE-Lite MT-Bench | tau
#   DFlash       5.36x  6.35x   5.93x  5.00x  5.19x    3.05x    3.01x   | 6.04
#   Draft-OPD    6.22x  7.22x   6.39x  5.40x  5.65x    3.18x    3.09x   | 6.60
#
# tau (acceptance length) is hardware-independent -- that is your replication
# target. The speedup multipliers depend on the memory-bandwidth/compute ratio
# and will NOT match H200 numbers on a 3090. Report them as "our hardware".
#
# NOTE: jobs run SEQUENTIALLY here. Do not use launch_sglang_bench_jobs.py with
# two jobs pinned to gpu 0 -- it Popen's them in parallel (line 317) and they
# will fight over the same 24 GB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/diffusion/dflash"

export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_ALLOW_CODE_EVAL=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TARGET=${TARGET:-Qwen/Qwen3-4B}
MODE=${MODE:-nothinking}          # nothinking | thinking
TEMP=${TEMP:-0.0}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/repro/eval/logs"}
mkdir -p "$LOG_DIR"

if [[ "$MODE" == "thinking" ]]; then
  THINK_FLAG=(--enable-think)
  MAX_NEW=${MAX_NEW:-8192}                       # paper: 8192 for thinking
  OPD_DRAFT=${OPD_DRAFT:-bingyang-lei/Qwen3-4B-Thinking-Draft-OPD}
else
  THINK_FLAG=()
  MAX_NEW=${MAX_NEW:-2048}                       # paper: 2048 for non-thinking
  OPD_DRAFT=${OPD_DRAFT:-bingyang-lei/Qwen3-4B-Ins-Draft-OPD}
fi
DFLASH_DRAFT=${DFLASH_DRAFT:-z-lab/Qwen3-4B-DFlash-b16}

# Table 1 columns, in order. Sample counts follow the authors' eval configs.
DATASETS=${DATASETS:-"gsm8k:128 math500:128 aime25:30 mbpp:128 humaneval:128 swe-bench:128 mt-bench:80"}

run_one () {
  local label="$1" draft="$2" extra_flags="${3:-}"
  local out="$LOG_DIR/${MODE}_temp${TEMP}_${label}.md"
  echo
  echo "###################################################################"
  echo "# $label"
  echo "#   draft      : ${draft:-<none, baseline only>}"
  echo "#   mode       : $MODE   temp: $TEMP   max_new_tokens: $MAX_NEW"
  echo "#   report     : $out"
  echo "###################################################################"
  # shellcheck disable=SC2086
  python benchmark_sglang.py \
    --target-model "$TARGET" \
    ${draft:+--draft-model "$draft"} \
    ${extra_flags} \
    --speculative-algorithm DFLASH \
    --dataset-name ${DATASETS} \
    --concurrencies 1 \
    --attention-backends flashinfer \
    --max-new-tokens "$MAX_NEW" \
    --temp "$TEMP" \
    --mem-fraction-static 0.80 \
    --max-running-requests 1 \
    --tp-size 1 \
    --dtype bfloat16 \
    "${THINK_FLAG[@]}" \
    --output-md "$out"
}

# 1. Vanilla autoregressive baseline -- the 1.00x denominator for every speedup.
run_one "baseline" "" "--only-baseline"

# 2. DFlash drafter (the paper's strongest prior-work baseline).
run_one "dflash" "$DFLASH_DRAFT"

# 3. Draft-OPD drafter (the paper's method).
run_one "draftopd" "$OPD_DRAFT"

echo
echo "===================================================================="
echo "Done. Reports in: $LOG_DIR"
echo "Compare the tau / accept-length column across the dflash and draftopd"
echo "reports -- that is the hardware-independent replication signal."
echo "===================================================================="
