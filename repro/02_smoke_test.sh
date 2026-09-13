#!/usr/bin/env bash
# 20-minute end-to-end smoke test: does the DFlash speculative stack run at all
# on this 3090? Runs 4 GSM8K prompts, 256 new tokens, drafter only (no baseline).
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
DRAFT=${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}
OUT=${OUT:-"$REPO_ROOT/repro/eval/logs/smoke.md"}
mkdir -p "$(dirname "$OUT")"

echo "target : $TARGET"
echo "draft  : $DRAFT"
echo "backend: flashinfer  (fa3/fa4 are SM90+/SM100+ only; benchmark_sglang.py:797 drops them)"
echo

python benchmark_sglang.py \
  --target-model "$TARGET" \
  --draft-model "$DRAFT" \
  --speculative-algorithm DFLASH \
  --dataset-name "gsm8k:4" \
  --concurrencies 1 \
  --attention-backends flashinfer \
  --max-new-tokens 256 \
  --temp 0.0 \
  --mem-fraction-static 0.80 \
  --max-running-requests 1 \
  --tp-size 1 \
  --dtype bfloat16 \
  --output-md "$OUT"

echo
echo "===================================================================="
echo "Smoke test finished. Report: $OUT"
echo
echo "What to check in the report:"
echo "  * an 'accept length' / tau value around 5-7  -> the drafter is working"
echo "  * tau ~= 1.0                                 -> drafter is NOT engaging;"
echo "                                                  check the DFLASH block size"
echo "                                                  and draft/target pairing"
echo "  * wall-clock speedup will NOT match the paper (H200 vs 3090). Expected."
echo "===================================================================="
