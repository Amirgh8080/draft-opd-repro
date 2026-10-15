#!/usr/bin/env bash
# Paper Table 1, Qwen3-4B row, on one H200 -- with the authors' eval settings:
# upstream benchmark_sglang.py, SGLang DFlash fork, attention backend fa3 (SM90),
# concurrency 1, tp 1, bf16, mem_fraction_static 0.75, temperature 0,
# max_new_tokens 2048 (non-thinking) / 8192 (thinking).
#
# On an H200 both the acceptance length tau AND the speedups are comparable to
# the paper, since this is the hardware the paper measured on.
#
#   bash h200/03_eval_table1.sh                          # released checkpoints
#   MODE=thinking bash h200/03_eval_table1.sh
#   DRAFTS="mine=/workspace/checkpoints/.../draft_model" SKIP_BASELINE=1 bash h200/03_eval_table1.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODE=${MODE:-nothinking}            # nothinking | thinking
TEMP=${TEMP:-0.0}
if [[ "$MODE" == "thinking" ]]; then
  THINK_FLAG=(--enable-think); MAX_NEW=${MAX_NEW:-8192}; OPD_ID_MODE=$OPD_THINK_ID
else
  THINK_FLAG=();               MAX_NEW=${MAX_NEW:-2048}; OPD_ID_MODE=$OPD_INS_ID
fi
# label=model pairs, space separated. Hub ids resolve from the offline cache.
DRAFTS=${DRAFTS:-"dflash=$DFLASH_ID draftopd=$OPD_ID_MODE"}
# Table 1 columns. Counts follow the authors' eval_config/*.yaml.
DATASETS=${DATASETS:-"gsm8k:128 math500:128 aime25:30 mbpp:128 humaneval:128 swe-bench:128 mt-bench:80"}
OUT_DIR=${OUT_DIR:-"$LOG_ROOT/eval/${MODE}_temp${TEMP}"}
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

sm=$(python -c "import torch; print('%d%d' % torch.cuda.get_device_capability())")
[[ "$sm" == "90" ]] || echo "WARN: SM$sm, not SM90 -- the bench will drop fa3 and use flashinfer; speedups will not be comparable to the paper."

cd "$UPSTREAM_DIR/diffusion/dflash"

bench() {   # bench <label> [benchmark args...]
  local label="$1"; shift
  local out="$OUT_DIR/${label}.md"
  echo; echo "### $label -> $out"
  # shellcheck disable=SC2086
  python benchmark_sglang.py \
    --target-model "$TARGET_ID" \
    --dataset-name $DATASETS \
    --concurrencies 1 \
    --attention-backends fa3 \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.75 \
    --max-new-tokens "$MAX_NEW" \
    --temp "$TEMP" \
    "${THINK_FLAG[@]}" \
    --output-md "$out" \
    "$@" 2>&1 | tee "$out.run.log"
}

# One baseline shared by every drafter, so all speedups have the same denominator.
if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
  bench baseline --only-baseline
fi
for pair in $DRAFTS; do
  bench "${pair%%=*}" --draft-model "${pair#*=}" --speculative-algorithm DFLASH
done

python "$H200_DIR/summarize_table1.py" "$OUT_DIR" --mode "$MODE" | tee "$OUT_DIR/summary.md"
