#!/usr/bin/env bash
# Install the Draft-OPD Python stack into the current interpreter.
#
# Written for the base image lmsysorg/sglang:v0.5.8-cu129-amd64 (Python 3.12,
# torch 2.9.1+cu129, CUDA 12.9 toolkit with nvcc). The Dockerfile runs it at
# build time; on a GPU cloud where you pick an image instead of building one,
# start a pod from that image and run this once.
#
# Unlike the 24 GB harness in repro/, nothing is optional here: flash-attn is
# installed so the frozen target runs with flash_attention_2 + packed sequences,
# exactly as the upstream launcher configures it.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PY=${PYTHON:-python3}
CONSTRAINTS="$H200_DIR/constraints.txt"
export PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 bases mark system Python as externally managed
PIP() { "$PY" -m pip install --retries 5 --timeout 120 -c "$CONSTRAINTS" "$@"; }

[[ -d "$UPSTREAM_DIR/sglang-dflash/python" ]] || {
  echo "ERROR: $UPSTREAM_DIR is not a Draft-OPD clone. Run h200/00_get_upstream.sh first." >&2; exit 1; }

pyver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ "$pyver" == "3.12" ]] || echo "WARN: python $pyver; this recipe is verified on 3.12 only." >&2

echo "=== system libraries ==============================================="
# torchcodec (hard sglang dependency) fails at import, not install, without FFmpeg.
if command -v apt-get >/dev/null && [[ $(id -u) -eq 0 ]]; then
  missing=()
  for p in ffmpeg git libnuma-dev; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if (( ${#missing[@]} )); then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
  fi
fi

echo "=== sglang-dflash (replaces the image's stock sglang) =============="
"$PY" -m pip uninstall -y sglang >/dev/null 2>&1 || true
PIP -e "$UPSTREAM_DIR/sglang-dflash/python"
PIP cachetools

# The fork pins flashinfer 0.6.4; the image ships a flashinfer-jit-cache built
# for its own (older) flashinfer, and flashinfer refuses to import on a version
# mismatch. Match the cache to 0.6.4, or drop it (kernels then JIT-compile on
# first use, which needs nvcc -- present in this image).
jit=$("$PY" -m pip show flashinfer-jit-cache 2>/dev/null | awk '/^Version:/ {print $2}')
if [[ -n "$jit" && "$jit" != 0.6.4* ]]; then
  echo "flashinfer-jit-cache $jit != 0.6.4; replacing"
  "$PY" -m pip install --no-deps "flashinfer-jit-cache==0.6.4" \
      --index-url "https://flashinfer.ai/whl/cu129" \
    || "$PY" -m pip uninstall -y flashinfer-jit-cache
fi

"$PY" - <<'PYEOF'
import sglang, os
from packaging import version
v = sglang.__version__
print(f"sglang {v} from {os.path.dirname(sglang.__file__)}")
assert "sglang-dflash" in sglang.__file__, "stock sglang still shadows the fork"
assert version.parse(v) >= version.parse("0.5.5"), f"sglang {v} < 0.5.5; verl asserts on this at rollout"
PYEOF

echo "=== verl (editable, --no-deps, deps under constraints) ============="
# Never `pip install -e ./verl[sglang]`: that extra pulls sglang==0.5.8 from PyPI
# and shadows the DFlash fork.
PIP -e "$UPSTREAM_DIR/verl" --no-deps
PIP accelerate codetiming dill hydra-core peft pybind11 pylatexenc \
    "ray[default]>=2.41.0" torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    wandb tensorboard "pyarrow>=19.0.0" pandas datasets
PIP latex2sympy2_extended math_verify || echo "WARN: math_verify failed; unused here (use_task_rewards=False)"

echo "=== flash-attn 2.8.3 for SM90 ======================================"
# The upstream launcher loads the frozen Qwen3-4B with flash_attention_2 and
# packs sequences (use_remove_padding=True -> flash_attn.bert_padding). Without
# it you would have to fall back to sdpa + padding, which is a deviation.
if "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "flash_attn already present: $("$PY" -c 'import flash_attn; print(flash_attn.__version__)')"
elif [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
  echo "WARN: SKIP_FLASH_ATTN=1 -- training will use sdpa + padded batches (not the paper's path)."
else
  command -v nvcc >/dev/null || {
    echo "ERROR: nvcc not found, cannot build flash-attn. Use a CUDA *devel* image" >&2
    echo "       (lmsysorg/sglang:v0.5.8-cu129-amd64 is one), or SKIP_FLASH_ATTN=1." >&2
    exit 1; }
  # Building only for sm90 cuts the compile from ~1-2 h to ~15-30 min.
  FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=${MAX_JOBS:-8} \
    PIP --no-build-isolation "flash-attn==2.8.3"
fi

echo "=== python alias ==================================================="
# Upstream scripts call both `python` and `python3`.
if ! command -v python >/dev/null; then
  ln -sf "$(command -v "$PY")" /usr/local/bin/python 2>/dev/null \
    || echo "WARN: no 'python' on PATH and /usr/local/bin is not writable"
fi

echo "=== import check ==================================================="
# GPU-bound extensions (sgl_kernel, the DFlash worker) may not load during a
# `docker build`, which has no driver. preflight.sh re-runs this check with the
# GPU attached and fails hard there.
"$PY" "$H200_DIR/check_imports.py" || {
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then exit 1; fi
  echo "WARN: no GPU visible; GPU-bound imports are re-checked by h200/preflight.sh at run time"; }
echo "install complete"
