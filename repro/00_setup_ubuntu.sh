#!/usr/bin/env bash
# Draft-OPD environment setup for a single-GPU Ubuntu box (RTX 3090, SM86).
# Run from the repository root:  bash repro/00_setup_ubuntu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME=${ENV_NAME:-draftopd}
PY_VERSION=${PY_VERSION:-3.12}

echo "=== 1/6 Driver check ==============================================="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install the NVIDIA driver (>= 550) first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
echo
echo "Expected: NVIDIA GeForce RTX 3090 / driver >= 550 / 24576 MiB / compute_cap 8.6"
echo "compute_cap 8.6 (SM86) means: FA3 and FA4 backends are unavailable."
echo "The benchmark auto-falls back to the flashinfer backend. This is expected."
echo

echo "=== 2/6 Conda environment =========================================="
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first:" >&2
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
  echo "  bash Miniconda3-latest-Linux-x86_64.sh" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi
conda activate "${ENV_NAME}"
python -V

echo "=== 3/6 sglang-dflash (editable) ==================================="
# Installs the vendored SGLang fork plus torch 2.9.1, sgl-kernel 0.3.21,
# flashinfer 0.6.4, transformers 4.57.1. This is the heavy step (~15 min).
pip install -e "./sglang-dflash/python"
pip install cachetools

echo "=== 4/6 verl (editable, NO extras) ================================="
# CRITICAL: never use `pip install -e ".[sglang]"` here. That extra pins
# sglang==0.5.8 from PyPI and would shadow the vendored DFlash fork.
pip install -e "./verl"

echo "=== 5/6 numpy pin reconciliation ==================================="
# verl pins numpy<2.0.0; sglang's dependency set generally expects numpy>=2.
# Installing verl second can silently downgrade numpy and break packages
# compiled against the numpy 2 ABI. Check and report.
python - <<'PYEOF'
import numpy, sys
print(f"numpy == {numpy.__version__}")
if numpy.__version__.startswith("1."):
    print("WARNING: numpy 1.x is installed (verl's pin won).")
    print("         If sglang/flashinfer imports raise an ABI error, run:")
    print("             pip install 'numpy>=2.0'")
    print("         verl tolerates numpy 2 in practice despite its setup.py pin.")
PYEOF

echo "=== 6/6 Import smoke check ========================================="
python - <<'PYEOF'
import torch
print(f"torch           {torch.__version__}")
print(f"cuda available  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"device          {torch.cuda.get_device_name(0)}")
    print(f"compute cap     sm{cap[0]}{cap[1]}")
    print(f"vram            {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
import sglang;      print(f"sglang          {getattr(sglang,'__version__','editable')}")
import verl;        print(f"verl            ok")
import transformers;print(f"transformers    {transformers.__version__}")
try:
    import flashinfer; print("flashinfer      ok")
except Exception as e:
    print(f"flashinfer      IMPORT FAILED: {e}")
from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
print("flex_attention  ok  (used by the DFlash student during training)")
PYEOF

echo
echo "Setup complete. Environment: ${ENV_NAME}"
echo "Next:  conda activate ${ENV_NAME} && bash repro/01_prefetch_assets.sh"
