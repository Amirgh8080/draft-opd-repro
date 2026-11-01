# shellcheck shell=bash disable=SC2034
# Shared settings for the H200 scripts. Sourced, never executed.
#
# Everything that must outlive the container (models, checkpoints, logs,
# generated data) lives under $WORKSPACE. Mount a persistent volume there.

H200_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "${H200_DIR}/.." && pwd)"

# Machine-local settings written by setup_bare_metal.sh (WORKSPACE, H200_VENV,
# CUDA_HOME). Absent in the Docker image and on pods; git- and docker-ignored.
if [[ -f "$H200_DIR/.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$H200_DIR/.local.env"
fi

# Upstream code, pinned. The paper's numbers were produced by this codebase; a
# later commit may change defaults under you, so do not float on main.
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/Simplified-Reasoning/Draft-OPD.git}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-d6f0ec859321959785c65b59eb311fd25f042158}

WORKSPACE=${WORKSPACE:-/workspace}
UPSTREAM_DIR=${UPSTREAM_DIR:-${WORKSPACE}/Draft-OPD}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/data}
CKPT_ROOT=${CKPT_ROOT:-${WORKSPACE}/checkpoints}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
STATE_DIR=${STATE_DIR:-${WORKSPACE}/.state}

export HF_HOME=${HF_HOME:-${WORKSPACE}/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}

# setuptools_scm cannot version the sglang fork (root="..", no v*.*.* tags), so
# it installs as 0.0.0.dev0 and verl's `sglang >= 0.5.5` assert fails at
# rollout time. The fork's base is 0.5.8.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG:-0.5.8}

# The upstream bench sets this too; harmless once the jit cache matches 0.6.4.
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}

# Ray reports cluster usage to usage-stats.ray.io unless told not to (its
# docstring says off by default; usage_lib.py returns ENABLED_BY_DEFAULT).
# Runs should need no network once assets are cached.
export RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED:-0}

# Hub ids. Paper Table 1, Qwen3-4B row.
TARGET_ID=${TARGET_ID:-Qwen/Qwen3-4B}
DFLASH_ID=${DFLASH_ID:-z-lab/Qwen3-4B-DFlash-b16}
OPD_INS_ID=${OPD_INS_ID:-bingyang-lei/Qwen3-4B-Ins-Draft-OPD}
OPD_THINK_ID=${OPD_THINK_ID:-bingyang-lei/Qwen3-4B-Thinking-Draft-OPD}

# Bare-metal installs (no Docker): use the Python 3.12 venv and the CUDA 12
# toolkit that setup_bare_metal.sh prepared, so every script and the upstream
# launcher (`python3 -m verl...`) resolve to them without manual activation.
if [[ -n "${H200_VENV:-}" && "${VIRTUAL_ENV:-}" != "$H200_VENV" && -f "$H200_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$H200_VENV/bin/activate"
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
fi

mkdir -p "$WORKSPACE" "$DATA_DIR" "$CKPT_ROOT" "$LOG_ROOT" "$STATE_DIR" "$HF_HOME" 2>/dev/null || true

# Resolve a hub id to its local snapshot dir (cache only, no network).
local_snapshot() {
  HF_HUB_OFFLINE=1 python - "$1" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PYEOF
}
