#!/usr/bin/env bash
# Prepare a plain Linux machine (no Docker) for the H200 harness.
#
#   bash h200/setup_bare_metal.sh --workspace /data/workspace
#
# Provides what the SGLang image would otherwise provide, without touching the
# NVIDIA driver or the system Python:
#   * system packages (git, ffmpeg for torchcodec, libnuma, build tools, tmux)
#   * a CUDA 12.8 toolkit, only to compile flash-attn (skipped if a CUDA 12 nvcc
#     already exists). 12.8 matches PyPI's torch 2.9.1, the build the fork's own
#     pyproject selects on x86_64.
#   * a Python 3.12 venv in <workspace>/venv, via uv (no root needed for this)
# then clones upstream and installs the stack (00_get_upstream.sh, 01_install.sh).
#
# The choices are saved in h200/.local.env, so every other h200 script picks up
# the workspace, the venv and the toolkit automatically. Safe to re-run.
set -euo pipefail
H200_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WS=""; ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WS="$2"; shift 2 ;;
    -y|--yes)    ASSUME_YES=1; shift ;;
    -h|--help)   sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
SUDO=""; [[ $(id -u) -eq 0 ]] || SUDO="sudo"

# ------------------------------------------------------------------ checks --
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
  || die "needs Linux x86_64 (sglang, sgl-kernel and flashinfer have no other builds)"
command -v nvidia-smi >/dev/null \
  || die "nvidia-smi not found. Install the NVIDIA driver (>= 550, 570+ recommended) first; this script never installs drivers."
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
(( ${drv%%.*} >= 550 )) || die "driver $drv is too old; need >= 550 (570+ recommended for CUDA 12.8)"
log "driver $drv, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if [[ -z "$WS" ]]; then
  if [[ -f "$H200_DIR/.local.env" ]]; then
    WS=$(sed -n 's/^WORKSPACE=//p' "$H200_DIR/.local.env" | tr -d '"')
  fi
  [[ -n "$WS" ]] || die "pass --workspace /path/on/a/big/disk (500 GB+; models, venv, checkpoints and logs go there)"
fi
mkdir -p "$WS" 2>/dev/null || { $SUDO mkdir -p "$WS" && $SUDO chown "$(id -u):$(id -g)" "$WS"; }
WS="$(cd "$WS" && pwd)"
[[ -w "$WS" ]] || die "$WS is not writable by $(id -un)"
log "workspace $WS ($(df -BG --output=avail "$WS" | tail -1 | tr -d ' ') free)"

# --------------------------------------------------------- system packages --
apt_install() {
  if ! command -v apt-get >/dev/null; then
    echo "  not an apt system; install these yourself: $*"; return 0
  fi
  local missing=() p
  for p in "$@"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  (( ${#missing[@]} )) || return 0
  if (( ! ASSUME_YES )) && [[ -t 0 ]]; then
    read -r -p "  apt-get install ${missing[*]} ? [Y/n] " r
    [[ "$r" =~ ^[Nn] ]] && { echo "  skipped"; return 0; }
  fi
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
}
log "system packages"
apt_install git git-lfs curl ca-certificates build-essential ffmpeg libnuma-dev pkg-config tmux

# ------------------------------------------------------------ CUDA toolkit --
# Only nvcc + headers for the flash-attn build. `cuda-toolkit-12-8` pulls no
# driver packages (the `cuda` meta-package would -- never install that one).
nvcc_major() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p'; }
CUDA_HOME_SEL=""
for cand in "${CUDA_HOME:-}" /usr/local/cuda-12.8 /usr/local/cuda /usr/local/cuda-12.*; do
  [[ -n "$cand" && -x "$cand/bin/nvcc" && "$(nvcc_major "$cand/bin/nvcc")" == 12 ]] && { CUDA_HOME_SEL="$cand"; break; }
done
if [[ -z "$CUDA_HOME_SEL" ]]; then
  log "installing CUDA toolkit 12.8 (nvcc for flash-attn; ~6 GB, no driver)"
  command -v apt-get >/dev/null || die "no CUDA 12 nvcc and not an apt system: install CUDA toolkit 12.8 manually, then rerun with CUDA_HOME=/path/to/cuda-12.8"
  # shellcheck disable=SC1091
  . /etc/os-release
  case "$VERSION_ID" in 22.04|24.04) ;; *) die "Ubuntu $VERSION_ID: NVIDIA's apt repo covers 22.04/24.04. Install CUDA toolkit 12.8 manually, then rerun with CUDA_HOME set." ;; esac
  repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/cuda-keyring.deb" "$repo/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i "$tmp/cuda-keyring.deb"; rm -rf "$tmp"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq cuda-toolkit-12-8
  CUDA_HOME_SEL=/usr/local/cuda-12.8
fi
[[ -x "$CUDA_HOME_SEL/bin/nvcc" ]] || die "nvcc missing under $CUDA_HOME_SEL"
log "CUDA toolkit $CUDA_HOME_SEL ($("$CUDA_HOME_SEL/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"

# --------------------------------------------------------- Python 3.12 venv --
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "installing uv (user-local, ~/.local/bin)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
VENV="$WS/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating Python 3.12 venv at $VENV"
  uv venv --seed --python 3.12 "$VENV"
fi
[[ "$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == 3.12 ]] \
  || die "$VENV is not Python 3.12; remove it and rerun"

# ---------------------------------------------------------- persist + install --
cat > "$H200_DIR/.local.env" <<EOF
# Written by h200/setup_bare_metal.sh on $(date -Is). Sourced by h200/env.sh.
WORKSPACE="$WS"
H200_VENV="$VENV"
CUDA_HOME="$CUDA_HOME_SEL"
EOF
log "saved settings to $H200_DIR/.local.env"

unset WORKSPACE UPSTREAM_DIR   # take them from .local.env
bash "$H200_DIR/00_get_upstream.sh"
MAX_JOBS=${MAX_JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))} bash "$H200_DIR/01_install.sh"

cat <<EOF

================================================================================
 Bare-metal setup complete.
   workspace : $WS
   python    : $VENV (activated automatically by every h200/ script)
   CUDA      : $CUDA_HOME_SEL

 Next (inside tmux):
   bash h200/preflight.sh
   bash h200/run_all.sh --eval-only
 For an interactive shell with the same environment:
   source $VENV/bin/activate
================================================================================
EOF
