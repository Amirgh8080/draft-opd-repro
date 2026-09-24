#!/usr/bin/env bash
# =============================================================================
# Draft-OPD: one-shot bootstrap + validation-data + training pipeline.
#
#   bash repro/10_bootstrap_and_run.sh                    # probe scale
#   bash repro/10_bootstrap_and_run.sh --scale reduced
#   bash repro/10_bootstrap_and_run.sh --skip-install     # env already built
#   bash repro/10_bootstrap_and_run.sh --train-only       # rerun training only
#
# Stages are idempotent: each records a marker in repro/.state/ and is skipped
# on re-run. Delete a marker file to force that stage to run again.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME=${ENV_NAME:-draftopd}
PY_VERSION=${PY_VERSION:-3.12}
SCALE=${SCALE:-probe}
CONSTRAINTS="$REPO_ROOT/repro/constraints.txt"
STATE_DIR="$REPO_ROOT/repro/.state"
LOG_DIR="$REPO_ROOT/repro/logs"
SKIP_INSTALL=0
WITH_FLASH_ATTN=0
WITH_SMOKE=0
ASSUME_YES=0
TRAIN_ONLY=0

# The fork is SGLang 0.5.8 (its CI release branches, the sgl-kernel 0.3.21
# pairing, and verl's own `sglang==0.5.8` extra all agree).
#
# This MUST be set. sglang-dflash/python/pyproject.toml sets setuptools_scm
# root=".." -- a directory with no .git -- and its git_describe_command looks for
# v*.*.* tags that this clone does not have. Without a pretend version the
# package installs as 0.0.0.dev0 and verl's
#     assert version.parse(sglang.__version__) >= version.parse("0.5.5")
# (sglang_rollout.py:194, async_sglang_server.py:241) fails at ROLLOUT time --
# long after the install looked successful.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG:-0.5.8}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)            SCALE="$2"; shift 2 ;;
    --env-name)         ENV_NAME="$2"; shift 2 ;;
    --skip-install)     SKIP_INSTALL=1; shift ;;
    --with-flash-attn)  WITH_FLASH_ATTN=1; shift ;;
    --with-smoke)       WITH_SMOKE=1; shift ;;
    --train-only)       TRAIN_ONLY=1; SKIP_INSTALL=1; shift ;;
    -y|--yes)           ASSUME_YES=1; shift ;;
    -h|--help)          sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$STATE_DIR" "$LOG_DIR"
RUN_LOG="$LOG_DIR/bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

log()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
die()  { echo "  FAIL  $*" >&2; echo "" >&2; echo "Full log: $RUN_LOG" >&2; exit 1; }

# stage <name> <function> -- run once, remember success
stage() {
  local name="$1" fn="$2" marker="$STATE_DIR/$1.done"
  if [[ -f "$marker" ]]; then
    log "stage '$name' already complete (rm $marker to redo)"
    return 0
  fi
  log "stage '$name' starting"
  "$fn"
  touch "$marker"
  log "stage '$name' complete"
}

PIP() { python -m pip install --retries 5 --timeout 60 -c "$CONSTRAINTS" "$@"; }

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
preflight() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "This stack is Linux-only. sglang, sgl-kernel and flashinfer publish no Windows or macOS builds."
  fi

  command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Install the NVIDIA driver (550+) first."
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

  local drv cap vram free_gb
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

  if (( drv < 525 )); then
    die "Driver $drv is too old for the CUDA 12.x torch wheels. Need 525 minimum, 550+ recommended."
  fi
  (( drv >= 550 )) || warn "Driver $drv works, but 550+ is recommended for this cu12.9-era stack."
  ok "driver $drv, compute capability $cap, ${vram} MiB VRAM"

  case "$cap" in
    8.6|8.9)
      ok "SM86/SM89: FA3 and FA4 are unavailable. benchmark_sglang.py:797 drops them and line 800 falls back to flashinfer. Expected." ;;
    9.0)
      ok "SM90 (Hopper): FA3 available." ;;
    *)
      warn "compute capability $cap is untested for this recipe." ;;
  esac

  (( vram >= 23000 )) || warn "${vram} MiB VRAM is below the 24 GB this recipe is tuned for. Expect to lower RESP_LEN and the mem fraction."

  free_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
  if (( free_gb < 80 )); then
    warn "Only ${free_gb} GB free. Budget: ~25 GB wheels + ~12 GB models + checkpoints. 80 GB+ recommended."
  else
    ok "${free_gb} GB free on the repo filesystem"
  fi

  if ! command -v conda >/dev/null; then
    die "conda not found. Install Miniconda first:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh"
  fi
  ok "conda $(conda --version | awk '{print $2}')"
}

# ---------------------------------------------------------------------------
# 1. System libraries.
#    torchcodec is a hard sglang dependency on linux-x86_64 and links against
#    FFmpeg shared libs. Missing FFmpeg fails at IMPORT time, not install time,
#    which makes it a confusing failure to diagnose later.
# ---------------------------------------------------------------------------
system_deps() {
  if ! command -v dpkg >/dev/null; then
    warn "not a dpkg-based distro; ensure these exist: build-essential cmake ninja ffmpeg libnuma"
    return 0
  fi
  local pkgs=(build-essential cmake ninja-build git git-lfs ffmpeg libnuma-dev pkg-config)
  local missing=()
  local p
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "system packages present"
    return 0
  fi

  warn "missing system packages: ${missing[*]}"
  if ! command -v sudo >/dev/null; then
    warn "no sudo. Install manually: apt-get install -y ${missing[*]}"
    return 0
  fi
  if (( ASSUME_YES == 0 )); then
    read -r -p "  Run 'sudo apt-get install -y ${missing[*]}'? [y/N] " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
      warn "skipped. torchcodec may fail to import later."
      return 0
    fi
  fi
  sudo apt-get update -qq || warn "apt-get update failed"
  sudo apt-get install -y "${missing[@]}" || warn "apt-get install failed; continuing"
  ok "system packages installed"
}

# ---------------------------------------------------------------------------
# 1b. Git safe.directory.
#     setuptools_scm shells out to git during the editable build of sglang --
#     both to resolve a version AND through its git-backed file finder, which
#     runs even when SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG is set. If the
#     repo sits on a mount whose owner differs from the current user (a
#     secondary /mnt partition, or a clone made by root), git aborts with
#     "detected dubious ownership", the build exits 1, and pip reports only a
#     generic "Failed to build ... when getting requirements to build editable".
# ---------------------------------------------------------------------------
git_safe() {
  if ! command -v git >/dev/null; then
    warn "git not found; skipping safe.directory check"
    return 0
  fi
  if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    ok "git can read $REPO_ROOT"
    return 0
  fi

  warn "git refuses to read $REPO_ROOT (almost certainly 'dubious ownership')"
  log "  repo owner: $(stat -c '%U:%G' "$REPO_ROOT" 2>/dev/null || echo unknown) | running as: $(id -un):$(id -gn)"
  if ! git config --global --add safe.directory "$REPO_ROOT"; then
    die "could not add safe.directory. Run manually:
    git config --global --add safe.directory $REPO_ROOT"
  fi

  if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    ok "added $REPO_ROOT to git safe.directory"
  else
    die "git still cannot read $REPO_ROOT after adding safe.directory.
    Check the mount: a noexec/nosuid or non-POSIX filesystem (NTFS, exFAT) can
    also break git introspection. Moving the clone onto an ext4 path is the
    reliable fix."
  fi
}

# ---------------------------------------------------------------------------
# 2. Conda environment
# ---------------------------------------------------------------------------
make_env() {
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ok "env '$ENV_NAME' already exists, reusing"
  else
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y || die "conda create failed"
    ok "env '$ENV_NAME' created (python $PY_VERSION)"
  fi
}

# activate_env MUST run on every invocation, never behind a stage marker.
#
# This used to live inside make_env(). On a resumed run make_env was skipped
# ("already complete") and the activation went with it, so `python` silently
# fell back to conda's BASE interpreter and everything installed against the
# wrong Python. Because outlines==0.1.11 pins outlines_core==0.1.26, which
# ships no cp313 wheel, pip then fell back to a source build and died on
# "can't find Rust compiler" -- a symptom, not the cause.
activate_env() {
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME" || die "could not activate env '$ENV_NAME'.
    Create it first by running without --skip-install."

  local py_path py_ver env_prefix
  py_path=$(command -v python) || die "no python on PATH after activating '$ENV_NAME'"
  py_ver=$(python -c "import sys; print('%d.%d' % sys.version_info[:2])")
  env_prefix="${CONDA_PREFIX:-}"

  if [[ -z "$env_prefix" || "$py_path" != "$env_prefix"/* ]]; then
    die "activated '$ENV_NAME' but python resolves outside it.
    python      : $py_path
    CONDA_PREFIX: ${env_prefix:-<unset>}
    Installing here would pollute the base environment. Aborting."
  fi
  if [[ "$py_ver" != "$PY_VERSION" ]]; then
    die "env '$ENV_NAME' has python $py_ver, expected $PY_VERSION.
    A mismatched interpreter breaks pinned wheels: outlines_core 0.1.26 has no
    cp313 wheel and falls back to a Rust source build.
    Fix: conda env remove -n $ENV_NAME  (then rerun), or pass --env-name <new>."
  fi

  python -m pip install -q --upgrade pip setuptools wheel "setuptools-scm>=8" packaging \
    || die "could not upgrade build tooling inside '$ENV_NAME'"
  ok "env '$ENV_NAME' active: python $(python -V 2>&1 | awk '{print $2}') at $py_path"
}

# ---------------------------------------------------------------------------
# 3. sglang-dflash. Brings torch 2.9.1, sgl-kernel 0.3.21, flashinfer 0.6.4,
#    transformers 4.57.1. This is the long step.
# ---------------------------------------------------------------------------
install_sglang() {
  log "installing sglang-dflash (editable). Expect 15-30 min and several GB of wheels."
  log "  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG"

  PIP -e "./sglang-dflash/python" || die "sglang-dflash install failed. Usual causes:
    * no wheel of sgl-kernel 0.3.21 / flashinfer 0.6.4 for python $PY_VERSION
      -> unlikely on 3.12: all 21 pins were verified to have cp312 linux wheels.
         If you changed PY_VERSION, change it back to 3.12.
    * network timeout on a multi-GB wheel -> just re-run, pip resumes downloads
    * missing nvcc/build tools for a source build -> re-run the system_deps stage
    * a 'can-t find Rust compiler' error -> you are on the WRONG python.
      Check wheel tags in the log above: cp313 means conda base, not '$ENV_NAME'.
      outlines_core 0.1.26 ships no cp313 wheel and falls back to a Rust source
      build. activate_env asserts against this now; if it still happens, run
      conda activate $ENV_NAME then python -V, and expect $PY_VERSION.
    * 'detected dubious ownership' in the build output -> the repo is on a mount
      git will not touch. Fix:
          git config --global --add safe.directory $REPO_ROOT
      then re-run. (The git_safe stage normally does this for you.)
    See $RUN_LOG"

  PIP cachetools || die "cachetools install failed"

  python - <<'PYEOF' || die "sglang installed but reports an unusable version; see the setuptools_scm note at the top of this script"
import sglang
from packaging import version
v = sglang.__version__
print(f"  sglang.__version__ = {v}")
if version.parse(v) < version.parse("0.5.5"):
    raise SystemExit(
        f"sglang reports {v}, but verl asserts >= 0.5.5 (sglang_rollout.py:194).\n"
        "  Fix: pip uninstall -y sglang, then re-run with\n"
        "       SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.8"
    )
PYEOF
  ok "sglang-dflash installed, version gate satisfied"
}

# ---------------------------------------------------------------------------
# 4. verl. Installed with --no-deps, then its dependencies explicitly under the
#    constraints file.
#
#    Why: verl's setup.py pins numpy<2.0.0 while the sglang stack expects
#    numpy 2.x. A plain `pip install -e ./verl` silently downgrades numpy and
#    breaks every C extension built against the numpy 2 ABI -- flashinfer and
#    sgl-kernel included. The same hazard applies to torch: tensordict and
#    torchdata can drag it off 2.9.1, which sgl-kernel is ABI-bound to.
# ---------------------------------------------------------------------------
install_verl() {
  local np_before np_after torch_after
  np_before=$(python -c "import numpy; print(numpy.__version__)")
  log "numpy before verl: $np_before"

  PIP -e "./verl" --no-deps || die "verl editable install failed (see $RUN_LOG)"

  # verl's install_requires, minus numpy (constrained to >=2 instead) and minus
  # what the sglang install already satisfied.
  PIP accelerate codetiming dill hydra-core peft pybind11 pylatexenc \
      "ray[default]>=2.41.0" torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
      wandb tensorboard "pyarrow>=19.0.0" pandas datasets \
    || die "verl dependency install failed (see $RUN_LOG)"

  # Only used by reward scoring, which this distillation pipeline never invokes.
  PIP latex2sympy2_extended math_verify \
    || warn "math-verify extras failed; harmless here (use_task_rewards=False)"

  np_after=$(python -c "import numpy; print(numpy.__version__)")
  torch_after=$(python -c "import torch; print(torch.__version__)")
  log "after verl -- numpy: $np_after, torch: $torch_after"

  case "$np_after" in
    2.*) : ;;
    *) die "numpy was downgraded to $np_after despite the constraints file.
    Fix: pip install 'numpy>=2.0' and re-run the verify stage." ;;
  esac
  case "$torch_after" in
    2.9.1*) : ;;
    *) die "torch moved to $torch_after. sgl-kernel 0.3.21 is ABI-bound to 2.9.1.
    Fix: pip install torch==2.9.1 torchaudio==2.9.1 and re-run the verify stage." ;;
  esac
  ok "verl installed without disturbing numpy or torch"
}

# ---------------------------------------------------------------------------
# 5. flash-attn (optional, but it does matter).
#    flex_attention covers only the DRAFT module's block-mask attention. The
#    frozen TARGET model inside the composed student is loaded by transformers
#    with flash_attention_2 by default, and verl's unpadded path imports
#    flash_attn.bert_padding. Without flash_attn, 04_train_1gpu.sh falls back to
#    sdpa + use_remove_padding=False -- correct, just slower.
#    No prebuilt wheel exists for torch 2.9 + cu12 (upstream ships cu13torch2.9
#    only), so this builds from source and needs a CUDA toolkit. Never fatal.
# ---------------------------------------------------------------------------
install_flash_attn() {
  if (( WITH_FLASH_ATTN == 0 )); then
    ok "skipping flash-attn. 04_train_1gpu.sh then auto-selects sdpa + padded"
    ok "  batches -- correct, just slower. flex_attention covers only the DRAFT"
    ok "  module; the frozen TARGET model would use flash_attn if it were present."
    ok "  Fast path needs a CUDA toolkit (no prebuilt wheel for torch 2.9 + cu12),"
    ok "  then rerun with --with-flash-attn."
    return 0
  fi
  warn "building flash-attn from source: 30-90 min, >16 GB RAM"
  MAX_JOBS=${MAX_JOBS:-4} PIP flash-attn --no-build-isolation \
    || warn "flash-attn build failed. Continuing -- it is not required for DFlash training."
}

# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------
verify() {
  python - <<'PYEOF' || die "environment verification failed (see the list above)"
import sys
from packaging import version

fail = []

def check(label, fn, required=True):
    try:
        print(f"  {label:<16} {fn()}")
    except Exception as e:
        print(f"  {label:<16} FAILED: {e}")
        if required:
            fail.append((label, e))

import torch
check("torch",       lambda: torch.__version__)
check("cuda",        lambda: f"available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}")
check("compute cap", lambda: "sm%d%d" % torch.cuda.get_device_capability())
check("vram",        lambda: f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

import numpy
check("numpy", lambda: numpy.__version__)
import transformers
check("transformers", lambda: transformers.__version__)
import sglang
check("sglang", lambda: sglang.__version__)

check("flashinfer", lambda: (__import__("flashinfer"), "ok")[1])
check("sgl_kernel", lambda: (__import__("sgl_kernel"), "ok")[1])
check("torchcodec", lambda: (__import__("torchcodec"), "ok")[1], required=False)
check("verl",       lambda: (__import__("verl"), "ok")[1])
check("ray",        lambda: __import__("ray").__version__)
check("tensordict", lambda: __import__("tensordict").__version__)
check("flex_attn",  lambda: (__import__("torch.nn.attention.flex_attention",
                                        fromlist=["create_block_mask"]), "ok")[1])

# The gate that bites at rollout time rather than import time.
v = sglang.__version__
if version.parse(v) < version.parse("0.5.5"):
    fail.append(("sglang version gate", f"{v} < 0.5.5"))
    print(f"  {'version gate':<16} FAILED: {v} < 0.5.5")
else:
    print(f"  {'version gate':<16} sglang {v} >= 0.5.5 (required by verl)")

try:
    from sglang.srt.speculative import dflash_worker  # noqa: F401
    print(f"  {'dflash_worker':<16} importable")
except Exception as e:
    fail.append(("dflash_worker", e))
    print(f"  {'dflash_worker':<16} FAILED: {e}")

if fail:
    print("\nBLOCKING FAILURES:")
    for label, e in fail:
        print(f"  - {label}: {e}")
    sys.exit(1)
print("\nAll checks passed.")
PYEOF
  ok "environment verified"
}

# ---------------------------------------------------------------------------
# 7. Assets, and resolve local snapshot paths for the trainer
# ---------------------------------------------------------------------------
prefetch() {
  bash repro/01_prefetch_assets.sh || die "asset prefetch failed (see $RUN_LOG)"
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python - > "$STATE_DIR/paths.env" <<'PYEOF' || die "could not resolve local snapshot paths"
from huggingface_hub import snapshot_download
print('MAIN_MODEL_PATH="%s"' % snapshot_download("Qwen/Qwen3-4B"))
print('DRAFT_MODEL_PATH="%s"' % snapshot_download("z-lab/Qwen3-4B-DFlash-b16"))
PYEOF
  cat "$STATE_DIR/paths.env"
  ok "assets cached, local paths resolved"
}

# ---------------------------------------------------------------------------
# 8/9/10. Validation data, optional smoke test, training
# ---------------------------------------------------------------------------
make_val() {
  python repro/make_val_jsonl.py --out-dir repro/data || die "validation JSONL generation failed"
  ok "validation data written to repro/data/"
}

smoke() {
  if (( WITH_SMOKE == 1 )); then
    bash repro/02_smoke_test.sh || warn "smoke test failed -- inspect before trusting training results"
  else
    ok "smoke test skipped (--with-smoke to run it)"
  fi
}

train() {
  [[ -f "$STATE_DIR/paths.env" ]] || die "missing $STATE_DIR/paths.env -- run once without --train-only first"
  # shellcheck disable=SC1090
  source "$STATE_DIR/paths.env"
  export MAIN_MODEL_PATH DRAFT_MODEL_PATH
  log "launching training at SCALE=$SCALE"
  log "  target (frozen) : $MAIN_MODEL_PATH"
  log "  draft  (trained): $DRAFT_MODEL_PATH"
  SCALE="$SCALE" bash repro/04_train_1gpu.sh
}

# ---------------------------------------------------------------------------
main() {
  log "Draft-OPD bootstrap | env=$ENV_NAME | scale=$SCALE"
  log "log file: $RUN_LOG"

  if (( SKIP_INSTALL == 0 )); then
    stage preflight   preflight
    stage system_deps system_deps
    stage git_safe    git_safe
    stage make_env    make_env
    activate_env      # NEVER stage-gated: a resumed run must still activate
    stage sglang      install_sglang
    stage verl        install_verl
    stage flash_attn  install_flash_attn
    stage verify      verify
  else
    log "--skip-install: activating existing env '$ENV_NAME'"
    activate_env
  fi

  if (( TRAIN_ONLY == 0 )); then
    stage prefetch prefetch
    stage make_val make_val
    smoke
  fi

  train   # deliberately not stage-gated: always re-runnable

  echo ""
  log "Pipeline finished. Log: $RUN_LOG"
  log "Checkpoints: verl/checkpoints/verl-dflash-opd/"
  log "Next: bash repro/05_extract_draft.sh   then   bash repro/03_eval_table1.sh"
}

main
