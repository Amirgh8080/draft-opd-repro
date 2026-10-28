#!/usr/bin/env bash
# Checks the container before hours of work: GPU, shared memory, disk, RAM.
# Fails on what would break the run; warns on what would only change numbers.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

fail=0
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }

command -v nvidia-smi >/dev/null || { bad "nvidia-smi missing: start the container with --gpus all"; exit 1; }
IFS=, read -r name drv mem cap < <(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader,nounits | head -1)
name=$(echo "$name" | xargs); cap=$(echo "$cap" | xargs); mem=$(echo "$mem" | xargs)
ngpu=$(nvidia-smi -L | wc -l)
echo "  GPU   $name | driver $drv | ${mem} MiB | sm $cap | visible: $ngpu"
[[ "$cap" == "9.0" ]] && ok "SM90: fa3 rollout/eval backend available, as in the paper" \
  || warn "compute capability $cap: the bench drops fa3 off SM90, so speedups are not comparable to the paper"
[[ "$name" == *H200* ]] || warn "not an H200; tau is still comparable, speedups are not"
(( mem >= 139000 )) && ok "${mem} MiB VRAM" \
  || bad "${mem} MiB VRAM: the fp32 paper recipe on one GPU is sized for a 141 GB H200"
(( ngpu == 1 )) || warn "$ngpu GPUs visible; the scripts use CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} only"

# Ray's object store and SGLang's IPC live in /dev/shm. Docker's default is 64 MB.
shm=$(df -BG --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
(( ${shm:-0} >= 16 )) && ok "/dev/shm ${shm} GB" \
  || bad "/dev/shm is ${shm:-?} GB; run the container with --shm-size=64g (or --ipc=host)"

ram=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
(( ram >= 120 )) && ok "${ram} GB RAM" \
  || warn "${ram} GB RAM; checkpoint saves gather the fp32 actor on CPU. 120 GB+ recommended"
cpus=$(nproc)
(( cpus >= 16 )) && ok "$cpus CPUs" || warn "$cpus CPUs; Ray workers + rollout clients want 16+"

free=$(df -BG --output=avail "$WORKSPACE" | tail -1 | tr -dc '0-9')
(( free >= 250 )) && ok "${free} GB free on $WORKSPACE" \
  || warn "${free} GB free on $WORKSPACE; budget ~15 GB models + ~30 GB per kept checkpoint (KEEP_CKPTS=4)"

# The cu129 image declares NVIDIA_REQUIRE_CUDA=cuda>=12.9 (driver >= 575). Older
# hosts refuse to start it unless NVIDIA_DISABLE_REQUIRE=1 is set; CUDA minor
# version compatibility then runs it on any R550+ driver.
(( ${drv%%.*} >= 550 )) && ok "driver $drv" || bad "driver $drv < 550"

# WSL2 (Windows host). /proc/version shows the host kernel inside containers too.
if grep -qi microsoft /proc/version 2>/dev/null; then
  warn "WSL2 detected: tau is comparable to the paper, speedups through GPU paravirtualisation may not be"
  # A Windows drive (/mnt/c, or a Docker Desktop bind of one) is slow and breaks
  # Linux permissions and git ownership checks.
  case "$(stat -f -c %T "$WORKSPACE" 2>/dev/null)" in
    v9fs|9p|drvfs|fuse*|smb*|cifs)
      bad "$WORKSPACE is on a Windows drive; keep it on the Linux filesystem (see h200/GUIDE_WINDOWS.md, Route 2 step 6)" ;;
  esac
  case "$(realpath "$H200_DIR")" in
    /mnt/[a-z]/*) bad "the repo is under $(realpath "$H200_DIR"); clone it inside the Linux filesystem instead" ;;
  esac
fi

# GPU-bound imports cannot be checked at image build time; check them here.
if python "$H200_DIR/check_imports.py" >/tmp/h200_imports.log 2>&1; then
  ok "python stack imports (sgl_kernel, DFlash worker, flash_attn, verl)"
elif python -c "import sglang" 2>/dev/null; then
  cat /tmp/h200_imports.log; bad "python stack is installed but broken (see above)"
else
  warn "python stack not installed yet (run_all.sh installs it)"
fi

exit $fail
