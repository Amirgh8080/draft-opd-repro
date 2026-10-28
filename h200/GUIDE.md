# Step-by-step: from a fresh H200 machine to the paper's numbers

This guide starts on a machine that has nothing installed yet and ends with
two results:

- **Phase 1:** a Table 1 comparison of the released checkpoints against the
  paper.
- **Phase 2:** a drafter you trained yourself with the paper's recipe.

Pick the path that matches your machine:

- **Path A: a VM or bare-metal server you SSH into** (AWS, GCP, Azure, Lambda
  on-demand, your own box). You install Docker and build the image yourself.
  Start at [A1](#a1-check-the-gpu-and-driver).
- **Path B: a GPU cloud "pod" where you choose a container image** (RunPod,
  Vast.ai, and similar). There is no Docker to install. Start at
  [B1](#b1-create-the-pod).

Both paths meet at [Step 5](#step-5-phase-1--evaluate-the-released-checkpoints).

---

## Before you start: what to rent

| | Minimum | Why |
|---|---|---|
| GPU | **1× H200 141 GB** | The paper's fp32 recipe on one GPU is sized for it. The speedups are only comparable to the paper on an H200. |
| RAM | 120 GB+ | Checkpoint saves gather the fp32 model on CPU |
| CPU | 16+ cores | Ray workers and rollout clients |
| Disk | **500 GB+** of persistent storage | ~30 GB image, ~15 GB models and data, ~25–30 GB per kept checkpoint (4 are kept) |
| OS (Path A) | Ubuntu 22.04 or 24.04, NVIDIA driver ≥ 550 | |

Phase 1 takes hours. Phase 2 takes days to weeks; you measure the real figure
in [Step 6](#step-6-measure-training-speed). Make sure the storage outlives the
machine if the provider lets you choose: everything the run produces goes to
one directory.

---

## Path A: VM or bare-metal server

### A1. Check the GPU and driver

```bash
nvidia-smi
```

You should see one H200 and a driver version. If the version is **below 575**,
note it: you will add `-e NVIDIA_DISABLE_REQUIRE=1` to every `docker run` below.
If `nvidia-smi` is not found, install the driver your provider recommends,
for example `sudo ubuntu-drivers install`, then reboot.

### A2. Install Docker and the NVIDIA Container Toolkit

Skip this step if `docker run --rm --gpus all ubuntu nvidia-smi` already works.

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"            # then log out and back in

# NVIDIA Container Toolkit (lets containers see the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### A3. Put Docker and the workspace on the big disk

The image build needs ~60 GB of scratch space. On many VMs the root disk is
small, so check where the large disk is mounted:

```bash
df -h / /var/lib/docker
lsblk                                       # find the large data disk
```

If `/var/lib/docker` is on a small disk, move Docker's storage onto the big
disk. Replace `/data` with the big disk's mount point:

```bash
sudo mkdir -p /data/docker
echo '{"data-root": "/data/docker"}' | sudo tee /etc/docker/daemon.json
sudo nvidia-ctk runtime configure --runtime=docker   # re-adds the GPU runtime to daemon.json
sudo systemctl restart docker
```

Next, create the workspace. Models, data, checkpoints and logs all go here:

```bash
sudo mkdir -p /data/workspace && sudo chown "$USER" /data/workspace
```

### A4. Test the GPU inside a container

This also pulls the base image (~15 GB), which the build needs anyway:

```bash
docker run --rm --gpus all lmsysorg/sglang:v0.5.8-cu129-amd64 nvidia-smi
```

It should print the same H200. If it fails with `unsatisfied condition:
cuda>=12.9`, your driver is older than 575; rerun with
`-e NVIDIA_DISABLE_REQUIRE=1`.

### A5. Build the image

```bash
git clone https://github.com/Amirgh8080/draft-opd-repro.git
cd draft-opd-repro
docker build -f h200/Dockerfile -t draft-opd-h200 .
```

This takes ~30–45 minutes, mostly compiling flash-attn. It ends with
`install complete`. A warning that GPU-bound imports will be re-checked at run
time is expected, because builds have no GPU.

If you `git pull` changes to this repo later, rebuild the image.

### A6. Define the run command

Every step below runs the same container with the workspace mounted. Save the
common flags once. Add `-e NVIDIA_DISABLE_REQUIRE=1` to the list if A1 showed a
driver below 575:

```bash
OPD_FLAGS=(--gpus '"device=0"' --shm-size=64g --ulimit memlock=-1 -v /data/workspace:/workspace)
opd() { docker run --rm "${OPD_FLAGS[@]}" "$@"; }      # for short, foreground tasks
# Put both lines in ~/.bashrc if you want them in new shells.
```

`--shm-size=64g` is required. Docker's 64 MB default breaks Ray and SGLang.

### A7. Preflight

```bash
opd draft-opd-h200 bash h200/preflight.sh
```

Every line should say `OK`. A `WARN` explains itself. A `FAIL` must be fixed
before you continue. Common failures:

- `/dev/shm`: you forgot `--shm-size`.
- VRAM: this is not a 141 GB GPU.
- `python stack is installed but broken`: rebuild the image.

Now continue at [Step 5](#step-5-phase-1--evaluate-the-released-checkpoints)
and use the **Path A** commands.

---

## Path B: GPU cloud pod

### B1. Create the pod

- **GPU:** 1× H200.
- **Image / template:** a custom image, `lmsysorg/sglang:v0.5.8-cu129-amd64`.
- **Volume:** 500 GB+, mounted at **`/workspace`**. This is the persistent
  volume; container disk is wiped on restart. Give the container disk itself
  ~50 GB.
- **Shared memory:** if the provider has a setting, set it to 64 GB.
- **Environment variable:** if the provider's hosts run drivers older than 575
  and the pod fails to start, add `NVIDIA_DISABLE_REQUIRE=1`.

Then open a terminal (web terminal or SSH).

### B2. Get the code and a terminal that survives disconnects

```bash
apt-get update && apt-get install -y tmux git
tmux new -s opd          # reattach later with: tmux attach -t opd
cd /workspace
git clone https://github.com/Amirgh8080/draft-opd-repro.git
cd draft-opd-repro
```

Run everything below inside this tmux session. If your browser or SSH drops,
the run keeps going.

### B3. Preflight

```bash
bash h200/preflight.sh
```

At this point `python stack not installed yet` is expected, because the next
step installs it. Fix any `FAIL`. If `/dev/shm` is too small and the provider
has no setting for it, `mount -o remount,size=64G /dev/shm` usually works on
pods that run as root.

Now continue at [Step 5](#step-5-phase-1--evaluate-the-released-checkpoints)
and use the **Path B** commands. The first run also installs the Python stack
(~30–45 minutes), and that install lives on the container disk. If the pod is
recreated, `run_all.sh` notices the stack is missing and reinstalls it.
Everything else stays on `/workspace`.

---

## Step 5. Phase 1: evaluate the released checkpoints

This single command clones upstream Draft-OPD at the pinned commit, applies the
launcher patch, downloads the models and datasets, and runs the Table 1
benchmarks. It runs a baseline, then DFlash, then Draft-OPD.

**Path A:**
```bash
opd -d --name opd-eval draft-opd-h200 bash h200/run_all.sh --eval-only
docker logs -f opd-eval          # Ctrl-C stops following; the run continues
```

**Path B:**
```bash
bash h200/run_all.sh --eval-only
```

When it finishes, read the summary:

```bash
cat /data/workspace/logs/eval/nothinking_temp0.0/summary.md    # Path A
cat /workspace/logs/eval/nothinking_temp0.0/summary.md         # Path B
```

**What you are checking.** In the τ column, the `dflash` and `draftopd` rows
should sit next to the paper's 6.04 and 6.60. The per-benchmark speedups should
be close to the paper rows printed beneath them. τ is the main signal: it
depends only on the models, prompts and temperature. Speedups also depend on
the machine, but this is the paper's hardware class, so they should land close
too.

Optional: to evaluate the thinking-mode rows as well (8192 tokens, roughly 4×
longer), add `--thinking` to the command above. Path A then reports in
`.../logs/eval/thinking_temp0.0/`.

If a stage fails, fix the cause and rerun the same command. Finished stages are
skipped through markers in `<workspace>/.state/`.

## Step 6. Measure training speed

Before committing the GPU for days, run 20 training steps. They go into a
separate `-timing` checkpoint and log directory, so the real run is unaffected:

**Path A:**
```bash
opd -e TIMING_PROBE=1 draft-opd-h200 bash h200/04_train_paper.sh
```

**Path B:**
```bash
TIMING_PROBE=1 bash h200/04_train_paper.sh
```

Read the per-step times from the log:

```bash
grep -o "timing_s/step:[0-9.]*" <workspace>/logs/train/qwen3-4b-ins-draftopd-h200x1-timing/train.log
```

Most steps only generate rollouts. Every 10th step also runs the model update,
so it is much slower. Average steps 11–20, one full cycle, to skip the
torch.compile and CUDA-graph warmup in the first cycle. Then:

**estimated run time ≈ average s/step × 6,088**

The real run has 761 steps per epoch × 8 epochs, minus any prompts filtered
for exceeding 512 tokens. Its log prints the exact count as `Total training
steps` near the top.

This probe also flushes out memory problems. If it fails with an out-of-memory
error, see the end of [`README.md`](README.md#if-something-goes-wrong).

## Step 7. Phase 2: train with the paper recipe

**Path A.** Run it detached, without `--rm`, which Docker does not allow
together with `--restart`. `--restart on-failure:3` restarts the container
after a crash, up to 3 times, and each restart resumes from the latest
checkpoint:

```bash
docker run -d --name opd-train --restart on-failure:3 "${OPD_FLAGS[@]}" \
  draft-opd-h200 bash h200/run_all.sh --train-only
docker logs -f opd-train          # Ctrl-C stops following; training continues
```

**Path B** (inside tmux):

```bash
bash h200/run_all.sh --train-only
```

After training finishes, the same command extracts the drafter from the last
checkpoint. It then benchmarks that drafter as `mine`, reusing Phase 1's
baseline, and adds it to the same `summary.md`.

### While it runs

```bash
# progress (Path A: docker logs -f opd-train also works)
tail -f <workspace>/logs/train/qwen3-4b-ins-draftopd-h200x1/train.log

# step lines carry the metrics; acceptance-length metrics show up at validation (every 250 steps)
grep -i "accept" <workspace>/logs/train/qwen3-4b-ins-draftopd-h200x1/train.log | tail

nvidia-smi                                   # GPU memory and utilisation
ls <workspace>/checkpoints/qwen3-4b-ins-draftopd-h200x1/   # global_step_250, _500, ...
df -h <workspace>                            # checkpoints are ~25-30 GB each
```

A checkpoint is saved every 250 steps, and the last 4 are kept. The final step
is always saved.

### If it stops

Run the **same command** again (Step 7). On Path A, run `docker rm opd-train`
first, because the stopped container still holds the name. It resumes from the latest checkpoint
in `<workspace>/checkpoints/qwen3-4b-ins-draftopd-h200x1/`, so at most 250
steps of work are lost. This also works on a different machine. Set it up with
Path A or B, attach the same storage as the workspace, and run Step 7.

## Step 8. Evaluate a specific checkpoint (optional)

`run_all.sh` evaluates the last checkpoint. To evaluate an earlier one, extract
it and benchmark it under its own label:

**Path B**, or **Path A** after you open a shell in the container with
`opd -it draft-opd-h200 bash`. Inside the container, the paths are the same
`/workspace/...`:

```bash
STEP_DIR=/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/global_step_5000 \
  bash h200/05_extract_draft.sh
DRAFTS="step5000=/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/global_step_5000/draft_model" \
  SKIP_BASELINE=1 bash h200/03_eval_table1.sh
```

## Step 9. Copy the results off the machine

Everything worth keeping is under the workspace. From your laptop:

```bash
rsync -avP user@machine:/data/workspace/logs ./draft-opd-results/           # Path A
rsync -avPm --include='*/' --include='**/draft_model/***' --exclude='*' \
  user@machine:/data/workspace/checkpoints ./draft-opd-results/            # extracted drafters only, not the ~30 GB checkpoints
```

On a pod (Path B), use the provider's file browser or `runpodctl send`, or
rsync over the pod's SSH port with the same patterns and `/workspace` paths.

The files to keep:

| File | What it is |
|---|---|
| `logs/eval/nothinking_temp0.0/summary.md` | your Table 1 next to the paper's |
| `logs/eval/nothinking_temp0.0/*.md` | full per-benchmark reports for each drafter |
| `logs/train/<exp>/launch_info.txt` | the exact training configuration and upstream commit |
| `logs/train/<exp>/train.log` | the training log |
| `checkpoints/<exp>/global_step_*/draft_model/` | your trained drafter(s) |

When you write up results, state the deviations from the paper's setup listed
in [`README.md`](README.md#what-does-differ-and-what-it-means-for-exact):
1 GPU instead of 7, and the public code version.

---

## Quick reference

| Task | Path A (Docker host) | Path B (pod) |
|---|---|---|
| Preflight | `opd draft-opd-h200 bash h200/preflight.sh` | `bash h200/preflight.sh` |
| Phase 1 | `opd -d --name opd-eval draft-opd-h200 bash h200/run_all.sh --eval-only` | `bash h200/run_all.sh --eval-only` |
| Timing probe | `opd -e TIMING_PROBE=1 draft-opd-h200 bash h200/04_train_paper.sh` | `TIMING_PROBE=1 bash h200/04_train_paper.sh` |
| Phase 2 / resume | `docker rm opd-train; docker run -d --name opd-train --restart on-failure:3 "${OPD_FLAGS[@]}" draft-opd-h200 bash h200/run_all.sh --train-only` | `bash h200/run_all.sh --train-only` |
| Interactive shell | `opd -it draft-opd-h200 bash` | (you are already in one) |
| Redo a stage | delete `<workspace>/.state/<stage>.done` and rerun | same |

`<workspace>` is `/data/workspace` on Path A (the host path; it is
`/workspace` inside the container) and `/workspace` on Path B.
