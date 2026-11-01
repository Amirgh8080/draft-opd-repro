# Draft-OPD on a Linux H200 machine, step by step

This guide is for one Linux machine with an H200 that you reach over SSH and
can `sudo` on. It follows a single path, top to bottom, without Docker. If you
would rather use Docker, or you rent a GPU "pod", see [`GUIDE.md`](GUIDE.md)
Paths A and B.

At the end you have:

- **Phase 1:** the paper's Table 1 row (Qwen3-4B) reproduced with the released
  checkpoints, printed next to the paper's numbers. This takes hours.
- **Phase 2:** a drafter you trained yourself with the paper's recipe, evaluated
  the same way. This takes days to weeks; Step 7 measures the real figure.

**Fresh Ubuntu install with no NVIDIA driver yet?** Do
[`INSTALL_UBUNTU.md`](INSTALL_UBUNTU.md) first. It installs the driver, sets up
the disk and runs the setup, then sends you back here at Step 5.

Commands in this guide run **on the Linux machine**, in your SSH session. The
guide uses `/data/workspace` as the workspace path; replace it everywhere if
you choose another.

---

## Step 0. What you need

| | Requirement |
|---|---|
| GPU | 1× H200 (141 GB). SXM or NVL both work. |
| OS | Ubuntu 22.04 or 24.04, x86_64 |
| Access | SSH, and `sudo` |
| NVIDIA driver | already installed, version ≥ 550 (570+ recommended). The setup never changes the driver. |
| Disk | **500 GB+** free on one filesystem, for the Python environment, models, checkpoints and logs |
| RAM / CPU | 120 GB+ / 16+ cores |
| Network | outbound HTTPS to PyPI, Hugging Face, GitHub, `developer.download.nvidia.com` and `astral.sh`. The full list is in [`NETWORK.md`](NETWORK.md). |

No Hugging Face account or token is needed: every model and dataset used is
public.

---

## Step 1. Connect and start tmux

Training runs for days, so everything runs inside `tmux`. If your SSH
connection drops, the run keeps going.

```bash
ssh user@your-h200-machine
sudo apt-get update && sudo apt-get install -y tmux git
tmux new -s opd
```

Reattach after a disconnect with `tmux attach -t opd`. To leave tmux running
and log out, press `Ctrl-b` then `d`.

## Step 2. Check the GPU and driver

```bash
nvidia-smi
```

You should see one **H200** and a **Driver Version** of 550 or higher.

- **`command not found`, or `couldn't communicate with the NVIDIA driver`:**
  the driver is missing or not loaded. On a cloud VM, use the provider's GPU
  image or driver instructions. On your own Ubuntu machine, follow
  [`INSTALL_UBUNTU.md`](INSTALL_UBUNTU.md) Part 3. If the driver is installed
  but will not load, check `mokutil --sb-state`: Secure Boot can block the
  driver until its key is enrolled.
- **Another process is using the GPU:** the Processes table in `nvidia-smi`
  lists it. Stop it first. This run needs the whole card.

Optional but recommended on a server: keep the driver loaded between runs.
This setting is reset at reboot.

```bash
sudo nvidia-smi -pm 1
```

## Step 3. Choose the disk

```bash
df -h
lsblk
```

Find a mounted filesystem with 500 GB+ free, for example `/data`. Then create
the workspace on it:

```bash
sudo mkdir -p /data/workspace && sudo chown "$USER" /data/workspace
```

If the large disk appears in `lsblk` but has no mount point, it is not
formatted or not mounted yet. Ask the machine's owner or provider. Formatting
the wrong disk destroys its data, so this guide does not do it for you.

## Step 4. Get the code and run the setup

```bash
cd ~
git clone https://github.com/Amirgh8080/draft-opd-repro.git
cd draft-opd-repro
bash h200/setup_bare_metal.sh --workspace /data/workspace
```

This takes about **45–60 minutes**. It asks before each `sudo` install; add
`-y` to accept them all. In order, it:

1. checks the driver and the workspace;
2. installs system packages: git, ffmpeg, libnuma, build tools, tmux;
3. installs the **CUDA 12.8 toolkit** (`/usr/local/cuda-12.8`), only to compile
   flash-attn. It skips this if a CUDA 12 toolkit is already installed. The
   package it installs, `cuda-toolkit-12-8`, contains no driver;
4. installs [uv](https://docs.astral.sh/uv/) in `~/.local/bin` and creates a
   **Python 3.12 environment** in `/data/workspace/venv`. The system Python is
   not touched;
5. clones upstream Draft-OPD at the pinned commit into
   `/data/workspace/Draft-OPD` and applies the launcher patch;
6. installs the pinned stack into that environment (torch 2.9.1, the SGLang
   DFlash fork, verl, flash-attn compiled for the H200). Most of the time is
   spent here.

It ends with **"Bare-metal setup complete."** Your choices are saved in
`h200/.local.env`, so every later command finds the workspace and the Python
environment by itself. There is nothing to activate.

If it stops with an error, fix the cause and run the same command again.
Finished parts are skipped. See [Troubleshooting](#troubleshooting) for the
usual causes.

## Step 5. Preflight

```bash
bash h200/preflight.sh
```

Every line should say `OK`, including the last one, the Python stack import
check. `WARN` lines explain themselves. Fix any `FAIL` before you continue.

## Step 6. Phase 1: reproduce Table 1 with the released checkpoints

```bash
bash h200/run_all.sh --eval-only
```

This command:

1. downloads Qwen3-4B, the DFlash drafter and the released Draft-OPD drafter
   (~10 GB), plus the seven benchmark datasets;
2. runs the autoregressive baseline;
3. runs DFlash;
4. runs Draft-OPD.

It uses the paper's evaluation settings: SGLang with the fa3 backend,
temperature 0, concurrency 1, and up to 2048 new tokens.

When it finishes:

```bash
cat /data/workspace/logs/eval/nothinking_temp0.0/summary.md
```

**What to look for.** Each of your rows has the paper's row printed directly
beneath it:

- **τ (acceptance length)** should be close to the paper's **6.04** for DFlash
  and **6.60** for Draft-OPD. τ depends only on the models, prompts and
  temperature, not on the hardware, so it is the main check.
- **Speedups** should be in the same range as the paper's, because this is the
  paper's GPU class. An H200 NVL (PCIe, 600 W) can come out somewhat lower
  than the SXM card the paper most likely used.

While a benchmark runs, check in a second tmux window (`Ctrl-b` then `c`) that
the card is not slowing itself down, since that would lower the speedups:

```bash
watch -n 5 nvidia-smi                       # temperature and power draw
nvidia-smi -q -d PERFORMANCE                # "Active" throttle reasons should all be "Not Active"
```

To add the thinking-mode rows (8192 tokens, roughly 4× longer), run
`bash h200/run_all.sh --eval-only --thinking`. The results land in
`/data/workspace/logs/eval/thinking_temp0.0/`.

## Step 7. Measure training speed

Before committing the GPU for days, run 20 training steps. They write to a
separate `-timing` directory, so the real run is unaffected:

```bash
TIMING_PROBE=1 bash h200/04_train_paper.sh
grep -o "timing_s/step:[0-9.]*" /data/workspace/logs/train/qwen3-4b-ins-draftopd-h200x1-timing/train.log
```

Most steps only generate rollouts. Every 10th step also updates the model, so
it is much slower. Average steps 11–20, which is one full cycle after warmup:

**estimated training time ≈ average s/step × 6,088**

(761 steps per epoch × 8 epochs. The real run prints the exact count as
`Total training steps` near the top of its log.)

This probe also catches memory problems early. If it fails with an
out-of-memory error, see [`README.md`](README.md#if-something-goes-wrong).

## Step 8. Phase 2: train with the paper's recipe

Inside tmux:

```bash
bash h200/run_all.sh --train-only
```

This trains with the paper's recipe: 16,000 prompts, 8 epochs, responses up
to 4,096 tokens, LR 3e-4, fp32 model, flash-attention. The one change is 1 GPU
instead of 7, which [`README.md`](README.md#why-one-gpu-does-not-change-the-optimisation)
explains. When training ends, the same command extracts your drafter,
benchmarks it as `mine`, and adds it to the Step 6 summary.

**Monitor it** from a second tmux window:

```bash
tail -f /data/workspace/logs/train/qwen3-4b-ins-draftopd-h200x1/train.log
grep -i "accept" /data/workspace/logs/train/qwen3-4b-ins-draftopd-h200x1/train.log | tail   # validation, every 250 steps
ls /data/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/                             # global_step_250, _500, ...
df -h /data                                                                               # ~25-30 GB per checkpoint
```

A checkpoint is saved every 250 steps, and the 4 newest are kept. The final
step is always saved.

**If it stops** (crash, reboot, power loss), run the same command again:

```bash
cd ~/draft-opd-repro && bash h200/run_all.sh --train-only
```

It resumes from the newest checkpoint, so you lose at most 250 steps.

## Step 9. Evaluate an earlier checkpoint (optional)

```bash
STEP=/data/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/global_step_5000
STEP_DIR=$STEP bash h200/05_extract_draft.sh
DRAFTS="step5000=$STEP/draft_model" SKIP_BASELINE=1 bash h200/03_eval_table1.sh
```

The new row is added to the same `summary.md`.

## Step 10. Copy the results off the machine

Run this **on your own computer**, not on the H200 machine:

```bash
rsync -avP user@your-h200-machine:/data/workspace/logs ./draft-opd-results/
rsync -avPm --include='*/' --include='**/draft_model/***' --exclude='*' \
  user@your-h200-machine:/data/workspace/checkpoints ./draft-opd-results/
```

The second command copies only the extracted drafters (~1 GB each), not the
full checkpoints. From Windows, use `scp` instead; see
[`GUIDE_WINDOWS.md`](GUIDE_WINDOWS.md#6-copy-results-to-windows).

Keep these:

| File | What it is |
|---|---|
| `logs/eval/nothinking_temp0.0/summary.md` | your Table 1 next to the paper's |
| `logs/eval/nothinking_temp0.0/*.md` | full per-benchmark reports |
| `logs/train/<exp>/launch_info.txt` | exact training configuration and upstream commit |
| `logs/train/<exp>/train.log` | training log |
| `checkpoints/<exp>/global_step_*/draft_model/` | your trained drafters |

## Step 11. Clean up before handing the machine back

The setup installed only these things. Remove the ones you don't want to leave
behind:

```bash
rm -rf /data/workspace                         # venv, models, data, checkpoints, logs (copy results first!)
rm -rf ~/draft-opd-repro                       # this repo, including h200/.local.env
rm -f ~/.local/bin/uv ~/.local/bin/uvx         # uv
rm -rf ~/.cache/uv ~/.local/share/uv           # uv's cache and its downloaded Python 3.12
sudo apt-get remove -y cuda-toolkit-12-8 && sudo apt-get autoremove -y   # only if the setup installed it
sudo nvidia-smi -pm 0                          # only if you enabled persistence mode in Step 2
```

`git config --global safe.directory` may also contain an entry for
`/data/workspace/Draft-OPD`. It is harmless, and
`git config --global --unset-all safe.directory` removes it. The driver and
the system Python were never modified.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvidia-smi: command not found` or `couldn't communicate with the NVIDIA driver` | Driver missing or not loaded (Step 2). With Secure Boot on, an unsigned driver module is blocked: check `mokutil --sb-state`. |
| Setup: `driver ... is too old` | Update the NVIDIA driver to 550+ (570+ recommended), reboot, rerun the setup. |
| Setup: `pass --workspace` | Run it with `--workspace /data/workspace` (Step 4). |
| Setup fails downloading from `developer.download.nvidia.com` or `astral.sh` | A firewall or proxy is blocking it. Allow those hosts. Or, if a CUDA 12 toolkit is already installed elsewhere, rerun with `CUDA_HOME=/path/to/cuda-12.x`. |
| `Ubuntu XX.YY: NVIDIA's apt repo covers 22.04/24.04` | Other distributions: install CUDA toolkit 12.x yourself, then rerun with `CUDA_HOME=/path/to/cuda-12.x`. |
| flash-attn build is killed, or the machine freezes during the install | The compile ran out of RAM. Rerun with fewer parallel jobs: `MAX_JOBS=4 bash h200/setup_bare_metal.sh --workspace /data/workspace`. |
| Install log shows `ERROR: ... verl ... requires numpy<2.0.0` | Expected and harmless. numpy 2 is kept on purpose; pip only reports the mismatch, and the install continues. |
| `the pinned stack needs 3.12` | Something other than the setup's environment is being used. Check that `h200/.local.env` exists, and rerun the setup. |
| `the system interpreter and has no torch` | You ran `h200/01_install.sh` directly on a fresh machine. Run `h200/setup_bare_metal.sh` instead. |
| Preflight: `python stack is installed but broken` | Read the module it names. Rerunning the setup repairs most cases. |
| Preflight: `/dev/shm` too small | Rare outside containers: `sudo mount -o remount,size=64G /dev/shm`. |
| Eval stops with an error about a dataset or model | The download was incomplete. Delete `/data/workspace/.state/prefetch.done` and rerun Step 6. |
| Out of memory during training | See [`README.md`](README.md#if-something-goes-wrong). |
| Speedups much lower than the paper while τ matches | The GPU is throttling. Check with `nvidia-smi -q -d PERFORMANCE` during a run, and fix the cooling or power limit. |
| Anything else | Every run writes a full log under `/data/workspace/logs/`. Look for the first `Traceback` or `ERROR`. |
