# draft-opd-repro

Single-GPU replication harness for **Draft-OPD: On-Policy Distillation for
Speculative Draft Models** ([arXiv:2605.29343](https://arxiv.org/abs/2605.29343),
Lei et al., 2026).

The paper's training recipe assumes 8× H200. This repo makes the work runnable
on **a single 24 GB consumer GPU** (RTX 3090 / 4090, SM86 / SM89): a staged
bootstrap that survives the stack's
dependency hazards, the validation data the upstream repo does not ship, a
single-GPU training configuration, and a sequential evaluation driver.

This repo contains **only** the replication harness. It does not redistribute
the upstream code — you clone that yourself, then apply a patch.

## What is here

| Path | Purpose |
|---|---|
| `repro/10_bootstrap_and_run.sh` | One-shot: env → verify → assets → validation data → train |
| `repro/00_setup_ubuntu.sh` | Standalone environment setup (manual path) |
| `repro/01_prefetch_assets.sh` | Pre-caches 3 models + 8 datasets — required, eval runs offline |
| `repro/02_smoke_test.sh` | 4 prompts, 256 tokens: does the stack work at all |
| `repro/03_eval_table1.sh` | Baseline → DFlash → Draft-OPD, sequentially on one GPU |
| `repro/04_train_1gpu.sh` | Single-GPU training, `SCALE=probe\|reduced\|paper` |
| `repro/05_extract_draft.sh` | FSDP checkpoint → loadable drafter |
| `repro/make_val_jsonl.py` | Builds the 4 validation JSONLs the trainer needs |
| `repro/constraints.txt` | Pins torch/numpy/transformers against silent drift |
| `patches/001-launcher-fixes.patch` | Three fixes to the upstream training launcher |

`repro/README.md` has the detailed guide: replication targets, memory tuning
order, and the full list of known friction points.

## Setup

```bash
git clone https://github.com/Simplified-Reasoning/Draft-OPD.git
cd Draft-OPD

git clone https://github.com/Amirgh8080/draft-opd-repro.git /tmp/repro
cp -r /tmp/repro/repro .
git apply /tmp/repro/patches/001-launcher-fixes.patch

bash repro/10_bootstrap_and_run.sh --scale probe -y --with-smoke
```

Requires Ubuntu, an NVIDIA driver ≥ 550, conda, and ~80 GB free disk.

## What the patch changes

Three fixes to `verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`:

1. `pkill -9 ray; pkill -9 python` on startup — gated behind `KILL_STALE_PROCS=True`
   (default off). Upstream SIGKILLed every Python process on the machine.
2. `GPU_STRESS_TEST_SCRIPT` defaulted to a path on the authors' cluster and ran
   in an `EXIT` trap, so it failed on every run. Now off by default.
3. `TEST_JSONLS` was hard-coded to the authors' `/mnt/shared-storage-user/` paths.
   Now points at `repro/data/` and is overridable per-file.

## Install hazards this handles

Six ways the stack installs *successfully* but wrong:

1. **setuptools-scm cannot version the sglang fork.** Its `pyproject.toml` sets
   `root = ".."` (no `.git` there) and greps for `v*.*.*` tags the clone lacks,
   so it falls back to `0.0.0.dev0`. verl then asserts
   `sglang.__version__ >= 0.5.5` (`sglang_rollout.py:194`) and dies *at rollout
   time*, long after the install looked fine. Fixed by exporting
   `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.8`.
2. **numpy gets downgraded.** verl pins `numpy<2.0.0`; the sglang stack needs
   numpy 2.x. A plain editable install of verl drops numpy to 1.x and breaks
   every extension built against the numpy 2 ABI. Fixed with `--no-deps` plus an
   explicit dependency install under `constraints.txt`.
3. **torch drifts off 2.9.1.** `tensordict` and `torchdata` can pull a different
   torch; `sgl-kernel 0.3.21` is ABI-bound to 2.9.1. Pinned and asserted.
4. **torchcodec has no FFmpeg.** A hard sglang dependency on linux-x86_64 that
   fails at *import*, not install. The `system_deps` stage installs it.
5. **Git `safe.directory`.** If the clone sits on a mount whose owner differs
   from the running user, git aborts with `detected dubious ownership`.
   `setuptools_scm` shells out to git during the sglang build — via its file
   finder, which runs even with the pretend version set — so the build dies
   with only a generic pip message. The `git_safe` stage detects and fixes it.
6. **Resumed runs installing into the wrong Python.** Env activation must not
   sit behind a stage marker. It originally lived inside `make_env`, so a
   resumed run skipped activation with the stage and installed into conda's
   *base* interpreter. Tell: cp313 wheel tags when the env is 3.12; visible
   error: `can't find Rust compiler`, because `outlines_core==0.1.26` has no
   cp313 wheel. `activate_env` now runs unconditionally and asserts the
   interpreter path and version.

## Reproducibility on a consumer GPU

Judge replication on **acceptance length τ**, not on speedup ratios.

τ is the mean number of draft tokens accepted per verification round — a
property of the draft/target pair, the prompts, and the temperature. It is
hardware-independent, so the paper's Qwen3-4B result (DFlash τ = 6.04 →
Draft-OPD τ = 6.60, non-thinking, temperature 0) is exactly reproducible here.

The speedup multipliers are not. They depend on the memory-bandwidth-to-compute
ratio (3090 ≈ 936 GB/s, 4090 ≈ 1008 GB/s, vs H200 ≈ 4.8 TB/s), so absolute ×
values will differ.
Report them as measured on your hardware.

**Training at paper scale is not feasible on one GPU.** The paper's recipe is
16K prompts × 8 epochs × up to 4096 tokens on 8× H200 — on the order of 100×
the wall clock here. `SCALE=paper` exists and will run, with a warning.
`SCALE=reduced` (2000 prompts, 2 epochs, 1024 tokens, ~1 day) tests the paper's
actual claim: Figure 1, where SFT plateaus and OPD keeps improving accepted
length.

## Status

Run on an **RTX 4090** (SM89, 24 GB, driver 595, Ubuntu noble) on 2026-09-20.

Confirmed working: preflight and GPU detection, the SM89 backend branch,
`system_deps` (ffmpeg, libnuma, ninja, git-lfs), and conda env creation
(python 3.12.14).

Two real bugs surfaced and are fixed:

1. Git `safe.directory` on a `/mnt` clone broke the sglang editable build
   (`git_safe` stage).
2. Env activation sat behind a stage marker, so resumed runs installed into
   conda's base Python 3.13 instead of the 3.12 env (`activate_env`, now
   unconditional and asserted).

All 21 pinned dependencies were verified against PyPI to have cp312 Linux
wheels, so nothing needs to compile from source on Python 3.12.

The stages after `sglang` (`verl`, `verify`, training) have not yet completed
end-to-end.

## Credit

All research credit belongs to the Draft-OPD authors:

```bibtex
@misc{lei2026draftopdonpolicydistillationspeculative,
      title={Draft-OPD: On-Policy Distillation for Speculative Draft Models},
      author={Haodi Lei and Yafu Li and Haoran Zhang and Shunkai Zhang and
              Qianjia Cheng and Xiaoye Qu and Ganqu Cui and Bowen Zhou and
              Ning Ding and Yun Luo and Yu Cheng},
      year={2026},
      eprint={2605.29343},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.29343},
}
```

Upstream: [Simplified-Reasoning/Draft-OPD](https://github.com/Simplified-Reasoning/Draft-OPD).
It builds on [DFlash](https://github.com/z-lab/dflash) (MIT),
[SGLang](https://github.com/sgl-project/sglang) (Apache 2.0),
[verl](https://github.com/volcengine/verl) (Apache 2.0), and
[EAGLE-3](https://github.com/SafeAILab/EAGLE).

The harness in this repo is MIT licensed (see `LICENSE`). It contains no
upstream code — only a patch file describing changes to it.
