# Draft-OPD replication on one RTX 3090

Single-GPU (24 GB, SM86) Ubuntu setup for arXiv 2605.29343.

## Quick start

One command: builds the env, verifies it, fetches assets, generates validation
data, and launches training.

```bash
bash repro/10_bootstrap_and_run.sh --scale probe -y --with-smoke
```

Stages are idempotent -- markers land in `repro/.state/`, so a re-run resumes
rather than reinstalling. Delete a marker to force that stage again.

```bash
bash repro/10_bootstrap_and_run.sh --scale reduced --train-only   # retrain only
bash repro/10_bootstrap_and_run.sh --skip-install                 # env exists
rm repro/.state/verify.done && bash repro/10_bootstrap_and_run.sh # recheck env
```

Flags: `--scale probe|reduced|paper`, `--env-name`, `--skip-install`,
`--train-only`, `--with-smoke`, `--with-flash-attn`, `-y`.

## Manual, step by step

If you would rather drive it yourself:

```bash
bash repro/00_setup_ubuntu.sh          # conda env + pinned deps + import check
conda activate draftopd
bash repro/01_prefetch_assets.sh       # REQUIRED: eval runs offline
bash repro/02_smoke_test.sh            # ~20 min, 4 prompts. Is tau ~5-7?
bash repro/03_eval_table1.sh           # Phase 1: reproduce Table 1's Q3-4B row
python repro/make_val_jsonl.py         # build the trainer's validation files
SCALE=probe   bash repro/04_train_1gpu.sh   # does training run without OOM?
SCALE=reduced bash repro/04_train_1gpu.sh   # Phase 2
bash repro/05_extract_draft.sh         # FSDP checkpoint -> loadable drafter
```

## Install hazards the bootstrap handles

The four ways this stack silently installs wrong:

1. **setuptools-scm cannot version the sglang fork.** Its `pyproject.toml` sets
   `root = ".."` (no `.git` there) and greps for `v*.*.*` tags this clone lacks,
   so it falls back to `0.0.0.dev0`. verl then **asserts**
   `sglang.__version__ >= 0.5.5` (`sglang_rollout.py:194`) and dies *at rollout
   time*, long after the install looked fine. Fixed by exporting
   `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.8`, the fork's real base.
2. **numpy gets downgraded.** verl pins `numpy<2.0.0`; the sglang stack needs
   numpy 2.x. `pip install -e ./verl` quietly drops numpy to 1.x and breaks
   every extension built against the numpy 2 ABI. Fixed by installing verl with
   `--no-deps`, then its dependencies explicitly under `constraints.txt`.
3. **torch drifts off 2.9.1.** `tensordict` and `torchdata` can pull a different
   torch; `sgl-kernel 0.3.21` is ABI-bound to 2.9.1. `constraints.txt` pins it,
   and the bootstrap asserts the version after the verl step.
4. **torchcodec has no FFmpeg.** It is a hard sglang dependency on linux-x86_64
   and fails at *import*, not install, without FFmpeg shared libs. The
   `system_deps` stage installs `ffmpeg` and friends.

`flash-attn` is **not** installed by default and is not required -- the DFlash
student uses torch `flex_attention`. Use `--with-flash-attn` if you want it.

## Phase 1 — evaluation (this is the real replication)

Target row, paper Table 1, **thinking disabled, temperature 0, Qwen3-4B**:

| Method | GSM8K | MATH-500 | AIME25 | MBPP | HumanEval | SWE-Lite | MT-Bench | mean | **τ** |
|---|---|---|---|---|---|---|---|---|---|
| DFlash | 5.36× | 6.35× | 5.93× | 5.00× | 5.19× | 3.05× | 3.01× | 4.84× | **6.04** |
| Draft-OPD | 6.22× | 7.22× | 6.39× | 5.40× | 5.65× | 3.18× | 3.09× | 5.31× | **6.60** |

**Judge your replication on τ, not on the speedups.** Acceptance length τ is the
mean number of draft tokens accepted per verification round — a property of the
draft/target pair, the prompts, and the temperature. It is *hardware-independent*,
so 6.04 → 6.60 is exactly reproducible on a 3090.

The speedup multipliers are not. They depend on the memory-bandwidth-to-compute
ratio (3090 ≈ 936 GB/s vs H200 ≈ 4.8 TB/s), so your absolute × values will
differ. Report them as measured on your hardware and make τ the headline.

Why the non-thinking row first: it is the one where both checkpoints are
unambiguously matched and public — `z-lab/Qwen3-4B-DFlash-b16` (baseline) and
`bingyang-lei/Qwen3-4B-Ins-Draft-OPD` (method). For thinking mode, run
`MODE=thinking bash repro/03_eval_table1.sh`, but confirm first that the DFlash
baseline drafter you use was trained in thinking mode — otherwise the comparison
is not matched.

**Out of reach on 24 GB:** Table 2's concurrency sweep to 32. At 32 concurrent
requests × 8192 tokens the KV cache alone needs ~40 GB. Concurrency 1–4 works
and shows the trend.

## Phase 2 — training

The paper's recipe is 16K prompts × 8 epochs × up to 4096 tokens on 8× H200.
On one 3090 that is on the order of 100× the wall clock — months. `SCALE=paper`
exists in the launcher and will run if you ask for it, with an ETA warning.

`SCALE=reduced` (2000 prompts, 2 epochs, 1024 tokens, anchor stride 2, ~1 day)
is the version that tests the paper's actual claim: **Figure 1** — SFT plateaus
while OPD keeps pushing accepted length up. Track `val/accept_length` across
steps; a rising curve against a flat SFT baseline reproduces the core result
even at reduced scale. One Table 3 ablation is nearly free on top, since it is
the same pipeline with different flags:

```bash
# acceptance-aware KL (default) vs all-forward KL
EXP_NAME=ablation-allforward SCALE=reduced bash repro/04_train_1gpu.sh \
  distillation.distillation_loss.rejected_draft_use_reverse_kl=False
```

Paper hyperparameters (Appendix A), already the defaults in `-ins.sh`: AdamW,
LR 3e-4, cosine schedule, warmup ratio 0.05, 8 epochs, γ = 0.8, λ_acc = λ_rej = 1.

## Upstream patches applied

`git diff` shows three edits to `run_qwen_gsm8k.sh`, all revertible with
`git checkout verl/examples/`:

1. `pkill -9 ray; pkill -9 python` on startup is now behind `KILL_STALE_PROCS=True`
   (default off). Upstream killed every Python process on the machine.
2. `GPU_STRESS_TEST_SCRIPT` defaulted to a path on the authors' cluster and ran
   in an `EXIT` trap, so it failed on every run. Now defaults to off.
3. `TEST_JSONLS` hard-coded `/mnt/shared-storage-user/leihaodi/...`. Now points
   at `repro/data/` and is overridable per-file.

Not patched, just avoided: `verl/scripts/fsdp_to_dflash.sh` does `rm -rf` on the
actor dir after conversion. Use `repro/05_extract_draft.sh` instead.

## Known friction

- **flash-attn is not optional for the fast path.** `flex_attention` covers only
  the DRAFT module's block-mask attention. The frozen TARGET model inside the
  composed student is loaded by transformers with `flash_attention_2` by default
  (`verl/workers/config/model.py:186`), and verl's unpadded path imports
  `flash_attn.bert_padding` (`transformer_impl.py:1735/2041/2215`). Without
  flash_attn both fail. `04_train_1gpu.sh` auto-detects and falls back to
  `attn_implementation=sdpa` + `use_remove_padding=False` — numerically
  equivalent, just slower. Override with `ATTN_IMPL=`. There is no prebuilt
  flash-attn wheel for torch 2.9 + cu12 (upstream ships cu13torch2.9 only), so
  the fast path needs a CUDA toolkit and `--with-flash-attn`.
  *Confirmed on an RTX 4090 box, 2026-09-24.*
- **Resumed runs and the wrong Python.** Environment activation must never sit
  behind a stage marker. It originally lived inside `make_env`, so a resumed run
  skipped the activation along with the stage and installed into conda's *base*
  interpreter. The tell is cp313 wheel tags in the log when the env is 3.12, and
  the visible error is `can't find Rust compiler` — `outlines==0.1.11` pins
  `outlines_core==0.1.26`, which has cp39–cp312 wheels but no cp313, so pip
  falls back to a Rust source build. `activate_env` now runs unconditionally and
  asserts both the interpreter path and its version.
  *Confirmed on an RTX 4090 box, 2026-09-23.*
- **Git `safe.directory`.** If the clone sits on a mount whose owner differs
  from the running user (a secondary `/mnt` partition, or a clone made by root),
  git aborts with `detected dubious ownership`. `setuptools_scm` shells out to
  git during the sglang editable build -- via its file finder, which runs even
  with `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG` set -- so the build dies with
  only a generic pip message. The `git_safe` stage detects and fixes this:
  `git config --global --add safe.directory <repo>`.
  *Confirmed on an RTX 4090 box, 2026-09-20.*
- **Offline mode.** `sglang_run_bench.sh` forces `HF_HUB_OFFLINE=1` and
  `HF_DATASETS_OFFLINE=1`. Anything not prefetched fails with a confusing error.
  Run `01_prefetch_assets.sh` first.
- **FA3 on SM86.** Every shipped eval config says `attention_backends: fa3`.
  `benchmark_sglang.py:797` drops fa3 on non-SM90 and line 800 falls back to
  flashinfer, so this self-corrects — but set `flashinfer` explicitly to be sure
  you know which backend produced your numbers.
- **numpy pin conflict.** verl's `setup.py` pins `numpy<2.0.0`; the sglang stack
  generally wants numpy ≥ 2. Installing verl second can downgrade numpy and
  break ABI. If flashinfer/sglang imports fail, `pip install 'numpy>=2.0'` —
  verl works fine with it in practice.
- **Never `pip install -e ".[sglang]"`** in verl. That extra pins
  `sglang==0.5.8` from PyPI and shadows the vendored DFlash fork.
- **Parallel bench jobs.** `launch_sglang_bench_jobs.py:317` `Popen`s all jobs at
  once. Two jobs on `gpu: 0` will fight over the same 24 GB.
  `03_eval_table1.sh` calls `benchmark_sglang.py` directly, sequentially.
- **`data_source` and rewards.** `reward_score/__init__.py:107` raises
  `NotImplementedError` for unknown sources. The shipped training data uses
  `"aops"`, which is not in the registry — confirming rewards are never computed
  in this pipeline (`use_task_rewards=False`, `rollout_speed_test_only=True`).
  The generated validation files use plain benchmark names for the same reason.
- **Long runs.** `trainer.resume_mode` is `disable` upstream; the launcher
  overrides it to `auto`.

## Memory tuning order on 24 GB

If training OOMs, relax in this order:

1. `TEACHER_GPU_MEMORY_UTILIZATION` (default 0.55) — rollout engine weights + KV
2. `RESP_LEN_OVERRIDE` — shorter responses
3. `ANCHOR_STRIDE_OVERRIDE` — fewer replay positions per sample (costs fidelity
   to the paper, which uses stride 1)

Already on by default in the launcher: gradient checkpointing, FSDP param
offload, FSDP optimizer offload, micro-batch 1.
