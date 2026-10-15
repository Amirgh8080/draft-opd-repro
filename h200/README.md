# Draft-OPD on one H200: replicating the paper

This directory runs **upstream Draft-OPD as published**, with the paper's
recipe, on a single H200 inside a container. It is separate from `repro/`,
which trims the recipe to fit 24 GB cards. Nothing here is scaled down.

The upstream code is pinned to
[`Simplified-Reasoning/Draft-OPD@d6f0ec8`](https://github.com/Simplified-Reasoning/Draft-OPD/commit/d6f0ec859321959785c65b59eb311fd25f042158).
The only change to it is `patches/001-launcher-fixes.patch`, which edits the
launcher and nothing else. It makes `pkill -9 python` opt-in, removes the GPU
stress test that pointed at the authors' cluster, and makes the validation
paths overridable.

## Two phases

| Phase | What it reproduces | Script | H200 time |
|---|---|---|---|
| 1. Eval | Table 1, Qwen3-4B row, using the **released** checkpoints | `03_eval_table1.sh` | hours |
| 2. Train | The released Draft-OPD drafter, retrained from DFlash with the paper recipe | `04_train_paper.sh` | up to ~7× the authors' 7-GPU wall clock; measure with `TIMING_PROBE=1` |

Phase 1 is the direct replication of the paper's numbers. On an H200 with the
fa3 backend, **both τ and the speedups** can be compared with the paper,
because the paper measured on this hardware. Phase 2 then checks that the
method itself produces that checkpoint.

## Quick start

### A. Build the image (Docker, Kubernetes, Slurm with enroot/pyxis)

```bash
git clone https://github.com/Amirgh8080/draft-opd-repro.git && cd draft-opd-repro
docker build -f h200/Dockerfile -t draft-opd-h200 .          # ~30-45 min (flash-attn compile)

docker run --gpus '"device=0"' --shm-size=64g --ulimit memlock=-1 \
  -v /big/disk:/workspace -it draft-opd-h200
# inside the container:
bash h200/run_all.sh --eval-only      # Phase 1
bash h200/run_all.sh --train-only     # Phase 2 (re-run the same command to resume)
```

### B. GPU cloud where you pick an image (RunPod, Vast, Lambda, …)

Start a 1×H200 pod from **`lmsysorg/sglang:v0.5.8-cu129-amd64`** with a large
volume on `/workspace`, then run:

```bash
cd /workspace
git clone https://github.com/Amirgh8080/draft-opd-repro.git && cd draft-opd-repro
bash h200/run_all.sh --eval-only      # clones upstream, installs, downloads, evaluates
bash h200/run_all.sh --train-only
```

`run_all.sh` installs into the image's Python the first time it runs.
Upstream is cloned into `/workspace/Draft-OPD`, so it persists across pod
restarts.

### Requirements

- 1× H200 (SM90, 141 GB). `preflight.sh` fails below 139 GB of VRAM, and warns
  on non-SM90 GPUs because the bench then drops fa3 and the speedups stop being
  comparable.
- `/dev/shm` ≥ 16 GB. Use `--shm-size=64g` or `--ipc=host`. Docker's 64 MB
  default breaks Ray and SGLang.
- Disk: ~15 GB of models and datasets, plus ~25–30 GB per kept training
  checkpoint (estimated; `KEEP_CKPTS=4`). The image itself is ~30 GB.
- RAM ≥ 120 GB recommended, because checkpoint saves gather the fp32 actor on
  CPU.
- NVIDIA driver ≥ 550. The cu129 image requests driver ≥ 575. On an older host,
  pass `-e NVIDIA_DISABLE_REQUIRE=1`, and CUDA minor-version compatibility then
  runs it.

## Phase 1: Table 1 with the released checkpoints

```bash
bash h200/03_eval_table1.sh                    # thinking disabled (Table 1 target row)
MODE=thinking bash h200/03_eval_table1.sh      # thinking rows, 8192 new tokens
```

The settings are those of the authors' `eval_config/*.yaml` and
`benchmark_sglang.py` defaults:

| Setting | Value |
|---|---|
| Target | `Qwen/Qwen3-4B` |
| Drafters | `z-lab/Qwen3-4B-DFlash-b16` (baseline), `bingyang-lei/Qwen3-4B-Ins-Draft-OPD` |
| Backend | SGLang DFlash fork, `fa3`, tp 1, bf16, `mem_fraction_static` 0.75 |
| Decoding | temperature 0 (top-k 1), concurrency 1 |
| Max new tokens | 2048 (non-thinking) / 8192 (thinking) |
| Benchmarks | gsm8k:128 math500:128 aime25:30 mbpp:128 humaneval:128 swe-bench:128 mt-bench:80 |

The autoregressive baseline runs once and is shared by every drafter, so all
speedups use the same denominator. `summary.md` puts your numbers next to the
paper's:

| Method | GSM8K | MATH-500 | AIME25 | MBPP | HumanEval | SWE-Lite | MT-Bench | mean | **τ** |
|---|---|---|---|---|---|---|---|---|---|
| DFlash (paper) | 5.36× | 6.35× | 5.93× | 5.00× | 5.19× | 3.05× | 3.01× | 4.84× | **6.04** |
| Draft-OPD (paper) | 6.22× | 7.22× | 6.39× | 5.40× | 5.65× | 3.18× | 3.09× | 5.31× | **6.60** |

The summary's τ is the mean of the per-benchmark acceptance lengths. Its mean
speedup is the mean of the seven columns, which is how the paper's `mean`
column is computed (5.36…3.01 averages to 4.84).

## Phase 2: training with the paper's recipe

```bash
TIMING_PROBE=1 bash h200/04_train_paper.sh   # 20 steps, separate dir: measure s/step first
bash h200/04_train_paper.sh                  # the real run; re-run to resume
bash h200/05_extract_draft.sh                # latest checkpoint -> draft_model/
DRAFTS="mine=/workspace/checkpoints/qwen3-4b-ins-draftopd-h200x1/global_step_N/draft_model" \
  SKIP_BASELINE=1 bash h200/03_eval_table1.sh
```

The recipe is upstream `run_qwen_gsm8k_forward-ins.sh` with its defaults. The
launch log records every value, and a dry run confirmed which Hydra override
wins for each one:

| | Paper / upstream | Here |
|---|---|---|
| Data | all 16,000 prompts in `data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl` | same |
| Epochs | 8 | 8 |
| Batch | 21 prompts per rollout step × 10 accumulated steps | same |
| Lengths | prompt 512, response 4096 | same |
| Optimiser | AdamW, LR 3e-4, cosine, warmup 0.05 | same |
| Loss | k3; accepted tokens forward KL, rejected reverse KL; γ = 0.8; stream weights 1/1 | same |
| Anchor stride | 1 (every response position) | 1 |
| Precision | fp32 model init (verl default), bf16 rollout | same |
| Memory tricks | none: no offload, no gradient checkpointing | none |
| Attention | target `flash_attention_2` + packed sequences; draft `flex_attention` | same |
| Rollout | SGLang DFlash, fa3, temperature 0, `mem_fraction_static` 0.2 | same |
| Teacher | `composed_main`: the frozen target inside the student | same |
| Seed | 42 (data shuffle, FSDP, SGLang) | same |
| **Student GPUs** | **7** | **1** |

### Why one GPU does not change the optimisation

- **The global batch is counted on the driver.** `data.train_batch_size=21` and
  `trainer.update_accumulation_steps=10` do not depend on world size. The
  trainer prints `Total training steps`. Expect `floor(N/21) × 8`, about 6,088,
  minus any prompts that `filter_overlong_prompts` drops for exceeding 512
  tokens.
- **The gradient is the same.** `workers/utils/losses.py` divides each rank's
  loss by the global token count × `dp_size` before DP's gradient average. One
  rank holding 21 samples therefore produces the same gradient as 7 ranks
  holding 3 each.
- **There is no teacher GPU at any world size.** With a single teacher equal to
  the target, `teacher_logprob_source=auto` resolves to `composed_main`
  (`trainer/distillation/teacher_source.py`). `TEACHER_WORLD_SIZE` is then
  never allocated.

### What does differ, and what it means for "exact"

Bit-identical weights are not attainable, even on the authors' own 8 GPUs:
SGLang and FSDP kernels are not deterministic. Two runs agreeing on τ within
noise is the standard to aim for. Against that standard, these are the
differences:

1. **Rollout batching.** All 21 prompts share one SGLang engine, where upstream
   ran 3 per engine on each of 7 engines. Decoding is greedy, but batch shape
   changes floating-point reduction order, so a near-tie can resolve
   differently.
2. **Mini-batch grouping within an update phase.** `balance_batch` reorders the
   210 rollouts across DP ranks before splitting them into 10 mini-batches.
   With 1 rank the same 210 rollouts form different groups of 21.
3. **Operational settings that do not touch the math.** Checkpoints are saved
   every 250 steps instead of every 500 from step 5000, and the last 4 are
   kept. `resume_mode=auto` replaces `disable`. `EXP_NAME` is fixed, where
   upstream embeds the date. Validation uses 1 worker × 14 samples per
   benchmark, where upstream used 7 × 2. Validation only measures speed and
   never feeds back into training.
4. **Code version.** The pinned commit is the public September 2026 code, not
   necessarily the exact tree that trained the May 2026 paper checkpoints. That
   is why Phase 1, the released checkpoints on the paper's eval settings, is
   the primary replication.

Upstream's README says `TEACHER_GPU_MEMORY_UTILIZATION` defaults to 0.1. The
`-ins.sh` script it documents uses 0.2, and that is what runs here.

### If something goes wrong

- **OOM when the rollout engine starts:** lower
  `TEACHER_GPU_MEMORY_UTILIZATION`, for example to 0.15. With a smaller KV
  cache, SGLang queues requests instead of failing, and greedy outputs do not
  depend on cache size.
- **OOM during the update:** the paper's per-GPU micro-batch (1 sample, up to
  4,608 tokens) is unchanged. Only the fp32 target (~16 GB) and the drafter's
  optimiser state are now unsharded. If it still does not fit, setting
  `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True` changes memory
  only, not the math. Pass it as an extra argument to `04_train_paper.sh`.
- **Thinking mode:** check that the DFlash baseline drafter matches the
  thinking-mode Draft-OPD checkpoint before you compare the two.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `lmsysorg/sglang:v0.5.8-cu129-amd64` + pinned upstream + install |
| `run_all.sh` | preflight → code → install → assets → eval → train → extract → eval |
| `preflight.sh` | GPU / shm / RAM / disk / driver checks, plus the GPU-bound import check |
| `check_imports.py` | Fails on version drift (torch, numpy, shadowed sglang) or a missing extension |
| `00_get_upstream.sh` | clone at the pinned commit, apply the launcher patch |
| `01_install.sh` | sglang fork, verl (`--no-deps`), flash-attn 2.8.3 for sm90 |
| `02_prefetch.sh` | models + datasets into `$HF_HOME`, validation JSONLs |
| `03_eval_table1.sh` | Table 1 eval, sequential on one GPU |
| `04_train_paper.sh` | paper recipe with `STUDENT_WORLD_SIZE=1` |
| `05_extract_draft.sh` | FSDP checkpoint → drafter (non-destructive) |
| `summarize_table1.py` | reports → Table 1 row next to the paper's |
| `env.sh`, `constraints.txt` | shared paths and version pins |

Every path can be overridden through the environment: `WORKSPACE`,
`UPSTREAM_DIR`, `HF_HOME`, `CKPT_ROOT`, `LOG_ROOT`, `EXP_NAME`.
