# Where everything downloads from

Every web domain the H200 setup contacts, grouped by stage. Use it to build a
firewall or proxy allowlist, or to check that a restricted network can run
this. All traffic is HTTPS on port 443, except the optional Ubuntu mirrors that
use HTTP on port 80.

This list was compiled from the scripts, the upstream code paths they run, and
the installed packages. It was not captured from a live run's network traffic.

## 1. Ubuntu itself: OS packages, NVIDIA driver, Fabric Manager

| Domain | What comes from it |
|---|---|
| `archive.ubuntu.com` (or your regional mirror, e.g. `de.archive.ubuntu.com`) | OS updates; tools (git, tmux, ffmpeg, build-essential, …); the **NVIDIA driver** (`nvidia-driver-*-server*`, `nvidia-utils-*-server`, signed `linux-modules-nvidia-*`) and **Fabric Manager**, all from Ubuntu's `restricted` component |
| `security.ubuntu.com` | security updates for the same packages |

The NVIDIA driver does **not** come from nvidia.com on this path: Ubuntu
builds, signs and hosts it. Your configured mirrors are listed in
`/etc/apt/sources.list.d/ubuntu.sources` (24.04) or `/etc/apt/sources.list`
(22.04).

## 2. This repository's setup (no Docker: `setup_bare_metal.sh`)

| Domain | What comes from it |
|---|---|
| `github.com` | this repo (`Amirgh8080/draft-opd-repro`); upstream code **and the 16K-prompt training data** (`Simplified-Reasoning/Draft-OPD`, no git-LFS, no submodules); fallback source for uv's Python; flash-attn's prebuilt-wheel lookup |
| `release-assets.githubusercontent.com`, `objects.githubusercontent.com` | where GitHub release downloads redirect: the uv binary, Python fallback, flash-attn wheel lookup |
| `developer.download.nvidia.com` | NVIDIA's apt repo: `cuda-keyring` and `cuda-toolkit-12-8` (compiler for flash-attn; no driver) |
| `astral.sh`, `releases.astral.sh` | uv installer, and Python 3.12 (uv fetches it from `releases.astral.sh` first, then GitHub) |
| `pypi.org`, `files.pythonhosted.org` | every Python package: torch 2.9.1 and its CUDA 12.8 runtime libraries (`nvidia-*-cu12`), the SGLang fork's dependencies (sgl-kernel, flashinfer, transformers, …), verl's dependencies (Ray, hydra, …), and the flash-attn source |

If `github.com` release downloads are blocked, flash-attn still installs: it
compiles from the PyPI source instead, which needs the CUDA toolkit above.

## 3. Docker paths instead (GUIDE.md Paths A and B)

These replace section 2's NVIDIA apt repo and astral entries. Sections 1, 4,
`github.com`, `pypi.org` and `files.pythonhosted.org` still apply.

| Domain | What comes from it |
|---|---|
| `get.docker.com`, `download.docker.com` | Docker Engine install (Path A) |
| `nvidia.github.io` | NVIDIA Container Toolkit apt repo and key (Path A) |
| `registry-1.docker.io`, `auth.docker.io`, `production.cloudflare.docker.com` | the base image `lmsysorg/sglang:v0.5.8-cu129-amd64` from Docker Hub |
| `flashinfer.ai` | only when the image's flashinfer kernel cache must be matched to 0.6.4. If blocked, the cache is removed and kernels compile on first use instead. |

## 4. Models and datasets (`02_prefetch.sh`, first run of `run_all.sh`)

| Domain | What comes from it |
|---|---|
| `huggingface.co` | metadata and downloads for all models and datasets below |
| `*.hf.co`, `*.huggingface.co` | Hugging Face's file storage, which the hub redirects to: `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`, `transfer.xethub.hf.co`, `cdn-lfs*.hf.co`, `cdn-lfs.huggingface.co`. The exact host varies by file, so allow the wildcards. |

What is fetched (~10 GB in total, no account or token needed):

| Kind | Hugging Face id |
|---|---|
| Target model | `Qwen/Qwen3-4B` |
| Drafters | `z-lab/Qwen3-4B-DFlash-b16`, `bingyang-lei/Qwen3-4B-Ins-Draft-OPD`, `bingyang-lei/Qwen3-4B-Thinking-Draft-OPD` |
| Benchmarks | `gsm8k` / `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `math-ai/aime25`, `HuggingFaceH4/aime_2024`, `google-research-datasets/mbpp`, `openai/openai_humaneval`, `princeton-nlp/SWE-bench_Lite`, `HuggingFaceH4/mt_bench_prompts` |

## 5. During evaluation and training

**No network is required.** Once section 4 has run, the eval and training
scripts set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, Weights & Biases
logs offline, and `env.sh` disables Ray's usage reporting
(`RAY_USAGE_STATS_ENABLED=0`; otherwise Ray would contact
`usage-stats.ray.io`).

One possible exception: flashinfer downloads a precompiled kernel from
`edge.urm.nvidia.com` if that kernel is missing locally. The installed
`flashinfer-cubin` package normally ships them, so this should not happen. If
the machine is fully offline during runs and a kernel is reported missing,
allow that domain for the first run.

## Minimal allowlist (Linux, no Docker)

```
archive.ubuntu.com            security.ubuntu.com
github.com                    release-assets.githubusercontent.com
objects.githubusercontent.com developer.download.nvidia.com
astral.sh                     releases.astral.sh
pypi.org                      files.pythonhosted.org
huggingface.co                *.hf.co
*.huggingface.co
```

Optional: `edge.urm.nvidia.com` (flashinfer kernels, see section 5). Replace
`archive.ubuntu.com` with your regional mirror if you use one.
