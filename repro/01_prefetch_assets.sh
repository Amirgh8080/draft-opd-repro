#!/usr/bin/env bash
# Pre-download every model and dataset Draft-OPD evaluation touches.
# REQUIRED: sglang_run_bench.sh forces HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1,
# so anything not cached ahead of time fails at run time with a confusing error.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export HF_HUB_ENABLE_HF_TRANSFER=1
# Force ONLINE for this script only.
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

echo "HF_HOME=$HF_HOME"
mkdir -p "$HF_HOME"

python - <<'PYEOF'
import os, traceback
from huggingface_hub import snapshot_download
from datasets import load_dataset

MODELS = [
    ("Qwen/Qwen3-4B",                       "target model (~8 GB)"),
    ("z-lab/Qwen3-4B-DFlash-b16",           "DFlash baseline drafter (~1 GB)"),
    ("bingyang-lei/Qwen3-4B-Ins-Draft-OPD", "Draft-OPD drafter, non-thinking (~1 GB)"),
    # Uncomment for the thinking-mode rows of Table 1:
    # ("bingyang-lei/Qwen3-4B-Thinking-Draft-OPD", "Draft-OPD drafter, thinking"),
]

# (hf_id, config, split) -- must match diffusion/dflash/model/utils.py exactly.
DATASETS = [
    ("openai/gsm8k",                     "main",      "test"),
    ("HuggingFaceH4/MATH-500",           None,        "test"),
    ("math-ai/aime25",                   None,        "test"),
    ("HuggingFaceH4/aime_2024",          None,        "train"),
    ("google-research-datasets/mbpp",    "sanitized", "test"),
    ("openai/openai_humaneval",          None,        "test"),
    ("princeton-nlp/SWE-bench_Lite",      None,        "test"),
    ("HuggingFaceH4/mt_bench_prompts",   None,        "train"),
]

failed = []

print("=" * 64); print("MODELS"); print("=" * 64)
for repo, note in MODELS:
    print(f"-> {repo}  ({note})")
    try:
        snapshot_download(repo_id=repo, resume_download=True)
        print(f"   ok")
    except Exception as e:
        failed.append((repo, repr(e))); print(f"   FAILED: {e}")

print(); print("=" * 64); print("DATASETS"); print("=" * 64)
for name, cfg, split in DATASETS:
    label = f"{name}" + (f" [{cfg}]" if cfg else "") + f" ({split})"
    print(f"-> {label}")
    try:
        ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        print(f"   ok, {len(ds)} rows")
    except Exception as e:
        failed.append((label, repr(e))); print(f"   FAILED: {e}")

print()
if failed:
    print("!" * 64)
    print(f"{len(failed)} asset(s) failed to download:")
    for name, err in failed:
        print(f"  - {name}: {err}")
    print("Fix these before running evaluation; offline mode will not recover.")
    print("Note: `gsm8k` is loaded by utils.py under its bare deprecated id.")
    print("      If that errors at eval time, the cached `openai/gsm8k` copy is")
    print("      the same data -- see repro/README.md, 'Known friction'.")
    raise SystemExit(1)
print("All assets cached. Offline evaluation will now work.")
PYEOF
