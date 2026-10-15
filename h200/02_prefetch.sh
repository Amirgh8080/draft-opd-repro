#!/usr/bin/env bash
# Download every model and dataset the eval and training touch into $HF_HOME,
# then build the trainer's validation JSONLs.
#
# Required: the upstream bench forces HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1,
# so anything not cached here fails later with a confusing error.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
echo "HF_HOME=$HF_HOME"

python - "$TARGET_ID" "$DFLASH_ID" "$OPD_INS_ID" "$OPD_THINK_ID" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset

failed = []
for repo in sys.argv[1:]:
    try:
        print(f"model   {repo:<45} {snapshot_download(repo)}")
    except Exception as e:
        failed.append(repo); print(f"model   {repo:<45} FAILED: {e}")

# (id, config, split) exactly as diffusion/dflash/model/utils.py loads them.
# utils.py loads GSM8K under the bare legacy id "gsm8k"; offline mode resolves
# the cache by the id it was fetched under, so cache it under both.
DATASETS = [
    ("gsm8k", "main", "test"),
    ("openai/gsm8k", "main", "test"),
    ("HuggingFaceH4/MATH-500", None, "test"),
    ("math-ai/aime25", None, "test"),
    ("HuggingFaceH4/aime_2024", None, "train"),
    ("google-research-datasets/mbpp", "sanitized", "test"),
    ("openai/openai_humaneval", None, "test"),
    ("princeton-nlp/SWE-bench_Lite", None, "test"),
    ("HuggingFaceH4/mt_bench_prompts", None, "train"),
]
for name, cfg, split in DATASETS:
    try:
        ds = load_dataset(name, cfg, split=split)
        print(f"dataset {name:<45} {len(ds)} rows")
    except Exception as e:
        failed.append(name); print(f"dataset {name:<45} FAILED: {e}")

if failed:
    raise SystemExit(f"\n{len(failed)} asset(s) failed: {failed}")
PYEOF

# Same prompt templates and shuffle(seed=0) subsets as the eval harness.
python "$HARNESS_ROOT/repro/make_val_jsonl.py" --out-dir "$DATA_DIR"

echo
echo "Offline smoke check (this is how the bench will see the cache):"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PYEOF'
from datasets import load_dataset
load_dataset("gsm8k", "main", split="test")
print("  gsm8k loads offline under its bare id: ok")
PYEOF
