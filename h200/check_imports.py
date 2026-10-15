#!/usr/bin/env python3
"""Fail if the installed stack drifted from what the recipe needs."""
import importlib
import os
import sys

import numpy
import sglang
import torch
import transformers

fail = []
print(f"  torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"  numpy        {numpy.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  sglang       {sglang.__version__} ({sglang.__file__})")
if not torch.__version__.startswith("2.9.1"):
    fail.append("torch drifted off 2.9.1; sgl-kernel 0.3.21 is ABI-bound to it")
if not numpy.__version__.startswith("2."):
    fail.append("numpy was downgraded; reinstall 'numpy>=2,<3'")
if "sglang-dflash" not in sglang.__file__:
    fail.append("stock sglang shadows the DFlash fork")

mods = ["flashinfer", "sgl_kernel", "verl", "ray", "tensordict",
        "sglang.srt.speculative.dflash_worker", "torch.nn.attention.flex_attention"]
if os.environ.get("SKIP_FLASH_ATTN") != "1":
    mods.insert(0, "flash_attn")  # target model path: flash_attention_2 + packed sequences
for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"  {mod:<38} ok")
    except Exception as e:  # noqa: BLE001 -- report every failure, not just the first
        print(f"  {mod:<38} FAILED: {e}")
        fail.append(f"{mod}: {e}")

if fail:
    print("\nBLOCKING:\n  - " + "\n  - ".join(fail))
    sys.exit(1)
print("  all imports ok")
