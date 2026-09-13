#!/usr/bin/env python3
"""Generate the validation JSONL files the OPD trainer expects.

The repo ships only the 16K training file. run_qwen_gsm8k.sh points TEST_JSONLS
at the authors' machine paths (/mnt/shared-storage-user/leihaodi/opd/data/...),
which do not exist for anyone else. This rebuilds equivalents.

Output schema matches data/apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt.jsonl:
    {"data_source": <name>, "prompt": [{"role": "user", "content": <text>}]}

Prompt templates are copied verbatim from diffusion/dflash/model/utils.py so the
training-time validation prompts match the evaluation-time prompts exactly.

Usage:
    python repro/make_val_jsonl.py --out-dir repro/data
"""
import argparse
import json
import pathlib
import sys

from datasets import load_dataset

MATH_FMT = "{problem}\nPlease reason step by step, and put your final answer within \boxed{{}}."
GSM8K_FMT = "{question}\nPlease reason step by step, and put your final answer within \boxed{{}}."


def _gsm8k():
    # utils.py uses the bare deprecated id "gsm8k"; "openai/gsm8k" is the same data.
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds, lambda x: GSM8K_FMT.format(question=x["question"])


SPECS = {
    # name        (loader,                                                    formatter,                          n)
    "aime24":     (lambda: load_dataset("HuggingFaceH4/aime_2024", split="train"),
                   lambda x: MATH_FMT.format(problem=x["problem"]), 30),
    "gsm8k":      (None, None, 128),  # special-cased below
    "math500":    (lambda: load_dataset("HuggingFaceH4/MATH-500", split="test"),
                   lambda x: MATH_FMT.format(problem=x["problem"]), 128),
    "mbpp":       (lambda: load_dataset("google-research-datasets/mbpp", "sanitized", split="test"),
                   lambda x: x["prompt"], 128),
}


def build(name, out_dir, seed):
    if name == "gsm8k":
        ds, fmt = _gsm8k()
        n = SPECS[name][2]
    else:
        loader, fmt, n = SPECS[name]
        ds = loader()

    # benchmark.py:252 uses shuffle(seed=0).select(range(max_samples)); match it
    # so the trainer's validation subset lines up with the eval harness subset.
    if n is not None and len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))

    out_path = out_dir / f"{name}_{len(ds)}_user_prompt.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = {
                "data_source": name,
                "prompt": [{"role": "user", "content": fmt(row)}],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {name:<9} {len(ds):>4} rows -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="repro/data")
    ap.add_argument("--seed", type=int, default=0, help="must match benchmark.py's shuffle seed (0)")
    ap.add_argument("--only", nargs="*", default=list(SPECS), choices=list(SPECS))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing validation JSONL to {out_dir.resolve()}")
    paths, failed = [], []
    for name in args.only:
        try:
            paths.append(build(name, out_dir, args.seed))
        except Exception as e:
            failed.append((name, e))
            print(f"  {name:<9} FAILED: {e}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)} dataset(s) failed. Run repro/01_prefetch_assets.sh first.", file=sys.stderr)
        raise SystemExit(1)

    print("\nHydra override for the trainer (paste into your launch command):")
    joined = ",".join(f"'{p.resolve()}'" for p in paths)
    print(f'  "data.val_files=[{joined}]"')


if __name__ == "__main__":
    main()
