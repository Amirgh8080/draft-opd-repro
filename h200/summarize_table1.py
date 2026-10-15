#!/usr/bin/env python3
"""Turn benchmark_sglang.py reports into a paper-style Table 1 row.

Reads <dir>/baseline.md and every other <dir>/<label>.md, takes the
concurrency-1 column, and prints per-benchmark speedup (drafter tok/s divided by
the shared baseline tok/s) and acceptance length tau, next to the paper's
Qwen3-4B non-thinking numbers.

    python h200/summarize_table1.py /workspace/logs/eval/nothinking_temp0.0
"""
import argparse
import pathlib
import re
import sys

COLUMNS = ["gsm8k", "math500", "aime25", "mbpp", "humaneval", "swe-bench", "mt-bench"]
HEADERS = ["GSM8K", "MATH-500", "AIME25", "MBPP", "HumanEval", "SWE-Lite", "MT-Bench"]

# Paper Table 1, Thinking disabled, temperature 0, Qwen3-4B, H200.
PAPER_NOTHINKING = {
    "dflash":   ([5.36, 6.35, 5.93, 5.00, 5.19, 3.05, 3.01], 4.84, 6.04),
    "draftopd": ([6.22, 7.22, 6.39, 5.40, 5.65, 3.18, 3.09], 5.31, 6.60),
}

DATASET_RE = re.compile(r"^### Dataset: `([^`:]+)(?::\d+)?`")
METRIC_RE = re.compile(r"^#### (.+)$")


def parse(path):
    """-> {dataset: {"base": tok/s, "spec": tok/s, "tau": float}} at concurrency 1."""
    out, ds, metric, conc_idx = {}, None, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if m := DATASET_RE.match(line):
            ds, metric = m.group(1), None
            out.setdefault(ds, {})
        elif m := METRIC_RE.match(line):
            h = m.group(1)
            metric = ("base" if h.startswith("Baseline output tok/s") else
                      "tau" if h.endswith("acceptance length") else
                      "spec" if h.endswith("output tok/s") else None)
        elif ds and metric and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == "conc":
                conc_idx = cells.index("1") if "1" in cells else None
            elif cells[0] == "value" and conc_idx is not None:
                v = cells[conc_idx]
                if v != "N/A":
                    out[ds][metric] = float(v.replace(",", ""))
                metric = None
    return out


def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir", type=pathlib.Path)
    ap.add_argument("--mode", default="nothinking")
    args = ap.parse_args()

    base_path = args.report_dir / "baseline.md"
    base = parse(base_path) if base_path.exists() else {}
    if not base:
        print(f"note: no baseline at {base_path}; speedups unavailable", file=sys.stderr)

    rows = []
    for path in sorted(args.report_dir.glob("*.md")):
        if path.name in ("baseline.md", "summary.md"):
            continue
        rep = parse(path)
        speed = [
            (rep.get(c, {}).get("spec") / base[c]["base"])
            if rep.get(c, {}).get("spec") and base.get(c, {}).get("base") else None
            for c in COLUMNS
        ]
        taus = [rep.get(c, {}).get("tau") for c in COLUMNS]
        rows.append((path.stem, speed, taus))

    print(f"## Table 1 reproduction — Qwen3-4B, {args.mode}, concurrency 1\n")
    print("Speedup over autoregressive baseline (same H200, same run settings); "
          "tau = mean of per-benchmark acceptance length.\n")
    print("| Method | " + " | ".join(HEADERS) + " | mean | τ |")
    print("|---" * (len(HEADERS) + 3) + "|")
    for label, speed, taus in rows:
        print(f"| {label} | " + " | ".join(fmt(s, ".2f") + ("×" if s else "") for s in speed)
              + f" | {fmt(mean(speed), '.2f')}× | **{fmt(mean(taus), '.2f')}** |")
        if args.mode == "nothinking" and label in PAPER_NOTHINKING:
            ps, pm, pt = PAPER_NOTHINKING[label]
            print(f"| *{label} (paper)* | " + " | ".join(f"{s:.2f}×" for s in ps)
                  + f" | {pm:.2f}× | **{pt:.2f}** |")

    print("\nPer-benchmark acceptance length τ:\n")
    print("| Method | " + " | ".join(HEADERS) + " |")
    print("|---" * (len(HEADERS) + 1) + "|")
    for label, _, taus in rows:
        print(f"| {label} | " + " | ".join(fmt(t, ".2f") for t in taus) + " |")

    if base:
        print("\nBaseline output tok/s: "
              + ", ".join(f"{h} {fmt(base.get(c, {}).get('base'), ',.1f')}"
                          for c, h in zip(COLUMNS, HEADERS)))


if __name__ == "__main__":
    main()
