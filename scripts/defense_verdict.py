#!/usr/bin/env python3
"""Score the defense experiment's pre-registered predictions (P-D1..P-D3).

Reads results/defense_eval/eval_{variant}_{model}.json (written by
defense_regen.slurm) and prints the R@1 grid plus the paired
D1-minus-ctrl-CAS gap per eval variant, then the verdict on each
registered prediction from scripts/defense_regen_eval.py's docstring.
"""

import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "defense_eval"
VARIANTS = ("orig", "exact", "paraphrase", "style", "mixed")
MODELS = ("base", "ctrl-cas", "d1-ctrl-llm", "d2-paraphrase", "d3-style", "d4-unrelated")


def r1(variant, model):
    p = RES / f"eval_{variant}_{model}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    for k in ("recall_at_1", "recall@1", "R@1"):
        if k in d:
            return 100 * d[k] if d[k] <= 1 else d[k]
    ov = d.get("overall") or d.get("recall") or {}
    for k in ("recall_at_1", "recall@1", "1"):
        if k in ov:
            return 100 * ov[k] if ov[k] <= 1 else ov[k]
    raise KeyError(f"no R@1 field in {p}: keys {sorted(d)[:12]}")


def main():
    grid = {v: {m: r1(v, m) for m in MODELS} for v in VARIANTS}
    print(f"{'variant':<12}" + "".join(f"{m:>15}" for m in MODELS))
    for v in VARIANTS:
        print(f"{v:<12}" + "".join(
            f"{grid[v][m]:>15.2f}" if grid[v][m] is not None else f"{'--':>15}"
            for m in MODELS))

    def gap(v):
        a, b = grid[v]["d1-ctrl-llm"], grid[v]["ctrl-cas"]
        return None if a is None or b is None else a - b

    gaps = {v: gap(v) for v in VARIANTS}
    print("\nD1 - ctrl-CAS gap per eval variant:")
    for v in VARIANTS:
        print(f"  {v:<12}{gaps[v] if gaps[v] is None else round(gaps[v], 2)}")

    if any(gaps[v] is None for v in VARIANTS):
        print("\n[verdict] incomplete grid -- run remaining evals first")
        return
    pd1 = gaps["exact"] >= 0.5 * gaps["orig"]
    pd2 = gaps["exact"] > gaps["paraphrase"] > gaps["style"]
    pd3 = min(gaps["exact"], gaps["style"]) < gaps["mixed"] < max(gaps["exact"], gaps["style"])
    fals = gaps["style"] >= gaps["exact"]
    print(f"\nP-D1 (attack survives same-recipe regeneration): {'PASS' if pd1 else 'FAIL'}")
    print(f"P-D2 (gap decays with eval-prompt distance):     {'PASS' if pd2 else 'FAIL'}")
    print(f"P-D3 (mixed eval strictly between endpoints):    {'PASS' if pd3 else 'FAIL'}")
    print(f"Falsifier gap(style) >= gap(exact):              {'TRIGGERED' if fals else 'not triggered'}")
    print("\nSecondary observable (not registered): D1-D3 gap on the style eval = "
          f"{grid['style']['d1-ctrl-llm'] - grid['style']['d3-style']:.2f}")


if __name__ == "__main__":
    main()
