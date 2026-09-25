#!/usr/bin/env python3
"""E-R4 registered verdict -- written BEFORE any
cleanfull number existed; the verdict branch is computed in code.

Reads results/ranks/{easy,medium,hard}_cleanfull-{llm,cas}-s{42..49}.summary.json
and the xling summaries, and prints/writes:

  PRIMARY  eight-seed mean easy-tier gap (llm - cas) at the fully-clean
           matched 6,145 budget.
           G1 CONFIRMS  gap >= 35.0 and positive in all 8 seeds
           G2 PARTIAL   15.0 <= gap < 35.0, all seeds positive
           G3 REFUTES   gap < 15.0 (contamination section must be rewritten)
  Context  hard-tier means, xling strict R@1 means, and the published
           reference points (45.33 headline, ~43 at the 4,101 clean budget).

Writes results/cleanfull_verdict.json.
"""
import json, os, sys

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(42, 50))


def read(tier, arm, seed):
    p = os.path.join(PROJECT, "results", "ranks",
                     f"{tier}_cleanfull-{arm}-s{seed}.summary.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if tier == "xling":
        return d["strict_crosslingual_gold"]["recall@1"]
    return d["overall"]["recall@1"]


def main():
    out = {"generated_by": "scripts/cleanfull_verdict.py", "seeds": SEEDS,
           "reference": {"headline_gap_easy": 45.33,
                         "clean4101_gap_easy": "about 43 (Table tab:cleangate)"},
           "tiers": {}}
    missing = []
    for tier in ("easy", "medium", "hard", "xling"):
        vals = {}
        for arm in ("llm", "cas"):
            per_seed = []
            for s in SEEDS:
                try:
                    per_seed.append(read(tier, arm, s))
                except FileNotFoundError:
                    missing.append((tier, arm, s))
                    per_seed.append(None)
            vals[arm] = per_seed
        got = [i for i in range(len(SEEDS))
               if vals["llm"][i] is not None and vals["cas"][i] is not None]
        if not got:
            continue
        gaps = [vals["llm"][i] - vals["cas"][i] for i in got]
        out["tiers"][tier] = {
            "llm_mean": round(float(np.mean([vals["llm"][i] for i in got])), 2),
            "cas_mean": round(float(np.mean([vals["cas"][i] for i in got])), 2),
            "gap_mean": round(float(np.mean(gaps)), 2),
            "gap_per_seed": [round(g, 2) for g in gaps],
            "n_seeds": len(got)}
    if missing:
        out["missing"] = [f"{t}_{a}_s{s}" for t, a, s in missing]
        print(f"[warn] {len(missing)} summaries missing; verdict is "
              f"provisional", file=sys.stderr)
    easy = out["tiers"].get("easy")
    if easy and easy["n_seeds"] == len(SEEDS):
        g, allpos = easy["gap_mean"], all(x > 0 for x in easy["gap_per_seed"])
        if g >= 35.0 and allpos:
            v = "G1 CONFIRMS: leakage is not a first-order driver; the gap survives a fully-clean matched 6,145 budget."
        elif g >= 15.0 and allpos:
            v = "G2 PARTIAL: effect real but materially smaller than published; revise the 1.24-point attribution."
        else:
            v = "G3 REFUTES: contamination section must be rewritten."
        out["verdict"] = v
        print(v)
    with open(os.path.join(PROJECT, "results/cleanfull_verdict.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out.get("tiers", {}), indent=1))


if __name__ == "__main__":
    main()
