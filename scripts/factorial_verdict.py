#!/usr/bin/env python3
"""E-R3 aggregation -- written before the new cells' numbers existed. No
directional bands are registered (the cells exist to deconfound positives from
negatives, not to test a direction); the table is reported as it lands.

Assembles the 2x2 positives x negatives factorial on every tier:

                      CAS negatives          no negatives
  CAS positives       ctrl-CAS (8 seeds)     fact-casnonegs (3 seeds)
  D4 positives        fact-d4casnegs (3)     dose-unrelated (3 seeds)

Reads results/ranks/*_<tag>-s<seed>.summary.json (+ xling) and writes
results/factorial_2x2.json.
"""
import json, os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = {
    "caspos_casnegs": ("ctrl-cas", list(range(42, 50))),
    "caspos_nonegs": ("fact-casnonegs", [42, 43, 44]),
    # E-R6 (2026-09-03): the reference cell extended to eight seeds, and two
    # cells with the recipe arm's own LLM-written negatives (count-matched to
    # d4casnegs, and full); registration in scripts/factorial_cells.slurm.
    "d4pos_casnegs": ("fact-d4casnegs", list(range(42, 50))),
    "d4pos_llmnegs_matched": ("fact-d4llmnegs", [42, 43, 44]),
    "d4pos_llmnegs_full": ("fact-d4llmnegsfull", [42, 43, 44]),
    "d4pos_nonegs": ("dose-unrelated", [42, 43, 44]),
    # E-R7 / E-R8 / E-R9 (2026-09-04); registered readings
    # R7-a/b, R8-a/b, R9 in scripts/factorial_cells.slurm (written before any of
    # these cells trained). Missing dumps are skipped, so the script can run
    # while cells are still training; quote nothing until all three have landed
    # and the three verdict files were regenerated in one pass.
    "d4pos_unrelnegs_matched": ("fact-d4unrelnegs", [42, 43, 44]),
    "d4pos_unrelnegs_full": ("fact-d4unrelnegsfull", [42, 43, 44]),
    "caspos_llmnegs": ("fact-casllmnegs", [42, 43, 44]),
    "btpos_casnegs": ("fact-btcasnegs", [42, 43, 44]),
    # E-R10 .. E-R13 (2026-09-06; registered readings R10-R13
    # in scripts/factorial_cells.slurm, written before any cell trained).
    "d5pos_nonegs": ("fact-d5nonegs", [42, 43, 44]),
    "d5pos_casnegs": ("fact-d5casnegs", [42, 43, 44]),
    "d1split": ("fact-d1split", [42, 43, 44]),
    "d1twojudge": ("fact-d1twojudge", [42, 43, 44]),
    "bt2pos_casnegs": ("fact-bt2casnegs", [42, 43, 44]),
    # Table fill (2026-09-08), E-T1..E-T7: the seven empty cells of tab:design,
    # built by scripts/build_review4_cells.py; no directional prediction registered.
    "caspos_unrelnegs": ("fact-casunrelnegs", [42, 43, 44]),
    "btpos_nonegs": ("fact-btnonegs", [42, 43, 44]),
    "btpos_unrelnegs": ("fact-btunrelnegs", [42, 43, 44]),
    "btpos_llmnegs": ("fact-btllmnegs", [42, 43, 44]),
    "d1pos_nonegs": ("fact-d1nonegs", [42, 43, 44]),
    "d1pos_casnegs": ("fact-d1casnegs", [42, 43, 44]),
    "d1pos_unrelnegs": ("fact-d1unrelnegs", [42, 43, 44]),
}


def read(tier, tag, seed):
    p = os.path.join(PROJECT, "results", "ranks",
                     f"{tier}_{tag}-s{seed}.summary.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if tier == "xling":
        return d["strict_crosslingual_gold"]["recall@1"]
    if tier == "samelang":
        return d["same_language_gold"]["recall@1"]
    return d["overall"]["recall@1"]


def main():
    out = {"generated_by": "scripts/factorial_verdict.py", "tiers": {}}
    for tier in ("easy", "medium", "hard", "xling", "samelang"):
        row = {}
        for cell, (tag, seeds) in CELLS.items():
            vals = []
            for s in seeds:
                try:
                    vals.append(read(tier, tag, s))
                except FileNotFoundError:
                    pass
            if vals:
                row[cell] = {"mean": round(float(np.mean(vals)), 2),
                             "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
                             "per_seed": [round(v, 2) for v in vals],
                             "n_seeds": len(vals)}
        out["tiers"][tier] = row
        cells = "  ".join(f"{c}={v['mean']}({v['n_seeds']}s)"
                          for c, v in row.items())
        print(f"{tier:6s} {cells}")
    with open(os.path.join(PROJECT, "results/factorial_2x2.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote results/factorial_2x2.json")


if __name__ == "__main__":
    main()
