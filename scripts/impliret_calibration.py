#!/usr/bin/env python3
"""
ImpliRet harness calibration + baseline table.

Every benchmark in this campaign either carries a calibration credential or
states plainly that it has none. ImpliRet can be calibrated: their Table 2
reports ReasonIR, and we can run ReasonIR-8B. This script compares our rows to
theirs and returns a verdict that governs how the paper is allowed to talk
about ImpliRet numbers.

The rule, fixed here before the comparison is run and deliberately strict,
because a cross-table result on MELD is one our harness cannot support:
  CALIBRATED      every category we can compare is within 2.0 nDCG@10 points of
                  their reported value -> we may compare our ImpliRet numbers
                  to their table.
  UNCALIBRATED    otherwise -> ImpliRet numbers are comparable ACROSS OUR OWN
                  MODELS ONLY, stated in the paper exactly as the BRIGHT and
                  MELD caveats are.

Usage:  python scripts/impliret_calibration.py
Output: results/impliret_calibration.json + a printed table.
"""

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# Their Table 2, nDCG@10, averaged over the two discourse styles.
THEIRS = {
    "reasonir-8b": {"wknow": 18.88, "arithmetic": 10.78, "temporal": 11.25, "avg": 13.64},
}
THEIR_BEST = {"model": "Dragon+", "avg": 14.91}
TOL = 2.0

OURS = ["base", "base_noprompt", "ctrl_llm", "ctrl_cas", "mpnet", "reasonir-8b"]


def load(tag):
    p = os.path.join(R, f"impliret_{tag}.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def by_category(d):
    """Average each reasoning category over the two discourse styles, as their table does."""
    if not d:
        return None
    per = d["per_subset"]
    out = {}
    for cat in ("wknow", "arithmetic", "temporal"):
        vals = [per[k]["ndcg@10"] for k in per if k.endswith(":" + cat)]
        if vals:
            out[cat] = round(sum(vals) / len(vals), 2)
    if out:
        out["avg"] = round(sum(out[c] for c in ("wknow", "arithmetic", "temporal")
                               if c in out) / len([c for c in ("wknow", "arithmetic", "temporal")
                                                   if c in out]), 2)
    return out


def main():
    os.chdir(ROOT)
    rows, cal = {}, {}
    for tag in OURS:
        d = load(tag)
        rows[tag] = by_category(d)

    verdict = "PENDING — ReasonIR row missing"
    if rows.get("reasonir-8b"):
        ours = rows["reasonir-8b"]
        theirs = THEIRS["reasonir-8b"]
        diffs = {c: round(ours[c] - theirs[c], 2) for c in theirs if c in ours}
        cal = {"ours": ours, "theirs": theirs, "diff": diffs, "tolerance": TOL}
        worst = max(abs(v) for v in diffs.values())
        cal["max_abs_diff"] = worst
        if worst <= TOL:
            verdict = (f"CALIBRATED — our ReasonIR rows sit within {worst:.2f} nDCG@10 of "
                       f"their Table 2 across every comparable category, so ImpliRet numbers "
                       f"may be compared to their table (their best is Dragon+ at "
                       f"{THEIR_BEST['avg']}).")
        else:
            verdict = (f"UNCALIBRATED — our ReasonIR rows differ from their Table 2 by up to "
                       f"{worst:.2f} nDCG@10, beyond the {TOL}-point tolerance. ImpliRet "
                       f"numbers are comparable ACROSS OUR OWN MODELS ONLY and the paper must "
                       f"say so, exactly as it does for BRIGHT and MELD. Do NOT claim anything "
                       f"against their published table.")

    doc = {
        "generated": str(date.today()),
        "script": "scripts/impliret_calibration.py",
        "benchmark": "ImpliRet (arXiv:2506.14407)",
        "rule_fixed_before_running": (
            "CALIBRATED iff every comparable category is within 2.0 nDCG@10 of their reported "
            "ReasonIR value; otherwise our numbers are internal-only. Written before the "
            "comparison, because round-2 finding C2 was exactly this mistake on MELD."),
        "their_table2_reasonir": THEIRS["reasonir-8b"],
        "their_best": THEIR_BEST,
        "calibration": cal,
        "verdict": verdict,
        "our_models_ndcg10_by_category": rows,
    }
    with open(os.path.join(R, "impliret_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"{'model':16s} {'W.Know':>8s} {'Arith':>8s} {'Temporal':>9s} {'Avg':>8s}")
    for tag in OURS:
        r = rows.get(tag)
        if not r:
            print(f"{tag:16s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>8s}")
            continue
        print(f"{tag:16s} {r.get('wknow', 0):8.2f} {r.get('arithmetic', 0):8.2f} "
              f"{r.get('temporal', 0):9.2f} {r.get('avg', 0):8.2f}")
    t = THEIRS["reasonir-8b"]
    print(f"{'THEIR ReasonIR':16s} {t['wknow']:8.2f} {t['arithmetic']:8.2f} "
          f"{t['temporal']:9.2f} {t['avg']:8.2f}   <- their Table 2")
    print(f"{'THEIR best':16s} {'':8s} {'':8s} {'':9s} {THEIR_BEST['avg']:8.2f}   "
          f"<- {THEIR_BEST['model']}")
    print(f"\nVERDICT: {verdict}")
    print("\nwrote results/impliret_calibration.json")


if __name__ == "__main__":
    main()
