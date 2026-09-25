#!/usr/bin/env python3
"""
Paired bootstrap for the SABER label-channel attack.

The label arm scores 0.6101 nDCG@10 against the base's 0.5747 and the
matched-budget document arm's 0.5515. Those are point estimates over 1,000
queries and the paper cannot use them without an interval -- round-2 findings
C2 and the closeout's block A were both exactly this mistake, so the rule here
is that no ordering is stated unless its paired CI excludes zero.

Resamples the 1,000 SABER queries with replacement, applying ONE shared
resample to both models of a pair so the per-query correlation is preserved.
This is an evaluation-sampling interval: it carries no training-seed variance,
and the label arm is a single run, so both caveats travel with any number
quoted from here.

Usage:  python scripts/saber_label_bootstrap.py
Output: results/saber_label_bootstrap.json + a printed table.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
B = 10000
RNG_SEED = 20260731

PAIRS = [
    ("label4733", "base", "the label channel against the untrained base"),
    ("label4733", "doc4733", "label vs document channel at a MATCHED 4,733-row budget"),
    ("doc4733", "base", "the document channel against the base (the published surface)"),
    ("pubattack", "base", "the published 6,145-row attack, for reference"),
]


def load(tag):
    p = os.path.join(R, f"saber_pq_{tag}.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return np.asarray(d["per_query_ndcg"], dtype=float)
    except (OSError, ValueError, KeyError):
        return None


def main():
    os.chdir(ROOT)
    arrays = {t: load(t) for t in {p for pair in PAIRS for p in pair[:2]}}
    missing = [t for t, v in arrays.items() if v is None]
    if missing:
        print("PENDING — missing per-query dumps:", ", ".join(sorted(missing)))
        return

    n = len(next(iter(arrays.values())))
    if any(len(v) != n for v in arrays.values()):
        raise SystemExit("[fatal] per-query arrays differ in length; not the same query set")

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(B, n))

    out = {}
    print(f"{'comparison':34s} {'diff':>8s} {'95% CI':>20s} {'p':>9s}  verdict")
    for a, b, why in PAIRS:
        da, db = arrays[a], arrays[b]
        diff = float(da.mean() - db.mean())
        boot = da[idx].mean(axis=1) - db[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # two-sided bootstrap p with the +1 small-sample correction, floored
        k = min((boot <= 0).sum(), (boot >= 0).sum())
        p = 2.0 * (k + 1) / (B + 1)
        excl = bool(lo > 0 or hi < 0)
        out[f"{a}_vs_{b}"] = {
            "what": why, "diff": round(diff, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided": round(float(min(p, 1.0)), 5),
            "ci_excludes_zero": excl, "n_queries": n,
        }
        print(f"{a+' vs '+b:34s} {diff:+8.4f} [{lo:+7.4f},{hi:+7.4f}] {min(p,1.0):9.5f}  "
              f"{'SIGNIFICANT' if excl else 'not resolvable'}")

    lab = out["label4733_vs_base"]
    doc = out["doc4733_vs_base"]
    verdict = (
        "LABEL SURFACE EXPOSED — the label channel beats the base by "
        f"{lab['diff']:+.4f} {lab['ci95']}, excluding zero, while the document channel at the "
        f"same budget gives {doc['diff']:+.4f} {doc['ci95']}. SABER's published negative tested "
        "the surface that could not work; the surface its relevance rule is actually computed "
        "over is exposed. The taxonomy's organic-corpus cell can no longer be described as "
        "resisting."
        if lab["ci_excludes_zero"] and lab["diff"] > 0 else
        "NOT RESOLVABLE — the label channel's gain does not survive a paired interval over the "
        "1,000 queries, so it must be reported as a point estimate that we could not resolve, "
        "and the paper's SABER negative stands with its channel caveat corrected."
    )
    out["verdict"] = verdict
    out["scope_warning"] = (
        "Evaluation-sampling only: resamples queries, not training seeds, and every arm here is a "
        "single training run. The correlation half of the original P-S1 is NOT met by the label arm "
        "(1.14x, essentially the same as the document attack), so this is a score result, not a "
        "demonstration that the model aligned to the summary-Jaccard signal specifically.")
    out["n_boot"] = B
    out["rng_seed"] = RNG_SEED

    with open(os.path.join(R, "saber_label_bootstrap.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}")
    print("\nwrote results/saber_label_bootstrap.json")


if __name__ == "__main__":
    main()
