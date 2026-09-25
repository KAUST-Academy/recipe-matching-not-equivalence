#!/usr/bin/env python3
"""
Within-document-class style tracking (round-2 review finding R2-M21).

WHY. The paper's stated control against a base-rate artifact was to recompute
the style-tracking correlation over *synthetic documents only*. That control
cannot fail: 115,420 of the 117,088 corpus documents (98.58%) are synthetic, so
restricting to them removes 1.42% of the corpus and moves rho by ~0.0001.

The base rate that actually exists is BETWEEN the two synthetic classes. The
benchmark's gold rewrites (`::eq::`) and its minimal-edit near-misses (`::nm::`)
have systematically different style scores, and ctrl-LLM is by construction the
model that ranks golds above near-misses -- so a correlation measured across
both classes could be driven by that ranking rather than by style tracking.

The control that can fail is therefore WITHIN a single document class: hold the
class fixed and ask whether similarity still tracks style. This script runs it,
reusing stage_correlate's exact estimator (same query sample, same centred-rank
Spearman) so the numbers are directly comparable to
results/style_probe_correlations.json.

Usage (login node, CPU, ~2 min/model):
  python scripts/style_within_class.py
Output: results/style_within_class.json + a printed table.
"""

import json
import os
import random
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from style_probe import load_npz, spearman_rows  # noqa: E402  (reuse, do not re-implement)

SCORES = os.path.join(ROOT, "results", "style_scores_retrieve.npz")
OUT = os.path.join(ROOT, "results", "style_within_class.json")

MODELS = {
    "ctrl-llm-6145": (".emb_cache/docs_models__ctrl-llm-6145__final_*.npz",
                      ".emb_cache/queries_models__ctrl-llm-6145__final_*.npz"),
    "ctrl-cas-6145": (".emb_cache/docs_models__ctrl-cas-6145__final_*.npz",
                      ".emb_cache/queries_models__ctrl-cas-6145__final_*.npz"),
    "base-0.6b": (".emb_cache/docs_Qwen__Qwen3-Embedding-0.6B_*.npz",
                  ".emb_cache/queries_Qwen__Qwen3-Embedding-0.6B_*.npz"),
}
N_QUERIES = 2000
SEED = 42


def centered_rank(v):
    r = rankdata(v).astype(np.float32)
    r -= r.mean()
    return r, float(np.sqrt((r * r).sum()))


def main():
    os.chdir(ROOT)
    z = np.load(SCORES, allow_pickle=False)
    doc_ids = [str(x) for x in z["doc_ids"]]
    style = z["doc_scores"].astype(np.float64)

    # document classes, from the id convention documented at style_probe.py:18
    cls = {
        "gold (::eq::)": np.array(["::eq::" in d for d in doc_ids]),
        "near-miss (::nm::)": np.array(["::nm::" in d for d in doc_ids]),
        "organic (::orig)": np.array([d.endswith("::orig") for d in doc_ids]),
    }
    print("document classes:")
    for k, m in cls.items():
        print(f"  {k:20s} n={m.sum():7d}  mean style={style[m].mean():+.4f}")

    # the base rate the vacuous control misses: gold vs near-miss separability
    g, n = style[cls["gold (::eq::)"]], style[cls["near-miss (::nm::)"]]
    lab = np.concatenate([np.ones(len(g)), np.zeros(len(n))])
    sc = np.concatenate([g, n])
    r = rankdata(sc)
    auc = (r[lab == 1].sum() - len(g) * (len(g) + 1) / 2) / (len(g) * len(n))
    print(f"\nstyle score alone separates gold from near-miss at AUC {auc:.4f}"
          f"  (this is the confound the within-class control removes)")

    ranks = {k: centered_rank(style[m]) for k, m in cls.items()}
    full_rank, full_den = centered_rank(style)

    rng = random.Random(SEED)
    loaded, shared = [], None
    for label, (dp, qp) in MODELS.items():
        d_ids, d_emb, _ = load_npz(dp)
        q_ids, q_emb, _ = load_npz(qp)
        if d_ids != doc_ids:
            pos = {c: i for i, c in enumerate(d_ids)}
            d_emb = d_emb[[pos[c] for c in doc_ids]]
        loaded.append((label, d_emb, dict(zip(q_ids, range(len(q_ids)))), q_emb))
        shared = set(q_ids) if shared is None else shared & set(q_ids)
    sample = rng.sample(sorted(shared), min(N_QUERIES, len(shared)))

    out = {}
    for label, d_emb, qidx, q_emb in loaded:
        qe = q_emb[[qidx[q] for q in sample]]
        res = {}
        for name, m in list(cls.items()) + [("all documents", None)]:
            sub = d_emb if m is None else d_emb[m]
            rk, den = (full_rank, full_den) if m is None else ranks[name]
            vals = np.zeros(len(sample))
            for s0 in range(0, len(sample), 256):
                vals[s0:s0 + 256] = spearman_rows(qe[s0:s0 + 256] @ sub.T, rk, den)
            res[name] = round(float(vals.mean()), 4)
        out[label] = res
        print(f"\n{label}:")
        for k, v in res.items():
            print(f"   {k:20s} rho = {v:+.4f}")

    doc = {
        "generated": "2026-07-31",
        "script": "scripts/style_within_class.py",
        "purpose": "R2-M21: replace the vacuous synthetic-only control with a "
                   "within-document-class one that can actually fail.",
        "n_queries_sampled": len(sample),
        "seed": SEED,
        "class_sizes": {k: int(m.sum()) for k, m in cls.items()},
        "class_mean_style": {k: round(float(style[m].mean()), 4) for k, m in cls.items()},
        "gold_vs_nearmiss_style_auc": round(float(auc), 4),
        "note": "The synthetic-only control the draft cited is uninformative: "
                "98.58% of the corpus is synthetic, so it removes 1.42% of the "
                "documents. The discriminating control holds the document class "
                "fixed; ctrl-LLM's margin over ctrl-CAS survives inside both "
                "synthetic classes, which is what rules out the gold-vs-near-miss "
                "ranking explaining the correlation.",
        "per_model": out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
