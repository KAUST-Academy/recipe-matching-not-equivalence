#!/usr/bin/env python3
"""Query-level paired bootstrap (2026-09-16) on the two BRIGHT
math splits, from the per-query nDCG@10 dumped by scripts/bright_perquery.slurm.

For each split: mean nDCG@10 of recipe arm (ctrl-llm-6145), verified arm
(ctrl-cas-6145) and untrained base; paired differences recipe-verified,
recipe-base, verified-base with 95% percentile intervals over B resamples of
the query set (queries resampled with replacement, the same draw applied to
both models of a pair). Also checks that the re-run aggregates reproduce the
committed results/bright_<split>_<tag>.json to 1e-4.

Usage: python scripts/bright_bootstrap.py [--B 10000] [--seed 0]
Writes results/bright_bootstrap.json and prints a summary.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "results", "bright_pq")
SPLITS = ["theoremqa_theorems", "aops"]
TAGS = {"recipe": "ctrl-llm-6145", "verified": "ctrl-cas-6145", "base": "qwen3-0.6b-base"}
PAIRS = [("recipe", "verified"), ("recipe", "base"), ("verified", "base")]


def load(split, tag):
    d = json.load(open(os.path.join(PQ, f"{split}_{tag}.perquery.json")))
    return {q: m["ndcg_cut_10"] for q, m in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {"B": a.B, "seed": a.seed, "splits": {}}
    for split in SPLITS:
        pq = {name: load(split, tag) for name, tag in TAGS.items()}
        qids = sorted(set.intersection(*(set(v) for v in pq.values())))
        assert all(len(v) == len(qids) for v in pq.values()), "query sets differ"
        X = {name: np.array([pq[name][q] for q in qids]) for name in TAGS}
        n = len(qids)
        idx = rng.integers(0, n, size=(a.B, n))
        rec = {"n_queries": n, "mean_ndcg10": {}, "reproduces_committed": {}, "paired": {}}
        for name, tag in TAGS.items():
            rec["mean_ndcg10"][name] = round(float(X[name].mean()), 5)
            committed = os.path.join(ROOT, "results", f"bright_{split}_{tag}.json")
            if os.path.exists(committed):
                c = json.load(open(committed))["metrics"]["NDCG@10"]
                rec["reproduces_committed"][name] = {"committed": c, "rerun": round(float(X[name].mean()), 5),
                                                     "match_1e-4": bool(abs(c - X[name].mean()) < 1e-4)}
        for a_, b_ in PAIRS:
            diff = X[a_] - X[b_]
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rec["paired"][f"{a_}-{b_}"] = {"point": round(float(diff.mean()), 5),
                                          "ci95": [round(float(lo), 5), round(float(hi), 5)],
                                          "excludes_zero": bool(lo > 0 or hi < 0),
                                          "n_queries_where_a_beats_b": int((diff > 0).sum()),
                                          "n_queries_where_b_beats_a": int((diff < 0).sum())}
        out["splits"][split] = rec
    path = os.path.join(ROOT, "results", "bright_bootstrap.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
