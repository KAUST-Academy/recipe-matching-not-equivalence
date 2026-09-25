#!/usr/bin/env python3
"""Same-language slices for the BM25 lexical baseline, read from the rank dump
results/ranks/samelang_bm25.ranks.jsonl (scripts/bm25_all.slurm) beside the
untrained base's results/ranks/samelang_qwen3-0.6b-base.ranks.jsonl. Slices as
in scripts/samelang_verdict.py: primary (reworded clusters clean of every
training file), its English queries, leaked (reworded clusters some training
file touches), exact (near-identical reprints) and all. BM25 and the base are
single deterministic runs, so the 95% interval of BM25 minus base resamples
duplicate clusters only (10,000 draws, seed 0, slices in the order below).
Writes results/bm25_samelang_slices.json.

    python scripts/bm25_samelang_readout.py
"""
import json
import os
from collections import defaultdict

import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10000


def ranks(tag):
    out = {}
    with open(f"{P}/results/ranks/samelang_{tag}.ranks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["qid"]] = r["same_rank"]
    return out


def main():
    queries = [json.loads(l) for l in open(f"{P}/data/samelang_eval/queries.jsonl", encoding="utf-8")]
    qids = [q["_id"] for q in queries]
    meta = {q["_id"]: q["metadata"] for q in queries}
    slices = {
        "primary": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"]],
        "primary_en": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"]
                       and meta[q]["lang"] == "en"],
        "leaked": [q for q in qids if not meta[q]["exact_text_cluster"] and not meta[q]["clean_of_training"]],
        "exact": [q for q in qids if meta[q]["exact_text_cluster"]],
        "all": list(qids),
    }
    bm, base = ranks("bm25"), ranks("qwen3-0.6b-base")
    assert len(bm) == len(base) == len(qids), (len(bm), len(base), len(qids))
    hit_bm = np.array([bm[q] == 0 for q in qids], dtype=float)
    hit_base = np.array([base[q] == 0 for q in qids], dtype=float)
    pos = {q: i for i, q in enumerate(qids)}
    rng = np.random.default_rng(0)
    out = {"generated_by": "scripts/bm25_samelang_readout.py", "bootstrap": B,
           "ranks": {"bm25": "results/ranks/samelang_bm25.ranks.jsonl",
                     "base": "results/ranks/samelang_qwen3-0.6b-base.ranks.jsonl"}}
    for name, qs in slices.items():
        ii = np.array([pos[q] for q in qs])
        groups = defaultdict(list)
        for q in qs:
            groups[meta[q]["cluster_id"]].append(pos[q])
        groups = [np.array(v) for v in groups.values()]
        reps = np.empty(B)
        for b in range(B):
            qi = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            reps[b] = hit_bm[qi].mean() - hit_base[qi].mean()
        r1, r1_base = 100 * hit_bm[ii].mean(), 100 * hit_base[ii].mean()
        out[name] = {"n": len(qs), "n_clusters": len(groups), "r1": round(float(r1), 2),
                     "base_r1": round(float(r1_base), 2), "gap_vs_base": round(float(r1 - r1_base), 2),
                     "gap_vs_base_ci95": [round(float(np.percentile(100 * reps, p)), 2) for p in (2.5, 97.5)]}
        print(f"{name:11s} n={len(qs):3d} BM25 {out[name]['r1']:6.2f}  base {out[name]['base_r1']:6.2f}  "
              f"gap {out[name]['gap_vs_base']:+6.2f} {out[name]['gap_vs_base_ci95']}")
    with open(f"{P}/results/bm25_samelang_slices.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
