#!/usr/bin/env python3
"""E-R2: quantify the paraphrase-depth confound.

The confound: CAS-verifiable positives (renames, term
moves, scaling) are intrinsically surface-preserving, while the recipe arm's
positives are free paraphrases -- so the arm gap might reflect paraphrase
STYLE rather than recipe PROXIMITY. This script measures, for every training
file, how far each positive actually sits from its anchor on plain surface
metrics: character-3-gram Jaccard, whitespace-token Jaccard, and length
ratio (|positive| / |anchor|).

If paraphrase depth alone ordered the benchmark results, the rungs' scores
should track these distances. The dose ladder says otherwise: D4 carries the
DEEPEST rewrites of the ladder yet posts ~30 easy R@1 with no recipe, and
D2/D3 sit between D1 and D4 on distance while their easy-tier scores are
statistically inseparable from D1's ordering only via prompt distance.

Writes results/paraphrase_depth.json.
"""
import json, os, sys

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "ctrl-cas": "data/cas_pairs/pairs.jsonl",
    "ctrl-llm(D1)": "data/llm_pairs/pairs.jsonl",
    "dose-paraphrase(D2)": "data/llm_pairs_paraphrase/pairs.jsonl",
    "dose-style(D3)": "data/llm_pairs_style/pairs.jsonl",
    "dose-unrelated(D4)": "data/llm_pairs_unrelated/pairs.jsonl",
    # E-R10 .. E-R12 (2026-09-06): the new pair files
    "survey(D5)": "data/llm_pairs_survey/pairs.jsonl",
    "exact-split(D1)": "data/llm_pairs_exact_split/pairs.jsonl",
    "twojudge(D1)": "data/llm_pairs_twojudge/pairs.jsonl",
}


def char_ngrams(t, n=3):
    t = " ".join((t or "").split())
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jacc(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def main():
    corpus = pd.read_parquet(os.path.join(PROJECT, "data",
                                          "mathnet_corpus.parquet"))
    text_of = dict(zip(corpus["id"], corpus["problem_markdown"]))
    out = {"generated_by": "scripts/paraphrase_depth.py",
           "metrics": ["char3_jaccard", "token_jaccard", "len_ratio"],
           "files": {}}
    for name, rel in FILES.items():
        c3, tj, lr = [], [], []
        with open(os.path.join(PROJECT, rel), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                a = r.get("anchor_text") or text_of.get(r["source_id"])
                p = r["positive_text"]
                if not a or not p:
                    continue
                c3.append(jacc(char_ngrams(a), char_ngrams(p)))
                tj.append(jacc(set(a.split()), set(p.split())))
                lr.append(len(p) / max(1, len(a)))
        stats = {}
        for mname, vals in (("char3_jaccard", c3), ("token_jaccard", tj),
                            ("len_ratio", lr)):
            v = np.array(vals)
            stats[mname] = {k: round(float(x), 3) for k, x in
                            (("mean", v.mean()), ("median", np.median(v)),
                             ("q25", np.percentile(v, 25)),
                             ("q75", np.percentile(v, 75)))}
        out["files"][name] = {"n_rows": len(c3), **stats}
        s = stats["char3_jaccard"]
        print(f"{name:22s} n={len(c3):5d}  char3 J mean={s['mean']:.3f} "
              f"median={s['median']:.3f} [q25 {s['q25']:.3f}, q75 {s['q75']:.3f}]  "
              f"tokenJ mean={stats['token_jaccard']['mean']:.3f}", flush=True)
    with open(os.path.join(PROJECT, "results", "paraphrase_depth.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote results/paraphrase_depth.json")


if __name__ == "__main__":
    main()
