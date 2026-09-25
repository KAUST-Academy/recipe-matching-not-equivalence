#!/usr/bin/env python3
"""Same-language R@1 (2026-09-16) of the BGE-large-en-v1.5
base and its two seed-42 arms, sliced as in scripts/samelang_verdict.py
(primary = non-exact clusters clean of every training file; primary_en = its
English queries; exact; all), with 95% cluster-bootstrap intervals (single
seed, so only duplicate clusters are resampled) for arm - base and recipe -
verified. Reads results/ranks/samelang_{bge-base,ctrl-cas-bge-s42,ctrl-llm-bge-s42}.ranks.jsonl
(scripts/bge_samelang.slurm) and writes results/bge_samelang.json."""
import json, os
from collections import defaultdict
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"base": "bge-base", "verified": "ctrl-cas-bge-s42", "recipe": "ctrl-llm-bge-s42"}
queries = [json.loads(l) for l in open(f"{P}/data/samelang_eval/queries.jsonl", encoding="utf-8")]
qids = [q["_id"] for q in queries]; meta = {q["_id"]: q["metadata"] for q in queries}
slices = {
  "primary": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"]],
  "primary_en": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"] and meta[q]["lang"] == "en"],
  "exact": [q for q in qids if meta[q]["exact_text_cluster"]],
  "all": list(qids)}
hit = {}
for arm, tag in TAGS.items():
    d = {}
    for line in open(f"{P}/results/ranks/samelang_{tag}.ranks.jsonl", encoding="utf-8"):
        r = json.loads(line); d[r["qid"]] = r["same_rank"]
    assert len(d) == len(qids), (tag, len(d))
    hit[arm] = {q: float(d[q] == 0) for q in qids}
rng = np.random.default_rng(0); B = 10000
out = {"B": B, "tags": TAGS, "slices": {}}
for name, qs in slices.items():
    groups = defaultdict(list)
    for q in qs: groups[meta[q]["cluster_id"]].append(q)
    groups = list(groups.values())
    rec = {"n_queries": len(qs), "n_clusters": len(groups),
           "r1": {arm: round(100 * np.mean([hit[arm][q] for q in qs]), 2) for arm in TAGS}}
    def gap(a, b):
        diffs = np.array([[hit[a][q] - hit[b][q] for q in g] for g in groups], dtype=object)
        point = 100 * np.mean([hit[a][q] - hit[b][q] for q in qs])
        boots = np.empty(B)
        for i in range(B):
            gi = rng.integers(0, len(groups), len(groups))
            vals = np.concatenate([np.array(diffs[j], dtype=float) for j in gi])
            boots[i] = 100 * vals.mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"point": round(point, 2), "ci95": [round(lo, 2), round(hi, 2)]}
    rec["gap"] = {"verified-base": gap("verified", "base"), "recipe-base": gap("recipe", "base"),
                  "recipe-verified": gap("recipe", "verified")}
    out["slices"][name] = rec
json.dump(out, open(f"{P}/results/bge_samelang.json", "w"), indent=2)
for name, rec in out["slices"].items():
    print(f"{name:11s} n={rec['n_queries']:3d} R@1 base {rec['r1']['base']:6.2f} verified {rec['r1']['verified']:6.2f} recipe {rec['r1']['recipe']:6.2f} | rec-ver {rec['gap']['recipe-verified']['point']:+6.2f} {rec['gap']['recipe-verified']['ci95']} | ver-base {rec['gap']['verified-base']['point']:+6.2f} {rec['gap']['verified-base']['ci95']} | rec-base {rec['gap']['recipe-base']['point']:+6.2f} {rec['gap']['recipe-base']['ci95']}")
