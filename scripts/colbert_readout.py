#!/usr/bin/env python3
"""Late-interaction arms (2026-09-17). Collects the
PyLate ColBERT scores written by scripts/colbert_arms.slurm
(results/colbert/{easy,medium,hard,crosslingual,samelang}_{tag}.json for the
untrained ColBERTv2 and the two arms fine-tuned on the recipe and verified
files), adds the same-language slices of scripts/bge_samelang_readout.py from
results/ranks/samelang_colbert_{tag}.ranks.jsonl with 95% cluster-bootstrap
intervals for the arm gaps, and writes results/colbert_summary.json plus the
rows of the appendix table."""
import json, os
from collections import defaultdict
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"base": "colbertv2-base", "verified": "ctrl-cas", "recipe": "ctrl-llm"}
TIERS = ["easy", "medium", "hard"]
out = {"tags": TAGS, "tiers": {}, "crosslingual": {}, "samelang": {}, "samelang_slices": {}}
for tier in TIERS:
    out["tiers"][tier] = {}
    for arm, tag in TAGS.items():
        f = f"{P}/results/colbert/{tier}_{tag}.json"
        out["tiers"][tier][arm] = json.load(open(f))["overall"] if os.path.exists(f) else None
for key, name in (("crosslingual", "crosslingual"), ("samelang", "samelang")):
    for arm, tag in TAGS.items():
        f = f"{P}/results/colbert/{name}_{tag}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            out[key][arm] = {"overall": d["overall"], "strict_crosslingual_gold": d["strict_crosslingual_gold"],
                             "same_language_gold": d["same_language_gold"], "n_corpus": d["n_corpus"], "n_queries": d["n_queries"]}
        else:
            out[key][arm] = None
# same-language slices with a cluster bootstrap, as in scripts/bge_samelang_readout.py
queries = [json.loads(l) for l in open(f"{P}/data/samelang_eval/queries.jsonl", encoding="utf-8")]
qids = [q["_id"] for q in queries]; meta = {q["_id"]: q["metadata"] for q in queries}
slices = {
  "primary": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"]],
  "primary_en": [q for q in qids if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"] and meta[q]["lang"] == "en"],
  "exact": [q for q in qids if meta[q]["exact_text_cluster"]],
  "all": list(qids)}
hit = {}
for arm, tag in TAGS.items():
    f = f"{P}/results/ranks/samelang_colbert_{tag}.ranks.jsonl"
    if not os.path.exists(f):
        continue
    d = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line); d[r["qid"]] = r["same_rank"]
    assert len(d) == len(qids), (tag, len(d))
    hit[arm] = {q: float(d[q] == 0) for q in qids}
if len(hit) == len(TAGS):
    rng = np.random.default_rng(0); B = 10000
    out["samelang_slices"] = {"B": B, "slices": {}}
    for name, qs in slices.items():
        groups = defaultdict(list)
        for q in qs: groups[meta[q]["cluster_id"]].append(q)
        groups = list(groups.values())
        rec = {"n_queries": len(qs), "n_clusters": len(groups),
               "r1": {arm: round(100 * float(np.mean([hit[arm][q] for q in qs])), 2) for arm in TAGS}}
        def gap(a, b):
            diffs = [np.array([hit[a][q] - hit[b][q] for q in g], dtype=float) for g in groups]
            point = 100 * float(np.mean([hit[a][q] - hit[b][q] for q in qs]))
            boots = np.empty(B)
            for i in range(B):
                gi = rng.integers(0, len(groups), len(groups))
                boots[i] = 100 * np.concatenate([diffs[j] for j in gi]).mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return {"point": round(point, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)]}
        rec["gap"] = {"verified-base": gap("verified", "base"), "recipe-base": gap("recipe", "base"),
                      "recipe-verified": gap("recipe", "verified")}
        out["samelang_slices"]["slices"][name] = rec
json.dump(out, open(f"{P}/results/colbert_summary.json", "w"), indent=2)
fmt = lambda b: "--" if b is None else f"{b['recall@1']:.2f} / {b['recall@5']:.2f} / {b['recall@10']:.2f}"
print("arm        | easy R@1/5/10        | medium               | hard                 | xling strict R@1 | samelang primary R@1")
for arm in TAGS:
    xl = out["crosslingual"].get(arm); sl = out["samelang_slices"].get("slices", {}).get("primary")
    print(f"{arm:10s} | {fmt(out['tiers']['easy'][arm]):20s} | {fmt(out['tiers']['medium'][arm]):20s} | {fmt(out['tiers']['hard'][arm]):20s} | "
          f"{'--' if not xl else xl['strict_crosslingual_gold']['recall@1']:>16} | {'--' if not sl else sl['r1'][arm]}")
if out["samelang_slices"]:
    for name, rec in out["samelang_slices"]["slices"].items():
        print(f"{name:11s} n={rec['n_queries']:3d} R@1 base {rec['r1']['base']:6.2f} verified {rec['r1']['verified']:6.2f} recipe {rec['r1']['recipe']:6.2f}"
              f" | rec-ver {rec['gap']['recipe-verified']['point']:+6.2f} {rec['gap']['recipe-verified']['ci95']}"
              f" | ver-base {rec['gap']['verified-base']['point']:+6.2f} {rec['gap']['verified-base']['ci95']}"
              f" | rec-base {rec['gap']['recipe-base']['point']:+6.2f} {rec['gap']['recipe-base']['ci95']}")
