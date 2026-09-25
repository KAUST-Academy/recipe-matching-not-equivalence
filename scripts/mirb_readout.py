#!/usr/bin/env python3
"""MIRB readout: collect results/mirb/<task>_<tag>.json into one table
(nDCG@10 per task and model, plus per-task paired query bootstrap of the arm
differences from the .perquery.json files) -> results/mirb_summary.json."""
import glob, json, os
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["msedup", "modup", "mseqa", "proofwikiqa", "stacksqa", "proofwikips", "stacksps",
         "realanalysisps", "numbertheoryps", "mseformula", "wikiformula"]
TAGS = {"base": "qwen3-0.6b-base", "verified": "ctrl-cas-6145", "recipe": "ctrl-llm-6145"}
KIND = {"msedup": "duplicate question", "modup": "duplicate question", "mseqa": "question to answer (ARQMath Task 1)",
        "proofwikiqa": "question to answer", "stacksqa": "question to answer", "proofwikips": "premise",
        "stacksps": "premise", "realanalysisps": "premise", "numbertheoryps": "premise",
        "mseformula": "formula", "wikiformula": "formula"}
rng = np.random.default_rng(0); B = 5000
# msedup was scored last (2026-09-17, after the other ten), so it is bootstrapped last: the ten
# earlier tasks then draw the same resamples as in the committed run and keep their intervals.
ORDER = [t for t in TASKS if t != "msedup"] + ["msedup"]
recs = {}
for t in ORDER:
    rec = {"kind": KIND[t], "ndcg10": {}, "n_queries": None, "paired": {}}
    pq = {}
    for name, tag in TAGS.items():
        f = f"{P}/results/mirb/{t}_{tag}.json"
        if not os.path.exists(f):
            rec["ndcg10"][name] = None; continue
        d = json.load(open(f)); rec["ndcg10"][name] = d["metrics"]["NDCG@10"]; rec["n_queries"] = d["n_queries"]
        pqf = f.replace(".json", ".perquery.json")
        if os.path.exists(pqf):
            pq[name] = {q: m["ndcg_cut_10"] for q, m in json.load(open(pqf)).items()}
    for a, b in (("recipe", "verified"), ("recipe", "base"), ("verified", "base")):
        if a in pq and b in pq:
            qs = sorted(set(pq[a]) & set(pq[b])); n = len(qs)
            diff = np.array([pq[a][q] - pq[b][q] for q in qs])
            idx = rng.integers(0, n, size=(B, n)); boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rec["paired"][f"{a}-{b}"] = {"point": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    recs[t] = rec
out = {"B": B, "tasks": {t: recs[t] for t in TASKS}}
json.dump(out, open(f"{P}/results/mirb_summary.json", "w"), indent=2)
print(f"{'task':16s} {'n':>5s} {'base':>7s} {'verif':>7s} {'recipe':>7s}   recipe-verified            recipe-base")
for t, r in out["tasks"].items():
    n = r["ndcg10"]; g = r["paired"]
    fmt = lambda k: (f"{g[k]['point']:+.4f} [{g[k]['ci95'][0]:+.4f},{g[k]['ci95'][1]:+.4f}]" if k in g else "-")
    print(f"{t:16s} {str(r['n_queries'] or ''):>5s} " + " ".join(f"{(n[k] if n[k] is not None else float('nan')):7.4f}" for k in ('base','verified','recipe')) + f"   {fmt('recipe-verified'):28s} {fmt('recipe-base')}")
