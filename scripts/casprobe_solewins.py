#!/usr/bin/env python3
"""R14-e (exploratory): do the recipe arm's hard-tier sole
wins predict success on the generator-free near-miss probe?

Sole wins: the 1,401 hard-tier queries the recipe arm ranks gold first at seed
42 while the verified arm does not (results/analyze_hits_perquery_hard.jsonl,
group == "hit", i.e. gold_rank_a == 0 and gold_rank_b != 0). Probe queries are
joined to benchmark queries through metadata.anchor_id in
data/casprobe_eval/queries.jsonl. For each model tag and each evaluation set
(casprobe = full public corpus + probe documents; casprobepairs = probe
documents only) the script reports R@1 on the probe queries that are sole wins
and on the other probe queries, per seed and as the seed mean, from the rank
dumps results/ranks/{casprobe,casprobepairs}_<tag>-s<seed>.ranks.jsonl
(gold_rank == 0 is a hit).

Writes results/casprobe_solewins.json (the authority for the R14-e sentence in
Appendix A of the paper).
"""
import glob
import json
import os
import re
import statistics

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {
    "recipe_arm": "ctrl-llm",
    "verified_arm": "ctrl-cas",
    "reference": "fact-d4casnegs",
    "base": "qwen3-0.6b-base",
}
SETS = ("casprobe", "casprobepairs")


def r1(rows, ids):
    sub = [r for r in rows if r["qid"] in ids]
    return 100.0 * sum(r["gold_rank"] == 0 for r in sub) / len(sub), len(sub)


def main():
    hits = [json.loads(l) for l in open(os.path.join(
        PROJECT, "results", "analyze_hits_perquery_hard.jsonl"))]
    sole = {h["qid"] for h in hits if h["gold_rank_a"] == 0 and h["gold_rank_b"] != 0}
    assert len(sole) == sum(h["group"] == "hit" for h in hits)
    queries = [json.loads(l) for l in open(os.path.join(
        PROJECT, "data", "casprobe_eval", "queries.jsonl"))]
    sole_probe = {q["_id"] for q in queries if q["metadata"]["anchor_id"] in sole}
    other_probe = {q["_id"] for q in queries} - sole_probe
    out = {
        "generated_by": "scripts/casprobe_solewins.py",
        "sole_wins_definition": "results/analyze_hits_perquery_hard.jsonl group == 'hit' "
                                "(recipe arm gold_rank 0, verified arm gold_rank != 0; seed 42)",
        "n_sole_wins_hard": len(sole),
        "n_probe_queries": len(queries),
        "n_probe_queries_in_sole_wins": len(sole_probe),
        "n_probe_queries_other": len(other_probe),
        "metric": "R@1 (percent), gold_rank == 0 in the rank dump",
        "models": {},
    }
    for name, tag in TAGS.items():
        out["models"][name] = {"tag": tag}
        for s in SETS:
            files = sorted(glob.glob(os.path.join(
                PROJECT, "results", "ranks", f"{s}_{tag}-s*.ranks.jsonl")))
            if not files:
                files = [os.path.join(PROJECT, "results", "ranks", f"{s}_{tag}.ranks.jsonl")]
            per_seed = {}
            for f in files:
                m = re.search(r"-s(\d+)\.ranks\.jsonl$", f)
                seed = m.group(1) if m else "single"
                rows = [json.loads(l) for l in open(f)]
                a, na = r1(rows, sole_probe)
                b, nb = r1(rows, other_probe)
                assert na == len(sole_probe) and nb == len(other_probe), (f, na, nb)
                per_seed[seed] = {"sole_wins": round(a, 2), "other": round(b, 2)}
            sw = [v["sole_wins"] for v in per_seed.values()]
            ot = [v["other"] for v in per_seed.values()]
            out["models"][name][s] = {
                "n_seeds": len(per_seed),
                "sole_wins_mean": round(statistics.mean(sw), 2),
                "other_mean": round(statistics.mean(ot), 2),
                "difference": round(statistics.mean(sw) - statistics.mean(ot), 2),
                "per_seed": per_seed,
            }
            print(f"{name:13s} {s:14s} seeds={len(per_seed)} sole-wins R@1 "
                  f"{statistics.mean(sw):6.2f}  other {statistics.mean(ot):6.2f}  "
                  f"diff {statistics.mean(sw) - statistics.mean(ot):+.2f}")
    out["reading"] = ("exploratory (no registration): the probe queries that are the recipe "
                      "arm's hard-tier sole wins are not easier for it on generator-free "
                      "near-misses, so its sole wins do not predict probe success")
    path = os.path.join(PROJECT, "results", "casprobe_solewins.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote results/casprobe_solewins.json "
          f"({len(sole_probe)} of {len(queries)} probe queries are sole wins)")


if __name__ == "__main__":
    main()
