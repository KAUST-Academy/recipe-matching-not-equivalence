#!/usr/bin/env python3
"""Readout (2026-09-16/17) of the V1 soups (models/v1soups,
scripts/v1_soups.slurm): per alpha, R@1/R@5/R@10 on the three tiers, strict
cross-language R@1/5/10, and same-language primary-slice R@1, beside the
verified-arm soups and V1 itself for comparison. Writes results/v1_soups.json."""
import json, os
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def J(p): return json.load(open(os.path.join(P, p)))
def tiers(tag):
    out = {}
    for t in ("easy", "medium", "hard"):
        m = J(f"results/eval_{t}_{tag}.json")["overall"]
        out[t] = [m["recall@1"], m["recall@5"], m["recall@10"]]
    return out
def xling(tag):
    m = J(f"results/crosslingual_{tag}.json")["strict_crosslingual_gold"]
    return [m["recall@1"], m["recall@5"], m["recall@10"]]
queries = [json.loads(l) for l in open(os.path.join(P, "data/samelang_eval/queries.jsonl"), encoding="utf-8")]
meta = {q["_id"]: q["metadata"] for q in queries}
primary = [q for q in meta if not meta[q]["exact_text_cluster"] and meta[q]["clean_of_training"]]
def samelang(tag):
    p = os.path.join(P, f"results/ranks/samelang_{tag}.ranks.jsonl")
    if not os.path.exists(p): return None
    r = {}
    for l in open(p, encoding="utf-8"):
        d = json.loads(l); r[d["qid"]] = d["same_rank"]
    return round(100 * sum(r[q] == 0 for q in primary) / len(primary), 2)
rows = {}
for label, tag in [("V1 soup a=0.3", "v1soup-a0.3"), ("V1 soup a=0.5", "v1soup-a0.5"), ("V1 soup a=0.7", "v1soup-a0.7"),
                   ("V1 (a=1)", "qwen3-0.6b-p2-mixed"), ("verified-arm soup a=0.3", "soup-a0.3"),
                   ("verified-arm soup a=0.5", "soup-a0.5"), ("verified-arm soup a=0.7", "soup-a0.7"),
                   ("uncapped verified arm (a=1)", "qwen3-0.6b-cas-cmnrl"), ("base", "qwen3-0.6b-base")]:
    try:
        rows[label] = {"tag": tag, "tiers": tiers(tag), "xling_strict": xling(tag), "samelang_primary_r1": samelang(tag)}
    except FileNotFoundError as e:
        rows[label] = {"tag": tag, "missing": str(e)}
json.dump(rows, open(os.path.join(P, "results/v1_soups.json"), "w"), indent=2)
for k, v in rows.items():
    if "missing" in v: print(f"{k:30s} MISSING {v['missing']}"); continue
    t = v["tiers"]; print(f"{k:30s} easy {t['easy'][0]:6.2f}/{t['easy'][1]:6.2f}/{t['easy'][2]:6.2f}  med {t['medium'][0]:5.2f}  hard {t['hard'][0]:5.2f}  xling {v['xling_strict'][0]:6.2f}  samelang {v['samelang_primary_r1']}")
