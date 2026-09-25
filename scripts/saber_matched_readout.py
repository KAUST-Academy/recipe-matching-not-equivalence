#!/usr/bin/env python3
"""Readout of the step- and positive-matched SABER summary arm
(models/saber-labelmatched-2368-s{42,43,44}, scripts/saber_matched.slurm):
nDCG@10 (statement-full, no prompt) per seed, mean +/- sd, gain over the base
0.5747, beside the original summary arm (105 steps, 4,706 positives) and the
document arm (57 steps, 2,368 positives); strict cross-language R@1 per seed.
Writes results/saber_matched.json."""
import json, os, numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nd = lambda f: json.load(open(os.path.join(P, f)))["settings"]["statement-full"]["ndcg@10"]
xl = lambda f: json.load(open(os.path.join(P, f)))["strict_crosslingual_gold"]["recall@1"]
base = nd("results/saber_base0.6b_noprompt.json")
seeds = ["s42", "s43", "s44"]
orig = {"summary": [nd(f"results/saber_label_4733{'' if s=='s42' else '_'+s}_noprompt.json") for s in seeds],
        "doc": [nd(f"results/saber_doc_4733{'' if s=='s42' else '_'+s}_noprompt.json") for s in seeds]}
matched = [nd(f"results/saber_labelmatched_2368_{s}_noprompt.json") for s in seeds]
mx = [xl(f"results/crosslingual_saber-labelmatched-2368-{s}.json") for s in seeds]
def stat(v): v = np.array(v); return {"per_seed": [round(float(x), 4) for x in v], "mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4), "gain_over_base": round(float(v.mean() - base), 4), "gain_sd": round(float(v.std(ddof=1)), 4), "all_seeds_above_base": bool((v > base).all())}
out = {"base": base, "matched_summary_arm": {"rows": 2368, "steps": 57, **stat(matched), "xling_strict_r1": mx},
       "original_summary_arm": {"rows": 4733, "steps": 105, "distinct_positives": 4706, **stat(orig["summary"])},
       "document_arm": {"rows": 4733, "steps": 57, "distinct_positives": 2368, **stat(orig["doc"])}}
json.dump(out, open(os.path.join(P, "results/saber_matched.json"), "w"), indent=2)
for k, v in out.items():
    if isinstance(v, dict): print(f"{k:22s} {v['per_seed']} mean {v['mean']:.4f} +/- {v['sd']:.4f}  gain {v['gain_over_base']:+.4f}  all>base {v['all_seeds_above_base']}" + (f"  xling {v['xling_strict_r1']}" if 'xling_strict_r1' in v else ""))
