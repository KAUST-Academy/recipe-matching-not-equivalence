#!/usr/bin/env python3
"""
Pre-registered verdict for the SABER label-channel seed replication.

The label result rewrote the paper's taxonomy on the strength of ONE training
run per channel. The dose ladder already showed what that can cost: easy D2-D3
looked decisive at seed 42 and reversed sign across three seeds. This evaluates
the readings fixed in scripts/saber_label_seeds.slurm's header before the runs.

  S1 REPLICATES  every label seed beats the base AND the three-seed label mean
                 beats the three-seed doc mean -> the taxonomy rewrite stands.
  S2 DOES NOT    any label seed at or below the base, or a label/doc flip in any
                 seed -> the finding is single-run, the rewrite is withdrawn to
                 a scoping correction, and ood/abstract/intro/conclusion/
                 discussion revert.

Usage:  python scripts/saber_label_seed_verdict.py
Output: results/saber_label_seed_stats.json + a printed table.
"""

import json
import os
import statistics
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
BASE = 0.5747
ROWCAP = 4733
SEEDS = (42, 43, 44)


def dig(p, *keys):
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def path(ch, s):
    # seed 42 keeps the original naming from saber_label_attack.slurm
    return os.path.join(R, f"saber_{ch}_{ROWCAP}_noprompt.json" if s == 42
                        else f"saber_{ch}_{ROWCAP}_s{s}_noprompt.json")


def main():
    os.chdir(ROOT)
    vals, missing = {}, []
    for ch in ("label", "doc"):
        vals[ch] = {}
        for s in SEEDS:
            v = dig(path(ch, s), "settings", "statement-full", "ndcg@10")
            if v is None:
                missing.append(os.path.relpath(path(ch, s), ROOT))
            vals[ch][s] = v

    lab = [vals["label"][s] for s in SEEDS if vals["label"][s] is not None]
    doc = [vals["doc"][s] for s in SEEDS if vals["doc"][s] is not None]

    out = {"generated": str(date.today()),
           "script": "scripts/saber_label_seed_verdict.py",
           "preregistered_in": "scripts/saber_label_seeds.slurm header",
           "base_noprompt": BASE, "per_seed": vals, "missing_inputs": sorted(set(missing)),
           "scope_warning": ("Training-seed statistics. The paired interval in "
                             "results/saber_label_bootstrap.json resamples the 1,000 SABER "
                             "queries and carries no seed variance; the two are additive.")}

    if len(lab) < 3 or len(doc) < 3:
        out["verdict"] = f"PENDING — label {len(lab)}/3, doc {len(doc)}/3 seeds present"
    else:
        lm, dm = statistics.mean(lab), statistics.mean(doc)
        all_above = all(v > BASE for v in lab)
        no_flip = all(vals["label"][s] > vals["doc"][s] for s in SEEDS)
        out.update({
            "label_mean": round(lm, 4), "label_sd": round(statistics.stdev(lab), 4),
            "doc_mean": round(dm, 4), "doc_sd": round(statistics.stdev(doc), 4),
            "label_minus_base_per_seed": [round(v - BASE, 4) for v in lab],
            "label_minus_doc_per_seed": [round(vals["label"][s] - vals["doc"][s], 4)
                                         for s in SEEDS],
            "all_label_seeds_above_base": all_above,
            "label_beats_doc_in_every_seed": no_flip,
        })
        out["verdict"] = (
            f"S1 REPLICATES — every label seed clears the base ({lm:.4f} mean vs {BASE}) and "
            f"beats the document channel in every seed ({dm:.4f} mean). The taxonomy rewrite "
            f"stands: SABER is exposed on its label surface."
            if all_above and no_flip else
            f"S2 DOES NOT REPLICATE — label mean {lm:.4f}, doc mean {dm:.4f}; all-above-base="
            f"{all_above}, no-flip={no_flip}. The label result is single-run. WITHDRAW the "
            f"taxonomy rewrite back to a scoping correction of our original negative and revert "
            f"ood.tex, the abstract, intro, conclusion and discussion.")

    with open(os.path.join(R, "saber_label_seed_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{'seed':>6s} {'label':>9s} {'doc':>9s} {'label-doc':>11s}")
    for s in SEEDS:
        l, d = vals["label"][s], vals["doc"][s]
        ld = f"{l - d:+.4f}" if (l is not None and d is not None) else "--"
        print(f"{s:>6d} {l if l is not None else '--':>9} {d if d is not None else '--':>9} {ld:>11s}")
    print(f"{'base':>6s} {BASE:>9.4f}")
    print(f"\nVERDICT: {out['verdict']}")
    if out["missing_inputs"]:
        print("\nmissing:", ", ".join(out["missing_inputs"]))
    print("\nwrote results/saber_label_seed_stats.json")


if __name__ == "__main__":
    main()
