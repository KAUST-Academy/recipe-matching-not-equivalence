#!/usr/bin/env python3
"""
Split the SABER attack pair file by CHANNEL, for the label-style re-attack.

WHY THIS EXISTS. The SABER attack reported in the paper (job 49589998,
models/saber-attack-6145) trained on `--max-rows 6145` of
data/saber_attack/pairs.jsonl, which holds 11,461 `saber_pair` (document
template) rows and 4,733 `saber_summary` (label substrate) rows.

CORRECTION 2026-07-31 (round-3 data audit, finding D-H1). An earlier version
of this docstring -- and of the paper -- asserted that because the file is
ordered channel-by-channel, the cap took a file-order PREFIX and so the run
saw ZERO summary rows. That is FALSE. train_invarembed.py shuffles with
random.Random(--seed) BEFORE truncating, so the cap is a seeded random
subsample. Replayed at seed 42 the published run trained on

    saber_pair 4,359  +  saber_summary 1,786   (29.1% of its budget)

and run_config.json's recorded n_source_ids=3606 / n_dev_source_ids=180 /
n_train_rows=5838 / n_train_triplets=11676 match that replay exactly, where a
prefix would have given 2,185 source ids. The label surface was DILUTED, not
absent, and the honest mechanism claim is channel PURITY rather than channel
presence: 29% label content scored -0.0129 vs base, a pure label file +0.0354.

This script writes the two channels separately so the re-attack can train on
each at a MATCHED budget and at full purity, which the original never had.

Outputs (row counts are measured, not assumed, and land in the manifest):
  data/saber_attack/pairs_summary_channel.jsonl   (label substrate)
  data/saber_attack/pairs_doc_channel.jsonl       (document template)
  results/saber_channel_split.json

Usage:  python scripts/make_saber_channels.py
"""

import json
import os
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "saber_attack", "pairs.jsonl")
OUT = os.path.join(ROOT, "data", "saber_attack")
MANIFEST = os.path.join(ROOT, "results", "saber_channel_split.json")


def main():
    os.chdir(ROOT)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    by = Counter(r["channel"] for r in rows)

    # What the published attack actually saw. The trainer shuffles with
    # random.Random(seed) and THEN truncates, so this must replay the shuffle;
    # rows[:6145] would model a file-order prefix and is wrong (finding D-H1).
    _kept = list(rows)
    random.Random(42).shuffle(_kept)
    trained = Counter(r["channel"] for r in _kept[:6145])

    paths = {}
    for chan, name in (("saber_summary", "pairs_summary_channel.jsonl"),
                       ("saber_pair", "pairs_doc_channel.jsonl")):
        sel = [r for r in rows if r["channel"] == chan]
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[chan] = {"path": os.path.relpath(p, ROOT), "rows": len(sel)}

    budget = min(v["rows"] for v in paths.values())
    doc = {
        "generated": "2026-07-31 (finding text corrected 2026-09-24)",
        "purpose": "Split data/saber_attack/pairs.jsonl by channel for the "
                   "label-style re-attack: the first SABER attack trained on a mix of "
                   "the two channels, so neither surface was tested alone.",
        "source_file": os.path.relpath(SRC, ROOT),
        "rows_total": len(rows),
        "rows_by_channel": dict(by),
        "what_the_published_attack_trained_on": dict(trained),
        "finding": ("The trainer shuffles with random.Random(seed) before applying "
                    "--max-rows, so the first attack's 6,145 rows are a seeded random draw "
                    "of both channels (counts in what_the_published_attack_trained_on): "
                    "mostly document-template rows with a minority of summary rows. The "
                    "channel arms train on each surface alone at a matched budget."),
        "channels": paths,
        "matched_budget_for_reattack": budget,
        "channel_semantics": {
            "saber_pair": "anchor = problem statement of x; positive = SABER's exact "
                          "statement-full DOCUMENT template for a partner y with "
                          "Jaccard(x,y) >= tau. Targets document style.",
            "saber_summary": "anchor = problem statement of x; positive = x's own "
                             "core-idea SUMMARY produced with SABER's verbatim "
                             "extraction prompt. Targets the labeling substrate: the "
                             "text SABER's relevance rule is actually computed over.",
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print("rows by channel:", dict(by))
    print("published attack trained on:", dict(trained))
    for c, v in paths.items():
        print(f"  {c:14s} -> {v['path']}  ({v['rows']} rows)")
    print("matched budget for the re-attack:", budget)


if __name__ == "__main__":
    main()
