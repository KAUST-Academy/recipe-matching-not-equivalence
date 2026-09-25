#!/usr/bin/env python3
"""
Derive the shared source-problem id list for the matched-budget experiment.

The supervision-controlled experiment (plan Rank 3) requires BOTH arms --
CAS-verified and LLM-judged -- to draw from the SAME source problems.
generate_cas_pairs.py ran over all eligible problems first, so its output
defines the shared set; generate_llm_pairs.slurm expects that list at
data/pairs/source_ids.txt.

This script writes source_ids.txt = unique source_ids of
data/cas_pairs/pairs.jsonl (problems with >= 1 CAS-verified positive), MINUS

  * anchor_to_corpus_mapping.json exclude_corpus_ids (8,698 MathNet-Retrieve
    eval-anchor matches -- should already be absent; asserted, and
    generate_llm_pairs.py would hard-abort on them anyway), and
  * data/crosslingual_eval/leakage_exclude_ids.json eval_member_ids (779
    query/gold problems of the cross-lingual duplicate eval; training on
    them would contaminate that eval -- train_invarembed.py drops them at
    load time too, so removing them here just avoids wasting LLM-generation
    budget on rows the trainer will discard).

Result: both arms' post-contamination-gate training sources are identical.

Usage:
  python scripts/make_source_ids.py            # writes data/pairs/source_ids.txt
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAS_PAIRS = os.path.join(PROJECT_ROOT, "data", "cas_pairs", "pairs.jsonl")
MAPPING = os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json")
XLING_LEAKAGE = os.path.join(PROJECT_ROOT, "data", "crosslingual_eval",
                             "leakage_exclude_ids.json")
OUT = os.path.join(PROJECT_ROOT, "data", "pairs", "source_ids.txt")


def main():
    ids = set()
    with open(CAS_PAIRS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["source_id"])
    print(f"[ids] {len(ids)} unique source_ids in {CAS_PAIRS}")

    anchor_excl = set(json.load(open(MAPPING))["exclude_corpus_ids"])
    bad = ids & anchor_excl
    assert not bad, (f"CAS pairs contain {len(bad)} anchor-excluded ids "
                     f"(e.g. {sorted(bad)[:5]}) -- regenerate them first")

    xling = set(json.load(open(XLING_LEAKAGE))["eval_member_ids"])
    n_before = len(ids)
    ids -= xling
    print(f"[ids] dropped {n_before - len(ids)} cross-lingual-eval "
          f"query/gold members (of {len(xling)} listed)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)) + "\n")
    print(f"[ids] wrote {len(ids)} ids -> {OUT}")


if __name__ == "__main__":
    main()
