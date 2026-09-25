#!/usr/bin/env python3
"""E-R3: build the two missing cells of the
positives x negatives factorial.

  D4-pos + CAS-negs  data/llm_pairs_unrelated_casnegs/pairs.jsonl
      Every dose-unrelated (D4) row -- whose sources are a strict subset of
      the CAS arm's -- gains the negative list of ONE seeded-randomly chosen
      CAS row of the same source_id, preserving the CAS arm's per-row
      negative-count distribution (matched negative volume).
  CAS-pos + no-negs  data/cas_pairs_nonegs/pairs.jsonl
      ctrl-CAS rows with their negatives stripped (D4's asymmetry mirrored).

Deterministic: --seed (default 0) drives the CAS-row choice only.
"""
import argparse, json, os
from collections import defaultdict

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cas_rows = defaultdict(list)
    with open(os.path.join(PROJECT, "data/cas_pairs/pairs.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cas_rows[r["source_id"]].append(r)

    # ---- D4 positives + CAS negatives ------------------------------------
    out_dir = os.path.join(PROJECT, "data/llm_pairs_unrelated_casnegs")
    os.makedirs(out_dir, exist_ok=True)
    n, n_negs, missing = 0, 0, 0
    with open(os.path.join(PROJECT, "data/llm_pairs_unrelated/pairs.jsonl"),
              encoding="utf-8") as fin, \
         open(os.path.join(out_dir, "pairs.jsonl"), "w",
              encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            donors = cas_rows.get(r["source_id"])
            if not donors:
                missing += 1
                continue
            donor = donors[rng.integers(0, len(donors))]
            r["negatives"] = donor["negatives"]
            r["negatives_from"] = "cas_pairs row (seeded choice), E-R3"
            n += 1
            n_negs += len(r["negatives"])
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[d4pos+casnegs] {n} rows, {n_negs} negatives "
          f"({n_negs/max(1,n):.2f}/row), {missing} D4 sources without CAS rows")

    # ---- CAS positives, negatives stripped -------------------------------
    out_dir = os.path.join(PROJECT, "data/cas_pairs_nonegs")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(os.path.join(PROJECT, "data/cas_pairs/pairs.jsonl"),
              encoding="utf-8") as fin, \
         open(os.path.join(out_dir, "pairs.jsonl"), "w",
              encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            r["negatives"] = []
            n += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[caspos+nonegs] {n} rows, 0 negatives")


if __name__ == "__main__":
    main()
