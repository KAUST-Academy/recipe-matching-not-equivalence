#!/usr/bin/env python3
"""E-R4 step 1: sample fresh v2-clean source ids.

Candidate pool = full public corpus
  minus anchor_to_corpus_mapping_v2.json exclude_corpus_ids   (corrected gate)
  minus data/crosslingual_eval leakage all_pair_member_ids    (eval reserve)
  minus data/pairs/source_ids.txt                             (original list)
Samples --n ids with numpy seed --seed and writes
data/pairs/source_ids_cleanfull.txt (sorted sample; generation order is the
generator's own seeded shuffle).
"""
import argparse, json, os

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        PROJECT, "data/pairs/source_ids_cleanfull.txt"))
    args = ap.parse_args()

    corpus = pd.read_parquet(os.path.join(PROJECT,
                                          "data/mathnet_corpus.parquet"))
    all_ids = set(corpus["id"])
    v2 = set(json.load(open(os.path.join(
        PROJECT, "anchor_to_corpus_mapping_v2.json")))["exclude_corpus_ids"])
    xl = set(json.load(open(os.path.join(
        PROJECT, "data/crosslingual_eval/leakage_exclude_ids.json")))
        ["all_pair_member_ids"])
    orig = {l.strip() for l in open(os.path.join(
        PROJECT, "data/pairs/source_ids.txt")) if l.strip()}

    pool = sorted(all_ids - v2 - xl - orig)
    print(f"[pool] corpus {len(all_ids)} - v2 {len(v2 & all_ids)} - "
          f"xling {len(xl & all_ids)} - original {len(orig & all_ids)} "
          f"-> {len(pool)} candidates")
    if len(pool) < args.n:
        raise SystemExit(f"FATAL: pool {len(pool)} < requested {args.n}")
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(pool, size=args.n, replace=False).tolist())
    with open(args.out, "w") as f:
        f.write("\n".join(sample) + "\n")
    print(f"[done] wrote {args.n} ids -> {args.out}")


if __name__ == "__main__":
    main()
