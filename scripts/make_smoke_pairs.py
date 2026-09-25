#!/usr/bin/env python3
"""
Generate a tiny dummy pairs file for smoke-testing train_invarembed.py.

Takes N (default 50) English problems from data/mathnet_corpus.parquet that
are NOT in the eval-anchor exclusion list, and builds trivially-transformed
"equivalence" rows in the exact cas_pairs schema:

  positive  = light paraphrase of the problem (synonym swaps + restatement
              prefix) -- trivially equivalent;
  negatives = (a) the same text with one digit perturbed (a fake "minimal
              non-equivalent edit"), (b) an unrelated problem's text.
              ~20% of rows get NO negatives to exercise the pair-only path.

This is TEST DATA ONLY -- the labels are heuristic, not verified. Never train
a real model on it.

Usage:
  python scripts/make_smoke_pairs.py                       # 50 rows
  python scripts/make_smoke_pairs.py --n 100 --output data/smoke_pairs/pairs.jsonl
"""

import argparse
import json
import os
import random
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SWAPS = [("Prove that", "Show that"), ("Find all", "Determine all"),
         ("Let ", "Suppose "), ("positive integer", "positive whole number"),
         ("such that", "with the property that"), ("Determine", "Compute")]


def paraphrase(text: str) -> str:
    out = text
    for a, b in SWAPS:
        out = out.replace(a, b)
    return "Consider the following problem. " + out


def perturb_digit(text: str, rng: random.Random) -> str:
    """Change one digit -> a minimally-edited NON-equivalent problem."""
    digits = [m.start() for m in re.finditer(r"\d", text)]
    if not digits:
        return text + " Assume additionally that $n$ is prime."
    i = rng.choice(digits)
    new = str((int(text[i]) + rng.randint(1, 8)) % 10)
    return text[:i] + new + text[i + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output",
                    default=os.path.join(PROJECT_ROOT, "data", "smoke_pairs",
                                         "pairs.jsonl"))
    args = ap.parse_args()

    import duckdb
    with open(os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json"),
              encoding="utf-8") as f:
        exclude = set(json.load(f)["exclude_corpus_ids"])
    rows = duckdb.sql(
        f"SELECT id, problem_markdown FROM "
        f"'{os.path.join(PROJECT_ROOT, 'data', 'mathnet_corpus.parquet')}' "
        f"WHERE language = 'English' AND length(problem_markdown) BETWEEN 80 AND 800"
    ).fetchall()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    picked = [(i, t) for i, t in rows if i not in exclude][:args.n]
    assert len(picked) == args.n, f"only {len(picked)} usable problems found"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for k, (cid, text) in enumerate(picked):
            negs = []
            if k % 5 != 4:  # ~20% pair-only rows
                other = picked[(k + 1) % len(picked)][1]
                negs = [{"text": perturb_digit(text, rng),
                         "kind": "digit_edit"},
                        {"text": other, "kind": "other_problem"}]
            f.write(json.dumps({"source_id": cid,
                                "positive_text": paraphrase(text),
                                "negatives": negs}, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} rows -> {args.output}")


if __name__ == "__main__":
    main()
