#!/usr/bin/env python3
"""
Convert LLM-arm pair rows (flat schema) into the trainer's grouped format.

generate_llm_pairs.py emits ONE ROW PER CANDIDATE (validate_pairs_schema.py
schema: pair_id / source_id / anchor_text / candidate_text / label /
verification{verified,...} / ...), while train_invarembed.py consumes ONE ROW
PER POSITIVE with the source's hard negatives attached:

    {"source_id": ..., "positive_text": ..., "negatives": [{"text": ...}, ...]}

(this grouped format is what generate_cas_pairs.py writes natively -- see the
train_invarembed.py docstring). This script is the bridge for the LLM-judged
supervision arm:

  * keeps only rows with verification.verified == true (judge-accepted) --
    rejected rows stay in the source file for rejection-rate analysis;
  * groups by source_id: every verified positive becomes one output row
    carrying ALL of that source's verified hard negatives (mirrors the CAS
    arm, where negatives are shared across a problem's positive records);
  * prints kept/dropped counts per label so the filtering is auditable.

Usage (after the generate_llm_pairs.slurm job has produced its output):
  python scripts/convert_llm_pairs.py                       # default paths
  python scripts/convert_llm_pairs.py --input data/pairs/llm_pairs.jsonl \
      --output data/llm_pairs/pairs.jsonl

The default output path data/llm_pairs/pairs.jsonl is exactly what the
train_invarembed.py docstring / train_invarembed.slurm name for the LLM arm:
  python scripts/train_invarembed.py --train-file data/llm_pairs/pairs.jsonl ...
"""

import argparse
import json
import os
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--input",
                    default=os.path.join(PROJECT_ROOT, "data", "pairs",
                                         "llm_pairs.jsonl"),
                    help="flat llm-arm jsonl from generate_llm_pairs.py")
    ap.add_argument("--output",
                    default=os.path.join(PROJECT_ROOT, "data", "llm_pairs",
                                         "pairs.jsonl"),
                    help="grouped trainer-format jsonl")
    ap.add_argument("--include-unverified", action="store_true",
                    help="keep judge-rejected rows too (ablation only; the "
                         "default matches the paper's LLM-judged arm)")
    args = ap.parse_args()

    stats = Counter()
    positives = defaultdict(list)   # source_id -> [candidate_text, ...]
    negatives = defaultdict(list)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["label"]
            verified = bool(row.get("verification", {}).get("verified"))
            stats[f"{label}_{'verified' if verified else 'unverified'}"] += 1
            if not (verified or args.include_unverified):
                continue
            text = (row.get("candidate_text") or "").strip()
            if not text:
                stats["empty_candidate_dropped"] += 1
                continue
            if label == "positive":
                positives[row["source_id"]].append(text)
            elif label == "hard_negative":
                negatives[row["source_id"]].append(text)
            else:
                stats["unknown_label_dropped"] += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_rows = n_neg = 0
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for sid in sorted(positives):
            negs = [{"text": t} for t in negatives.get(sid, [])]
            for pos in positives[sid]:
                out.write(json.dumps({"source_id": sid, "positive_text": pos,
                                      "negatives": negs},
                                     ensure_ascii=False) + "\n")
                n_rows += 1
                n_neg += len(negs)
    os.replace(tmp, args.output)

    orphan_negs = sum(len(v) for s, v in negatives.items()
                      if s not in positives)
    print(f"[convert] input label/verification counts: {dict(stats)}")
    print(f"[convert] wrote {n_rows} trainer rows "
          f"({len(positives)} source problems, {n_neg} attached negatives, "
          f"{orphan_negs} negatives dropped for lack of a verified positive) "
          f"-> {args.output}")
    print("[convert] matched-budget reminder: pass --max-rows "
          "min(#rows cas_pairs, #rows here) to train_invarembed.py on BOTH "
          "arms (rows are counted in THIS grouped format).")


if __name__ == "__main__":
    main()
