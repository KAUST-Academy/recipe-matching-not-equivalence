#!/usr/bin/env python3
"""E-R4 step 3: assemble the fully-clean 6,145-row
recipe-arm training file.

  data/llm_pairs_cleanfull/pairs.jsonl =
      the v2-surviving rows of data/llm_pairs/pairs.jsonl   (must be 4,101)
    + the first (6,145 - 4,101) = 2,044 verified rows of the fresh-source
      regeneration (data/pairs/llm_pairs_cleanfull_raw.jsonl, converted with
      scripts/convert_llm_pairs.py), in generation order.

Fails loudly if the survivor count is not 4,101, if the regeneration yield
is short, or if any assembled source_id hits the v2 gate or the original
source list. Writes results/llm_pairs_cleanfull_build.json.
"""
import json, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = 6145


def main():
    v2 = set(json.load(open(os.path.join(
        PROJECT, "anchor_to_corpus_mapping_v2.json")))["exclude_corpus_ids"])
    orig_sources = {l.strip() for l in
                    open(os.path.join(PROJECT, "data/pairs/source_ids.txt"))}

    survivors = []
    with open(os.path.join(PROJECT, "data/llm_pairs/pairs.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source_id"] not in v2:
                survivors.append(r)
    if len(survivors) != 4101:
        sys.exit(f"FATAL: expected 4,101 v2-surviving rows, got "
                 f"{len(survivors)} (gate or file changed?)")

    conv = os.path.join(PROJECT, "data/llm_pairs_cleanfull/new_rows.jsonl")
    os.makedirs(os.path.dirname(conv), exist_ok=True)
    subprocess.run([sys.executable,
                    os.path.join(PROJECT, "scripts/convert_llm_pairs.py"),
                    "--input", os.path.join(
                        PROJECT, "data/pairs/llm_pairs_cleanfull_raw.jsonl"),
                    "--output", conv], check=True)
    new_rows = [json.loads(l) for l in open(conv, encoding="utf-8")]

    need = TARGET - len(survivors)
    if len(new_rows) < need:
        sys.exit(f"FATAL: regeneration yielded {len(new_rows)} verified rows "
                 f"< {need} needed. Rerun generation with more sources.")
    picked = new_rows[:need]

    bad = [r["source_id"] for r in picked
           if r["source_id"] in v2 or r["source_id"] in orig_sources]
    if bad:
        sys.exit(f"FATAL: {len(bad)} new rows hit the v2 gate or the "
                 f"original source list, e.g. {bad[:5]}")
    ids = [r["source_id"] for r in survivors + picked]
    if len(set(ids)) != len(ids):
        sys.exit("FATAL: duplicate source_ids in assembled file")

    out = os.path.join(PROJECT, "data/llm_pairs_cleanfull/pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in survivors + picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    negs = sum(len(r.get("negatives") or []) for r in survivors + picked)
    summary = {"survivor_rows": len(survivors), "new_rows_available":
               len(new_rows), "new_rows_used": len(picked),
               "total_rows": TARGET, "total_negatives": negs,
               "negs_per_row": round(negs / TARGET, 3)}
    with open(os.path.join(PROJECT,
                           "results/llm_pairs_cleanfull_build.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
