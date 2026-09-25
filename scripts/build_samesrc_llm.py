#!/usr/bin/env python3
"""Same-source regenerated control, step 2c (2026-09-17): assemble the same-source
6,145-row recipe-arm training file.

  data/llm_pairs_samesrc/pairs.jsonl =
      the 4,101 v2-surviving rows of data/llm_pairs/pairs.jsonl
    + the first 2,044 verified rows of the second-rewrite generation over the
      SAME 4,101 sources (data/pairs/llm_pairs_samesrc_raw.jsonl, seed 1,
      converted with scripts/convert_llm_pairs.py), in the converted file's
      order (sorted by source id, the generator's own order), exactly as
      scripts/build_cleanfull_llm.py picked its 2,044.

Fails loudly if the survivor count is not 4,101, if the regeneration yield is
short, if any new row's source is not a survivor, if a source would get more
than one new row, if the file does not hold 6,145 rows over exactly 4,101
sources, or if anything hits the v2 gate. Two rows per source is what the
trainer already handles for the verified arm: the dev split is by source_id and
the NO_DUPLICATES sampler keeps rows sharing an anchor text out of one batch.
Writes results/llm_pairs_samesrc_build.json.
"""
import json, os, subprocess, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET, SURVIVORS = 6145, 4101
CLEANFULL_NEGS_PER_ROW = 2.622          # results/llm_pairs_cleanfull_build.json


def main():
    v2 = set(json.load(open(os.path.join(PROJECT, "anchor_to_corpus_mapping_v2.json")))["exclude_corpus_ids"])
    surv_ids = {l.strip() for l in open(os.path.join(PROJECT, "data/pairs/source_ids_survivors.txt")) if l.strip()}
    if len(surv_ids) != SURVIVORS:
        sys.exit(f"FATAL: survivor list has {len(surv_ids)} ids, expected {SURVIVORS}")
    survivors = [json.loads(l) for l in open(os.path.join(PROJECT, "data/llm_pairs/pairs.jsonl"), encoding="utf-8")
                 if l.strip() and json.loads(l)["source_id"] not in v2]
    if len(survivors) != SURVIVORS or {r["source_id"] for r in survivors} != surv_ids:
        sys.exit(f"FATAL: {len(survivors)} surviving rows do not match the survivor list")

    conv = os.path.join(PROJECT, "data/llm_pairs_samesrc/new_rows.jsonl")
    os.makedirs(os.path.dirname(conv), exist_ok=True)
    # convert_llm_pairs.py writes rows sorted by source_id, which is also the generator's
    # order over a sorted source list; build_cleanfull_llm.py took its 2,044 from that order too
    subprocess.run([sys.executable, os.path.join(PROJECT, "scripts/convert_llm_pairs.py"),
                    "--input", os.path.join(PROJECT, "data/pairs/llm_pairs_samesrc_raw.jsonl"),
                    "--output", conv], check=True)
    new_rows = [json.loads(l) for l in open(conv, encoding="utf-8")]

    need = TARGET - len(survivors)
    if len(new_rows) < need:
        sys.exit(f"FATAL: regeneration yielded {len(new_rows)} verified rows < {need} needed")
    picked = new_rows[:need]
    bad = [r["source_id"] for r in picked if r["source_id"] not in surv_ids or r["source_id"] in v2]
    if bad:
        sys.exit(f"FATAL: {len(bad)} new rows are not from a survivor source or hit the v2 gate, e.g. {bad[:5]}")
    new_ids = [r["source_id"] for r in picked]
    if len(set(new_ids)) != len(new_ids):
        sys.exit("FATAL: a source would receive more than one new row")
    rows = survivors + picked
    ids = [r["source_id"] for r in rows]
    if len(rows) != TARGET or len(set(ids)) != SURVIVORS:
        sys.exit(f"FATAL: assembled {len(rows)} rows over {len(set(ids))} sources, expected {TARGET} over {SURVIVORS}")
    same_text = sum(1 for r in picked if r["positive_text"].strip() ==
                    next(s["positive_text"] for s in survivors if s["source_id"] == r["source_id"]).strip())

    out = os.path.join(PROJECT, "data/llm_pairs_samesrc/pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    negs = sum(len(r.get("negatives") or []) for r in rows)
    summary = {"survivor_rows": len(survivors), "new_rows_available": len(new_rows), "new_rows_used": len(picked),
               "total_rows": len(rows), "sources": len(set(ids)), "sources_with_two_rows": len(set(new_ids)),
               "new_rows_identical_to_first_rewrite": same_text,
               "total_negatives": negs, "negs_per_row": round(negs / len(rows), 3),
               "cleanfull_negs_per_row": CLEANFULL_NEGS_PER_ROW}
    json.dump(summary, open(os.path.join(PROJECT, "results/llm_pairs_samesrc_build.json"), "w"), indent=2)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
