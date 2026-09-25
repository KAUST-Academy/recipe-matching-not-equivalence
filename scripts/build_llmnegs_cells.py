#!/usr/bin/env python3
"""E-R6 (2026-09-03): the two cells that separate LLM-written negatives from
verified negatives in the positives x negatives factorial (experiment E-R3).
With recipe-free positives fixed, pairing them once with verified negatives
and once with LLM-written minimal-edit negatives isolates the negatives' source.

  D4-pos + LLM-negs, count-matched   data/llm_pairs_unrelated_llmnegs/pairs.jsonl
      Every D4 row whose source problem also has a ctrl-LLM row (D1: the
      Appendix-F prompt, which writes three near-miss negatives per source)
      takes the FIRST k negatives of that D1 row, k being the negative count
      the same source carries in the E-R3 d4casnegs file (data/
      llm_pairs_unrelated_casnegs). Against d4casnegs the only change is the
      provenance of the negatives (Appendix-F LLM-written near-misses vs
      SymPy counterexamples): positives identical, per-row counts identical
      except where k exceeds D1's cap of 3 (then 3).
  D4-pos + LLM-negs, full            data/llm_pairs_unrelated_llmnegs_full/pairs.jsonl
      The same rows with ALL of the D1 row's negatives (<= 3 per row, the
      recipe arm's own volume). Against D1 (= ctrl-LLM) this isolates the
      positive prompt at the recipe arm's own negatives.

790 of D4's 6,768 sources have no D1 row (D1 kept 6,145 of 7,089 sources after
judging); those D4 rows are DROPPED, so both new files carry 5,978 rows. That
is 2.7% fewer rows than the 6,145 d4casnegs trains under --max-rows; the
shortfall is disclosed, not hidden. No randomness: "first k" in D1's stored
order. Writes results/llmnegs_cells_build.json.
"""
import json
import os
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    with open(os.path.join(PROJECT, rel), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    d1 = {r["source_id"]: r for r in load("data/llm_pairs/pairs.jsonl")}
    d4 = load("data/llm_pairs_unrelated/pairs.jsonl")
    d4c = {r["source_id"]: r
           for r in load("data/llm_pairs_unrelated_casnegs/pairs.jsonl")}
    assert len(d1) == 6145 and len(d4) == 6768 and len(d4c) == 6768, \
        (len(d1), len(d4), len(d4c))

    outs = {
        "matched": os.path.join(PROJECT, "data/llm_pairs_unrelated_llmnegs"),
        "full": os.path.join(PROJECT, "data/llm_pairs_unrelated_llmnegs_full"),
    }
    for p in outs.values():
        os.makedirs(p, exist_ok=True)
    fm = open(os.path.join(outs["matched"], "pairs.jsonl"), "w", encoding="utf-8")
    ff = open(os.path.join(outs["full"], "pairs.jsonl"), "w", encoding="utf-8")

    stats = {"rows_in_d4": len(d4), "rows_dropped_no_d1_row": 0, "rows_written": 0,
             "k_truncated_to_d1_cap": 0, "matched": Counter(), "full": Counter(),
             "matched_negatives": 0, "full_negatives": 0,
             "matched_shortfall_vs_d4casnegs": 0}
    for r in d4:
        s = r["source_id"]
        if s not in d1:
            stats["rows_dropped_no_d1_row"] += 1
            continue
        d1_negs = d1[s]["negatives"]
        k = len(d4c[s]["negatives"])
        if k > len(d1_negs):
            stats["k_truncated_to_d1_cap"] += 1
            stats["matched_shortfall_vs_d4casnegs"] += k - len(d1_negs)
        matched = d1_negs[:k]
        base = {"source_id": s, "positive_text": r["positive_text"]}
        rm = dict(base, negatives=matched,
                  negatives_from="ctrl-LLM (D1, Appendix-F) row of the same source, "
                                 "first k where k = d4casnegs count, E-R6")
        rf = dict(base, negatives=list(d1_negs),
                  negatives_from="ctrl-LLM (D1, Appendix-F) row of the same source, "
                                 "all negatives, E-R6")
        fm.write(json.dumps(rm, ensure_ascii=False) + "\n")
        ff.write(json.dumps(rf, ensure_ascii=False) + "\n")
        stats["rows_written"] += 1
        stats["matched"][len(matched)] += 1
        stats["full"][len(d1_negs)] += 1
        stats["matched_negatives"] += len(matched)
        stats["full_negatives"] += len(d1_negs)
    fm.close()
    ff.close()

    n = stats["rows_written"]
    summary = {
        "generated_by": "scripts/build_llmnegs_cells.py",
        "inputs": ["data/llm_pairs_unrelated/pairs.jsonl (D4 positives)",
                   "data/llm_pairs/pairs.jsonl (D1 negatives)",
                   "data/llm_pairs_unrelated_casnegs/pairs.jsonl (per-row counts)"],
        "rows_in_d4": stats["rows_in_d4"],
        "rows_dropped_no_d1_row": stats["rows_dropped_no_d1_row"],
        "rows_written": n,
        "rows_truncated_to_d1_cap_of_3": stats["k_truncated_to_d1_cap"],
        "matched_negatives_short_of_d4casnegs": stats["matched_shortfall_vs_d4casnegs"],
        "count_matched": {"file": "data/llm_pairs_unrelated_llmnegs/pairs.jsonl",
                          "negatives": stats["matched_negatives"],
                          "negatives_per_row": round(stats["matched_negatives"] / n, 3),
                          "histogram": dict(sorted(stats["matched"].items()))},
        "full": {"file": "data/llm_pairs_unrelated_llmnegs_full/pairs.jsonl",
                 "negatives": stats["full_negatives"],
                 "negatives_per_row": round(stats["full_negatives"] / n, 3),
                 "histogram": dict(sorted(stats["full"].items()))},
        "reference_d4casnegs_negatives_per_row": round(
            sum(len(v["negatives"]) for v in d4c.values()) / len(d4c), 3),
    }
    with open(os.path.join(PROJECT, "results", "llmnegs_cells_build.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
