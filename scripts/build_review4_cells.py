#!/usr/bin/env python3
"""Table-fill cells (2026-09-08), E-T1 .. E-T7: the seven empty cells of the
positives x negatives grid (paper Table tab:design), all trained by
scripts/factorial_cells.slurm at three seeds (42-44). No directional
prediction is registered for any of them; the grid is reported as it lands,
and nothing is quoted until all three seeds of a cell have landed and
scripts/factorial_verdict.py has been regenerated in one pass.

Conventions follow the existing cells exactly:
  * CAS negatives = the reference file's negatives for that source
    (data/llm_pairs_unrelated_casnegs, one seeded CAS row per source, E-R3),
    so a CAS-negative cell shares its negatives row-for-row with the reference
    and with bt_pairs_casnegs (about 1.27 per row).
  * LLM near-misses = ALL negatives of the D1 (Appendix-F) row, or of the
    unrelated-prompt row, for that source (<= 3 per row): the volume the recipe
    arm itself trains on, as in cas_pairs_llmnegs (E-R8).
  * CAS positives = the positive of cas_pairs_llmnegs for that source (the E-R8
    seeded choice), so the two CAS-positive LLM-negative cells differ only in
    who wrote the negatives.
  * Rows whose source lacks the needed partner file, or whose negatives list
    would be empty, are dropped and counted.

  --which casunrelnegs  CAS positives + unrelated-prompt near-misses   -> data/cas_pairs_unrelnegs
  --which btnonegs      back-translated positives, no negatives         -> data/bt_pairs_nonegs
  --which btunrelnegs   back-translated positives + unrelated near-misses -> data/bt_pairs_unrelnegs
  --which btllmnegs     back-translated positives + D1 near-misses      -> data/bt_pairs_llmnegs
  --which d1nonegs      D1 positives, negatives stripped                 -> data/llm_pairs_nonegs
  --which d1casnegs     D1 positives + the reference's CAS negatives     -> data/llm_pairs_casnegs
  --which d1unrelnegs   D1 positives + unrelated-prompt near-misses      -> data/llm_pairs_unrelnegs
  --which all           every cell above

Deterministic (no randomness: every choice is inherited from an existing
file). Writes results/review4_cells_build.json (merged across invocations).
"""
import argparse
import json
import os
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(PROJECT, "results", "review4_cells_build.json")

D1 = "data/llm_pairs/pairs.jsonl"
UNREL = "data/llm_pairs_unrelated_negs/pairs.jsonl"
REF = "data/llm_pairs_unrelated_casnegs/pairs.jsonl"
BT = "data/bt_pairs/pairs.jsonl"
CASPOS = "data/cas_pairs_llmnegs/pairs.jsonl"


def load(rel):
    with open(os.path.join(PROJECT, rel), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def by_source(rel, need_positive=False):
    out = {}
    for r in load(rel):
        if need_positive and not str(r.get("positive_text", "")).strip():
            continue
        out.setdefault(r["source_id"], r)          # first row per source
    return out


def write_rows(rel_dir, rows):
    d = os.path.join(PROJECT, rel_dir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.join(rel_dir, "pairs.jsonl")


def merge_stats(key, obj):
    allstats = {}
    if os.path.exists(STATS):
        with open(STATS, encoding="utf-8") as f:
            allstats = json.load(f)
    allstats[key] = obj
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(allstats, f, indent=2)
    print(f"[{key}] rows {obj['rows_written']}  dropped {obj['rows_dropped']}  "
          f"negs/row {obj['negatives_per_row']}  -> {obj['file']}")


def combine(key, pos_rows, pos_from, neg_by_source, neg_from, out_dir, inputs, allow_empty=False):
    """One output row per positive row whose source has a partner negatives row."""
    rows, dropped, negs_total, hist = [], 0, 0, Counter()
    for r in pos_rows:
        s = r["source_id"]
        if neg_by_source is None:
            negs = []
        else:
            if s not in neg_by_source:
                dropped += 1
                continue
            negs = list(neg_by_source[s].get("negatives") or [])
            if not negs and not allow_empty:
                dropped += 1
                continue
        rows.append({"source_id": s, "positive_text": r["positive_text"], "negatives": negs,
                     "positive_from": r.get("positive_from", pos_from),
                     "negatives_from": neg_from})
        negs_total += len(negs)
        hist[len(negs)] += 1
    p = write_rows(out_dir, rows)
    merge_stats(key, {"inputs": inputs, "file": p, "rows_written": len(rows), "rows_dropped": dropped,
                      "negatives": negs_total,
                      "negatives_per_row": round(negs_total / max(len(rows), 1), 3),
                      "hist": dict(sorted(hist.items()))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["casunrelnegs", "btnonegs", "btunrelnegs", "btllmnegs",
                             "d1nonegs", "d1casnegs", "d1unrelnegs", "all"])
    a = ap.parse_args()
    which = (["casunrelnegs", "btnonegs", "btunrelnegs", "btllmnegs", "d1nonegs", "d1casnegs", "d1unrelnegs"]
             if a.which == "all" else [a.which])
    d1 = load(D1)
    unrel = by_source(UNREL)
    ref = by_source(REF)
    bt = [r for r in load(BT) if str(r.get("positive_text", "")).strip()]
    caspos = load(CASPOS)
    d1_negs = {r["source_id"]: r for r in d1}
    for w in which:
        if w == "casunrelnegs":
            combine("casunrelnegs", caspos, "cas_pairs row of the same source (E-R8 seeded choice)", unrel,
                    "unrelated_negs row of the same source, all negatives, E-T1", "data/cas_pairs_unrelnegs",
                    [f"{CASPOS} (CAS positives, E-R8 choice)", f"{UNREL} (unrelated-prompt negatives)"])
        elif w == "btnonegs":
            combine("btnonegs", bt, "back-translation, E-R9", None, "none (E-T2)", "data/bt_pairs_nonegs",
                    [f"{BT} (back-translated positives)"])
        elif w == "btunrelnegs":
            combine("btunrelnegs", bt, "back-translation, E-R9", unrel,
                    "unrelated_negs row of the same source, all negatives, E-T3", "data/bt_pairs_unrelnegs",
                    [f"{BT} (back-translated positives)", f"{UNREL} (unrelated-prompt negatives)"])
        elif w == "btllmnegs":
            combine("btllmnegs", bt, "back-translation, E-R9", d1_negs,
                    "ctrl-LLM (D1, Appendix-F) row of the same source, all negatives, E-T4", "data/bt_pairs_llmnegs",
                    [f"{BT} (back-translated positives)", f"{D1} (D1 negatives)"])
        elif w == "d1nonegs":
            combine("d1nonegs", d1, "ctrl-LLM (D1, Appendix-F) positive", None, "none (E-T5)", "data/llm_pairs_nonegs",
                    [f"{D1} (D1 positives)"])
        elif w == "d1casnegs":
            combine("d1casnegs", d1, "ctrl-LLM (D1, Appendix-F) positive", ref,
                    "cas_pairs row (seeded choice), E-R3 reference negatives, E-T6", "data/llm_pairs_casnegs",
                    [f"{D1} (D1 positives)", f"{REF} (reference rows: CAS negatives)"],
                    allow_empty=True)   # the reference keeps rows whose CAS row has no counterexample; so does this cell
        elif w == "d1unrelnegs":
            combine("d1unrelnegs", d1, "ctrl-LLM (D1, Appendix-F) positive", unrel,
                    "unrelated_negs row of the same source, all negatives, E-T7", "data/llm_pairs_unrelnegs",
                    [f"{D1} (D1 positives)", f"{UNREL} (unrelated-prompt negatives)"])


if __name__ == "__main__":
    main()
