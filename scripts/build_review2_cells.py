#!/usr/bin/env python3
"""E-R7 / E-R8 / E-R9 (2026-09-04): three further cells of the factorial, all
trained by scripts/factorial_cells.slurm.

  --which r7   D4 positives + near-misses written under an UNRELATED prompt
      Source of the negatives: data/llm_pairs_unrelated_negs/pairs.jsonl, the
      trainer-format conversion of the E-R7 generation run (prompt variant
      unrelated_negs in scripts/generate_llm_pairs.py: the D4 audience-adaptation
      persona plus a "spot the difference" drill; judge pass identical). Two
      cells mirror E-R6 exactly, with the D1 (Appendix-F) negatives replaced:
        data/llm_pairs_unrelated_unrelnegs/pairs.jsonl       count-matched
            first k of the unrelated-prompt negatives, k = the count the same
            source carries in the reference file (d4casnegs), capped at 3
        data/llm_pairs_unrelated_unrelnegs_full/pairs.jsonl  full (<= 3/row)
      D4 rows whose source has no unrelated_negs row are dropped (disclosed).
      Against d4llmnegs / d4llmnegsfull the only change is the PROMPT that
      wrote the near-misses.

  --which r8   CAS positives + the recipe arm's own near-misses
      data/cas_pairs_llmnegs/pairs.jsonl: for every D1 (ctrl-LLM) source, ONE
      seeded-randomly chosen CAS row of that source supplies the positive
      (exactly as build_factorial_cells.py chose CAS rows for the reference)
      and the D1 row supplies ALL its negatives (<= 3, the recipe arm's own
      volume). Against ctrl-CAS this swaps only the negatives' provenance
      (SymPy counterexamples -> LLM near-misses) at near-copy positives;
      against D1 it swaps only the positives (tests whether the CAS negatives
      drive the cross-language collapse).

  --which r9   back-translated positives + CAS negatives (non-LLM paraphrase)
      data/bt_pairs_casnegs/pairs.jsonl: the reference file (d4casnegs) with
      each D4 positive replaced by a back-translation of the SOURCE problem
      through an NMT model (scripts/backtranslate_anchors.py writes
      data/bt_pairs/pairs.jsonl); rows whose source has no back-translation
      are dropped (disclosed). Against the reference this swaps only the
      positives' authorship (LLM restatement -> NMT round trip), the non-LLM
      deep-paraphrase control.

  --which r10  E-R10 (2026-09-05): the reference file with
      each D4 positive replaced by the D5 (survey-restatement, prompt variant
      `survey`) positive of the same source -> data/llm_pairs_survey_casnegs/
      pairs.jsonl; rows whose source has no verified D5 positive are dropped
      (disclosed). Against the reference this swaps only the recipe-free
      prompt that wrote the positives (R5 W2 / Q1). D5 alone (no negatives)
      is data/llm_pairs_survey/pairs.jsonl straight from convert_llm_pairs.py.

  --which r13  E-R13: as r9 with the multi-hop
      back-translations of data/bt2_pairs/pairs.jsonl ->
      data/bt2_pairs_casnegs/pairs.jsonl (R5 Q4).

Deterministic; --seed drives the r8 CAS-row choice only. Writes
results/review2_cells_build.json (merged across invocations).
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(PROJECT, "results", "review2_cells_build.json")


def load(rel):
    with open(os.path.join(PROJECT, rel), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


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
    print(json.dumps(obj, indent=2))


def build_r7():
    d4 = load("data/llm_pairs_unrelated/pairs.jsonl")
    d4c = {r["source_id"]: r for r in load("data/llm_pairs_unrelated_casnegs/pairs.jsonl")}
    un = {r["source_id"]: r for r in load("data/llm_pairs_unrelated_negs/pairs.jsonl")}
    assert len(d4) == 6768 and len(d4c) == 6768, (len(d4), len(d4c))
    matched, full = [], []
    st = {"rows_in_d4": len(d4), "rows_dropped_no_unrelated_negs_row": 0,
          "rows_dropped_zero_negatives": 0, "k_truncated_to_cap_3": 0,
          "matched_negatives": 0, "full_negatives": 0,
          "matched_hist": Counter(), "full_hist": Counter()}
    for r in d4:
        s = r["source_id"]
        if s not in un:
            st["rows_dropped_no_unrelated_negs_row"] += 1
            continue
        negs = list(un[s]["negatives"])
        if not negs:
            st["rows_dropped_zero_negatives"] += 1
            continue
        k = len(d4c[s]["negatives"])
        if k > 3:
            st["k_truncated_to_cap_3"] += 1
            k = 3
        m = negs[:k]
        base = {"source_id": s, "positive_text": r["positive_text"]}
        matched.append(dict(base, negatives=m,
                            negatives_from="unrelated_negs row of the same source, "
                                           "first k (k = d4casnegs count), E-R7"))
        full.append(dict(base, negatives=negs,
                         negatives_from="unrelated_negs row of the same source, "
                                        "all negatives, E-R7"))
        st["matched_negatives"] += len(m)
        st["full_negatives"] += len(negs)
        st["matched_hist"][len(m)] += 1
        st["full_hist"][len(negs)] += 1
    n = len(full)
    pm = write_rows("data/llm_pairs_unrelated_unrelnegs", matched)
    pf = write_rows("data/llm_pairs_unrelated_unrelnegs_full", full)
    ref_npr = sum(len(v["negatives"]) for v in d4c.values()) / len(d4c)
    merge_stats("r7", {
        "inputs": ["data/llm_pairs_unrelated/pairs.jsonl (D4 positives)",
                   "data/llm_pairs_unrelated_negs/pairs.jsonl (unrelated-prompt negatives)",
                   "data/llm_pairs_unrelated_casnegs/pairs.jsonl (per-row counts)"],
        "rows_written": n,
        "rows_dropped_no_unrelated_negs_row": st["rows_dropped_no_unrelated_negs_row"],
        "rows_dropped_zero_negatives": st["rows_dropped_zero_negatives"],
        "k_truncated_to_cap_3": st["k_truncated_to_cap_3"],
        "count_matched": {"file": pm, "negatives": st["matched_negatives"],
                          "negatives_per_row": round(st["matched_negatives"] / n, 3),
                          "hist": dict(st["matched_hist"])},
        "full": {"file": pf, "negatives": st["full_negatives"],
                 "negatives_per_row": round(st["full_negatives"] / n, 3),
                 "hist": dict(st["full_hist"])},
        "reference_d4casnegs_negatives_per_row": round(ref_npr, 3),
        "e_r6_count_matched_rows_for_comparison": 5978,
    })


def build_r8(seed):
    rng = np.random.default_rng(seed)
    d1 = load("data/llm_pairs/pairs.jsonl")
    cas = defaultdict(list)
    for r in load("data/cas_pairs/pairs.jsonl"):
        cas[r["source_id"]].append(r)
    rows = []
    st = {"rows_in_d1": len(d1), "rows_dropped_no_cas_row": 0, "negatives": 0,
          "hist": Counter(), "cas_rows_per_source": Counter()}
    for r in d1:
        s = r["source_id"]
        if s not in cas:
            st["rows_dropped_no_cas_row"] += 1
            continue
        choice = cas[s][int(rng.integers(len(cas[s])))]
        st["cas_rows_per_source"][len(cas[s])] += 1
        negs = list(r["negatives"])
        rows.append({"source_id": s, "positive_text": choice["positive_text"],
                     "negatives": negs,
                     "positive_from": "cas_pairs row of the same source (seeded choice), E-R8",
                     "negatives_from": "ctrl-LLM (D1, Appendix-F) row of the same source, "
                                       "all negatives, E-R8"})
        st["negatives"] += len(negs)
        st["hist"][len(negs)] += 1
    p = write_rows("data/cas_pairs_llmnegs", rows)
    merge_stats("r8", {
        "inputs": ["data/llm_pairs/pairs.jsonl (D1 negatives and source list)",
                   "data/cas_pairs/pairs.jsonl (CAS positives)"],
        "seed": seed, "file": p, "rows_written": len(rows),
        "rows_dropped_no_cas_row": st["rows_dropped_no_cas_row"],
        "negatives": st["negatives"],
        "negatives_per_row": round(st["negatives"] / max(len(rows), 1), 3),
        "hist": dict(st["hist"]),
        "cas_rows_per_source_hist": dict(sorted(st["cas_rows_per_source"].items())),
    })


def build_r9():
    d4c = load("data/llm_pairs_unrelated_casnegs/pairs.jsonl")
    bt = {r["source_id"]: r for r in load("data/bt_pairs/pairs.jsonl")}
    rows, dropped = [], 0
    negs_total, hist = 0, Counter()
    for r in d4c:
        s = r["source_id"]
        if s not in bt or not str(bt[s].get("positive_text", "")).strip():
            dropped += 1
            continue
        rows.append({"source_id": s, "positive_text": bt[s]["positive_text"],
                     "negatives": list(r["negatives"]),
                     "positive_from": bt[s].get("positive_from", "back-translation, E-R9"),
                     "negatives_from": r.get("negatives_from", "cas_pairs row (seeded choice), E-R3")})
        negs_total += len(r["negatives"])
        hist[len(r["negatives"])] += 1
    p = write_rows("data/bt_pairs_casnegs", rows)
    merge_stats("r9", {
        "inputs": ["data/llm_pairs_unrelated_casnegs/pairs.jsonl (reference rows: CAS negatives)",
                   "data/bt_pairs/pairs.jsonl (back-translated positives)"],
        "file": p, "rows_written": len(rows), "rows_dropped_no_backtranslation": dropped,
        "negatives": negs_total,
        "negatives_per_row": round(negs_total / max(len(rows), 1), 3),
        "hist": dict(hist),
    })


def build_replace_positives(key, pos_file, out_dir, pos_from_default, label):
    """Reference rows (D4 positives + CAS negatives) with the positive replaced,
    per source, by the positive of `pos_file` (trainer or BT format)."""
    d4c = load("data/llm_pairs_unrelated_casnegs/pairs.jsonl")
    alt = {}
    for r in load(pos_file):
        if str(r.get("positive_text", "")).strip() and r["source_id"] not in alt:
            alt[r["source_id"]] = r                      # first positive per source
    rows, dropped = [], 0
    negs_total, hist = 0, Counter()
    for r in d4c:
        s = r["source_id"]
        if s not in alt:
            dropped += 1
            continue
        rows.append({"source_id": s, "positive_text": alt[s]["positive_text"],
                     "negatives": list(r["negatives"]),
                     "positive_from": alt[s].get("positive_from", pos_from_default),
                     "negatives_from": r.get("negatives_from", "cas_pairs row (seeded choice), E-R3")})
        negs_total += len(r["negatives"])
        hist[len(r["negatives"])] += 1
    p = write_rows(out_dir, rows)
    merge_stats(key, {
        "inputs": ["data/llm_pairs_unrelated_casnegs/pairs.jsonl (reference rows: CAS negatives)",
                   f"{pos_file} ({label})"],
        "file": p, "rows_written": len(rows), "rows_dropped_no_positive": dropped,
        "sources_in_positive_file": len(alt),
        "negatives": negs_total,
        "negatives_per_row": round(negs_total / max(len(rows), 1), 3),
        "hist": dict(hist),
    })


def build_r10():
    build_replace_positives("r10", "data/llm_pairs_survey/pairs.jsonl",
                            "data/llm_pairs_survey_casnegs",
                            "D5 survey restatement (prompt variant survey), E-R10",
                            "D5 positives")


def build_r13():
    build_replace_positives("r13", "data/bt2_pairs/pairs.jsonl", "data/bt2_pairs_casnegs",
                            "multi-hop back-translation, E-R13", "multi-hop back-translated positives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["r7", "r8", "r9", "r10", "r13"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"r7": build_r7, "r8": lambda: build_r8(args.seed), "r9": build_r9,
     "r10": build_r10, "r13": build_r13}[args.which]()


if __name__ == "__main__":
    main()
