#!/usr/bin/env python3
"""Clean-slice contamination analysis: is the ctrl-LLM vs ctrl-CAS gap an
artifact of near-verbatim benchmark text leaking into the LLM arm's anchors?

Background (review finding C1)
------------------------------
The always-on contamination gate in `train_invarembed.py` keys on
`anchor_to_corpus_mapping.json` (v1), which matched anchors by *exact*
whitespace-normalized text and therefore missed every corpus row carrying a
literal "Problem:" header.  `build_anchor_mapping_v2.py` recovers those:
14,921/15,000 anchors have a corpus twin, 15,244 corpus ids should have been
excluded rather than 8,698.  Both controlled arms take their *anchor* text
from the corpus row (`corpus.get(source_id)`), so a trained row whose
source_id is a v2-only twin of a benchmark query is a row that trained on
benchmark-query text.

This script partitions the 15,000 easy-tier queries by leakage status and
recomputes the arm gap inside each slice, per seed:

  v1_excluded        anchor matched by v1  -> its corpus id was dropped by the
                     gate ("guaranteed clean" as the paper claims; 8,761)
  v1_excluded_strict  ... and no v2 twin of it was trained in either arm
                     (removes the corpus-internal-duplicate channel)
  leaked_<arm>       anchor NOT v1-matched and at least one v2 twin id is in
                     that arm's trained source_ids  -> the arm literally
                     trained on this query's text
  untrained_twin     anchor NOT v1-matched, has v2 twins, none trained
                     (the correct comparison population for `leaked`)
  no_corpus_twin     the 79 anchors with no corpus twin under any v2 rule

Trained source_id sets are reconstructed exactly from the pair files by
replaying `train_invarembed.build_datasets`' filter -> `random.Random(seed)`
shuffle -> `--max-rows` truncation -> dev-split logic, and are *validated*
against each model's `run_config.json` data_stats (row/source/negative
histogram counts must match, else the script aborts).

Outputs results/clean_slice_analysis.json.  CPU only, ~1 min.

Usage: python scripts/clean_slice_analysis.py
"""

import argparse
import collections
import json
import os
import random
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import train_invarembed as ti  # noqa: E402  (pure-stdlib at import time)

QUERIES = f"{PROJECT}/data/retrieve/easy/queries.jsonl"
V1_MAP = f"{PROJECT}/anchor_to_corpus_mapping.json"
V2_MAP = f"{PROJECT}/anchor_to_corpus_mapping_v2.json"
RANKS = f"{PROJECT}/results/ranks"
HITS = f"{PROJECT}/results/analyze_hits_perquery_hard.jsonl"
BENCH_CORPUS = f"{PROJECT}/data/retrieve/easy/corpus.jsonl"
OUT = f"{PROJECT}/results/clean_slice_analysis.json"

ARMS = {"ctrl-llm": f"{PROJECT}/data/llm_pairs/pairs.jsonl",
        "ctrl-cas": f"{PROJECT}/data/cas_pairs/pairs.jsonl"}
MAX_ROWS = 6145
DEV_FRAC = 0.05


# --------------------------------------------------------------------------
# exact replay of the training-data selection
# --------------------------------------------------------------------------
def trained_sets(train_file, seed, corpus, exclude_anchor, exclude_xling,
                 max_rows=MAX_ROWS, dev_frac=DEV_FRAC):
    """Return (used_source_ids, trained_source_ids, stats) for one run."""
    raw = ti.load_pairs(train_file)
    kept = []
    n_excluded = n_xling = n_missing = 0
    for r in raw:
        if r["source_id"] in exclude_anchor:
            n_excluded += 1
            continue
        if r["source_id"] in exclude_xling:
            n_xling += 1
            continue
        anchor = r.get("anchor_text") or corpus.get(r["source_id"])
        if not anchor:
            n_missing += 1
            continue
        kept.append(r)
    rng = random.Random(seed)
    rng.shuffle(kept)
    if max_rows and max_rows < len(kept):
        kept = kept[:max_rows]
    source_ids = sorted({r["source_id"] for r in kept})
    rng.shuffle(source_ids)
    n_dev = max(1, int(round(dev_frac * len(source_ids)))) if dev_frac > 0 else 0
    dev_ids = set(source_ids[:n_dev])
    train_rows = [r for r in kept if r["source_id"] not in dev_ids]
    neg_hist = collections.Counter(len(r["negatives"]) for r in kept)
    stats = {
        "input_rows": len(raw),
        "rows_dropped_eval_anchor_overlap": n_excluded,
        "rows_dropped_crosslingual_eval_overlap": n_xling,
        "rows_dropped_source_id_not_in_corpus": n_missing,
        "rows_used": len(kept),
        "n_source_ids": len(source_ids),
        "n_dev_source_ids": len(dev_ids),
        "n_train_rows": len(train_rows),
        "n_dev_rows": len(kept) - len(train_rows),
        "negatives_per_row_histogram": {str(k): v for k, v
                                        in sorted(neg_hist.items())},
    }
    used = {r["source_id"] for r in kept}
    trained = {r["source_id"] for r in train_rows}
    return used, trained, stats, kept


def model_dir(arm, seed):
    return (f"{PROJECT}/models/{arm}-6145" if seed == 42
            else f"{PROJECT}/models/{arm}-6145-s{seed}")


def validate(arm, seed, stats):
    """Compare the replay against the model's recorded data_stats."""
    cfg = f"{model_dir(arm, seed)}/run_config.json"
    if not os.path.exists(cfg):
        return {"run_config": cfg, "status": "missing"}
    rec = json.load(open(cfg, encoding="utf-8"))["data_stats"]
    keys = ["input_rows", "rows_dropped_eval_anchor_overlap",
            "rows_dropped_crosslingual_eval_overlap", "rows_used",
            "n_source_ids", "n_dev_source_ids", "n_train_rows", "n_dev_rows",
            "negatives_per_row_histogram"]
    diffs = {k: {"replay": stats[k], "recorded": rec.get(k)}
             for k in keys if stats[k] != rec.get(k)}
    return {"run_config": cfg,
            "status": "match" if not diffs else "MISMATCH",
            "diffs": diffs}


def doc_type(doc_id):
    if doc_id.endswith("::orig"):
        return "orig"
    if "::eq::" in doc_id:
        return "eq_gold"
    if "::nm::" in doc_id:
        return "nm_hard_negative"
    return "other"


def document_side_audit(runs, seeds, norm):
    """Do the *positives/negatives* of the training files reproduce benchmark
    corpus documents?  The paper claims "no benchmark query or corpus document,
    generated or otherwise, appears in any training set"; this measures the
    document side of that claim under the v2 aggressive normalizer."""
    docs = [json.loads(l) for l in open(BENCH_CORPUS, encoding="utf-8")
            if l.strip()]
    idx = collections.defaultdict(list)
    for d in docs:
        k = norm(d["text"])
        if len(k) >= 12:
            idx[k].append(d["_id"])
    corpus_keys = set()
    for cid, text in ti.load_corpus_texts().items():
        k = norm(text)
        if len(k) >= 12:
            corpus_keys.add(k)
    tot = collections.Counter(doc_type(d["_id"]) for d in docs)
    organic = collections.Counter()
    for d in docs:
        k = norm(d["text"])
        if len(k) >= 12 and k in corpus_keys:
            organic[doc_type(d["_id"])] += 1

    out = {
        "benchmark_corpus": BENCH_CORPUS,
        "n_docs": len(docs),
        "docs_by_type": dict(tot),
        "docs_that_are_organic_corpus_problems": {
            t: {"n": organic[t], "of": tot[t],
                "pct": round(100.0 * organic[t] / tot[t], 2)} for t in tot},
        "arms": {},
    }
    for arm, path in ARMS.items():
        rows = ti.load_pairs(path)
        used42 = runs[(arm, seeds[0])]["used"]
        rec = {"rows_in_file": len(rows),
               "n_positives": len(rows),
               "n_negatives": sum(len(r["negatives"]) for r in rows),
               "positives_matching_benchmark_doc": collections.Counter(),
               "negatives_matching_benchmark_doc": collections.Counter(),
               "positives_matching_in_seed42_rows": 0,
               "negatives_matching_in_seed42_rows": 0}
        for r in rows:
            k = norm(r["positive_text"])
            if len(k) >= 12 and k in idx:
                for t in {doc_type(i) for i in idx[k]}:
                    rec["positives_matching_benchmark_doc"][t] += 1
                if r["source_id"] in used42:
                    rec["positives_matching_in_seed42_rows"] += 1
            for neg in r["negatives"]:
                k = norm(neg)
                if len(k) >= 12 and k in idx:
                    for t in {doc_type(i) for i in idx[k]}:
                        rec["negatives_matching_benchmark_doc"][t] += 1
                    if r["source_id"] in used42:
                        rec["negatives_matching_in_seed42_rows"] += 1
        rec["positives_matching_benchmark_doc"] = dict(
            rec["positives_matching_benchmark_doc"])
        rec["negatives_matching_benchmark_doc"] = dict(
            rec["negatives_matching_benchmark_doc"])
        out["arms"][arm] = rec
    out["reading"] = (
        "normalization-equal, not merely similar. ::orig docs are the "
        "benchmark's 1,668 organic originals (99.8% are corpus problems), so "
        "any corpus-sampled negative can reproduce one; ::nm:: hits are hard-"
        "negative documents; the handful of ::eq:: (gold) hits sit mostly in "
        "the NEGATIVES column, where the effect is to push the gold document "
        "away -- conservative for the LLM arm, not favourable.")
    return out


def load_ranks(path):
    """qid -> gold_rank (0-based; 0 == top-1 hit)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["qid"]] = d["gold_rank"]
    return out


def r_at_1(ranks, qids):
    qids = [q for q in qids if q in ranks]
    if not qids:
        return None, 0
    return 100.0 * sum(1 for q in qids if ranks[q] == 0) / len(qids), len(qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    queries = [json.loads(l) for l in open(QUERIES, encoding="utf-8")
               if l.strip()]
    qids = [q["_id"] for q in queries]
    v1 = json.load(open(V1_MAP, encoding="utf-8"))
    v2 = json.load(open(V2_MAP, encoding="utf-8"))
    v1_anchor_matched = {m["anchor_id"] for m in v1["mapping"]}
    v1_ids = set(v1["exclude_corpus_ids"])
    a2i = {k: set(v) for k, v in v2["anchor_to_corpus_ids"].items()}

    corpus = ti.load_corpus_texts()
    exclude_anchor, exclude_xling = ti.load_exclude_ids()
    assert exclude_anchor == v1_ids, "gate mapping is not the v1 file"

    # ---- replay every arm x seed ----
    runs, validations = {}, {}
    for arm, path in ARMS.items():
        for seed in seeds:
            used, trained, stats, kept = trained_sets(
                path, seed, corpus, exclude_anchor, exclude_xling)
            runs[(arm, seed)] = {"used": used, "trained": trained,
                                 "stats": stats, "kept": kept}
            validations[f"{arm}-s{seed}"] = validate(arm, seed, stats)
    bad = {k: v for k, v in validations.items() if v["status"] == "MISMATCH"}
    if bad:
        print(json.dumps(bad, indent=2))
        sys.exit("FATAL: training-data replay does not match run_config")
    print(f"replay validated against {sum(1 for v in validations.values() if v['status']=='match')}"
          f"/{len(validations)} run_config files")

    # ---- per-arm contamination accounting (task item 3) ----
    # a trained row is "on benchmark-query text" iff its source_id is a v2
    # twin of some anchor (its anchor text IS that query's text)
    v2_id_to_anchors = collections.defaultdict(set)
    for a, ids in a2i.items():
        for pid in ids:
            v2_id_to_anchors[pid].add(a)
    v2_ids = set(v2_id_to_anchors)
    # ids matched by a *text-key* rule only (R1-R3): the anchor text and the
    # corpus text are the same problem statement up to formatting.  R5 mined
    # partners are equivalent-problem reprints, i.e. paraphrase-level leakage.
    key_rule_ids = collections.defaultdict(set)
    for m in v2["mapping"]:
        if m["rule"] in ("R1_exact_v1", "R2_boilerplate", "R3_aggressive"):
            key_rule_ids[m["corpus_id"]].add(m["anchor_id"])

    arm_rowstats = {}
    for arm in ARMS:
        for seed in seeds:
            r = runs[(arm, seed)]
            for scope in ("used", "trained"):
                ids = r[scope]
                n_v2 = sum(1 for s in ids if s in v2_ids)
                n_key = sum(1 for s in ids if s in key_rule_ids)
                n_v1 = sum(1 for s in ids if s in v1_ids)
                arm_rowstats[f"{arm}-s{seed}-{scope}"] = {
                    "n_source_ids": len(ids),
                    "n_v1_excluded_ids": n_v1,
                    "n_v2_matched_ids": n_v2,
                    "pct_v2_matched": round(100.0 * n_v2 / len(ids), 2),
                    "n_v2_textkey_matched_ids": n_key,
                    "pct_v2_textkey_matched": round(100.0 * n_key / len(ids), 2),
                }
    # rows (not source ids) for the headline fraction, split by leakage channel
    #   missed_exclusion       source_id is a v2 twin of an anchor v1 never
    #                          matched -> the C1 defect proper
    #   corpus_internal_dupe   source_id is a twin only of anchors v1 DID
    #                          match (v1 excluded one id per text, not all)
    v2_only_ids = {pid: a for pid, a in v2_id_to_anchors.items()
                   if a - v1_anchor_matched}
    dupe_only_ids = {pid for pid, a in v2_id_to_anchors.items()
                     if pid not in v2_only_ids}
    row_frac = {}
    for arm, path in ARMS.items():
        for seed in seeds:
            kept = runs[(arm, seed)]["kept"]
            dev_excl = runs[(arm, seed)]["trained"]
            n_rows = len(kept)
            n_leak = sum(1 for r in kept if r["source_id"] in v2_ids)
            n_leak_key = sum(1 for r in kept if r["source_id"] in key_rule_ids)
            n_missed = sum(1 for r in kept if r["source_id"] in v2_only_ids)
            n_dupe = sum(1 for r in kept if r["source_id"] in dupe_only_ids)
            tr = [r for r in kept if r["source_id"] in dev_excl]
            row_frac[f"{arm}-s{seed}"] = {
                "rows_used": n_rows,
                "rows_on_v2_matched_benchmark_text": n_leak,
                "pct_rows_on_v2_matched_benchmark_text":
                    round(100.0 * n_leak / n_rows, 2),
                "rows_on_v2_textkey_benchmark_text": n_leak_key,
                "pct_rows_on_v2_textkey_benchmark_text":
                    round(100.0 * n_leak_key / n_rows, 2),
                "channel_missed_exclusion_rows": n_missed,
                "channel_missed_exclusion_pct":
                    round(100.0 * n_missed / n_rows, 2),
                "channel_corpus_internal_dupe_rows": n_dupe,
                "channel_corpus_internal_dupe_pct":
                    round(100.0 * n_dupe / n_rows, 2),
                "distinct_benchmark_queries_whose_text_was_used":
                    len({a for r in kept
                         for a in v2_id_to_anchors.get(r["source_id"], ())}),
                "train_rows_excl_dev": len(tr),
                "train_rows_on_v2_matched_benchmark_text":
                    sum(1 for r in tr if r["source_id"] in v2_ids),
                "pct_train_rows_on_v2_matched_benchmark_text":
                    round(100.0 * sum(1 for r in tr if r["source_id"] in v2_ids)
                          / max(1, len(tr)), 2),
            }

    # ---- sensitivity of the headline "% of rows trained on benchmark text"
    #      to the matching-rule set (the review reported 31.7% for ctrl-llm) --
    rule_pid_anchors = collections.defaultdict(lambda:
                                               collections.defaultdict(set))
    for m in v2["mapping"]:
        rule_pid_anchors[m["rule"]][m["corpus_id"]].add(m["anchor_id"])

    def ids_for(rules, non_v1_anchors_only):
        out = set()
        for r in rules:
            for pid, ancs in rule_pid_anchors[r].items():
                if (ancs - v1_anchor_matched) if non_v1_anchors_only else ancs:
                    out.add(pid)
        return out

    TEXT_KEYS = ["R1_exact_v1", "R2_boilerplate", "R3_aggressive"]
    VARIANTS = [
        ("exact_or_prefix_only__missed_exclusions",
         ["R1_exact_v1", "R2_boilerplate"], True),
        ("all_text_keys__missed_exclusions", TEXT_KEYS, True),
        ("all_rules__missed_exclusions", TEXT_KEYS + ["R5_mined_partner"], True),
        ("all_text_keys__any_anchor", TEXT_KEYS, False),
        ("all_rules__any_anchor", TEXT_KEYS + ["R5_mined_partner"], False),
    ]
    row_variants = {}
    for arm in ARMS:
        for seed in seeds:
            kept = runs[(arm, seed)]["kept"]
            d = {}
            for label, rules, nonv1 in VARIANTS:
                ids = ids_for(rules, nonv1)
                n = sum(1 for r in kept if r["source_id"] in ids)
                d[label] = {"rows": n,
                            "pct": round(100.0 * n / len(kept), 2)}
            row_variants[f"{arm}-s{seed}"] = d

    # ---- what a v2-gated regeneration would cost (for the paper's fix note)
    v2_ids_all = set(v2["exclude_corpus_ids"])
    gate_impact = {}
    for arm, path in ARMS.items():
        raw = ti.load_pairs(path)
        n_v1 = sum(1 for r in raw if r["source_id"] in v1_ids)
        n_x = sum(1 for r in raw if r["source_id"] not in v1_ids
                  and r["source_id"] in exclude_xling)
        surv_v1 = [r for r in raw if r["source_id"] not in v1_ids
                   and r["source_id"] not in exclude_xling]
        surv_v2 = [r for r in surv_v1 if r["source_id"] not in v2_ids_all]
        gate_impact[arm] = {
            "pair_file": path,
            "rows_in_file": len(raw),
            "rows_dropped_by_v1_gate": n_v1,
            "rows_dropped_by_crosslingual_gate": n_x,
            "rows_surviving_v1_gate": len(surv_v1),
            "rows_surviving_v2_gate": len(surv_v2),
            "additional_rows_v2_would_drop": len(surv_v1) - len(surv_v2),
            "pct_of_v1_surviving_rows_v2_would_drop":
                round(100.0 * (len(surv_v1) - len(surv_v2)) / len(surv_v1), 2),
            "distinct_source_ids_surviving_v2": len({r["source_id"]
                                                     for r in surv_v2}),
            "matched_budget_6145_still_feasible": len(surv_v2) >= MAX_ROWS,
        }

    # ---- document-side leakage audit (positives / negatives) ----
    sys.path.insert(0, os.path.join(PROJECT, "scripts"))
    import build_anchor_mapping_v2 as bam2
    doc_audit = document_side_audit(runs, seeds, bam2.norm_aggressive)

    # ---- slice construction + per-seed metrics ----
    hits = {}
    if os.path.exists(HITS):
        for line in open(HITS, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                hits[d["qid"]] = (d["group"] == "hit")


    def run_slices(scope):
        """One pass of the slice analysis.

        scope='trained' -> leakage judged against the source_ids the weights
        actually saw (dev rows removed); scope='used' -> against every row the
        arm's gated data file contributed (the definition the 2026-07-30
        internal audit used, and the one comparable across seeds for
        ctrl-llm, whose file is used in full).
        """
        per_seed = {}
        for seed in seeds:
            rk = {}
            for arm in ARMS:
                p = f"{RANKS}/easy_{arm}-s{seed}.ranks.jsonl"
                if os.path.exists(p):
                    rk[arm] = load_ranks(p)
            if len(rk) < len(ARMS):
                print(f"[skip] seed {seed}: rank dumps missing")
                continue
            tr_llm = runs[("ctrl-llm", seed)][scope]
            tr_cas = runs[("ctrl-cas", seed)][scope]

            slices = collections.defaultdict(list)
            for qid in qids:
                twins = a2i.get(qid, set())
                l_llm = bool(twins & tr_llm)
                l_cas = bool(twins & tr_cas)
                if qid in v1_anchor_matched:
                    slices["v1_excluded"].append(qid)
                    (slices["v1_excluded_leaked_dupe"] if (l_llm or l_cas)
                     else slices["v1_excluded_strict"]).append(qid)
                elif not twins:
                    slices["no_corpus_twin"].append(qid)
                elif l_llm or l_cas:
                    slices["leaked_either"].append(qid)
                else:
                    slices["untrained_twin"].append(qid)
                if qid not in v1_anchor_matched and twins:
                    slices["leaked_llm" if l_llm
                           else "notleaked_llm"].append(qid)
                    slices["leaked_cas" if l_cas
                           else "notleaked_cas"].append(qid)
                if l_llm:
                    slices["leaked_llm_any"].append(qid)
                if l_cas:
                    slices["leaked_cas_any"].append(qid)
            slices["strict_clean_all"] = [q for q in qids
                                          if not (a2i.get(q, set())
                                                  & (tr_llm | tr_cas))]

            out_slices = {}
            for name, ids in sorted(slices.items()):
                row = {"n_queries": len(ids),
                       "share_of_queries_pct": round(100.0 * len(ids)
                                                     / len(qids), 2)}
                for arm in ARMS:
                    v, n = r_at_1(rk[arm], ids)
                    row[f"{arm}_r_at_1"] = None if v is None else round(v, 2)
                    row["n_scored"] = n
                if row.get("ctrl-llm_r_at_1") is not None:
                    row["arm_gap_llm_minus_cas"] = round(
                        row["ctrl-llm_r_at_1"] - row["ctrl-cas_r_at_1"], 2)
                if hits and seed == 42:
                    sub = [q for q in ids if q in hits]
                    row["hard_p_recipe_hit_pct"] = (
                        round(100.0 * sum(hits[q] for q in sub) / len(sub), 2)
                        if sub else None)
                    row["hard_n_hits"] = int(sum(hits[q] for q in sub))
                    row["hard_n_scored"] = len(sub)
                out_slices[name] = row

            overall = {}
            for arm in ARMS:
                v, n = r_at_1(rk[arm], qids)
                overall[f"{arm}_r_at_1"] = round(v, 2)
                overall["n_scored"] = n
            overall["arm_gap_llm_minus_cas"] = round(
                overall["ctrl-llm_r_at_1"] - overall["ctrl-cas_r_at_1"], 2)
            if hits and seed == 42:
                overall["hard_p_recipe_hit_pct"] = round(
                    100.0 * sum(hits.values()) / len(hits), 2)

            # ---- leakage attribution ----
            # excess R@1 the leaked slice buys an arm relative to the matched
            # comparison population (same "not v1-excluded, has a twin" pool)
            attrib = {}
            for arm, ln, nn in (("ctrl-llm", "leaked_llm", "notleaked_llm"),
                                ("ctrl-cas", "leaked_cas", "notleaked_cas")):
                L, NL = out_slices.get(ln), out_slices.get(nn)
                if not L or not NL or not L["n_queries"]:
                    continue
                w = L["n_queries"] / len(qids)
                delta = L[f"{arm}_r_at_1"] - NL[f"{arm}_r_at_1"]
                attrib[arm] = {
                    "leaked_n": L["n_queries"],
                    "leaked_share_pct": round(100 * w, 2),
                    "leaked_r_at_1": L[f"{arm}_r_at_1"],
                    "untrained_twin_r_at_1": NL[f"{arm}_r_at_1"],
                    "delta_leaked_minus_untrained": round(delta, 2),
                    "overall_r_at_1_excess_pts": round(w * delta, 2),
                }
            if len(attrib) == 2:
                net = (attrib["ctrl-llm"]["overall_r_at_1_excess_pts"]
                       - attrib["ctrl-cas"]["overall_r_at_1_excess_pts"])
                attrib["net_gap_attributable_to_leakage_pts"] = round(net, 2)
                attrib["gap_after_removing_leakage_excess"] = round(
                    overall["arm_gap_llm_minus_cas"] - net, 2)
                attrib["note"] = (
                    "excess = (leaked share) x (leaked R@1 - untrained-twin "
                    "R@1), i.e. the overall-R@1 points the leaked slice buys "
                    "the arm above its own untrained-twin baseline; net = LLM "
                    "excess - CAS excess (both arms take anchors from the "
                    "corpus, so both leak)")
            per_seed[str(seed)] = {"overall": overall, "slices": out_slices,
                                   "leakage_attribution": attrib}
            print(f"[{scope}] seed {seed}: overall gap "
                  f"{overall['arm_gap_llm_minus_cas']} | v1_excluded "
                  f"{out_slices['v1_excluded']['arm_gap_llm_minus_cas']} "
                  f"(n={out_slices['v1_excluded']['n_queries']}) | "
                  f"strict_clean "
                  f"{out_slices['strict_clean_all']['arm_gap_llm_minus_cas']} "
                  f"(n={out_slices['strict_clean_all']['n_queries']}) | "
                  f"leaked_llm n={out_slices['leaked_llm']['n_queries']} "
                  f"R@1 {out_slices['leaked_llm']['ctrl-llm_r_at_1']} vs "
                  f"untrained {out_slices['notleaked_llm']['ctrl-llm_r_at_1']}"
                  f" | net leak attribution "
                  f"{per_seed[str(seed)]['leakage_attribution'].get('net_gap_attributable_to_leakage_pts')}")
        return per_seed

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        sd = ((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
              if len(vals) > 1 else 0.0)
        return {"n_seeds": len(vals), "mean": round(mean, 2),
                "sd": round(sd, 2), "min": min(vals), "max": max(vals),
                "values": vals}

    def summarize(per_seed):
        def across(name, key):
            return agg([per_seed[s]["slices"][name][key] for s in per_seed
                        if name in per_seed[s]["slices"]])
        return {
            "overall_gap": agg([per_seed[s]["overall"]["arm_gap_llm_minus_cas"]
                                for s in per_seed]),
            "overall_llm_r_at_1": agg([per_seed[s]["overall"]["ctrl-llm_r_at_1"]
                                       for s in per_seed]),
            "overall_cas_r_at_1": agg([per_seed[s]["overall"]["ctrl-cas_r_at_1"]
                                       for s in per_seed]),
            "gap_v1_excluded": across("v1_excluded", "arm_gap_llm_minus_cas"),
            "gap_v1_excluded_strict": across("v1_excluded_strict",
                                             "arm_gap_llm_minus_cas"),
            "gap_strict_clean_all": across("strict_clean_all",
                                           "arm_gap_llm_minus_cas"),
            "gap_leaked_either": across("leaked_either",
                                        "arm_gap_llm_minus_cas"),
            "gap_leaked_llm": across("leaked_llm", "arm_gap_llm_minus_cas"),
            "gap_untrained_twin": across("untrained_twin",
                                         "arm_gap_llm_minus_cas"),
            "gap_no_corpus_twin": across("no_corpus_twin",
                                         "arm_gap_llm_minus_cas"),
            "llm_r1_leaked_llm": across("leaked_llm", "ctrl-llm_r_at_1"),
            "llm_r1_notleaked_llm": across("notleaked_llm", "ctrl-llm_r_at_1"),
            "cas_r1_leaked_cas": across("leaked_cas", "ctrl-cas_r_at_1"),
            "cas_r1_notleaked_cas": across("notleaked_cas",
                                           "ctrl-cas_r_at_1"),
            "n_leaked_llm": across("leaked_llm", "n_queries"),
            "net_gap_attributable_to_leakage_pts": agg(
                [per_seed[s]["leakage_attribution"].get(
                    "net_gap_attributable_to_leakage_pts") for s in per_seed]),
            "llm_only_leakage_excess_pts": agg(
                [per_seed[s]["leakage_attribution"]["ctrl-llm"][
                    "overall_r_at_1_excess_pts"] for s in per_seed]),
        }

    results = {}
    for scope in ("trained", "used"):
        ps = run_slices(scope)
        results[scope] = {"per_seed": ps, "cross_seed_summary": summarize(ps)}

    # ---- explicit cross-check against the review's reported values ----
    rv = results["used"]["per_seed"].get("42", {})
    rvs, rvo = rv.get("slices", {}), rv.get("overall", {})
    reviewer = [
        ("seed-42 overall easy arm gap", 44.6,
         rvo.get("arm_gap_llm_minus_cas")),
        ("seed-42 arm gap on v1-excluded (guaranteed-clean) slice", 45.7,
         rvs.get("v1_excluded", {}).get("arm_gap_llm_minus_cas")),
        ("seed-42 ctrl-llm easy R@1 on leaked slice", 68.8,
         rvs.get("leaked_llm", {}).get("ctrl-llm_r_at_1")),
        ("seed-42 ctrl-llm easy R@1 on untrained-twin slice", 58.0,
         rvs.get("notleaked_llm", {}).get("ctrl-llm_r_at_1")),
        ("gap points attributable to leakage (LLM arm only)", 1.4,
         rv.get("leakage_attribution", {}).get("ctrl-llm", {})
           .get("overall_r_at_1_excess_pts")),
        ("hard-tier P(recipe hit) on leaked slice, %", 6.2,
         rvs.get("leaked_llm", {}).get("hard_p_recipe_hit_pct")),
        ("hard-tier P(recipe hit) on clean slices, %", 9.85,
         rvs.get("v1_excluded", {}).get("hard_p_recipe_hit_pct")),
        ("v1-excluded slice size", 8761,
         rvs.get("v1_excluded", {}).get("n_queries")),
        ("% of ctrl-llm rows whose anchor is benchmark-query text", 31.7,
         row_variants["ctrl-llm-s42"]
         ["exact_or_prefix_only__missed_exclusions"]["pct"]),
        ("count of such ctrl-llm rows", 1950,
         row_variants["ctrl-llm-s42"]
         ["exact_or_prefix_only__missed_exclusions"]["rows"]),
    ]
    # tolerance: 0.5 pt for percentage-point quantities, 1% relative for
    # raw counts (slice sizes / row counts)
    cross_check = []
    for q, r, o in reviewer:
        is_count = r > 100
        tol = max(1.0, 0.01 * r) if is_count else 0.5
        cross_check.append({
            "quantity": q, "reviewer": r, "ours": o,
            "unit": "count" if is_count else "percentage_points",
            "tolerance": tol,
            "abs_diff": None if o is None else round(abs(o - r), 3),
            "within_tolerance": None if o is None else abs(o - r) <= tol})

    out = {
        "analysis": "clean-slice contamination analysis (review finding C1)",
        "generated": "2026-07-30",
        "inputs": {"queries": QUERIES, "v1_mapping": V1_MAP,
                   "v2_mapping": V2_MAP, "ranks_dir": RANKS,
                   "hard_hits": HITS,
                   "hard_hit_definition": ("ctrl-llm gold_rank==0 AND ctrl-cas "
                                           "gold_rank!=0 on the hard tier, "
                                           "seed 42 (analyze_hits.py); 1,401 "
                                           "of 15,000 queries")},
        "slice_definitions": {
            "v1_excluded": ("anchor matched by the v1 exact-text mapping, so "
                            "its corpus id was dropped by the always-on "
                            "training gate -- the paper's 'guaranteed clean' "
                            "set (8,761 queries)"),
            "v1_excluded_strict": ("v1_excluded and no v2 twin of the anchor "
                                   "appears in either arm's rows"),
            "v1_excluded_leaked_dupe": ("v1_excluded but a corpus-internal "
                                        "duplicate/twin WAS trained (v1 kept "
                                        "only the first id per text)"),
            "leaked_llm / leaked_cas": ("not v1-matched, has a v2 twin, and at "
                                        "least one twin id is in that arm's "
                                        "rows -> the arm trained on this "
                                        "query's own text"),
            "notleaked_llm / notleaked_cas": ("not v1-matched, has v2 twins, "
                                              "none in that arm's rows -- the "
                                              "matched comparison population"),
            "leaked_either": "not v1-matched and leaked into either arm",
            "untrained_twin": ("not v1-matched, has v2 twins, none trained in "
                               "either arm"),
            "no_corpus_twin": ("no corpus twin under any v2 rule (79 "
                               "anchors)"),
            "strict_clean_all": ("no v2 twin of the anchor is in either arm's "
                                 "rows -- the strictest clean slice"),
        },
        "scopes": {
            "trained": ("leakage judged against source_ids the weights saw "
                        "(5% dev split removed)"),
            "used": ("leakage judged against every gated row of the arm's "
                     "data file -- the review's definition"),
        },
        "training_replay_validation": validations,
        "contamination_of_training_rows": row_frac,
        "contamination_of_training_source_ids": arm_rowstats,
        "document_side_leakage_audit": doc_audit,
        "v2_gate_impact_if_regenerated": gate_impact,
        "row_contamination_rule_sensitivity": {
            "variants": {label: {"rules": rules,
                                 "non_v1_anchors_only": nonv1}
                         for label, rules, nonv1 in VARIANTS},
            "per_run": row_variants,
        },
        "reviewer_cross_check": cross_check,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nsaved {args.out}")
    print(json.dumps({"reviewer_cross_check": cross_check,
                      "used_scope_summary": results["used"]["cross_seed_summary"],
                      "rows_llm_s42": row_frac.get("ctrl-llm-s42"),
                      "rows_cas_s42": row_frac.get("ctrl-cas-s42")}, indent=2))


if __name__ == "__main__":
    main()
