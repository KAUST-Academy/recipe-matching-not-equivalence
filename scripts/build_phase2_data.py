#!/usr/bin/env python3
"""
Build the phase-2 MIXED training set for InvarEmbed.

WHY: phase-1 (models/qwen3-0.6b-cas-cmnrl) trained ONLY on difficulty
extremes -- CAS minimal-pair hard negatives + random in-batch negatives.
R@1 doubled on every tier but recall depth collapsed (medium R@5
60.91 -> 40.80). This script builds a breadth-preserving mix.

OUTPUT: data/phase2/mixed_pairs.jsonl in the SAME grouped schema
scripts/train_invarembed.py consumes:
    {"source_id": ..., "positive_text": ..., "negatives": [{"text":...},...],
     "anchor_text": <optional override>, "meta": {"channel": ...}}
plus data/phase2/build_stats.json (channel counts, drops, histograms).

FOUR CHANNELS (row meta.channel tags enable ablation):
  cas             CAS rows from data/cas_pairs/pairs.jsonl as-is, but
                  minimal-pair negatives CAPPED at 2 per row (extras dropped
                  at random, --seed) so minimal pairs stop dominating.
  cas_midneg      ~--midneg-rows of the CAS rows additionally get 1
                  same-topic different-problem negative (matched on the
                  2-level prefix of the source's first topics_flat path,
                  e.g. "Geometry > Plane Geometry") -- difficulty between
                  minimal-edit and random in-batch.
  replay          (problem, its own first official solution) pairs with NO
                  explicit negatives -- pure in-batch diversity. First
                  solution only, kept iff len >= --min-solution-chars.
                  Capped at --replay-cap rows, proportionally stratified by
                  top-level domain (largest-remainder allocation).
  cross_transform (positive_i, positive_j) for CAS sources with >= 2
                  verified positives, 1 pair per source_id. positive_i is
                  carried in "anchor_text" (the trainer resolves anchors
                  from the corpus parquet unless anchor_text is present --
                  see the matching train_invarembed.py patch).

CONTAMINATION: rows whose source_id is in
anchor_to_corpus_mapping.json["exclude_corpus_ids"] OR
data/crosslingual_eval/leakage_exclude_ids.json["eval_member_ids"] are
excluded AT BUILD TIME from every channel (the trainer gates again -- belt
and braces). Mid-difficulty negative POOLS also exclude both lists, so no
eval text appears anywhere in training rows.

USAGE (login node, ~1 min):
    python scripts/build_phase2_data.py            # defaults below
Then validate end-to-end (CPU, ~2 min):
    python scripts/train_invarembed.py --train-file data/phase2/mixed_pairs.jsonl \
        --max-rows 200 --model sentence-transformers/all-MiniLM-L6-v2 \
        --loss mnrl --epochs 0.02 --output-dir <scratch>
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAS_PAIRS = os.path.join(PROJECT_ROOT, "data", "cas_pairs", "pairs.jsonl")
CORPUS_PARQUET = os.path.join(PROJECT_ROOT, "data", "mathnet_corpus.parquet")
MAPPING_JSON = os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json")
XLING_LEAKAGE_JSON = os.path.join(PROJECT_ROOT, "data", "crosslingual_eval",
                                  "leakage_exclude_ids.json")
OUT_DEFAULT = os.path.join(PROJECT_ROOT, "data", "phase2", "mixed_pairs.jsonl")


def load_exclusions():
    with open(MAPPING_JSON, encoding="utf-8") as f:
        anchor = set(json.load(f)["exclude_corpus_ids"])
    with open(XLING_LEAKAGE_JSON, encoding="utf-8") as f:
        xling = set(json.load(f)["eval_member_ids"])
    return anchor, xling


def load_corpus():
    """id -> (problem_markdown, first_solution, topics_flat list)."""
    import duckdb
    rows = duckdb.sql(
        "SELECT id, problem_markdown, solutions_markdown, topics_flat "
        f"FROM '{CORPUS_PARQUET}'").fetchall()
    out = {}
    for cid, prob, sols, topics in rows:
        first_sol = sols[0] if sols else None
        out[cid] = (prob, first_sol, list(topics) if topics else [])
    return out


def topic_prefix(topics, levels=2):
    """2-level prefix of the FIRST topics_flat path ('A > B'), or None."""
    if not topics:
        return None
    segs = [s.strip() for s in topics[0].split(">") if s.strip()]
    if not segs:
        return None
    return " > ".join(segs[:levels])


def top_domain(topics):
    if not topics:
        return "Unknown"
    seg = topics[0].split(">")[0].strip()
    return seg or "Unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cas-file", default=CAS_PAIRS)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cas-negatives", type=int, default=2,
                    help="cap on minimal-pair negatives per CAS row")
    ap.add_argument("--midneg-rows", type=int, default=5000,
                    help="number of CAS rows that get 1 mid-difficulty "
                         "same-topic negative")
    ap.add_argument("--replay-cap", type=int, default=12000,
                    help="max replay (problem -> own solution) rows")
    ap.add_argument("--min-solution-chars", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    excl_anchor, excl_xling = load_exclusions()
    excluded = excl_anchor | excl_xling
    corpus = load_corpus()
    print(f"[build] corpus {len(corpus)} | exclusions: {len(excl_anchor)} "
          f"anchor-matched + {len(excl_xling)} xling-eval "
          f"(union {len(excluded)})", flush=True)

    stats = {"seed": args.seed,
             "exclusion_union_size": len(excluded),
             "params": {k: v for k, v in vars(args).items()}}

    # ------------------------------------------------------------------ CAS
    cas_rows, drop_excl, drop_missing = [], 0, 0
    negs_dropped_by_cap = 0
    by_source = defaultdict(list)   # source_id -> positive texts (for c)
    with open(args.cas_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r["source_id"]
            if sid in excluded:
                drop_excl += 1
                continue
            if sid not in corpus or not (corpus[sid][0] or "").strip():
                drop_missing += 1
                continue
            negs = []
            for n in r.get("negatives") or []:
                t = (n.get("text") if isinstance(n, dict) else n) or ""
                if t.strip():
                    negs.append({"text": t.strip(), "kind": "cas_minimal",
                                 "edit": n.get("edit") if isinstance(n, dict)
                                 else None})
            if len(negs) > args.max_cas_negatives:
                keep_idx = sorted(rng.sample(range(len(negs)),
                                             args.max_cas_negatives))
                negs_dropped_by_cap += len(negs) - args.max_cas_negatives
                negs = [negs[i] for i in keep_idx]
            pos = (r.get("positive_text") or "").strip()
            if not pos:
                drop_missing += 1
                continue
            cas_rows.append({
                "source_id": sid,
                "positive_text": pos,
                "negatives": negs,
                "meta": {"channel": "cas",
                         "transform_family": r.get("transform_family"),
                         "positive_transforms": r.get("positive_transforms")},
            })
            by_source[sid].append(pos)
    print(f"[cas] kept {len(cas_rows)} rows "
          f"(dropped {drop_excl} excluded + {drop_missing} missing/empty; "
          f"capped away {negs_dropped_by_cap} extra minimal negatives)",
          flush=True)
    stats["cas"] = {"rows": len(cas_rows),
                    "dropped_excluded": drop_excl,
                    "dropped_missing_or_empty": drop_missing,
                    "minimal_negatives_dropped_by_cap": negs_dropped_by_cap}

    # -------------------------------------------- mid-difficulty negatives
    # pool: prefix -> [corpus ids] usable as same-topic negatives
    pool = defaultdict(list)
    for cid, (prob, _sol, topics) in corpus.items():
        if cid in excluded or not (prob or "").strip():
            continue
        p = topic_prefix(topics)
        if p:
            pool[p].append(cid)
    for p in pool:
        pool[p].sort()          # determinism (dict order is insertion order)

    eligible_idx = []
    for i, row in enumerate(cas_rows):
        p = topic_prefix(corpus[row["source_id"]][2])
        if p and len(pool.get(p, [])) >= 2:   # someone besides itself
            eligible_idx.append(i)
    n_mid = min(args.midneg_rows, len(eligible_idx))
    chosen = set(rng.sample(eligible_idx, n_mid))
    n_attached = 0
    for i in sorted(chosen):
        row = cas_rows[i]
        sid = row["source_id"]
        prob_text = corpus[sid][0]
        p = topic_prefix(corpus[sid][2])
        cands = pool[p]
        for _attempt in range(10):
            nid = rng.choice(cands)
            if nid != sid and corpus[nid][0] != prob_text:
                row["negatives"].append({
                    "text": corpus[nid][0].strip(),
                    "kind": "mid_topic",
                    "negative_source_id": nid,
                    "topic_prefix": p})
                row["meta"]["channel"] = "cas_midneg"
                n_attached += 1
                break
    print(f"[midneg] attached same-topic negatives to {n_attached} CAS rows "
          f"({len(eligible_idx)} eligible, requested {args.midneg_rows})",
          flush=True)
    stats["cas_midneg"] = {"rows": n_attached,
                           "eligible_rows": len(eligible_idx)}

    # ---------------------------------------------------------------- replay
    elig_by_domain = defaultdict(list)
    for cid, (prob, sol, topics) in corpus.items():
        if cid in excluded:
            continue
        if not (prob or "").strip():
            continue
        if not sol or len(sol) < args.min_solution_chars:
            continue
        elig_by_domain[top_domain(topics)].append(cid)
    total_elig = sum(len(v) for v in elig_by_domain.values())
    cap = min(args.replay_cap, total_elig)
    # largest-remainder proportional allocation per domain
    shares = {d: cap * len(v) / total_elig for d, v in elig_by_domain.items()}
    alloc = {d: int(s) for d, s in shares.items()}
    for d, _ in sorted(shares.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)[:cap - sum(alloc.values())]:
        alloc[d] += 1
    replay_rows = []
    for d in sorted(elig_by_domain):
        ids = sorted(elig_by_domain[d])
        take = rng.sample(ids, alloc[d]) if alloc[d] < len(ids) else ids
        for cid in take:
            replay_rows.append({
                "source_id": cid,
                "positive_text": corpus[cid][1].strip(),
                "negatives": [],
                "meta": {"channel": "replay", "domain": d}})
    print(f"[replay] {len(replay_rows)} rows from {total_elig} eligible "
          f"(alloc: { {d: alloc[d] for d in sorted(alloc)} })", flush=True)
    stats["replay"] = {"rows": len(replay_rows), "eligible": total_elig,
                       "allocation_by_domain": alloc}

    # ------------------------------------------------------- cross-transform
    xform_rows, skipped_dup = [], 0
    for sid in sorted(by_source):
        poss = by_source[sid]
        if len(poss) < 2:
            continue
        pi, pj = rng.sample(poss, 2)
        if pi == pj:
            skipped_dup += 1
            continue
        xform_rows.append({
            "source_id": sid,
            "anchor_text": pi,
            "positive_text": pj,
            "negatives": [],
            "meta": {"channel": "cross_transform"}})
    print(f"[xform] {len(xform_rows)} (positive_i, positive_j) rows "
          f"(1 per source with >=2 positives; {skipped_dup} identical-text "
          f"pairs skipped)", flush=True)
    stats["cross_transform"] = {"rows": len(xform_rows),
                                "skipped_identical": skipped_dup}

    # ----------------------------------------------------------------- write
    all_rows = cas_rows + replay_rows + xform_rows
    rng.shuffle(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)

    channel_counts = Counter(r["meta"]["channel"] for r in all_rows)
    neg_hist = Counter(len(r["negatives"]) for r in all_rows)
    stats["channel_counts"] = dict(channel_counts)
    stats["total_rows"] = len(all_rows)
    stats["negatives_per_row_histogram"] = {
        str(k): v for k, v in sorted(neg_hist.items())}
    stats["output"] = args.out
    with open(os.path.join(os.path.dirname(args.out),
                           "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n[done] channel counts:")
    for ch in ("cas", "cas_midneg", "replay", "cross_transform"):
        print(f"    {ch:16s} {channel_counts.get(ch, 0):6d}")
    print(f"    {'TOTAL':16s} {len(all_rows):6d}")
    print(f"[done] negatives/row histogram: "
          f"{dict(sorted(neg_hist.items()))}")
    print(f"[done] wrote {args.out} + build_stats.json", flush=True)

    # ------------------------------------------- loader-compat sanity check
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_invarembed import load_pairs      # the trainer's own parser
    parsed = load_pairs(args.out)
    assert len(parsed) == len(all_rows), \
        f"trainer loader parsed {len(parsed)} != {len(all_rows)} written"
    n_anchor_override = sum(1 for r in parsed if r.get("anchor_text"))
    assert n_anchor_override == len(xform_rows), \
        (f"anchor_text rows seen by trainer loader {n_anchor_override} != "
         f"{len(xform_rows)} cross_transform rows")
    print(f"[check] trainer load_pairs(): {len(parsed)} rows OK "
          f"({n_anchor_override} with anchor_text override)", flush=True)


if __name__ == "__main__":
    main()
