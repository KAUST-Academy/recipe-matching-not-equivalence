#!/usr/bin/env python3
"""Recompute a tier's Figure-6 near-miss separation from CACHED embeddings (CPU).

Why this exists: `figure6_separation` blocks written by eval_retrieve.py before
2026-07-30 paired each query's positive similarity with a *later* query's
near-miss similarities whenever a query had a gold document but no near-miss
documents (7 such queries on the full 15,000-query tiers -- see the module
docstring of scripts/eval_retrieve.py). This script recomputes the block from
the `--emb-cache-dir` npz files of an already-finished eval, in both the buggy
(misaligned) and the fixed (per-query aligned) accounting, so a published
credential can be corrected without spending GPU time.

It only needs 4 dot products per query (gold + up to 3 near-misses), so a full
tier costs seconds once the npz files are loaded.

Usage (the paper's harness credential, Qwen3-Embedding-4B on the hard tier):
  python scripts/recompute_separation.py --tier hard \
      --docs-npz  .emb_cache/docs_Qwen__Qwen3-Embedding-4B_169406da1e116196.npz \
      --queries-npz .emb_cache/queries_Qwen__Qwen3-Embedding-4B_5f58052584c6753a.npz \
      --compare-json results/eval_hard_qwen3-embedding-4b.json
  -> buggy pct_pos_above_all_hard_negs 0.09 (13/14,993; reproduces the released
     JSON), fixed 0.07 (11/14,993).

The npz cache keys are md5(model|dtype|pooling|append_eos|max_seq_length|
prompt|md5(data_file))[:16]; `--list-cache` prints the candidates so the right
pair can be picked (docs are encoded without a query prompt, queries with one).
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ids(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["_id"] for line in f]


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels[qid].add(cid)
    return dict(qrels)


def block(pos, nmean, nmax, truncate):
    """`truncate=True` reproduces the pre-fix `n_sep = min(len(...))` pairing."""
    n = min(len(pos), len(nmean)) if truncate else len(pos)
    p = np.array(pos[:n]); m = np.array(nmean[:n]); x = np.array(nmax[:n])
    above = int((p > x).sum())
    return {
        "n_queries_with_pos_and_neg": n,
        "mean_positive_sim": round(float(p.mean()), 4),
        "mean_hard_negative_sim": round(float(m.mean()), 4),
        "mean_gap_pos_minus_meanneg": round(float((p - m).mean()), 4),
        "mean_gap_pos_minus_maxneg": round(float((p - x).mean()), 4),
        "pct_pos_above_all_hard_negs": round(100 * above / n, 2),
        "n_pos_above_all_hard_negs": above,
        "std_positive_sim": round(float(p.std()), 4),
        "std_hard_negative_sim": round(float(m.std()), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], default="hard")
    ap.add_argument("--data-dir", default=None,
                    help="BEIR-style dir (default: data/retrieve/<tier>)")
    ap.add_argument("--docs-npz", default=None)
    ap.add_argument("--queries-npz", default=None)
    ap.add_argument("--compare-json", default=None,
                    help="released eval JSON whose figure6_separation block "
                         "should be reproduced by the buggy accounting")
    ap.add_argument("--output", default=None, help="write the two blocks as JSON")
    ap.add_argument("--list-cache", action="store_true",
                    help="list .emb_cache candidates and exit")
    args = ap.parse_args()

    if args.list_cache:
        for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, ".emb_cache", "*.npz"))):
            print(os.path.basename(p))
        return 0
    if not (args.docs_npz and args.queries_npz):
        ap.error("--docs-npz and --queries-npz are required (see --list-cache)")

    tier_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", "retrieve", args.tier)
    corpus_ids = load_ids(os.path.join(tier_dir, "corpus.jsonl"))
    qrels = load_qrels(os.path.join(tier_dir, "qrels", "test.tsv"))
    query_ids = [q for q in load_ids(os.path.join(tier_dir, "queries.jsonl")) if q in qrels]
    corpus_pos = {c: i for i, c in enumerate(corpus_ids)}
    print(f"[data] tier={args.tier} corpus={len(corpus_ids)} queries={len(query_ids)}",
          flush=True)

    zq = np.load(args.queries_npz, allow_pickle=False)
    if list(zq["ids"]) != query_ids:
        raise SystemExit("query-embedding cache does not match this tier's query ids")
    qemb = zq["emb"].astype(np.float32); del zq
    zd = np.load(args.docs_npz, allow_pickle=False)
    if list(zd["ids"]) != corpus_ids:
        raise SystemExit("doc-embedding cache does not match this tier's corpus ids")
    demb = zd["emb"].astype(np.float32); del zd
    print(f"[cache] queries={qemb.shape} docs={demb.shape}", flush=True)

    # aligned records (fixed) and the pre-fix independent-append lists (buggy)
    a_pos, a_mean, a_max = [], [], []
    b_pos, b_mean, b_max = [], [], []
    no_nm, no_gold = [], []
    for i, qid in enumerate(query_ids):
        gold_id = next(iter(qrels[qid]))
        if gold_id not in corpus_pos:
            no_gold.append(qid)
            continue
        q = qemb[i]
        p = float(q @ demb[corpus_pos[gold_id]])
        negs = [float(q @ demb[corpus_pos[f"{qid}::nm::{j}"]])
                for j in range(3) if f"{qid}::nm::{j}" in corpus_pos]
        b_pos.append(p)                      # pre-fix: positive appended always
        if negs:
            b_mean.append(float(np.mean(negs))); b_max.append(float(np.max(negs)))
            a_pos.append(p)
            a_mean.append(float(np.mean(negs))); a_max.append(float(np.max(negs)))
        else:
            no_nm.append(qid)

    buggy = block(b_pos, b_mean, b_max, truncate=True)
    fixed = block(a_pos, a_mean, a_max, truncate=False)
    out = {
        "tier": args.tier,
        "docs_npz": os.path.basename(args.docs_npz),
        "queries_npz": os.path.basename(args.queries_npz),
        "n_queries_gold_but_no_near_miss": len(no_nm),
        "qids_gold_but_no_near_miss": no_nm,
        "n_queries_gold_missing_from_corpus": len(no_gold),
        "buggy_misaligned": buggy,
        "fixed_aligned": fixed,
    }
    if args.compare_json:
        released = json.load(open(args.compare_json, encoding="utf-8"))["figure6_separation"]
        out["released_json"] = released
        out["released_vs_buggy_max_abs_diff"] = max(
            abs(released[k] - buggy[k]) for k in released
            if k in buggy and isinstance(released[k], (int, float)))
    print(json.dumps(out, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
