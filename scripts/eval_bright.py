#!/usr/bin/env python3
"""
BRIGHT external-OOD evaluation harness (Recipe-Matching follow-up project).

Evaluates an embedding model on one split of xlangai/BRIGHT (arXiv:2407.12883,
CC-BY-4.0) following the OFFICIAL protocol of github.com/xlang-ai/BRIGHT
run.py + retrievers.py (fetched 2026-07-29):
  * queries  = config 'examples',  split <task>   (query, id, excluded_ids, gold_ids)
  * corpus   = config 'documents', split <task>   (id, content)
  * cosine similarity over L2-normalized embeddings (their sbert path);
    instruction (if any) concatenated to the QUERY side only (their
    add_instruct_concatenate) -- pass it via --query-prompt;
  * per query: drop excluded_ids from the score dict, keep top 1000;
  * metrics via pytrec_eval exactly as their calculate_retrieval_metrics
    (k = 1,5,10,25,50,100; headline = NDCG@10) -- if pytrec_eval is not
    installed, a local reimplementation of the same trec_eval measures is
    used (validated against pytrec_eval to 1e-9 on random data + real smoke
    scores; ties broken like trec_eval: score desc, then doc-id desc);
    the output JSON records which implementation produced the numbers.

Math-relevant splits for this project: theoremqa_theorems (76 queries /
23,839 docs) and aops (111 queries / 188,002 docs).

BRIGHT's official instruction strings for E5-style instruct models
(configs/e5/<task>.json), for use with --query-prompt on Qwen3-Embedding
family models:
  aops:               'Instruct: Given a Math problem, retrieve relevant
                       examples that help answer the problem\\nQuery: '
  theoremqa_theorems: 'Instruct: Given a Math problem, retrieve relevant
                       theorems that help answer the problem\\nQuery: '
(their sbert / all-mpnet-base-v2 run uses NO instruction).

Documented deviations from the official run (recorded in the output JSON):
  * --max-seq-length truncates both sides (official used per-model
    doc_max_length up to 8192 for 7B instruct models; our campaign standard
    is 1024 -- numbers are internally consistent across our models but not
    byte-comparable to the official leaderboard for long documents);
  * only sentence-transformers-loadable checkpoints are supported (no
    manual pooling stack -- every model in eval_external_ood.slurm loads
    natively); --max-queries/--max-corpus subsample for CPU smokes and mark
    the output comparable_to_official=false.

Usage (CPU smoke, subsampled):
  python scripts/eval_bright.py --split theoremqa_theorems \
      --model sentence-transformers/all-MiniLM-L6-v2 --device cpu \
      --max-corpus 2000 --output results/bright_smoke.json

Usage (GPU, see eval_external_ood.slurm):
  python scripts/eval_bright.py --split aops --model models/ctrl-llm-6145/final \
      --device cuda --model-dtype bfloat16 --query-prompt-name query \
      --max-seq-length 1024 --output results/bright_aops_ctrl-llm-6145.json
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_VALUES = [1, 5, 10, 25, 50, 100]
TASKS = ["biology", "earth_science", "economics", "psychology", "robotics",
         "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
         "theoremqa_theorems", "theoremqa_questions"]


# --------------------------------------------------------------------------
# Metrics: exact reimplementation of BRIGHT's calculate_retrieval_metrics
# (retrievers.py L544), which itself follows BEIR/trec_eval. Uses pytrec_eval
# when available; otherwise a validated local fallback with identical
# semantics (linear gains, log2(rank+1) discount, trec_eval tie-breaking:
# score desc then doc-id desc; MRR over the full submitted ranking).
# --------------------------------------------------------------------------
def _local_trec_metrics(results, qrels, k_values):
    per_query = {}
    for qid, doc_scores in results.items():
        rel = {d: r for d, r in qrels.get(qid, {}).items() if r > 0}
        n_rel = len(rel)
        # trec_eval tie-break: score descending, then doc-id DESCENDING
        # (stable two-pass sort: doc-id desc first, then score desc)
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[0], reverse=True)
        ranked = sorted(ranked, key=lambda kv: kv[1], reverse=True)
        rels_at = [rel.get(d, 0) for d, _ in ranked]
        m = {}
        # recip_rank (full ranking)
        m["recip_rank"] = 0.0
        for i, r in enumerate(rels_at):
            if r > 0:
                m["recip_rank"] = 1.0 / (i + 1)
                break
        ideal = sorted(rel.values(), reverse=True)
        for k in k_values:
            top = rels_at[:k]
            dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top) if r > 0)
            idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
            m[f"ndcg_cut_{k}"] = dcg / idcg if idcg > 0 else 0.0
            n_rel_ret = sum(1 for r in top if r > 0)
            m[f"recall_{k}"] = n_rel_ret / n_rel if n_rel else 0.0
            m[f"P_{k}"] = n_rel_ret / k
            ap, seen = 0.0, 0
            for i, r in enumerate(top):
                if r > 0:
                    seen += 1
                    ap += seen / (i + 1)
            m[f"map_cut_{k}"] = ap / n_rel if n_rel else 0.0
        per_query[qid] = m
    return per_query


def calculate_retrieval_metrics(results, qrels, k_values=K_VALUES, per_query_sink=None):
    try:
        import pytrec_eval
        map_string = "map_cut." + ",".join(str(k) for k in k_values)
        ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
        recall_string = "recall." + ",".join(str(k) for k in k_values)
        precision_string = "P." + ",".join(str(k) for k in k_values)
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string,
                    "recip_rank"})
        scores = evaluator.evaluate(results)
        impl = "pytrec_eval"
    except ImportError:
        scores = _local_trec_metrics(results, qrels, k_values)
        impl = "local_fallback (validated against pytrec_eval)"
    if per_query_sink is not None:
        per_query_sink.update(scores)
    agg = {"MRR": 0.0}
    for k in k_values:
        agg[f"NDCG@{k}"] = 0.0
        agg[f"MAP@{k}"] = 0.0
        agg[f"Recall@{k}"] = 0.0
        agg[f"P@{k}"] = 0.0
    for qid in scores:
        for k in k_values:
            agg[f"NDCG@{k}"] += scores[qid][f"ndcg_cut_{k}"]
            agg[f"MAP@{k}"] += scores[qid][f"map_cut_{k}"]
            agg[f"Recall@{k}"] += scores[qid][f"recall_{k}"]
            agg[f"P@{k}"] += scores[qid][f"P_{k}"]
        agg["MRR"] += scores[qid]["recip_rank"]
    n = len(scores)
    out = {key: round(v / n, 5) for key, v in agg.items()}
    out["n_queries_scored"] = n
    out["metrics_impl"] = impl
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", choices=TASKS, default="theoremqa_theorems",
                    help="BRIGHT task (math-relevant: theoremqa_theorems, aops)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--query-prompt-name", default=None,
                    help="sentence-transformers prompt name for queries")
    ap.add_argument("--query-prompt", default=None,
                    help="raw instruction prepended to every query (BRIGHT "
                         "add_instruct_concatenate convention)")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override tokenizer truncation length (0 = model default)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="subsample queries (0 = all; smoke only)")
    ap.add_argument("--max-corpus", type=int, default=0,
                    help="subsample corpus keeping all gold docs of kept "
                         "queries (0 = all). WARNING: not comparable to the "
                         "official protocol; smoke only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-query-out", default=None,
                    help="also write per-query trec metrics (ndcg_cut_k, recall_k, P_k, map_cut_k, recip_rank) as JSON")
    ap.add_argument("--output", default=None,
                    help="result JSON path (default results/bright_<split>_<model>.json)")
    args = ap.parse_args()
    if args.query_prompt and args.query_prompt_name:
        ap.error("--query-prompt and --query-prompt-name are mutually exclusive")

    t0 = time.time()

    # ---------------- data (official source: HF xlangai/BRIGHT) -----------
    from datasets import load_dataset
    examples = load_dataset("xlangai/BRIGHT", "examples", split=args.split)
    docs = load_dataset("xlangai/BRIGHT", "documents", split=args.split)
    doc_ids = [d["id"] for d in docs]
    documents = [d["content"] for d in docs]
    queries, query_ids, excluded_ids, gold_ids = [], [], {}, {}
    for e in examples:
        queries.append(e["query"])
        query_ids.append(e["id"])
        excluded_ids[e["id"]] = e["excluded_ids"]
        gold_ids[e["id"]] = e["gold_ids"]
        # official run.py asserts golds are never excluded
        assert not set(e["excluded_ids"]) & set(e["gold_ids"])
    print(f"[data] BRIGHT/{args.split}: {len(queries)} queries, "
          f"{len(documents)} documents", flush=True)

    import random
    rng = random.Random(args.seed)
    subsampled = False
    if args.max_queries and args.max_queries < len(query_ids):
        keep = set(rng.sample(range(len(query_ids)), args.max_queries))
        queries = [q for i, q in enumerate(queries) if i in keep]
        query_ids = [q for i, q in enumerate(query_ids) if i in keep]
        subsampled = True
        print(f"[sample] {len(query_ids)} queries kept (seed={args.seed})", flush=True)
    if args.max_corpus and args.max_corpus < len(doc_ids):
        required = set()
        for qid in query_ids:
            required |= set(gold_ids[qid])
        pool = [i for i, d in enumerate(doc_ids) if d not in required]
        keep_idx = set(rng.sample(pool, max(0, args.max_corpus - len(required))))
        keep_idx |= {i for i, d in enumerate(doc_ids) if d in required}
        doc_ids = [d for i, d in enumerate(doc_ids) if i in keep_idx]
        documents = [t for i, t in enumerate(documents) if i in keep_idx]
        subsampled = True
        print(f"[sample] WARNING corpus subsampled to {len(doc_ids)} docs -- "
              f"NOT comparable to the official protocol", flush=True)

    # ---------------- model ----------------
    from sentence_transformers import SentenceTransformer
    model_kwargs = {}
    if args.model_dtype and args.model_dtype != "float32":
        import torch
        model_kwargs["torch_dtype"] = getattr(torch, args.model_dtype)
    model = SentenceTransformer(args.model, device=args.device,
                                trust_remote_code=args.trust_remote_code,
                                model_kwargs=model_kwargs or None)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    print(f"[model] {args.model} device={model.device} "
          f"max_seq_length={model.max_seq_length}", flush=True)

    enc = dict(batch_size=args.batch_size, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=True)
    q_kwargs = dict(enc)
    if args.query_prompt_name:
        q_kwargs["prompt_name"] = args.query_prompt_name
    if args.query_prompt:
        q_kwargs["prompt"] = args.query_prompt

    t_enc = time.time()
    doc_emb = model.encode(documents, **enc).astype(np.float32)
    query_emb = model.encode(queries, **q_kwargs).astype(np.float32)
    print(f"[encode] {len(documents)} docs + {len(queries)} queries in "
          f"{time.time() - t_enc:.1f}s", flush=True)

    # ------------- scoring, official get_scores semantics ------------------
    # cosine scores; drop excluded ids; keep top 1000 per query
    results = {}
    for i, qid in enumerate(query_ids):
        sims = query_emb[i] @ doc_emb.T
        cur = dict(zip(doc_ids, sims.tolist()))
        for did in set(excluded_ids[qid]):
            if did != "N/A":
                cur.pop(did, None)
        top = sorted(cur.items(), key=lambda x: x[1], reverse=True)[:1000]
        results[str(qid)] = {d: float(s) for d, s in top}

    ground_truth = {}
    for qid in query_ids:
        ground_truth[str(qid)] = {gid: 1 for gid in gold_ids[qid]}
        for did in excluded_ids[qid]:
            assert did not in results[str(qid)]
            assert did not in ground_truth[str(qid)]
    # golds absent from a subsampled corpus would silently zero recall; guard
    kept = set(doc_ids)
    missing_golds = sum(1 for qid in query_ids
                        for g in gold_ids[qid] if g not in kept)

    per_query = {}
    metrics = calculate_retrieval_metrics(results=results, qrels=ground_truth,
                                         per_query_sink=per_query)
    if args.per_query_out:
        os.makedirs(os.path.dirname(args.per_query_out) or ".", exist_ok=True)
        with open(args.per_query_out, "w", encoding="utf-8") as f:
            json.dump({qid: {k: float(v) for k, v in m.items()} for qid, m in per_query.items()},
                      f, indent=1)
        print(f"[done] wrote per-query metrics for {len(per_query)} queries to {args.per_query_out}", flush=True)

    result = {
        "benchmark": f"xlangai/BRIGHT / {args.split} (external, arXiv:2407.12883)",
        "protocol": "official run.py + retrievers.py (github.com/xlang-ai/BRIGHT)",
        "model": args.model,
        "device": str(model.device),
        "n_corpus": len(doc_ids),
        "n_queries": len(query_ids),
        "subsampled": subsampled,
        "comparable_to_official": (not subsampled) and args.max_seq_length == 0,
        "missing_gold_docs": missing_golds,
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "model_dtype": args.model_dtype,
        "max_seq_length": int(model.max_seq_length) if model.max_seq_length else None,
        "seed": args.seed,
        "metrics": metrics,
        "headline": {"NDCG@10": metrics["NDCG@10"]},
        "protocol_notes": [
            "truncation at --max-seq-length applies to docs and queries "
            "(official used per-model doc_max_length up to 8192; ours is a "
            "documented deviation, consistent across our models)",
            "excluded_ids removed from each query's candidates before "
            "ranking; top-1000 kept; metrics averaged over all queries",
        ],
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "results",
        f"bright_{args.split}_{args.model.rstrip('/').split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
