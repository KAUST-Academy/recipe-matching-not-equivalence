#!/usr/bin/env python3
"""
Cross-lingual duplicate-retrieval evaluation harness (InvarEmbed project).

Evaluates an embedding model on the mined real-duplicate cross-lingual eval
set built by scripts/build_crosslingual_eval.py (data/crosslingual_eval/):
393 non-English queries against the FULL 27,817-problem public MathNet
corpus; golds are officially-reprinted duplicates of the query problem
(transitively clustered, so a query may have several golds).

Adapted from scripts/eval_retrieve.py (validated against MathNet v2 Table 4)
with the same cosine-similarity retrieval and Recall@k = hit-rate@k of the
best-ranked gold. Differences:
  * SELF-MASKING (required for correctness): every query problem also exists
    verbatim in corpus.jsonl under the same _id; that document's similarity
    is set to -inf for its own query, otherwise it would trivially rank #1.
  * strict cross-lingual metrics: rank of the best gold whose language
    differs from the query language (covers 100% of queries by construction).
  * breakdowns: per query language, per confidence, and by the
    overlaps_retrieve_anchor flag (queries/golds that also appear among
    MathNet-Retrieve anchor problems -- reported, not dropped).

Usage (smoke test, CPU, ~10 min on 8 threads):
  python scripts/eval_crosslingual.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --device cpu --output results/crosslingual_smoke_test.json

Usage (GPU -- see eval_crosslingual.slurm):
  python scripts/eval_crosslingual.py --model Qwen/Qwen3-Embedding-4B \
      --query-prompt-name query --model-dtype bfloat16 --batch-size 32 \
      --max-seq-length 1024

NOTE: --max-corpus subsamples the corpus (always keeping every gold and every
query doc); recall numbers obtained that way are NOT comparable to full-corpus
runs and are flagged in the output JSON.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "crosslingual_eval")
K_VALUES = (1, 5, 10)


def load_corpus(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"]
    return out


def load_queries(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def recall_at_k(rank_of_gold, k):
    """rank_of_gold: best (0-based) rank of any relevant doc per query."""
    return float(np.mean(np.asarray(rank_of_gold) < k))


def metric_block(ranks):
    return {f"recall@{k}": round(100 * recall_at_k(ranks, k), 2) for k in K_VALUES}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-corpus", type=int, default=0,
                    help="random subsample of corpus docs, always keeping gold "
                         "+ query docs (0 = all). WARNING: not comparable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-prompt-name", default=None,
                    help="sentence-transformers prompt name for queries "
                         "(e.g. 'query' for Qwen3-Embedding models)")
    ap.add_argument("--query-prompt", default=None,
                    help="raw prompt text prepended to every query "
                         "(mutually exclusive with --query-prompt-name)")
    ap.add_argument("--doc-prompt", default=None,
                    help="raw prompt text prepended to every corpus document "
                         "(e.g. 'passage: ' for multilingual-e5 models)")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override tokenizer truncation length (0 = model default)")
    ap.add_argument("--output", default=None,
                    help="result JSON path (default results/crosslingual_<model>.json)")
    ap.add_argument("--dump-ranks", default=None, metavar="PATH",
                    help="also write a per-query gold-rank JSONL alongside the "
                         "summary (one row per query: qid, cluster_id, lang, "
                         "min_confidence, overlaps_retrieve_anchor, gold_rank, "
                         "xling_rank, same_rank; ranks are EXACT and 0-based, "
                         "a value equal to n_docs means 'no gold ranked'). "
                         "Consumed by scripts/bootstrap_stats.py (hierarchical "
                         "cluster bootstrap). Flag name shared with eval_retrieve.py.")
    ap.add_argument("--eval-dir", default=EVAL_DIR,
                    help="BEIR-style eval directory (corpus.jsonl, queries.jsonl); "
                         "default data/crosslingual_eval. Added 2026-09-03 for the "
                         "same-language set data/samelang_eval (E-R5); the "
                         "cross-lingual path and its outputs are unchanged.")
    args = ap.parse_args()

    t0 = time.time()

    # ---------------- data ----------------
    eval_dir = args.eval_dir
    corpus_path = os.path.join(eval_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        # The same-language set shares the cross-lingual corpus and ships no
        # copy (its symlink broke Overleaf's GitHub sync, 2026-09-04).
        corpus_path = os.path.join(EVAL_DIR, "corpus.jsonl")
        print(f"[data] no corpus in {eval_dir}; using {corpus_path}", flush=True)
    corpus = load_corpus(corpus_path)
    queries = load_queries(os.path.join(eval_dir, "queries.jsonl"))
    print(f"[data] corpus={len(corpus)} queries={len(queries)}", flush=True)

    corpus_subsampled = False
    if args.max_corpus and args.max_corpus < len(corpus):
        import random
        rng = random.Random(args.seed)
        required = {q["_id"] for q in queries}
        for q in queries:
            required |= set(q["metadata"]["gold_ids"])
        pool = [c for c in corpus if c not in required]
        n_extra = max(0, args.max_corpus - len(required))
        keep = required | set(rng.sample(pool, min(n_extra, len(pool))))
        corpus = {c: t for c, t in corpus.items() if c in keep}
        corpus_subsampled = True
        print(f"[sample] WARNING corpus subsampled to {len(corpus)} docs -- "
              f"NOT comparable to full-corpus runs", flush=True)

    corpus_ids = list(corpus)
    corpus_pos = {c: i for i, c in enumerate(corpus_ids)}

    # ---------------- model ----------------
    from sentence_transformers import SentenceTransformer
    model_kwargs = {}
    if args.model_dtype and args.model_dtype != "float32":
        import torch
        model_kwargs["torch_dtype"] = getattr(torch, args.model_dtype)
    model = SentenceTransformer(args.model, device=args.device,
                                model_kwargs=model_kwargs or None)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    print(f"[model] {args.model} device={model.device} "
          f"max_seq_length={model.max_seq_length}", flush=True)

    enc = dict(batch_size=args.batch_size, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=True)
    t_enc = time.time()
    d_kwargs = dict(enc)
    if args.doc_prompt:
        d_kwargs["prompt"] = args.doc_prompt
    doc_emb = model.encode([corpus[c] for c in corpus_ids], **d_kwargs).astype(np.float32)
    q_kwargs = dict(enc)
    if args.query_prompt_name and args.query_prompt:
        raise SystemExit("--query-prompt-name and --query-prompt are mutually exclusive")
    if args.query_prompt_name:
        q_kwargs["prompt_name"] = args.query_prompt_name
    if args.query_prompt:
        q_kwargs["prompt"] = args.query_prompt
    query_emb = model.encode([q["text"] for q in queries], **q_kwargs).astype(np.float32)
    print(f"[encode] {len(corpus_ids)} docs + {len(queries)} queries in "
          f"{time.time() - t_enc:.1f}s", flush=True)

    # ---------------- retrieval: full similarity rows (only 393 queries) ----
    n_docs = len(corpus_ids)
    gold_rank = np.full(len(queries), n_docs, dtype=np.int64)      # any gold
    xling_rank = np.full(len(queries), n_docs, dtype=np.int64)     # gold w/ lang != query lang
    same_rank = np.full(len(queries), n_docs, dtype=np.int64)      # gold w/ lang == query lang
    chunk = 128
    for s in range(0, len(queries), chunk):
        sims = query_emb[s:s + chunk] @ doc_emb.T                  # (chunk, N)
        for i, q in enumerate(queries[s:s + chunk]):
            row = sims[i]
            # SELF-MASK: the query's own corpus document must not be retrievable
            if q["_id"] in corpus_pos:
                row[corpus_pos[q["_id"]]] = -np.inf
            meta = q["metadata"]
            golds = [g for g in meta["gold_ids"] if g in corpus_pos]
            if not golds:
                continue
            gold_sims = {g: row[corpus_pos[g]] for g in golds}
            best = max(gold_sims.values())
            gold_rank[s + i] = int((row > best).sum())             # 0-based rank
            xl = [gold_sims[g] for g, gl in zip(meta["gold_ids"], meta["gold_langs"])
                  if g in gold_sims and gl != meta["lang"]]
            if xl:
                xling_rank[s + i] = int((row > max(xl)).sum())
            sl = [gold_sims[g] for g, gl in zip(meta["gold_ids"], meta["gold_langs"])
                  if g in gold_sims and gl == meta["lang"]]
            if sl:
                same_rank[s + i] = int((row > max(sl)).sum())

    # ---------------- optional per-query rank dump (bootstrap machinery) ----
    if args.dump_ranks:
        dump_dir = os.path.dirname(os.path.abspath(args.dump_ranks))
        os.makedirs(dump_dir, exist_ok=True)
        with open(args.dump_ranks, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries):
                meta = q["metadata"]
                f.write(json.dumps({
                    "qid": q["_id"],
                    "cluster_id": meta["cluster_id"],
                    "lang": meta["lang"],
                    "min_confidence": meta["min_confidence"],
                    "overlaps_retrieve_anchor": bool(meta["overlaps_retrieve_anchor"]),
                    "gold_rank": int(gold_rank[i]),
                    "xling_rank": int(xling_rank[i]),
                    "same_rank": int(same_rank[i]),
                    "n_docs": n_docs,
                }) + "\n")
        print(f"[dump-ranks] wrote {len(queries)} rows to {args.dump_ranks}",
              flush=True)

    # ---------------- metrics ----------------
    overall = metric_block(gold_rank)
    strict = metric_block(xling_rank)
    same_language = metric_block(same_rank)

    def breakdown(key_fn):
        groups = defaultdict(list)
        for i, q in enumerate(queries):
            groups[key_fn(q)].append(i)
        return {k: {"n_queries": len(idx), **metric_block(gold_rank[idx])}
                for k, idx in sorted(groups.items(), key=lambda kv: -len(kv[1]))}

    per_language = breakdown(lambda q: q["metadata"]["lang"])
    per_confidence = breakdown(lambda q: q["metadata"]["min_confidence"])
    per_anchor_overlap = breakdown(
        lambda q: "overlaps_retrieve_anchor" if q["metadata"]["overlaps_retrieve_anchor"]
        else "no_overlap")

    result = {
        "benchmark": ("InvarEmbed cross-lingual duplicate retrieval "
                      "(data/crosslingual_eval, mined real duplicates)"
                      if os.path.abspath(eval_dir) == os.path.abspath(EVAL_DIR)
                      else f"InvarEmbed duplicate retrieval ({eval_dir})"),
        "eval_dir": eval_dir,
        "model": args.model,
        "device": str(model.device),
        "n_corpus": n_docs,
        "n_queries": len(queries),
        "corpus_subsampled": corpus_subsampled,
        "comparable_full_corpus": not corpus_subsampled,
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "doc_prompt": args.doc_prompt,
        "self_masking": True,
        "overall_any_gold": overall,
        "strict_crosslingual_gold": strict,
        "same_language_gold": same_language,
        "median_gold_rank": int(np.median(gold_rank)) + 1,  # 1-based
        "per_language": per_language,
        "per_confidence": per_confidence,
        "per_anchor_overlap": per_anchor_overlap,
        "dump_ranks": args.dump_ranks,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "results",
        f"crosslingual_{args.model.split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
