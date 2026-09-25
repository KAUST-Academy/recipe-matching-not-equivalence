#!/usr/bin/env python3
"""
MathNet-Retrieve evaluation harness (InvarEmbed project).

Evaluates an embedding model on one tier (easy / medium / hard) of the
ShadenA/MathNet-Retrieve benchmark with cosine-similarity retrieval.

Reports:
  * Recall@1/5/10 overall (one gold doc per query -> identical to hit-rate,
    which is what the MathNet paper reports).
  * Recall@1/5/10 per math domain (Algebra, Geometry, ...) for the subset of
    queries whose anchor text was exact-matched to the public MathNet corpus
    (anchor_to_corpus_mapping.json -> topics_flat in mathnet_corpus.parquet).
  * Figure-6-style separation metric: similarity of each query to its gold
    equivalent rephrasing (positive) vs. its near-miss distractors (hard
    negatives), plus the mean gap and the fraction of queries where the
    positive outranks every hard negative.

_id SCHEME of MathNet-Retrieve (inspected 2026-07-29, easy tier):
  * queries.jsonl  _id = "<anchor>"  e.g. "rus_2019_d99efd"
        anchor = <country/competition>_<year?>_<hash> ; the query text is the
        original competition problem.
  * corpus.jsonl   _id = "<anchor>::<kind>::<variant>" with kinds
        eq::easy | eq::medium | eq::hard   equivalent rephrasings of the
                                           anchor problem (the positives;
                                           the tier's qrels point at
                                           "<anchor>::eq::<tier>")
        nm::0 | nm::1 | nm::2              near-miss problems: superficially
                                           similar but NOT equivalent (the
                                           hard negatives for that anchor)
        orig                               a small number (1,668 on easy) of
                                           original texts of other problems
    Easy-tier corpus: 117,088 docs, 21,000 distinct anchors, 15,000 queries;
    every query id is an anchor that also has eq::* and nm::* docs.
  * qrels/test.tsv: header "query-id\tcorpus-id\tscore"; exactly one gold per
    query: <anchor> -> <anchor>::eq::<tier>, score 1.

Usage (smoke test, CPU):
  python eval_retrieve.py --tier easy --model sentence-transformers/all-MiniLM-L6-v2 \
      --max-queries 500 --output ../results/eval_smoke_test.json

Usage (full eval, GPU -- see eval_retrieve.slurm / eval_baselines.slurm):
  python eval_retrieve.py --tier easy --model Qwen/Qwen3-Embedding-4B \
      --query-prompt-name query --model-dtype bfloat16 --batch-size 32

Trained-baseline support (added 2026-07-29 for eval_baselines.slurm):
  --query-prompt / --doc-prompt      raw prompt strings prepended by
                                     sentence-transformers (e.g. RaDeR's
                                     "query: " / "document: ", ReasonIR's
                                     "<|user|>\\n...\\n<|embed|>\\n")
  --doc-prompt-name                  registered prompt name for documents
                                     (e.g. "document" for MathLeap models)
  --pooling lasttoken --append-eos   build a manual Transformer+Pooling stack
                                     for plain-transformers checkpoints with
                                     no sentence-transformers config (RaDeR:
                                     EOS-token pooling, "{text}<eos>" inputs)
  --trust-remote-code                required by ReasonIR-8B (custom
                                     bidirectional Llama) and the gte-Qwen2
                                     bidirectional attention of RaDeR-gte
  --emb-cache-dir DIR                cache full-set doc/query embeddings; the
                                     three MathNet-Retrieve tiers share
                                     byte-identical corpus.jsonl and
                                     queries.jsonl (verified by md5), so one
                                     encode serves all tiers (~3x speedup).
                                     Keys include model+prompt+dtype+seqlen.
  --data-dir DIR                     evaluate a local BEIR-style directory
                                     (corpus.jsonl / queries.jsonl /
                                     qrels/test.tsv) instead of downloading a
                                     MathNet-Retrieve tier -- used for
                                     data/crosslingual_eval.

SELF-MASKING (on by default since 2026-07-30; --no-self-mask opts out):
  any corpus document whose _id is identical to the query's _id has its
  similarity set to -inf for that query, so a corpus that also contains the
  query problem verbatim cannot be "retrieved" trivially. This is a strict
  no-op on all three MathNet-Retrieve tiers (verified: 0 of the 15,000 query
  ids occur among the 117,088 corpus ids, whose scheme is
  "<anchor>::eq|nm|orig::*"), and it is REQUIRED for local --data-dir sets
  such as data/crosslingual_eval, where every query problem also sits in the
  27,817-doc corpus under the same _id. Running that set without masking
  yields the self-hit signature R@1 ~0.25 / R@5 ~96 (the gold is pushed to
  rank 2 by the query's own document); the artifact produced that way before
  this flag existed is kept as
  results/eval_crosslingual_rader-qwen25-7b.json.INVALID.
  scripts/eval_crosslingual.py is still the preferred harness for that set
  (it also reports the strict cross-language metric and the cluster
  breakdowns).

NOTE: --max-corpus subsamples the corpus; recall numbers obtained that way are
NOT comparable to the paper (easier retrieval pool). --max-queries alone keeps
the full corpus and stays comparable up to query-sampling noise.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "retrieve")
HF_BASE = "https://huggingface.co/datasets/ShadenA/MathNet-Retrieve/resolve/main"
MAPPING_JSON = os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json")
MATHNET_PARQUET = os.path.join(PROJECT_ROOT, "data", "mathnet_corpus.parquet")
K_VALUES = (1, 5, 10)


def download_tier(tier: str) -> str:
    """Download (and cache) the raw jsonl/tsv files for one tier."""
    tier_dir = os.path.join(DATA_DIR, tier)
    os.makedirs(os.path.join(tier_dir, "qrels"), exist_ok=True)
    for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        dest = os.path.join(tier_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{HF_BASE}/{tier}/{rel}"
        print(f"[download] {url} -> {dest}", flush=True)
        urllib.request.urlretrieve(url, dest)
    return tier_dir


def load_jsonl(path: str) -> dict:
    """Return {_id: text} preserving file order."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"]
    return out


def load_qrels(path: str) -> dict:
    """Return {query_id: set(relevant corpus ids)}."""
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # "query-id\tcorpus-id\tscore"
        assert header.lower().startswith("query"), f"unexpected qrels header: {header!r}"
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels[qid].add(cid)
    return dict(qrels)


def load_domain_map() -> dict:
    """anchor_id -> top-level math domain ('Algebra', 'Geometry', ...).

    Uses the exact-text anchor->corpus mapping plus topics_flat from the
    public MathNet corpus parquet. Covers only exact-matched anchors
    (8,761 / 15,000); returns {} if either file is missing.
    """
    if not (os.path.exists(MAPPING_JSON) and os.path.exists(MATHNET_PARQUET)):
        return {}
    try:
        import duckdb
    except ImportError:
        return {}
    with open(MAPPING_JSON, encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]  # [{anchor_id, corpus_id}, ...]
    rows = duckdb.sql(
        f"SELECT id, topics_flat FROM '{MATHNET_PARQUET}'"
    ).fetchall()
    corpus_domain = {}
    for cid, topics in rows:
        if topics:  # primary domain = top level of the first topic path
            corpus_domain[cid] = topics[0].split(" > ")[0].strip()
    return {
        m["anchor_id"]: corpus_domain[m["corpus_id"]]
        for m in mapping
        if m["corpus_id"] in corpus_domain
    }


def recall_at_k(rank_of_gold: np.ndarray, k: int) -> float:
    """rank_of_gold: best (0-based) rank of any relevant doc per query."""
    return float(np.mean(rank_of_gold < k))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="random subsample of queries (0 = all)")
    ap.add_argument("--max-corpus", type=int, default=0,
                    help="random subsample of corpus docs, always keeping the "
                         "gold + near-miss docs of the sampled queries "
                         "(0 = all). WARNING: recall not comparable to paper.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-prompt-name", default=None,
                    help="sentence-transformers prompt name for queries "
                         "(e.g. 'query' for Qwen3-Embedding models)")
    ap.add_argument("--query-prompt", default=None,
                    help="raw prompt string prepended to every query "
                         "(mutually exclusive with --query-prompt-name)")
    ap.add_argument("--doc-prompt-name", default=None,
                    help="registered prompt name for corpus documents")
    ap.add_argument("--doc-prompt", default=None,
                    help="raw prompt string prepended to every document "
                         "(mutually exclusive with --doc-prompt-name)")
    ap.add_argument("--pooling", default=None,
                    choices=[None, "lasttoken", "mean", "cls"],
                    help="build a manual Transformer+Pooling stack instead of "
                         "the repo's sentence-transformers config (needed for "
                         "plain-transformers checkpoints such as RaDeR)")
    ap.add_argument("--append-eos", action="store_true",
                    help="append the tokenizer EOS token string to every text "
                         "(RaDeR convention: '{text}<eos>')")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="pass trust_remote_code=True (ReasonIR-8B, gte-Qwen2)")
    ap.add_argument("--emb-cache-dir", default=None,
                    help="directory for cached full-set embeddings, reused "
                         "across tiers (corpus/queries are tier-identical)")
    ap.add_argument("--data-dir", default=None,
                    help="local BEIR-style dir (corpus.jsonl, queries.jsonl, "
                         "qrels/test.tsv); overrides --tier download")
    ap.add_argument("--no-self-mask", action="store_true",
                    help="DISABLE self-masking (default: on). With masking on, "
                         "a corpus doc whose _id equals the query _id is set "
                         "to -inf for that query. No-op on the MathNet tiers; "
                         "required for --data-dir sets whose corpus contains "
                         "the query problems themselves (data/crosslingual_eval). "
                         "Opting out reproduces the pre-2026-07-30 behaviour "
                         "and is recorded as self_masking=false in the output.")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"],
                    help="torch dtype for model weights (GPU runs)")
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override tokenizer truncation length (0 = model default)")
    ap.add_argument("--output", default=None,
                    help="path of the result JSON (default results/eval_<tier>_<model>.json)")
    ap.add_argument("--dump-ranks", default=None, metavar="PATH",
                    help="also write per-query ranks as JSONL: {qid, gold_rank "
                         "(EXACT 0-based rank of the best gold, ties broken "
                         "pessimistically = #docs strictly above), gold_sim, "
                         "top10_ids}. Input for the recipe-sensitivity audit "
                         "(scripts/analyze_hits.py). Backward compatible: "
                         "omitting the flag changes nothing.")
    args = ap.parse_args()

    if args.query_prompt and args.query_prompt_name:
        ap.error("--query-prompt and --query-prompt-name are mutually exclusive")
    if args.doc_prompt and args.doc_prompt_name:
        ap.error("--doc-prompt and --doc-prompt-name are mutually exclusive")

    rng = random.Random(args.seed)
    t0 = time.time()

    # ---------------- data ----------------
    if args.data_dir:
        tier_dir = args.data_dir
        tier_label = os.path.basename(os.path.normpath(args.data_dir))
    else:
        tier_dir = download_tier(args.tier)
        tier_label = args.tier
    corpus = load_jsonl(os.path.join(tier_dir, "corpus.jsonl"))
    queries = load_jsonl(os.path.join(tier_dir, "queries.jsonl"))
    qrels = load_qrels(os.path.join(tier_dir, "qrels", "test.tsv"))
    print(f"[data] tier={tier_label} corpus={len(corpus)} queries={len(queries)} "
          f"qrels={len(qrels)}", flush=True)

    query_ids = [q for q in queries if q in qrels]
    if args.max_queries and args.max_queries < len(query_ids):
        query_ids = rng.sample(query_ids, args.max_queries)
        print(f"[sample] subsampled to {len(query_ids)} queries (seed={args.seed})",
              flush=True)

    corpus_subsampled = False
    if args.max_corpus and args.max_corpus < len(corpus):
        # keep every doc needed for metrics on the sampled queries ...
        required = set()
        for qid in query_ids:
            required |= qrels[qid]
            required |= {f"{qid}::nm::{i}" for i in range(3) if f"{qid}::nm::{i}" in corpus}
        pool = [c for c in corpus if c not in required]
        n_extra = max(0, args.max_corpus - len(required))
        keep = required | set(rng.sample(pool, min(n_extra, len(pool))))
        corpus = {c: t for c, t in corpus.items() if c in keep}
        corpus_subsampled = True
        print(f"[sample] WARNING corpus subsampled to {len(corpus)} docs -- "
              f"recall is NOT comparable to the paper", flush=True)

    corpus_ids = list(corpus)
    corpus_pos = {c: i for i, c in enumerate(corpus_ids)}

    # ---------------- model ----------------
    from sentence_transformers import SentenceTransformer
    model_kwargs = {}
    if args.model_dtype and args.model_dtype != "float32":
        import torch
        model_kwargs["torch_dtype"] = getattr(torch, args.model_dtype)
    if args.pooling:
        # Manual stack for plain-transformers checkpoints (e.g. RaDeR repos,
        # which ship no sentence-transformers config; their released eval code
        # uses last-hidden-state-of-last-token pooling + L2 norm).
        # chat_template=None is REQUIRED: sentence-transformers 5.6 otherwise
        # wraps plain strings in the tokenizer's chat template for decoder
        # checkpoints (verified 2026-07-29: Qwen2.5 inputs silently became
        # "<|im_start|>system..." messages), which corrupts the embeddings.
        from sentence_transformers.sentence_transformer.modules import (
            Pooling as STPooling, Transformer as STTransformer)
        trc = {"trust_remote_code": True} if args.trust_remote_code else {}
        word = STTransformer(
            args.model,
            max_seq_length=args.max_seq_length or None,
            model_kwargs={**model_kwargs, **trc},
            processor_kwargs={"chat_template": None, **trc},
            config_kwargs=dict(trc),
        )
        pool = STPooling(word.get_word_embedding_dimension(),
                         pooling_mode=args.pooling)
        model = SentenceTransformer(modules=[word, pool], device=args.device)
    else:
        model = SentenceTransformer(args.model, device=args.device,
                                    trust_remote_code=args.trust_remote_code,
                                    model_kwargs=model_kwargs or None)
        if args.max_seq_length:
            model.max_seq_length = args.max_seq_length
    print(f"[model] {args.model} device={model.device} "
          f"max_seq_length={model.max_seq_length} pooling={args.pooling or 'repo-default'}",
          flush=True)

    eos = ""
    if args.append_eos:
        eos = model.tokenizer.eos_token or ""
        print(f"[model] appending EOS token {eos!r} to every text", flush=True)

    enc = dict(batch_size=args.batch_size, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=True)
    d_kwargs = dict(enc)
    if args.doc_prompt_name:
        d_kwargs["prompt_name"] = args.doc_prompt_name
    if args.doc_prompt:
        d_kwargs["prompt"] = args.doc_prompt
    q_kwargs = dict(enc)
    if args.query_prompt_name:
        q_kwargs["prompt_name"] = args.query_prompt_name
    if args.query_prompt:
        q_kwargs["prompt"] = args.query_prompt

    # -------- embedding cache (full sets only; tiers share corpus/queries) ---
    def cache_path(kind, prompt_desc, data_file):
        if not args.emb_cache_dir:
            return None
        sig = "|".join([args.model, str(args.model_dtype), str(args.pooling),
                        str(args.append_eos), str(args.max_seq_length),
                        str(prompt_desc), file_md5(data_file)])
        key = hashlib.md5(sig.encode()).hexdigest()[:16]
        safe = args.model.replace("/", "__")
        os.makedirs(args.emb_cache_dir, exist_ok=True)
        return os.path.join(args.emb_cache_dir, f"{kind}_{safe}_{key}.npz")

    def encode_with_cache(kind, ids, texts, kwargs, cacheable, prompt_desc, data_file):
        path = cache_path(kind, prompt_desc, data_file) if cacheable else None
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            if list(z["ids"]) == ids:
                print(f"[cache] loaded {kind} embeddings from {path}", flush=True)
                return z["emb"].astype(np.float32), True
            print(f"[cache] id mismatch for {path}; re-encoding", flush=True)
        emb = model.encode([t + eos for t in texts], **kwargs).astype(np.float32)
        if path:
            np.savez(path, ids=np.array(ids), emb=emb)
            print(f"[cache] saved {kind} embeddings to {path}", flush=True)
        return emb, False

    t_enc = time.time()
    corpus_file = os.path.join(tier_dir, "corpus.jsonl")
    queries_file = os.path.join(tier_dir, "queries.jsonl")
    doc_emb, doc_cached = encode_with_cache(
        "docs", corpus_ids, [corpus[c] for c in corpus_ids], d_kwargs,
        cacheable=not corpus_subsampled,
        prompt_desc=args.doc_prompt_name or args.doc_prompt, data_file=corpus_file)
    query_emb, query_cached = encode_with_cache(
        "queries", query_ids, [queries[q] for q in query_ids], q_kwargs,
        cacheable=not args.max_queries,
        prompt_desc=args.query_prompt_name or args.query_prompt, data_file=queries_file)
    print(f"[encode] {len(corpus_ids)} docs + {len(query_ids)} queries in "
          f"{time.time() - t_enc:.1f}s (doc_cached={doc_cached} "
          f"query_cached={query_cached})", flush=True)

    # ---------------- retrieval (chunked cosine sim; embeddings are unit-norm)
    # SELF-MASK bookkeeping: which queries have a corpus doc under their own _id?
    # A doc that is itself relevant per qrels is never masked (masking it would
    # destroy the metric instead of protecting it); that case is flagged instead.
    self_mask = not args.no_self_mask
    self_hit_ids = [q for q in query_ids if q in corpus_pos]
    self_gold_ids = [q for q in self_hit_ids if q in qrels[q]]
    mask_ids = set(self_hit_ids) - set(self_gold_ids) if self_mask else set()
    if self_hit_ids:
        msg = (f"[self-mask] {len(self_hit_ids)} of {len(query_ids)} queries have a "
               f"corpus document under their own _id (e.g. {self_hit_ids[0]!r})")
        if self_mask:
            print(msg + f" -- {len(mask_ids)} masked to -inf for their own query",
                  flush=True)
        else:
            print(msg + " -- NOT masked (--no-self-mask): recall numbers are "
                  "INVALID as a retrieval measurement", flush=True)
    if self_gold_ids:
        print(f"[self-mask] WARNING {len(self_gold_ids)} queries list their own "
              f"document as relevant in qrels (e.g. {self_gold_ids[0]!r}); left "
              f"unmasked -- check the qrels of this eval set", flush=True)
    max_k = max(K_VALUES)
    gold_rank = np.full(len(query_ids), len(corpus_ids), dtype=np.int64)
    # Figure-6 separation is accumulated as ALIGNED per-query records: a query
    # contributes iff it has BOTH its gold doc and >=1 near-miss doc in the
    # corpus. Before 2026-07-30 the positive was appended whenever the gold was
    # present while the negatives were appended only when near-misses existed,
    # so every query with a gold but no nm docs shifted all later positives
    # against another query's negatives (7 such queries on the full hard tier
    # shifted 13,872 of 14,993 pairs; pct_pos_above_all_hard_negs for
    # Qwen3-Embedding-4B read 0.09 instead of 0.07).
    pos_sims, neg_sims_mean, neg_sims_max = [], [], []
    n_pos_without_negs = 0
    dump_rows = [] if args.dump_ranks else None
    chunk = 256
    for s in range(0, len(query_ids), chunk):
        qs = query_emb[s:s + chunk]
        sims = qs @ doc_emb.T                                   # (chunk, N)
        if mask_ids:
            for i, qid in enumerate(query_ids[s:s + chunk]):
                if qid in mask_ids:
                    sims[i, corpus_pos[qid]] = -np.inf
        top = np.argpartition(-sims, max_k - 1, axis=1)[:, :max_k]
        # exact order inside the top-k slice
        order = np.take_along_axis(
            top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
        for i, qid in enumerate(query_ids[s:s + chunk]):
            gold_idx = {corpus_pos[c] for c in qrels[qid] if c in corpus_pos}
            ranks = [r for r, d in enumerate(order[i]) if d in gold_idx]
            if ranks:
                gold_rank[s + i] = ranks[0]
            # Figure-6-style positive / hard-negative similarities (aligned:
            # all three lists get exactly one entry per contributing query)
            gold_id = next(iter(qrels[qid]))
            if gold_id in corpus_pos:
                negs = [float(sims[i, corpus_pos[f"{qid}::nm::{j}"]])
                        for j in range(3) if f"{qid}::nm::{j}" in corpus_pos]
                if negs:
                    pos_sims.append(float(sims[i, corpus_pos[gold_id]]))
                    neg_sims_mean.append(float(np.mean(negs)))
                    neg_sims_max.append(float(np.max(negs)))
                else:
                    n_pos_without_negs += 1
            if dump_rows is not None:
                # exact rank of the best gold (not capped at top-k): number of
                # docs with strictly higher similarity.
                best_rank, best_sim = None, None
                for g in qrels[qid]:
                    if g not in corpus_pos:
                        continue
                    gs = float(sims[i, corpus_pos[g]])
                    r = int((sims[i] > gs).sum())
                    if best_rank is None or r < best_rank:
                        best_rank, best_sim = r, gs
                dump_rows.append({
                    "qid": qid,
                    "gold_rank": best_rank,
                    "gold_sim": best_sim,
                    "top10_ids": [corpus_ids[d] for d in order[i][:10]],
                })

    if dump_rows is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_ranks)), exist_ok=True)
        with open(args.dump_ranks, "w", encoding="utf-8") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[dump-ranks] wrote {len(dump_rows)} per-query rows to "
              f"{args.dump_ranks}", flush=True)

    # ---------------- metrics ----------------
    overall = {f"recall@{k}": round(100 * recall_at_k(gold_rank, k), 2)
               for k in K_VALUES}

    domain_map = load_domain_map()
    per_domain = {}
    if domain_map:
        by_dom = defaultdict(list)
        for i, qid in enumerate(query_ids):
            dom = domain_map.get(qid)
            if dom:
                by_dom[dom].append(i)
        for dom, idxs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
            sub = gold_rank[idxs]
            per_domain[dom] = {"n_queries": len(idxs),
                               **{f"recall@{k}": round(100 * recall_at_k(sub, k), 2)
                                  for k in K_VALUES}}

    assert len(pos_sims) == len(neg_sims_mean) == len(neg_sims_max), (
        "figure6 record lists must stay aligned per query")
    n_sep = len(pos_sims)
    pos = np.array(pos_sims); nmean = np.array(neg_sims_mean)
    nmax = np.array(neg_sims_max)
    separation = {
        "n_queries_with_pos_and_neg": n_sep,
        "n_queries_gold_but_no_near_miss": n_pos_without_negs,
        "mean_positive_sim": round(float(pos.mean()), 4) if n_sep else None,
        "mean_hard_negative_sim": round(float(nmean.mean()), 4) if n_sep else None,
        "mean_gap_pos_minus_meanneg": round(float((pos - nmean).mean()), 4) if n_sep else None,
        "mean_gap_pos_minus_maxneg": round(float((pos - nmax).mean()), 4) if n_sep else None,
        "pct_pos_above_all_hard_negs": round(100 * float((pos > nmax).mean()), 2) if n_sep else None,
        "std_positive_sim": round(float(pos.std()), 4) if n_sep else None,
        "std_hard_negative_sim": round(float(nmean.std()), 4) if n_sep else None,
    }

    result = {
        "benchmark": (args.data_dir or "ShadenA/MathNet-Retrieve"),
        "tier": tier_label,
        "model": args.model,
        "device": str(model.device),
        "n_corpus": len(corpus_ids),
        "n_queries_evaluated": len(query_ids),
        "corpus_subsampled": corpus_subsampled,
        "comparable_to_paper": not corpus_subsampled,
        "self_masking": self_mask,
        "n_queries_with_own_doc_in_corpus": len(self_hit_ids),
        "n_queries_self_masked": len(mask_ids),
        "n_queries_own_doc_is_relevant": len(self_gold_ids),
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "doc_prompt_name": args.doc_prompt_name,
        "doc_prompt": args.doc_prompt,
        "pooling": args.pooling or "repo-default",
        "append_eos": args.append_eos,
        "trust_remote_code": args.trust_remote_code,
        "model_dtype": args.model_dtype,
        "max_seq_length": int(model.max_seq_length) if model.max_seq_length else None,
        "seed": args.seed,
        "overall": overall,
        "per_domain": per_domain,
        "per_domain_coverage": (f"{sum(d['n_queries'] for d in per_domain.values())}"
                                f"/{len(query_ids)} queries have a derivable math "
                                f"domain (exact-match anchors only)") if per_domain else "unavailable",
        "figure6_separation": separation,
        "dump_ranks": args.dump_ranks,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "results",
        f"eval_{tier_label}_{args.model.split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
