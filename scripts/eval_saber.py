#!/usr/bin/env python3
"""
SABER-Math evaluation harness (MathNet follow-up: out-of-distribution evaluation).

Evaluates any embedding model / trained checkpoint on SABER-Math
(arXiv:2606.29894, INSAIT), the automated math-IR RERANKING benchmark:
1,000 queries x 150 graded candidates, metric = nDCG@10 with EXPONENTIAL gain
(gain 2^rel - 1, discount 1/log2(rank+1), IDCG from all 150 candidate
relevances sorted desc). Protocol verified line-by-line against the official
implementation (github.com/insait-institute/sabermath, master @ 2026-07-25;
reference copies in data/saber/upstream_ref/):

  * candidates are POSITIONAL row indices into SaberMath-documents
    (benchmark.py: `ds.select(doc_ids)`), NOT the `original_index` column;
  * text construction (benchmark.py `transform`):
      statement version = row["problem"]  (verbatim)
      full version      = f"Problem: {problem}\\n\\nSolution: {solution}"
  * main setting = statement-full (query: statement; docs: full);
    also statement-statement and full-full (Table 4);
  * ranking = np.argsort(-cosine_scores) over the 150 candidates;
  * overall nDCG = mean over all 1,000 queries; per-domain = mean over
    queries whose `domains` list contains the branch (queries can carry
    multiple domains, so branch n's sum to 1,156, not 1,000);
  * their model runs use NO instruction prompts (models.txt: Qwen3-Embedding
    via vLLM raw, e5 via ST without "query:"/"passage:" prefixes). Prompt
    flags below exist for OUR trained checkpoints, whose training prompt must
    be mirrored at eval time — always reported in the output JSON;
  * over-long texts: their ST driver optionally splits into non-overlapping
    token chunks and averages the chunk embeddings (`chunk_to_context`, used
    for e5/BERT/RoBERTa). Mirrored here as --chunk-long-docs.

Published anchors (their Table 1, statement-full overall nDCG@10):
  Qwen3-Embedding-0.6B 0.575 | multilingual-e5-large 0.488 | BGE-m3 0.511

Extra diagnostics per run (not part of their protocol, used by the
recipe-matching attack analysis): mean per-query Spearman correlation of the
model's candidate scores with (a) the final relevance_scores, (b) the
summary-Jaccard construction signal, (c) the topic-BMA construction signal.
A model that games the construction recipe shows (b) >> honest models.

Usage (CPU smoke, subsample):
  python scripts/eval_saber.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --max-queries 15 --output results/saber_smoke.json

Usage (full, GPU):
  python scripts/eval_saber.py --model Qwen/Qwen3-Embedding-0.6B \
      --device cuda --model-dtype bfloat16 --batch-size 64 \
      --output results/saber_qwen3-0.6b-base.json
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_REPO = "INSAIT-Institute/SaberMath-queries"
DOCS_REPO = "INSAIT-Institute/SaberMath-documents"
BRANCHES = ("Algebra", "Geometry", "Number Theory", "Combinatorics",
            "Calculus and Analysis")
SETTINGS = ("statement-full", "statement-statement", "full-full")


def load_saber():
    """Download (cached via HF_HOME) + load both parquets. Returns (q, d)."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    frames = []
    for repo in (QUERIES_REPO, DOCS_REPO):
        snap = snapshot_download(repo_id=repo, repo_type="dataset")
        pq = os.path.join(snap, "data", "train-00000-of-00001.parquet")
        frames.append(pd.read_parquet(pq))
    q, d = frames
    # positional-candidate invariants (fail loudly if upstream re-shuffles)
    n_docs = len(d)
    assert len(q) == 1000 and n_docs == 71117, \
        f"unexpected sizes: {len(q)} queries / {n_docs} docs (repo updated?)"
    cmax = max(int(np.max(c)) for c in q["candidates"])
    assert cmax < n_docs, f"candidate index {cmax} >= n_docs {n_docs}"
    assert all(len(c) == 150 and len(r) == 150
               for c, r in zip(q["candidates"], q["relevance_scores"]))
    return q, d


def full_text(problem: str, solution: str) -> str:
    """EXACT template of the official benchmark.py transform()."""
    return f"Problem: {problem}\n\nSolution: {solution}"


def dcg_at_k(rel: np.ndarray, k: int, variant: str) -> float:
    rel = np.asarray(rel[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = np.power(2.0, rel) - 1.0 if variant == "exponent" else rel
    return float(np.sum(gains * disc))


def ndcg_at_k(rel_in_model_order: np.ndarray, k: int, variant: str) -> float:
    dcg = dcg_at_k(rel_in_model_order, k, variant)
    idcg = dcg_at_k(np.sort(rel_in_model_order)[::-1], k, variant)
    return 0.0 if idcg == 0 else dcg / idcg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--setting", default="statement-full",
                    choices=list(SETTINGS) + ["all"],
                    help="main setting = statement-full (their Table 1)")
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--per-query-out", default=None,
                    help="write per-query nDCG so differences can be bootstrapped")
    ap.add_argument("--dcg-variant", default="exponent",
                    choices=["exponent", "linear"],
                    help="paper reports exponent (gain 2^rel - 1)")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="random subsample of queries (0 = all 1000); only "
                         "the sampled queries' candidate docs are encoded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-prompt-name", default=None,
                    help="registered ST prompt name for queries (our trained "
                         "checkpoints; their baseline runs use none)")
    ap.add_argument("--query-prompt", default=None,
                    help="raw prompt string prepended to every query")
    ap.add_argument("--doc-prompt-name", default=None)
    ap.add_argument("--doc-prompt", default=None)
    ap.add_argument("--pooling", default=None,
                    choices=[None, "lasttoken", "mean", "cls"],
                    help="manual Transformer+Pooling stack (see eval_retrieve.py)")
    ap.add_argument("--append-eos", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override truncation length (0 = model default)")
    ap.add_argument("--chunk-long-docs", action="store_true",
                    help="mirror their chunk_to_context: split over-long "
                         "texts into non-overlapping max-length token chunks, "
                         "embed each, average (they enable it for e5/BERT/"
                         "RoBERTa; irrelevant for 32K-context models)")
    ap.add_argument("--emb-cache-dir", default=None,
                    help="cache full-set embeddings per (model,cfg,version)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.query_prompt and args.query_prompt_name:
        ap.error("--query-prompt and --query-prompt-name are mutually exclusive")
    if args.doc_prompt and args.doc_prompt_name:
        ap.error("--doc-prompt and --doc-prompt-name are mutually exclusive")

    rng = random.Random(args.seed)
    t0 = time.time()
    settings = list(SETTINGS) if args.setting == "all" else [args.setting]

    # ---------------- data ----------------
    q_df, d_df = load_saber()
    print(f"[data] SABER-Math: {len(q_df)} queries, {len(d_df)} documents",
          flush=True)
    q_idx = list(range(len(q_df)))
    if args.max_queries and args.max_queries < len(q_idx):
        q_idx = sorted(rng.sample(q_idx, args.max_queries))
        print(f"[sample] subsampled to {len(q_idx)} queries (seed={args.seed}) "
              f"-- overall nDCG has query-sampling noise", flush=True)
    needed_docs = sorted({int(c) for i in q_idx
                          for c in q_df["candidates"].iloc[i]})
    print(f"[data] {len(needed_docs)} unique candidate docs needed", flush=True)

    # ---------------- model (same loading contract as eval_retrieve.py) ----
    from sentence_transformers import SentenceTransformer
    model_kwargs = {}
    if args.model_dtype and args.model_dtype != "float32":
        import torch
        model_kwargs["torch_dtype"] = getattr(torch, args.model_dtype)
    if args.pooling:
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
          f"max_seq_length={model.max_seq_length} "
          f"pooling={args.pooling or 'repo-default'}", flush=True)

    eos = ""
    if args.append_eos:
        eos = model.tokenizer.eos_token or ""
        print(f"[model] appending EOS token {eos!r} to every text", flush=True)

    def chunked_encode(texts, kwargs):
        """Their chunk_to_context: non-overlapping token chunks, mean-pooled.
        Chunks inherit the same prompt kwargs as whole texts."""
        tok = model.tokenizer
        max_len = int(model.max_seq_length)
        n_special = tok.num_special_tokens_to_add(pair=False)
        budget = max_len - n_special
        chunks, owner = [], []
        for ti, t in enumerate(texts):
            ids = tok.encode(t, add_special_tokens=False, truncation=False)
            if not ids:
                chunks.append("")
                owner.append(ti)
                continue
            for s in range(0, len(ids), budget):
                chunks.append(tok.decode(ids[s:s + budget],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False))
                owner.append(ti)
        emb = model.encode([c + eos for c in chunks], **kwargs)
        owner = np.asarray(owner)
        out = np.stack([emb[owner == ti].mean(axis=0)
                        for ti in range(len(texts))])
        # re-normalize after averaging (cosine is computed on these directly)
        out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out.astype(np.float32)

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

    def cache_path(kind, version, prompt_desc):
        if not args.emb_cache_dir:
            return None
        sig = "|".join([args.model, str(args.model_dtype), str(args.pooling),
                        str(args.append_eos), str(args.max_seq_length),
                        str(args.chunk_long_docs), str(prompt_desc),
                        f"saber-{kind}-{version}"])
        key = hashlib.md5(sig.encode()).hexdigest()[:16]
        safe = args.model.replace("/", "__")
        os.makedirs(args.emb_cache_dir, exist_ok=True)
        return os.path.join(args.emb_cache_dir, f"saber_{kind}_{safe}_{key}.npz")

    def encode_texts(texts, kwargs):
        if args.chunk_long_docs:
            return chunked_encode(texts, kwargs)
        return model.encode([t + eos for t in texts], **kwargs).astype(np.float32)

    emb_store = {}   # (kind, version) -> ndarray aligned with ids below

    def get_embeddings(kind, version):
        """kind in {docs, queries}; version in {statement, full}."""
        mkey = (kind, version)
        if mkey in emb_store:
            return emb_store[mkey]
        if kind == "docs":
            rows, ids = d_df.iloc[needed_docs], needed_docs
            kwargs, pdesc = d_kwargs, (args.doc_prompt_name or args.doc_prompt)
            cacheable = len(needed_docs) == len(d_df)
        else:
            rows, ids = q_df.iloc[q_idx], q_idx
            kwargs, pdesc = q_kwargs, (args.query_prompt_name or args.query_prompt)
            cacheable = len(q_idx) == len(q_df)
        if version == "full":
            texts = [full_text(p, s) for p, s in zip(rows["problem"],
                                                     rows["solution"])]
        else:
            texts = list(rows["problem"])
        path = cache_path(kind, version, pdesc) if cacheable else None
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            if list(z["ids"]) == ids:
                print(f"[cache] loaded {kind}/{version} from {path}", flush=True)
                emb_store[mkey] = z["emb"].astype(np.float32)
                return emb_store[mkey]
            print(f"[cache] id mismatch for {path}; re-encoding", flush=True)
        t = time.time()
        emb = encode_texts(texts, kwargs)
        print(f"[encode] {kind}/{version}: {len(texts)} texts in "
              f"{time.time() - t:.1f}s", flush=True)
        if path:
            np.savez(path, ids=np.array(ids), emb=emb)
            print(f"[cache] saved {kind}/{version} to {path}", flush=True)
        emb_store[mkey] = emb
        return emb

    doc_pos = {d: i for i, d in enumerate(needed_docs)}

    # ---------------- evaluate ----------------
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    results = {}
    for setting in settings:
        q_version = "full" if setting == "full-full" else "statement"
        d_version = "statement" if setting == "statement-statement" else "full"
        q_emb = get_embeddings("queries", q_version)
        d_emb = get_embeddings("docs", d_version)

        ndcgs, by_branch = [], {b: [] for b in BRANCHES}
        sp_rel, sp_jac, sp_bma = [], [], []
        for row_i, qi in enumerate(q_idx):
            row = q_df.iloc[qi]
            cand = [doc_pos[int(c)] for c in row["candidates"]]
            scores = d_emb[cand] @ q_emb[row_i]          # cosine (unit norm)
            order = np.argsort(-scores)                  # their exact ranking
            rel = np.asarray(row["relevance_scores"], dtype=float)[order]
            nd = ndcg_at_k(rel, args.ndcg_k, args.dcg_variant)
            ndcgs.append(nd)
            for b in row["domains"]:
                if b in by_branch:
                    by_branch[b].append(nd)
            if spearmanr is not None:
                sp_rel.append(spearmanr(scores, row["relevance_scores"]).statistic)
                sp_jac.append(spearmanr(scores, row["jaccard_scores"]).statistic)
                sp_bma.append(spearmanr(scores, row["bma_scores"]).statistic)

        # Per-query nDCG, so a difference between two models can carry a paired
        # interval. Added 2026-07-31: the SABER label-channel attack produced a
        # +0.0354 point estimate over the base and there was no way to test it
        # without re-running: a "point estimates only" gap.
        if args.per_query_out:
            os.makedirs(os.path.dirname(args.per_query_out) or ".", exist_ok=True)
            with open(args.per_query_out, "w", encoding="utf-8") as _f:
                json.dump({"model": args.model, "setting": setting,
                           "ndcg_k": args.ndcg_k, "dcg_variant": args.dcg_variant,
                           "query_row_index": [int(x) for x in q_idx],
                           "per_query_ndcg": [float(x) for x in ndcgs]}, _f)
            print(f"[saber] per-query nDCG -> {args.per_query_out}", flush=True)

        results[setting] = {
            f"ndcg@{args.ndcg_k}": round(float(np.mean(ndcgs)), 4),
            "per_domain": {b: {"n_queries": len(v),
                               f"ndcg@{args.ndcg_k}": round(float(np.mean(v)), 4)}
                           for b, v in by_branch.items() if v},
            "score_correlations_spearman_mean": ({
                "with_final_relevance": round(float(np.nanmean(sp_rel)), 4),
                "with_summary_jaccard_signal": round(float(np.nanmean(sp_jac)), 4),
                "with_topic_bma_signal": round(float(np.nanmean(sp_bma)), 4),
            } if spearmanr is not None else "scipy unavailable"),
        }
        print(f"[result] {setting}: nDCG@{args.ndcg_k} "
              f"({args.dcg_variant}) = {results[setting][f'ndcg@{args.ndcg_k}']}",
              flush=True)

    # ---------------- write ----------------
    out = {
        "benchmark": "SABER-Math (INSAIT-Institute/SaberMath-{queries,documents})",
        "protocol": "official reranking protocol, positional candidates, "
                    "np.argsort(-cos); anchors: Qwen3-0.6B 0.575, e5-large "
                    "0.488, bge-m3 0.511 (Table 1, statement-full)",
        "model": args.model,
        "device": str(model.device),
        "n_queries_evaluated": len(q_idx),
        "n_docs_encoded": len(needed_docs),
        "comparable_to_paper": not args.max_queries,
        "ndcg_k": args.ndcg_k,
        "dcg_variant": args.dcg_variant,
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "doc_prompt_name": args.doc_prompt_name,
        "doc_prompt": args.doc_prompt,
        "pooling": args.pooling or "repo-default",
        "append_eos": args.append_eos,
        "trust_remote_code": args.trust_remote_code,
        "model_dtype": args.model_dtype,
        "max_seq_length": int(model.max_seq_length) if model.max_seq_length else None,
        "chunk_long_docs": args.chunk_long_docs,
        "seed": args.seed,
        "settings": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out_path = args.output or os.path.join(
        PROJECT_ROOT, "results",
        f"saber_{args.model.split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
