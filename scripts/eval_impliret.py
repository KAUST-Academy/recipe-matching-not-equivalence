#!/usr/bin/env python3
"""
ImpliRet evaluation harness (arXiv:2506.14407, HF zeinabTaghavi/ImpliRet).

WHY THIS BENCHMARK. ImpliRet is the cross-domain test of this paper's claim.
Everything we have attacked so far is mathematics, so the strongest remaining
objection is that recipe-matching is something about math rather than about
LLM-constructed evaluation. ImpliRet is open-domain (forum/chat text about
shopping, travel and scheduling), its documents are LLM-generated with a
PUBLISHED prompt, and -- like MathNet-Retrieve's hard tier -- it looks
unsolvable: the paper reports a best nDCG@10 of 14.91 across the benchmark.

PROTOCOL, as described in their paper. Six subsets = 3 reasoning categories
(arithmetic, world-knowledge, temporal) x 2 discourse styles (uni-speaker chat,
multi-speaker forum). Each subset holds 1,500 documents and 1,500 queries, one
positive per query, and "at test time, each query is compared to all its
discourse style documents" -- so the candidate pool is the 1,500 documents of
that (category, style) subset, NOT the global 9,000. Their headline metric is
nDCG@10; with exactly one relevant document per query that reduces to
1/log2(rank+1) for rank <= 10 and 0 otherwise. We report R@1/5/10 alongside so
the numbers sit on the same scale as the rest of our campaign.

CALIBRATION. Their Table 2 reports several retrievers we can also run. Any
model we evaluate here that they also report is a calibration check, and we
state the result whether or not it matches -- the same rule the MathNet, SABER
and MELD harnesses follow. Run --calibrate to print the comparison.

Usage:
  python scripts/eval_impliret.py --model Qwen/Qwen3-Embedding-0.6B --device cuda \\
      --model-dtype bfloat16 --max-seq-length 1024 --query-prompt-name query \\
      --output results/impliret_base.json
  python scripts/eval_impliret.py --model ... --subsets unispeaker:arithmetic
"""

import argparse
import json
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_ID = "zeinabTaghavi/ImpliRet"
CONFIGS = ("unispeaker", "multispeaker")
SPLITS = ("arithmetic", "wknow", "temporal")

# Their Table 2, for the calibration check. Best reported nDCG@10 across the
# benchmark is 14.91; per-cell numbers are filled in as we confirm them from
# the paper rather than guessed.
PUBLISHED = {"best_overall_ndcg@10": 14.91}


LOCAL_DIR = os.path.join(ROOT, "data", "impliret_eval")


def dump_local():
    """Materialise the six subsets as JSONL.

    The `baselines-tf4` env exists to pin transformers 4.51.3 for ReasonIR and
    gte-Qwen2, and it has no `datasets` module. Rather than install into a
    deliberately pinned environment -- which is how the earlier torch/CUDA
    breakages happened -- we dump once from the main env and let every env read
    plain files. It also makes the evaluation reproducible without network.
    """
    from datasets import load_dataset
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for cfg in CONFIGS:
        for sp in SPLITS:
            d = load_dataset(HF_ID, cfg, split=sp)
            p = os.path.join(LOCAL_DIR, f"{cfg}__{sp}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for doc, q in zip(d["pos_document"], d["question"]):
                    f.write(json.dumps({"pos_document": str(doc),
                                        "question": str(q)}, ensure_ascii=False) + "\n")
            print(f"[dump] {p} ({len(d)} rows)")


def load_subsets(selected, local=False):
    out = {}
    for cfg in CONFIGS:
        for sp in SPLITS:
            name = f"{cfg}:{sp}"
            if selected and name not in selected:
                continue
            lp = os.path.join(LOCAL_DIR, f"{cfg}__{sp}.jsonl")
            if local or os.path.exists(lp):
                rows = [json.loads(l) for l in open(lp, encoding="utf-8") if l.strip()]
                docs = [r["pos_document"] for r in rows]
                queries = [r["question"] for r in rows]
            else:
                from datasets import load_dataset
                d = load_dataset(HF_ID, cfg, split=sp)
                docs = [str(x) for x in d["pos_document"]]
                queries = [str(x) for x in d["question"]]
            if len(docs) != len(queries):
                raise SystemExit(f"[fatal] {name}: {len(docs)} docs vs {len(queries)} queries")
            out[name] = {"docs": docs, "queries": queries}
    if not out:
        raise SystemExit("[fatal] no subsets selected")
    return out


def encode(model, texts, bs, prompt_name=None, prompt=None):
    kw = {}
    if prompt_name:
        kw["prompt_name"] = prompt_name
    elif prompt:
        kw["prompt"] = prompt
    return model.encode(texts, batch_size=bs, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False, **kw)


def score_subset(q_emb, d_emb):
    """One positive per query, aligned by index. Returns per-query gold rank (1-based)."""
    sims = q_emb @ d_emb.T
    n = sims.shape[0]
    gold = sims[np.arange(n), np.arange(n)]
    # rank = 1 + number of documents strictly better than the gold
    ranks = 1 + (sims > gold[:, None]).sum(axis=1)
    return ranks


def metrics(ranks):
    n = len(ranks)
    ndcg10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]))
    return {
        "ndcg@10": round(100 * ndcg10, 4),
        "recall@1": round(100 * float(np.mean(ranks <= 1)), 4),
        "recall@5": round(100 * float(np.mean(ranks <= 5)), 4),
        "recall@10": round(100 * float(np.mean(ranks <= 10)), 4),
        "mrr": round(float(np.mean(1.0 / ranks)), 4),
        "mean_gold_rank": round(float(np.mean(ranks)), 3),
        "n_queries": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model-dtype", default=None)
    ap.add_argument("--max-seq-length", type=int, default=0)
    ap.add_argument("--query-prompt-name", default=None)
    ap.add_argument("--query-prompt", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--subsets", default="", help="comma list like 'unispeaker:arithmetic'")
    ap.add_argument("--local", action="store_true",
                    help="read data/impliret_eval/*.jsonl instead of the HF hub")
    ap.add_argument("--dump-data", action="store_true",
                    help="materialise the six subsets as JSONL and exit")
    ap.add_argument("--output", default=None)
    ap.add_argument("--per-query-out", default=None)
    args = ap.parse_args()
    if args.query_prompt_name and args.query_prompt:
        raise SystemExit("[fatal] --query-prompt-name and --query-prompt are mutually exclusive")

    os.chdir(ROOT)
    if args.dump_data:
        dump_local()
        return
    sel = {s.strip() for s in args.subsets.split(",") if s.strip()}
    subsets = load_subsets(sel, args.local)

    import torch
    from sentence_transformers import SentenceTransformer
    kw = {}
    if args.model_dtype:
        kw["model_kwargs"] = {"torch_dtype": getattr(torch, args.model_dtype)}
    if args.trust_remote_code:
        kw["trust_remote_code"] = True
    t0 = time.time()
    model = SentenceTransformer(args.model, device=args.device, **kw)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    per_subset, all_ranks, pq = {}, [], {}
    for name, d in subsets.items():
        d_emb = encode(model, d["docs"], args.batch_size)
        q_emb = encode(model, d["queries"], args.batch_size,
                       args.query_prompt_name, args.query_prompt)
        ranks = score_subset(q_emb, d_emb)
        per_subset[name] = metrics(ranks)
        all_ranks.append(ranks)
        pq[name] = [int(r) for r in ranks]
        print(f"[impliret] {name:26s} nDCG@10={per_subset[name]['ndcg@10']:6.2f} "
              f"R@1={per_subset[name]['recall@1']:6.2f} "
              f"n={per_subset[name]['n_queries']}", flush=True)

    pooled = metrics(np.concatenate(all_ranks))
    # their headline is the average across subsets; we give both, since a macro
    # average over equal-sized subsets and a pooled one coincide here but would
    # not if a subset were ever subsampled.
    macro = {k: round(float(np.mean([per_subset[s][k] for s in per_subset])), 4)
             for k in ("ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")}

    out = {
        "benchmark": f"ImpliRet ({HF_ID}, arXiv:2506.14407)",
        "protocol": ("per-subset retrieval: each query ranks the 1,500 documents of its own "
                     "(category, discourse-style) subset, one positive per query, as the paper "
                     "specifies. nDCG@10 with a single relevant document."),
        "model": args.model,
        "device": str(args.device),
        "model_dtype": args.model_dtype,
        "max_seq_length": args.max_seq_length or getattr(model, "max_seq_length", None),
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "per_subset": per_subset,
        "macro_avg": macro,
        "pooled": pooled,
        "published_reference": PUBLISHED,
        "comparable_to_paper": ("protocol reimplemented from the paper; no official harness was "
                                "run, so treat cross-paper comparisons with the same caution as "
                                "our BRIGHT and MELD numbers"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {args.output}")
    if args.per_query_out:
        with open(args.per_query_out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "gold_ranks": pq}, f)
        print(f"[done] wrote {args.per_query_out}")

    print(f"\n[impliret] MACRO nDCG@10 = {macro['ndcg@10']:.2f} "
          f"(their best reported: {PUBLISHED['best_overall_ndcg@10']})")


if __name__ == "__main__":
    main()
