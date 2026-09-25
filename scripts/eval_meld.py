#!/usr/bin/env python3
"""
MELD external-OOD evaluation harness (Recipe-Matching follow-up project).

Evaluates an embedding model on MELD (uw-math-ai/MELD-dataset, Apache-2.0):
270 human-reviewed pairs of mathematically equivalent statements written in
DIFFERENT subfield dialects (9 domains x 30 pairs, 18 framings) plus 541
lexically-similar-but-false distractor statements. Introduced by the
MathLeap paper (arXiv:2606.23959). We did NOT
build this benchmark -- that is the point: it supplies the independence our
mined real-duplicate set lacks.

Metrics reported (all cosine over L2-normalized embeddings):
  1. equivalence-pair separation AUC:
       positives = cos(query-role entry, doc-role partner)     (540 scores)
       negatives = cos(query-role entry, doc-role distractor)
     two negative pools: ALL 541 distractors ("auc_all_distractors") and
     only the distractors written in the partner's framing -- the hardest
     setting per the dataset card ("auc_same_framing_distractors"); plus
     pct_pos_above_all_same_framing_distractors (Figure-6-style).
  2. retrieval, protocol "pairs_only" (the MathLeap paper's own Table 2/5
     protocol: "we embed all 540 statements ... whether equivalent pairs are
     grouped near each other"): each of the 540 statements queries the other
     539; gold = its partner. Recall@1/3/5/10/20, MRR, mean 1-based gold
     rank -- directly comparable to their Table 2 (e.g. Qwen3-Embedding-4B
     13.7 R@1) and Table 5 (MRR 0.28, mean rank 21.4).
  3. retrieval, protocol "with_distractors" (the dataset card's suggested
     protocol): entry_1 queries a pool of all 270 entry_2 + all 541
     distractors (and the reverse direction). Recall@1/3/5/10/20, MRR.
  4. per-domain Recall@1 (pairs_only protocol) for the 9 domains.

Protocol notes / deviations (documented, not silently applied):
  * The paper does not state whether Table 2 includes distractors in the
    candidate pool nor which instruction each model got; we report BOTH
    protocols and record the exact prompt used in the output JSON.
  * For instruction/prompted models the prompt is applied to the QUERY role
    only (house convention == BRIGHT convention == Qwen3-Embedding usage);
    every statement is therefore encoded twice when a prompt is set (query
    role with prompt, doc role without).

Usage (CPU smoke, ~2 min -- the set is tiny, no subsampling needed):
  python scripts/eval_meld.py --model sentence-transformers/all-MiniLM-L6-v2 \
      --device cpu --output results/meld_smoke_minilm.json

Usage (GPU, see eval_external_ood.slurm):
  python scripts/eval_meld.py --model models/ctrl-llm-6145/final \
      --device cuda --model-dtype bfloat16 --query-prompt-name query \
      --max-seq-length 1024 --output results/meld_ctrl-llm-6145.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MELD_DIR = os.path.join(PROJECT_ROOT, "data", "external", "meld")
HF_BASE = "https://huggingface.co/datasets/uw-math-ai/MELD-dataset/resolve/main"
FILES = ("adversarial_theorem_pairs_2.json", "distractors_all.json")
K_VALUES = (1, 3, 5, 10, 20)


def download_meld():
    os.makedirs(MELD_DIR, exist_ok=True)
    for name in FILES:
        dest = os.path.join(MELD_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{HF_BASE}/{name}"
        print(f"[download] {url} -> {dest}", flush=True)
        urllib.request.urlretrieve(url, dest)


def rank_metrics(gold_rank_1based):
    """gold_rank_1based: np.array of the gold's 1-based rank per query."""
    r = np.asarray(gold_rank_1based, dtype=np.float64)
    out = {f"recall@{k}": round(100 * float(np.mean(r <= k)), 2) for k in K_VALUES}
    out["mrr"] = round(float(np.mean(1.0 / r)), 4)
    out["mean_gold_rank"] = round(float(np.mean(r)), 2)
    out["n_queries"] = int(len(r))
    return out


def auc(pos, neg):
    """Exact ROC-AUC via rank statistic (ties counted 0.5)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(len(all_scores), dtype=np.float64)
    # average ranks for ties
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--query-prompt-name", default=None,
                    help="sentence-transformers prompt name for the query role "
                         "(e.g. 'query' for Qwen3-Embedding models)")
    ap.add_argument("--query-prompt", default=None,
                    help="raw prompt string prepended to every query-role text "
                         "(mutually exclusive with --query-prompt-name)")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override tokenizer truncation length (0 = model default)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--output", default=None,
                    help="result JSON path (default results/meld_<model>.json)")
    ap.add_argument("--per-query-out", default=None,
                    help="ALSO dump per-query pairs_only gold ranks and the "
                         "same-framing hard indicator to this JSON, so paired "
                         "cluster-bootstrap CIs can be computed between two "
                         "models afterwards (generate_meld_pairs.py --stage "
                         "compare). Purely additive: the main result JSON and "
                         "every number in it are unchanged.")
    args = ap.parse_args()
    if args.query_prompt and args.query_prompt_name:
        ap.error("--query-prompt and --query-prompt-name are mutually exclusive")

    t0 = time.time()

    # ---------------- data ----------------
    download_meld()
    with open(os.path.join(MELD_DIR, FILES[0]), encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    with open(os.path.join(MELD_DIR, FILES[1]), encoding="utf-8") as f:
        distractors = json.load(f)  # framing -> [statement, ...]
    assert len(pairs) == 270, f"expected 270 MELD pairs, got {len(pairs)}"
    n_distr = sum(len(v) for v in distractors.values())
    print(f"[data] MELD: {len(pairs)} pairs, {n_distr} distractors, "
          f"{len(distractors)} framings", flush=True)

    # statement lists (fixed order)
    stmts = []          # 540 pair statements: index 2*i = entry_1, 2*i+1 = entry_2
    for p in pairs:
        stmts.append(p["entry_1"]["statement"])
        stmts.append(p["entry_2"]["statement"])
    distr_texts, distr_framing = [], []
    for fr, lst in distractors.items():
        for s in lst:
            distr_texts.append(s)
            distr_framing.append(fr)
    distr_idx_by_framing = defaultdict(list)
    for i, fr in enumerate(distr_framing):
        distr_idx_by_framing[fr].append(i)

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
    prompted = bool(args.query_prompt_name or args.query_prompt)

    t_enc = time.time()
    doc_stmt = model.encode(stmts, **enc).astype(np.float32)          # (540, d)
    doc_distr = model.encode(distr_texts, **enc).astype(np.float32)   # (541, d)
    # query-role embeddings (with prompt if set; else reuse doc embeddings)
    qry_stmt = (model.encode(stmts, **q_kwargs).astype(np.float32)
                if prompted else doc_stmt)
    print(f"[encode] {len(stmts)}+{len(distr_texts)} texts "
          f"(x2 roles={prompted}) in {time.time() - t_enc:.1f}s", flush=True)

    n = len(pairs)
    e1_q, e2_q = qry_stmt[0::2], qry_stmt[1::2]      # query role
    e1_d, e2_d = doc_stmt[0::2], doc_stmt[1::2]      # doc role

    # ---------------- 1. pair-separation AUC ----------------
    pos_12 = np.sum(e1_q * e2_d, axis=1)     # entry_1 queries its partner
    pos_21 = np.sum(e2_q * e1_d, axis=1)
    pos = np.concatenate([pos_12, pos_21])   # 540 positive scores

    neg_all = np.concatenate([(e1_q @ doc_distr.T).ravel(),
                              (e2_q @ doc_distr.T).ravel()])
    neg_same, above_all_hard = [], []
    for i, p in enumerate(pairs):
        for qv, target_framing, pos_s in (
                (e1_q[i], p["entry_2"]["framing"], pos_12[i]),
                (e2_q[i], p["entry_1"]["framing"], pos_21[i])):
            idx = distr_idx_by_framing.get(target_framing, [])
            if not idx:
                continue
            sims = doc_distr[idx] @ qv
            neg_same.extend(sims.tolist())
            above_all_hard.append(float(pos_s > sims.max()))
    separation = {
        "n_positive_scores": int(len(pos)),
        "auc_all_distractors": round(auc(pos, neg_all), 4),
        "auc_same_framing_distractors": round(auc(pos, np.array(neg_same)), 4),
        "pct_pos_above_all_same_framing_distractors":
            round(100 * float(np.mean(above_all_hard)), 2),
        "mean_positive_sim": round(float(pos.mean()), 4),
        "mean_all_distractor_sim": round(float(neg_all.mean()), 4),
        "mean_same_framing_distractor_sim": round(float(np.mean(neg_same)), 4),
    }

    # ---------------- 2. retrieval: pairs_only (paper Table 2/5) ----------
    sims = qry_stmt @ doc_stmt.T                       # (540, 540)
    np.fill_diagonal(sims, -np.inf)                    # exclude self
    gold_col = np.arange(2 * n)
    gold_col = gold_col + 1 - 2 * (gold_col % 2)       # partner index (i^1)
    gold_sim = sims[np.arange(2 * n), gold_col]
    pairs_only_rank = (sims > gold_sim[:, None]).sum(axis=1) + 1   # 1-based
    pairs_only = rank_metrics(pairs_only_rank)

    per_domain = {}
    dom_idx = defaultdict(list)
    for i, p in enumerate(pairs):
        dom_idx[p["domain"]] += [2 * i, 2 * i + 1]
    for dom, idx in sorted(dom_idx.items()):
        r = pairs_only_rank[idx]
        per_domain[dom] = {"n_queries": len(idx),
                           "recall@1": round(100 * float(np.mean(r <= 1)), 2),
                           "recall@5": round(100 * float(np.mean(r <= 5)), 2)}

    # ---------------- 3. retrieval: with_distractors (dataset card) -------
    def directed(q_emb, target_doc_emb):
        """q_emb (n,d) queries pool = target_doc_emb (n,d) + all distractors;
        gold = row-aligned target. Returns 1-based gold ranks."""
        s_pair = q_emb @ target_doc_emb.T              # (n, n)
        s_dist = q_emb @ doc_distr.T                   # (n, 541)
        gold = np.diag(s_pair)
        return ((s_pair > gold[:, None]).sum(axis=1)
                + (s_dist > gold[:, None]).sum(axis=1) + 1)

    rank_12 = directed(e1_q, e2_d)
    rank_21 = directed(e2_q, e1_d)
    with_distractors = {
        "pool": "270 partner statements + 541 distractors (811 candidates)",
        "entry1_to_entry2": rank_metrics(rank_12),
        "entry2_to_entry1": rank_metrics(rank_21),
        "both_directions": rank_metrics(np.concatenate([rank_12, rank_21])),
    }

    result = {
        "benchmark": "uw-math-ai/MELD-dataset (external, arXiv:2606.23959)",
        "model": args.model,
        "device": str(model.device),
        "n_pairs": n,
        "n_distractors": n_distr,
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "model_dtype": args.model_dtype,
        "max_seq_length": int(model.max_seq_length) if model.max_seq_length else None,
        "separation_auc": separation,
        "retrieval_pairs_only": pairs_only,
        "retrieval_pairs_only_per_domain": per_domain,
        "retrieval_with_distractors": with_distractors,
        "protocol_notes": [
            "pairs_only = MathLeap paper Table 2/5 protocol (540 statements, "
            "query the other 539, gold = partner); their Qwen3-Embedding-4B "
            "reference row: R@1 13.7 / R@5 45.0 / R@20 70.2, MRR 0.28, mean "
            "rank 21.4 (prompt they used is unstated -- exact match not "
            "guaranteed).",
            "with_distractors = dataset-card suggested protocol.",
            "prompt applied to query role only; doc role always unprompted.",
        ],
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "results", f"meld_{args.model.rstrip('/').split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}", flush=True)

    if args.per_query_out:
        # Query i corresponds to statement i (i = 2*pair + 0/1), so the
        # resampling unit "pair" is just i // 2. above_all_hard is in the same
        # statement order (entry_1 query then entry_2 query, per pair).
        os.makedirs(os.path.dirname(os.path.abspath(args.per_query_out)) or ".",
                    exist_ok=True)
        with open(args.per_query_out, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "query_prompt_name": args.query_prompt_name,
                "query_prompt": args.query_prompt,
                "n_pairs": n,
                "order": "statement index i; pair = i // 2; i even = entry_1 "
                         "query, i odd = entry_2 query",
                "pairs_only_gold_rank_1based": [int(r) for r in pairs_only_rank],
                "above_all_same_framing_distractors": [int(v) for v in above_all_hard],
                "with_distractors_gold_rank_entry1_to_entry2": [int(r) for r in rank_12],
                "with_distractors_gold_rank_entry2_to_entry1": [int(r) for r in rank_21],
            }, f)
        print(f"[done] wrote per-query dump {args.per_query_out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
