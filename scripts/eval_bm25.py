#!/usr/bin/env python3
"""A sparse lexical baseline (BM25, 2026-09-16) on every
evaluation set of the paper, scored with the SAME semantics as the dense
harnesses so the rows are comparable:
  * MathNet-Retrieve tiers (scripts/eval_retrieve.py): one gold per query,
    R@k = gold rank < k, rank = number of docs scoring STRICTLY above the gold;
  * duplicate sets (scripts/eval_crosslingual.py): the query's own corpus entry
    is masked, gold_rank / xling_rank (gold in another language: the "strict"
    metric) / same_rank from the query metadata, and the per-query rank dump
    in the harness's format so the same-language slices can be read out.
BM25 = bm25s (Lucene variant, k1=1.5, b=0.75), tokens = lowercased \\w+ runs
with English stopwords removed, no stemming; LaTeX commands survive as tokens.

  python scripts/eval_bm25.py --data-dir data/retrieve/easy --output results/bm25_easy.json
  python scripts/eval_bm25.py --data-dir data/crosslingual_eval --dup \
      --output results/bm25_crosslingual.json --dump-ranks results/ranks/xling_bm25.ranks.jsonl
"""
import argparse, json, os, time
import numpy as np
import bm25s

K_VALUES = (1, 5, 10)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dup", action="store_true", help="duplicate-set mode (metadata gold_ids/gold_langs)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dump-ranks", default=None)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    a = ap.parse_args()
    t0 = time.time()
    corpus_path = os.path.join(a.data_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):          # samelang_eval shares the cross-lingual corpus (as in eval_crosslingual.py)
        corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "crosslingual_eval", "corpus.jsonl")
    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(os.path.join(a.data_dir, "queries.jsonl"))
    doc_ids = [d["_id"] for d in corpus]
    pos = {d: i for i, d in enumerate(doc_ids)}
    n_docs = len(doc_ids)
    tok_docs = bm25s.tokenize([d["text"] for d in corpus], stopwords="en", show_progress=False)
    retriever = bm25s.BM25(k1=a.k1, b=a.b)
    retriever.index(tok_docs, show_progress=False)
    print(f"[index] {n_docs} docs in {time.time()-t0:.1f}s", flush=True)
    tok_q = bm25s.tokenize([q["text"] for q in queries], stopwords="en", show_progress=False, return_ids=False)
    if not a.dup:
        qrels = {}
        with open(os.path.join(a.data_dir, "qrels", "test.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                q, d, s = line.rstrip("\n").split("\t")
                if int(s) > 0:
                    qrels.setdefault(q, []).append(d)
    gold_rank = np.full(len(queries), n_docs, dtype=np.int64)
    xling_rank = np.full(len(queries), n_docs, dtype=np.int64)
    same_rank = np.full(len(queries), n_docs, dtype=np.int64)
    for i, q in enumerate(queries):
        row = retriever.get_scores(tok_q[i]).astype(np.float64)
        if q["_id"] in pos:                       # self-mask, as in the harness
            row[pos[q["_id"]]] = -np.inf
        if a.dup:
            meta = q["metadata"]
            golds = [g for g in meta["gold_ids"] if g in pos]
            if not golds:
                continue
            gs = {g: row[pos[g]] for g in golds}
            gold_rank[i] = int((row > max(gs.values())).sum())
            xl = [gs[g] for g, gl in zip(meta["gold_ids"], meta["gold_langs"]) if g in gs and gl != meta["lang"]]
            if xl:
                xling_rank[i] = int((row > max(xl)).sum())
            sl = [gs[g] for g, gl in zip(meta["gold_ids"], meta["gold_langs"]) if g in gs and gl == meta["lang"]]
            if sl:
                same_rank[i] = int((row > max(sl)).sum())
        else:
            golds = [g for g in qrels.get(q["_id"], []) if g in pos]
            if golds:
                gold_rank[i] = int((row > max(row[pos[g]] for g in golds)).sum())
        if i % 2000 == 0:
            print(f"[score] {i}/{len(queries)} queries, {time.time()-t0:.0f}s", flush=True)

    def block(r):
        return {f"recall@{k}": round(100 * float(np.mean(r < k)), 2) for k in K_VALUES}
    out = {"benchmark": a.data_dir, "model": "BM25 (bm25s lucene, k1=%.2f, b=%.2f, lowercased \\w+ tokens, English stopwords removed, no stemming)" % (a.k1, a.b),
           "n_corpus": n_docs, "n_queries": len(queries), "self_masking": True,
           "overall": block(gold_rank), "runtime_seconds": round(time.time() - t0, 1)}
    if a.dup:
        out["strict_crosslingual_gold"] = block(xling_rank)
        out["same_language_gold"] = block(same_rank)
    if a.dump_ranks:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_ranks)), exist_ok=True)
        with open(a.dump_ranks, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries):
                m = q.get("metadata", {})
                f.write(json.dumps({"qid": q["_id"], "cluster_id": m.get("cluster_id"), "lang": m.get("lang"),
                                    "min_confidence": m.get("min_confidence"),
                                    "overlaps_retrieve_anchor": m.get("overlaps_retrieve_anchor"),
                                    "gold_rank": int(gold_rank[i]), "xling_rank": int(xling_rank[i]),
                                    "same_rank": int(same_rank[i]), "n_docs": n_docs}) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    print(json.dumps(out, indent=2)); print(f"[done] wrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
