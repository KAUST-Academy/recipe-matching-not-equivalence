#!/usr/bin/env python3
"""Late-interaction arms: score a PyLate ColBERT model on one BEIR-style directory
with the dense harness's semantics: the query's own corpus entry is masked;
tiers use the qrels gold; duplicate sets (--dup) use metadata gold_ids /
gold_langs for the overall, strict cross-language and same-language ranks.
Ranks come from a PLAID index (top --k candidates, exact MaxSim rerank inside
PyLate); a gold outside the top --k counts as rank n_docs (a miss at every k).

  python scripts/colbert_eval.py --data-dir data/retrieve/easy --model models/colbert-ctrl-llm-6145/final \
      --output results/colbert/easy_ctrl-llm-6145.json
"""
import argparse, json, os, shutil, time
import numpy as np

K_VALUES = (1, 5, 10)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--query-length", type=int, default=256)
    ap.add_argument("--document-length", type=int, default=512)
    ap.add_argument("--index-root", default="/tmp/colbert_index")
    ap.add_argument("--max-docs", type=int, default=0, help="debug subsample (keeps golds)")
    ap.add_argument("--dump-ranks", default=None)
    a = ap.parse_args()
    t0 = time.time()
    from pylate import indexes, models, retrieve
    corpus_path = os.path.join(a.data_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "crosslingual_eval", "corpus.jsonl")
    corpus = jsonl(corpus_path); queries = jsonl(os.path.join(a.data_dir, "queries.jsonl"))
    qrels = {}
    if not a.dup:
        with open(os.path.join(a.data_dir, "qrels", "test.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                q, d, s = line.rstrip("\n").split("\t")
                if int(s) > 0:
                    qrels.setdefault(q, []).append(d)
    if a.max_docs and a.max_docs < len(corpus):
        gold = {g for q in queries for g in (qrels.get(q["_id"], []) if not a.dup else q["metadata"]["gold_ids"])}
        keep = [c for c in corpus if c["_id"] in gold]; rest = [c for c in corpus if c["_id"] not in gold]
        corpus = keep + rest[: max(0, a.max_docs - len(keep))]
    doc_ids = [c["_id"] for c in corpus]; n_docs = len(doc_ids)
    try:
        model = models.ColBERT(model_name_or_path=a.model, query_length=a.query_length, document_length=a.document_length)
    except TypeError:
        model = models.ColBERT(model_name_or_path=a.model)
    name = f"idx_{os.getpid()}"
    index = indexes.PLAID(index_folder=a.index_root, index_name=name, override=True)
    d_emb = model.encode([c["text"] for c in corpus], batch_size=a.batch_size, is_query=False, show_progress_bar=False)
    index.add_documents(documents_ids=doc_ids, documents_embeddings=d_emb)
    del d_emb
    q_emb = model.encode([q["text"] for q in queries], batch_size=a.batch_size, is_query=True, show_progress_bar=False)
    retriever = retrieve.ColBERT(index=index)
    k = min(a.k + 1, n_docs)
    results = retriever.retrieve(queries_embeddings=q_emb, k=k)
    print(f"[retrieve] {n_docs} docs, {len(queries)} queries, top-{k} in {time.time()-t0:.0f}s", flush=True)
    gold_rank = np.full(len(queries), n_docs); xling_rank = np.full(len(queries), n_docs); same_rank = np.full(len(queries), n_docs)
    for i, q in enumerate(queries):
        ranked = [r["id"] for r in results[i] if r["id"] != q["_id"]]   # self-mask
        pos = {d: j for j, d in enumerate(ranked)}
        if a.dup:
            meta = q["metadata"]
            golds = [(g, gl) for g, gl in zip(meta["gold_ids"], meta["gold_langs"])]
            rs = [pos[g] for g, _ in golds if g in pos]
            if rs: gold_rank[i] = min(rs)
            xl = [pos[g] for g, gl in golds if g in pos and gl != meta["lang"]]
            if xl: xling_rank[i] = min(xl)
            sl = [pos[g] for g, gl in golds if g in pos and gl == meta["lang"]]
            if sl: same_rank[i] = min(sl)
        else:
            rs = [pos[g] for g in qrels.get(q["_id"], []) if g in pos]
            if rs: gold_rank[i] = min(rs)
    block = lambda r: {f"recall@{kk}": round(100 * float(np.mean(r < kk)), 2) for kk in K_VALUES}
    out = {"benchmark": a.data_dir, "model": a.model, "backend": "pylate PLAID (top-%d candidates)" % a.k,
           "query_length": a.query_length, "document_length": a.document_length, "n_corpus": n_docs,
           "n_queries": len(queries), "self_masking": True, "overall": block(gold_rank),
           "runtime_seconds": round(time.time() - t0, 1)}
    if a.dup:
        out["strict_crosslingual_gold"] = block(xling_rank); out["same_language_gold"] = block(same_rank)
    if a.dump_ranks:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_ranks)), exist_ok=True)
        with open(a.dump_ranks, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries):
                m = q.get("metadata", {})
                f.write(json.dumps({"qid": q["_id"], "cluster_id": m.get("cluster_id"), "lang": m.get("lang"),
                                    "min_confidence": m.get("min_confidence"), "overlaps_retrieve_anchor": m.get("overlaps_retrieve_anchor"),
                                    "gold_rank": int(gold_rank[i]), "xling_rank": int(xling_rank[i]), "same_rank": int(same_rank[i]), "n_docs": n_docs}) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    shutil.rmtree(os.path.join(a.index_root, name), ignore_errors=True)
    print(json.dumps(out["overall"]), f"[done] wrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
