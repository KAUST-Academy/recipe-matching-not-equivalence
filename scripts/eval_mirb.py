#!/usr/bin/env python3
"""2026-09-16: score a sentence-transformers checkpoint on one
MIRB task (Ju & Dong 2025, arXiv:2505.15585; data = the hcju/* Hugging Face
datasets downloaded to data/external/mirb/<task>/, BEIR layout with graded
qrels/test.jsonl and per-query excluded_ids.jsonl). Protocol as in MIRB's mteb
fork: documents are "title text", each query's excluded ids are dropped from
its ranking, main score nDCG@10 over the top 1000 (pytrec_eval semantics via
scripts/eval_bright.py's calculate_retrieval_metrics). Queries use the model's
"query" prompt (the setting the paper uses for every external benchmark).

  python scripts/eval_mirb.py --task modup --model models/ctrl-llm-6145/final \
      --output results/mirb/modup_ctrl-llm-6145.json

2026-09-17 (msedup, 1.35M documents and 992M excluded ids): three changes that
leave the scores and the output files unchanged.
  * --doc-cache PATH: document embeddings are saved as fp16 (.npy) with the
    document ids next to them (.ids.json) and reloaded on the next run in the
    model's dtype; bf16 values survive the fp16 round trip exactly. Ignored
    under --max-docs; refused when the cached ids differ from the corpus.
  * The per-query exclusion is one index_fill_ per chunk of 64 queries
    instead of one Python assignment per excluded id.
  * excluded_ids.jsonl is streamed in the ranking loop (ExcludedIds) instead
    of being held in memory; the file is expected in query order, and a
    byte-offset index is built in one pass if it is not.
"""
import argparse, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bright import calculate_retrieval_metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRB = os.path.join(ROOT, "data", "external", "mirb")
QID_RE = re.compile(rb'^\s*\{\s*"query-id"\s*:\s*(?:"([^"]*)"|(-?\d+))')


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class ExcludedIds:
    """Per-query excluded ids, read from excluded_ids.jsonl as the ranking asks
    for them. Rows for queries that are not scored are skipped; if a scored
    query's row turns up out of order, or a scored query has no row before the
    end of the file, a byte-offset index of the whole file is built once and
    rows are read by seek from then on. Queries without a row get []."""

    def __init__(self, path, wanted):
        self.path, self.wanted = path, set(wanted)
        self.f = open(path, "rb") if path and os.path.exists(path) else None
        self.offsets = None
        self.n_rows = self.n_ids = 0

    @staticmethod
    def _parse(line):
        r = json.loads(line)
        return str(r["query-id"]), [str(x) for x in (r.get("excluded-ids") or [])]

    def _index(self):
        t0 = time.time(); self.offsets = {}
        self.f.seek(0)
        while True:
            pos = self.f.tell(); line = self.f.readline()
            if not line:
                break
            m = QID_RE.match(line)
            qid = (m.group(1) or m.group(2)).decode() if m else str(json.loads(line)["query-id"])
            self.offsets.setdefault(qid, pos)
        print(f"[excluded] file not in query order: built an offset index of "
              f"{len(self.offsets)} rows in {time.time()-t0:.0f}s", flush=True)

    def get(self, qid):
        if self.f is None:
            return []
        if self.offsets is not None:
            off = self.offsets.get(qid)
            if off is None:
                return []
            self.f.seek(off); q, ids = self._parse(self.f.readline())
            assert q == qid
        else:
            while True:
                line = self.f.readline()
                if not line:
                    self._index(); return self.get(qid)
                q, ids = self._parse(line)
                if q == qid:
                    break
                if q in self.wanted:          # a scored query out of order
                    self._index(); return self.get(qid)
        self.n_rows += 1; self.n_ids += len(ids)
        return ids


def rank_queries(q_emb, d_emb, q_ids, doc_ids, excl, topk, chunk=64):
    """{qid: {doc_id: score}} over the top-k unexcluded documents per query.
    sims = (q @ d.T).float() exactly as before; the exclusion is one
    index_fill_ per chunk on the flattened (chunk, n_docs) score block."""
    n = d_emb.shape[0]
    doc_pos = {did: j for j, did in enumerate(doc_ids)}
    results = {}
    k = min(topk, n)
    for s in range(0, len(q_ids), chunk):
        qb = q_ids[s:s + chunk]
        sims = (q_emb[s:s + chunk] @ d_emb.T).float()
        flat = []
        for i, qid in enumerate(qb):
            base = i * n
            flat.extend(base + j for j in map(doc_pos.get, excl.get(qid)) if j is not None)
        if flat:
            idx = torch.tensor(flat, dtype=torch.int64, device=sims.device)
            sims.view(-1).index_fill_(0, idx, float("-inf"))
        top = torch.topk(sims, k, dim=1)
        vals, inds = top.values.tolist(), top.indices.tolist()
        for i, qid in enumerate(qb):
            results[qid] = {doc_ids[j]: float(v) for v, j in zip(vals[i], inds[i])}
    return results


def load_task(task, max_docs=0):
    """(doc_ids, doc_txt, q_ids, q_txt, qrels, excluded_ids_path, subsampled) for one task."""
    d = os.path.join(MIRB, task)
    corpus = jsonl(os.path.join(d, "corpus.jsonl")); queries = jsonl(os.path.join(d, "queries.jsonl"))
    qrels = {}
    for r in jsonl(os.path.join(d, "qrels", "test.jsonl")):
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(float(r["score"]))
    queries = [q for q in queries if str(q["_id"]) in qrels]
    subsampled = bool(max_docs and max_docs < len(corpus))
    if subsampled:
        gold = {cid for q in queries for cid in qrels[str(q["_id"])]}
        keep = [c for c in corpus if str(c["_id"]) in gold]
        rest = [c for c in corpus if str(c["_id"]) not in gold]
        corpus = keep + rest[: max(0, max_docs - len(keep))]
    doc_ids = [str(c["_id"]) for c in corpus]
    doc_txt = [((c.get("title") or "").strip() + " " + (c.get("text") or "").strip()).strip() for c in corpus]
    q_ids = [str(q["_id"]) for q in queries]; q_txt = [q["text"] for q in queries]
    return doc_ids, doc_txt, q_ids, q_txt, qrels, os.path.join(d, "excluded_ids.jsonl"), subsampled


def load_model(path, device=None, model_dtype="bfloat16", max_seq_length=1024):
    from sentence_transformers import SentenceTransformer
    kw = {}
    if model_dtype and model_dtype != "float32":
        kw["torch_dtype"] = getattr(torch, model_dtype)
    model = SentenceTransformer(path, device=device, model_kwargs=kw)
    model.max_seq_length = max_seq_length
    return model


def encode_queries(model, q_txt, batch_size, prompt_name="query", no_prompt=False):
    """(q_emb, prompt used or None): the model's own query prompt when it has one."""
    enc = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    if no_prompt or prompt_name not in (model.prompts or {}):
        return model.encode(q_txt, **enc), None
    return model.encode(q_txt, prompt_name=prompt_name, **enc), model.prompts[prompt_name]


def encode_docs(model, doc_txt, doc_ids, batch_size, cache=None):
    """Document embeddings, from the fp16 cache when it exists and lists exactly doc_ids."""
    ids_path = os.path.splitext(cache)[0] + ".ids.json" if cache else None
    if cache and os.path.exists(cache) and os.path.exists(ids_path):
        if json.load(open(ids_path)) != doc_ids:
            sys.exit(f"[cache] {ids_path} does not list the corpus's document ids in order; "
                     f"delete the cache or point --doc-cache elsewhere")
        arr = np.load(cache)
        d_emb = torch.from_numpy(arr).to(device=model.device, dtype=model[0].auto_model.dtype)
        print(f"[cache] loaded {cache} {tuple(arr.shape)} {arr.dtype} -> {d_emb.dtype}", flush=True)
        return d_emb
    d_emb = model.encode(doc_txt, batch_size=batch_size, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        tmp = cache + ".tmp.npy"
        np.save(tmp, d_emb.half().cpu().numpy()); os.replace(tmp, cache)
        json.dump(doc_ids, open(ids_path, "w"))
        print(f"[cache] wrote {cache} ({d_emb.shape[0]} x {d_emb.shape[1]} fp16) and {ids_path}", flush=True)
    return d_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--model-dtype", default="bfloat16")
    ap.add_argument("--query-prompt-name", default="query")
    ap.add_argument("--no-query-prompt", action="store_true")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--max-docs", type=int, default=0, help="debug: subsample corpus (keeps golds)")
    ap.add_argument("--doc-cache", default=None,
                    help="fp16 .npy of the document embeddings (+ .ids.json); reused when present")
    a = ap.parse_args()
    t0 = time.time()
    doc_ids, doc_txt, q_ids, q_txt, qrels, ex_path, subsampled = load_task(a.task, a.max_docs)
    print(f"[data] {a.task}: {len(doc_ids)} docs, {len(q_ids)} queries with qrels, "
          f"excluded ids {'streamed from ' + ex_path if os.path.exists(ex_path) else 'absent'}", flush=True)
    model = load_model(a.model, a.device, a.model_dtype, a.max_seq_length)
    q_emb, used_prompt = encode_queries(model, q_txt, a.batch_size, a.query_prompt_name, a.no_query_prompt)
    cache = a.doc_cache if a.doc_cache and not subsampled else None
    if a.doc_cache and not cache:
        print("[cache] --max-docs set: --doc-cache ignored", flush=True)
    d_emb = encode_docs(model, doc_txt, doc_ids, a.batch_size, cache)
    print(f"[encode] done in {time.time()-t0:.0f}s (prompt={used_prompt!r}, dtype={d_emb.dtype})", flush=True)
    excl = ExcludedIds(ex_path, q_ids)
    results = rank_queries(q_emb, d_emb, q_ids, doc_ids, excl, a.topk)
    print(f"[rank] done in {time.time()-t0:.0f}s ({excl.n_rows} exclusion rows, {excl.n_ids} ids applied)", flush=True)
    per_query = {}
    metrics = calculate_retrieval_metrics(results=results, qrels=qrels, per_query_sink=per_query)
    out = {"benchmark": f"MIRB/{a.task} (Ju & Dong 2025)", "model": a.model, "n_corpus": len(doc_ids),
           "n_queries": len(q_ids), "subsampled": subsampled,
           "query_prompt": used_prompt, "model_dtype": a.model_dtype, "max_seq_length": a.max_seq_length,
           "excluded_ids_applied": excl.f is not None, "metrics": metrics, "headline": {"NDCG@10": metrics["NDCG@10"]},
           "runtime_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2)
    json.dump({q: {k: float(v) for k, v in m.items()} for q, m in per_query.items()},
              open(a.output.replace(".json", ".perquery.json"), "w"))
    print(json.dumps(out["headline"]), f"[done] wrote {a.output} in {out['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
