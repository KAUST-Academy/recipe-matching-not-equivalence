#!/usr/bin/env python3
"""MIRB msedup (2026-09-17): the two sanity checks the
rewritten scripts/eval_mirb.py must pass on modup before msedup is scored.

  1. Reproduction. The rewritten scorer, run on modup for the untrained base
     with the same settings as job 51976966 (cuda, bf16, batch 128, 1,024
     tokens, --doc-cache), must reproduce every metric of the committed
     results/mirb/modup_qwen3-0.6b-base.json to within 1e-4.
  2. Equivalence. From the cached document embeddings and freshly encoded
     queries, the new masked path (rank_queries: one index_fill_ per chunk,
     exclusions streamed) must give the same nDCG@10 as the old per-id loop
     (copied verbatim below, exclusions held in memory).

Writes results/mirb_modup_check.json and exits non-zero on any failure, so a
msedup job may depend on it with --dependency=afterok.

  python scripts/mirb_modup_check.py            # needs a GPU, about 12 min
"""
import json, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_mirb as em  # noqa: E402
from eval_bright import calculate_retrieval_metrics  # noqa: E402

P = em.ROOT
TASK, TAG, MODEL, TOL = "modup", "qwen3-0.6b-base", "Qwen/Qwen3-Embedding-0.6B", 1e-4
COMMITTED = os.path.join(P, "results", "mirb", f"{TASK}_{TAG}.json")
CACHE = os.path.join(P, "results", "mirb_cache", f"{TASK}_{TAG}.npy")
CHECK_OUT = os.path.join(P, "results", "mirb_cache", f"check_{TASK}_{TAG}.json")
ARGS = ["--device", "cuda", "--model-dtype", "bfloat16", "--batch-size", "128", "--max-seq-length", "1024"]


def rank_old(q_emb, d_emb, q_ids, doc_ids, excluded, topk, chunk=64):
    """The ranking loop of scripts/eval_mirb.py before the msedup rewrite (no cache, per-id Python masking)."""
    doc_pos = {did: j for j, did in enumerate(doc_ids)}
    results = {}
    for s in range(0, len(q_ids), chunk):
        sims = (q_emb[s:s + chunk] @ d_emb.T).float()
        for i in range(sims.shape[0]):
            qid = q_ids[s + i]
            row = sims[i].clone()
            for x in excluded.get(qid, ()):
                j = doc_pos.get(x)
                if j is not None:
                    row[j] = -float("inf")
            k = min(topk, row.numel())
            top = torch.topk(row, k)
            results[qid] = {doc_ids[j]: float(v) for v, j in zip(top.values.tolist(), top.indices.tolist())}
    return results


def main():
    t0 = time.time()
    report = {"generated_by": "scripts/mirb_modup_check.py", "task": TASK, "model": MODEL, "tolerance": TOL}
    # ---- 1. reproduction of the committed modup result ------------------------
    cmd = [sys.executable, os.path.join(P, "scripts", "eval_mirb.py"), "--task", TASK, "--model", MODEL,
           *ARGS, "--doc-cache", CACHE, "--output", CHECK_OUT]
    print("[check 1]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=P)
    ref, new = json.load(open(COMMITTED)), json.load(open(CHECK_OUT))
    diffs = {k: abs(new["metrics"][k] - ref["metrics"][k]) for k, v in ref["metrics"].items() if isinstance(v, float)}
    pq_ref = json.load(open(COMMITTED.replace(".json", ".perquery.json")))
    pq_new = json.load(open(CHECK_OUT.replace(".json", ".perquery.json")))
    pq_diff = max(abs(pq_new[q]["ndcg_cut_10"] - pq_ref[q]["ndcg_cut_10"]) for q in pq_ref)
    report["reproduction"] = {"committed_ndcg10": ref["metrics"]["NDCG@10"], "new_ndcg10": new["metrics"]["NDCG@10"],
                              "max_abs_metric_diff": max(diffs.values()), "worst_metric": max(diffs, key=diffs.get),
                              "max_abs_perquery_ndcg10_diff": pq_diff, "n_queries": new["n_queries"],
                              "same_prompt": new["query_prompt"] == ref["query_prompt"],
                              "pass": max(diffs.values()) <= TOL and new["query_prompt"] == ref["query_prompt"]}
    print("[check 1]", json.dumps(report["reproduction"]), flush=True)
    # ---- 2. new masked path against the old per-id loop -----------------------
    doc_ids, doc_txt, q_ids, q_txt, qrels, ex_path, _ = em.load_task(TASK)
    model = em.load_model(MODEL, "cuda", "bfloat16", 1024)
    q_emb, _ = em.encode_queries(model, q_txt, 128)
    d_emb = em.encode_docs(model, doc_txt, doc_ids, 128, CACHE)
    excluded = {str(r["query-id"]): set(map(str, r.get("excluded-ids", []))) for r in em.jsonl(ex_path)}
    t1 = time.time(); r_old = rank_old(q_emb, d_emb, q_ids, doc_ids, excluded, 1000); t_old = time.time() - t1
    t1 = time.time(); r_new = em.rank_queries(q_emb, d_emb, q_ids, doc_ids, em.ExcludedIds(ex_path, q_ids), 1000)
    t_new = time.time() - t1
    m_old = calculate_retrieval_metrics(results=r_old, qrels=qrels)
    m_new = calculate_retrieval_metrics(results=r_new, qrels=qrels)
    n_diff_sets = sum(set(r_old[q]) != set(r_new[q]) for q in q_ids)
    n_diff_scores = sum(r_old[q] != r_new[q] for q in q_ids)
    report["equivalence"] = {"old_ndcg10": m_old["NDCG@10"], "new_ndcg10": m_new["NDCG@10"],
                             "all_metrics_identical": m_old == m_new,
                             "queries_with_different_top1000_sets": n_diff_sets,
                             "queries_with_different_top1000_scores": n_diff_scores,
                             "n_excluded_ids": sum(len(v) for v in excluded.values()),
                             "seconds_old_loop": round(t_old, 1), "seconds_new_path": round(t_new, 1),
                             "pass": m_old["NDCG@10"] == m_new["NDCG@10"]}
    print("[check 2]", json.dumps(report["equivalence"]), flush=True)
    report["pass"] = report["reproduction"]["pass"] and report["equivalence"]["pass"]
    report["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(P, "results", "mirb_modup_check.json"), "w"), indent=2)
    print("[done]", "PASS" if report["pass"] else "FAIL", "-> results/mirb_modup_check.json", flush=True)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
