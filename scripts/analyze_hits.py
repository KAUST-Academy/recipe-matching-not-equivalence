#!/usr/bin/env python3
"""Recipe-sensitivity audit, part 1: per-query hit accounting (analyze_hits).

Turns the suspected recipe-matching mechanism into a demonstrated one.
For a pair of models A (suspected recipe-matched, e.g. ctrl-llm-6145) and
B (control, e.g. ctrl-cas-6145) evaluated on one MathNet-Retrieve tier, this
script:

  1. obtains per-query gold ranks + top-10 lists for both models, either from
       * --ranks-a/--ranks-b   JSONL dumps written by
                               `eval_retrieve.py --dump-ranks` (GPU path), or
       * --emb-docs-*/--emb-queries-*  cached full-set embedding .npz files
                               (written by eval_retrieve.py --emb-cache-dir;
                               CPU-only path, ~2-4 min per model at 15,000 x
                               117,088 x 1024 on 8 BLAS threads).
     In the embedding mode the recomputed dumps are saved (same schema as
     --dump-ranks) so later runs and the paper can reuse them.

  2. defines the RECIPE-HIT set H = queries where A ranks the gold top-1 but
     B does not, and the complement set N (all other queries), and compares
     surface statistics of (query text vs gold doc text) between H and N:
       * token Jaccard (unicode word tokens, lowercased)
       * LaTeX-span Jaccard ($...$ / $$...$$ spans, whitespace-normalized)
       * length ratio  len(gold)/len(query)  in characters
       * template-phrase count in the gold doc (recipe-fingerprint n-grams
         mined by scripts/style_probe.py --stage fit, or a built-in fallback)
     with Mann-Whitney U tests and AUC effect sizes.

  3. reports full hit accounting: per-model rank histograms, the 2x2 top-1
     contingency table A vs B, and (if style scores exist) the mean
     synthetic-style score of gold docs in H vs N.

Usage (CPU, from cached embeddings -- the ctrl-llm vs ctrl-cas hard-tier audit):
  python scripts/analyze_hits.py --tier hard \
      --label-a ctrl-llm-6145 --label-b ctrl-cas-6145 \
      --emb-docs-a    .emb_cache/docs_models__ctrl-llm-6145__final_*.npz \
      --emb-queries-a .emb_cache/queries_models__ctrl-llm-6145__final_*.npz \
      --emb-docs-b    .emb_cache/docs_models__ctrl-cas-6145__final_*.npz \
      --emb-queries-b .emb_cache/queries_models__ctrl-cas-6145__final_*.npz \
      --template-phrases results/style_probe_features.json \
      --output results/analyze_hits_ctrl-llm_vs_ctrl-cas_hard.json

Usage (from eval_retrieve.py --dump-ranks output):
  python scripts/analyze_hits.py --tier hard --label-a A --label-b B \
      --ranks-a results/ranks_A_hard.jsonl --ranks-b results/ranks_B_hard.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "retrieve")

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
LATEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)

# Fallback only -- prefer the mined list from style_probe.py --stage fit.
BUILTIN_TEMPLATE_PHRASES = [
    "determine", "prove that", "show that", "find the value",
    "let ", "consider ", "suppose ", "denote", "such that",
    "compute", "positive integer", "real number",
]


def word_tokens(text):
    return {w.lower() for w in WORD_RE.findall(text)}


def latex_spans(text):
    spans = set()
    for m in LATEX_RE.finditer(text):
        s = re.sub(r"\s+", "", m.group(1) or m.group(2) or "")
        if s:
            spans.add(s)
    return spans


def jaccard(a, b):
    if not a and not b:
        return None
    u = len(a | b)
    return len(a & b) / u if u else None


def load_jsonl_map(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"]
    return out


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels.setdefault(qid, set()).add(cid)
    return qrels


def resolve_glob(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        sys.exit(f"[error] no file matches {pattern!r}")
    if len(hits) > 1:
        print(f"[warn] {len(hits)} files match {pattern!r}; using newest "
              f"{hits[-1]}", flush=True)
    return hits[-1]


def load_npz(pattern):
    path = resolve_glob(pattern)
    z = np.load(path, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    emb = z["emb"].astype(np.float32)
    print(f"[npz] {path}: {emb.shape[0]} x {emb.shape[1]}", flush=True)
    return ids, emb


def ranks_from_embeddings(doc_pat, query_pat, corpus_ids, qrels, label,
                          save_path=None, chunk=512):
    """Exact gold rank + top-10 ids per query, recomputed on CPU."""
    d_ids, d_emb = load_npz(doc_pat)
    if d_ids != corpus_ids:
        if set(d_ids) != set(corpus_ids):
            sys.exit(f"[error] doc embedding ids for {label} do not cover the "
                     f"corpus ({len(set(corpus_ids) - set(d_ids))} missing)")
        order = {c: i for i, c in enumerate(d_ids)}
        d_emb = d_emb[[order[c] for c in corpus_ids]]
        print(f"[npz] reordered doc embeddings to corpus order", flush=True)
    q_ids, q_emb = load_npz(query_pat)
    keep = [i for i, q in enumerate(q_ids) if q in qrels]
    q_ids = [q_ids[i] for i in keep]
    q_emb = q_emb[keep]
    corpus_pos = {c: i for i, c in enumerate(corpus_ids)}
    rows = {}
    t0 = time.time()
    for s in range(0, len(q_ids), chunk):
        sims = q_emb[s:s + chunk] @ d_emb.T
        top = np.argpartition(-sims, 9, axis=1)[:, :10]
        top = np.take_along_axis(
            top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1),
            axis=1)
        for i, qid in enumerate(q_ids[s:s + chunk]):
            best_rank, best_sim = None, None
            for g in qrels[qid]:
                if g not in corpus_pos:
                    continue
                gs = float(sims[i, corpus_pos[g]])
                r = int((sims[i] > gs).sum())
                if best_rank is None or r < best_rank:
                    best_rank, best_sim = r, gs
            rows[qid] = {"qid": qid, "gold_rank": best_rank,
                         "gold_sim": best_sim,
                         "top10_ids": [corpus_ids[d] for d in top[i]]}
        if (s // chunk) % 8 == 0:
            print(f"[{label}] {min(s + chunk, len(q_ids))}/{len(q_ids)} "
                  f"queries ({time.time() - t0:.0f}s)", flush=True)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for qid in q_ids:
                f.write(json.dumps(rows[qid], ensure_ascii=False) + "\n")
        print(f"[{label}] saved rank dump to {save_path}", flush=True)
    return rows


def load_ranks(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_template_phrases(path):
    if not path:
        return BUILTIN_TEMPLATE_PHRASES, "builtin"
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(p).lower() for p in obj], path
    if "top_synthetic_ngrams" in obj:
        return [d["ngram"].lower() for d in obj["top_synthetic_ngrams"]], path
    if "phrases" in obj:
        return [str(p).lower() for p in obj["phrases"]], path
    sys.exit(f"[error] unrecognized template-phrase file format: {path}")


def rank_hist(ranks):
    arr = np.array([r if r is not None else 10 ** 9 for r in ranks])
    return {"top1": int((arr < 1).sum()), "top5": int((arr < 5).sum()),
            "top10": int((arr < 10).sum()), "beyond10": int((arr >= 10).sum()),
            "recall@1": round(100 * float((arr < 1).mean()), 2),
            "recall@5": round(100 * float((arr < 5).mean()), 2),
            "recall@10": round(100 * float((arr < 10).mean()), 2)}


def group_compare(name, vals_h, vals_n):
    from scipy.stats import mannwhitneyu
    vh = np.array([v for v in vals_h if v is not None], dtype=np.float64)
    vn = np.array([v for v in vals_n if v is not None], dtype=np.float64)
    out = {"stat": name, "n_hit": int(vh.size), "n_rest": int(vn.size),
           "hit_mean": round(float(vh.mean()), 4) if vh.size else None,
           "hit_median": round(float(np.median(vh)), 4) if vh.size else None,
           "rest_mean": round(float(vn.mean()), 4) if vn.size else None,
           "rest_median": round(float(np.median(vn)), 4) if vn.size else None}
    if vh.size and vn.size:
        u, p = mannwhitneyu(vh, vn, alternative="two-sided")
        out["auc_hit_vs_rest"] = round(float(u / (vh.size * vn.size)), 4)
        out["mannwhitney_p"] = float(f"{p:.3g}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], default="hard")
    ap.add_argument("--data-dir", default=None,
                    help="BEIR-style dir override (corpus/queries/qrels)")
    ap.add_argument("--label-a", default="model-a",
                    help="name of the suspected recipe-matched model")
    ap.add_argument("--label-b", default="model-b", help="name of the control")
    ap.add_argument("--ranks-a", default=None,
                    help="JSONL from eval_retrieve.py --dump-ranks for model A")
    ap.add_argument("--ranks-b", default=None)
    ap.add_argument("--emb-docs-a", default=None,
                    help="doc-embedding .npz (glob ok) for model A; used when "
                         "--ranks-a is absent")
    ap.add_argument("--emb-queries-a", default=None)
    ap.add_argument("--emb-docs-b", default=None)
    ap.add_argument("--emb-queries-b", default=None)
    ap.add_argument("--template-phrases", default=None,
                    help="JSON with mined recipe n-grams "
                         "(results/style_probe_features.json)")
    ap.add_argument("--style-scores", default=None,
                    help="optional results/style_scores_retrieve.npz -- adds "
                         "gold-doc synthetic-style score to the H vs N table")
    ap.add_argument("--per-query-out", default=None,
                    help="optional JSONL of per-query stats (for plots)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tier_dir = args.data_dir or os.path.join(DATA_DIR, args.tier)
    corpus = load_jsonl_map(os.path.join(tier_dir, "corpus.jsonl"))
    queries = load_jsonl_map(os.path.join(tier_dir, "queries.jsonl"))
    qrels = load_qrels(os.path.join(tier_dir, "qrels", "test.tsv"))
    corpus_ids = list(corpus)
    print(f"[data] tier={args.tier} corpus={len(corpus)} queries={len(queries)} "
          f"qrels={len(qrels)}", flush=True)

    def get_ranks(ranks_path, doc_pat, query_pat, label):
        if ranks_path:
            return load_ranks(ranks_path)
        if not (doc_pat and query_pat):
            sys.exit(f"[error] need --ranks-{label[-1]} or both emb patterns "
                     f"for model {label}")
        save = os.path.join(PROJECT_ROOT, "results",
                            f"ranks_{label}_{args.tier}.jsonl")
        return ranks_from_embeddings(doc_pat, query_pat, corpus_ids, qrels,
                                     label, save_path=save)

    ranks_a = get_ranks(args.ranks_a, args.emb_docs_a, args.emb_queries_a,
                        args.label_a)
    ranks_b = get_ranks(args.ranks_b, args.emb_docs_b, args.emb_queries_b,
                        args.label_b)
    shared = [q for q in ranks_a if q in ranks_b and q in qrels]
    print(f"[align] {len(shared)} shared queries", flush=True)

    phrases, phrase_src = load_template_phrases(args.template_phrases)

    style = None
    if args.style_scores and os.path.exists(args.style_scores):
        z = np.load(args.style_scores, allow_pickle=False)
        style = {str(i): float(s) for i, s in zip(z["doc_ids"], z["doc_scores"])}

    # ---------------- per-query accounting + surface stats ----------------
    a_top1 = {q for q in shared if ranks_a[q]["gold_rank"] == 0}
    b_top1 = {q for q in shared if ranks_b[q]["gold_rank"] == 0}
    hits = sorted(a_top1 - b_top1)          # H: A top-1, B not
    rest = sorted(set(shared) - set(hits))  # N: everything else
    contingency = {
        "A_top1_and_B_top1": len(a_top1 & b_top1),
        "A_top1_only": len(hits),
        "B_top1_only": len(b_top1 - a_top1),
        "neither_top1": len(shared) - len(a_top1 | b_top1),
    }

    def surface(qid):
        gold_id = next(iter(qrels[qid]))
        q_text, g_text = queries[qid], corpus.get(gold_id, "")
        qt, gt = word_tokens(q_text), word_tokens(g_text)
        gl = g_text.lower()
        n_words = max(1, len(WORD_RE.findall(g_text)))
        tmpl = sum(gl.count(p) for p in phrases)
        row = {
            "qid": qid,
            "token_jaccard": jaccard(qt, gt),
            "latex_jaccard": jaccard(latex_spans(q_text), latex_spans(g_text)),
            "len_ratio_gold_over_query": (len(g_text) / len(q_text)
                                          if len(q_text) else None),
            "template_hits_per_100w": round(100.0 * tmpl / n_words, 4),
            "gold_style_score": style.get(gold_id) if style else None,
            "gold_rank_a": ranks_a[qid]["gold_rank"],
            "gold_rank_b": ranks_b[qid]["gold_rank"],
        }
        return row

    t0 = time.time()
    stats_h = [surface(q) for q in hits]
    stats_n = [surface(q) for q in rest]
    print(f"[surface] computed stats for {len(shared)} queries "
          f"({time.time() - t0:.0f}s)", flush=True)

    stat_names = ["token_jaccard", "latex_jaccard",
                  "len_ratio_gold_over_query", "template_hits_per_100w"]
    if style:
        stat_names.append("gold_style_score")
    comparisons = [group_compare(n, [r[n] for r in stats_h],
                                 [r[n] for r in stats_n]) for n in stat_names]

    result = {
        "audit": "recipe-sensitivity audit -- per-query hit accounting",
        "tier": args.tier,
        "model_a": args.label_a,
        "model_b": args.label_b,
        "n_queries": len(shared),
        "rank_hist_a": rank_hist([ranks_a[q]["gold_rank"] for q in shared]),
        "rank_hist_b": rank_hist([ranks_b[q]["gold_rank"] for q in shared]),
        "top1_contingency": contingency,
        "hit_set_definition": f"{args.label_a} gold_rank==0 AND "
                              f"{args.label_b} gold_rank>0",
        "n_recipe_hits": len(hits),
        "template_phrase_source": phrase_src,
        "n_template_phrases": len(phrases),
        "hit_vs_rest": comparisons,
    }

    if args.per_query_out:
        with open(args.per_query_out, "w", encoding="utf-8") as f:
            for r in stats_h:
                f.write(json.dumps({**r, "group": "hit"}, ensure_ascii=False) + "\n")
            for r in stats_n:
                f.write(json.dumps({**r, "group": "rest"}, ensure_ascii=False) + "\n")
        result["per_query_out"] = args.per_query_out

    out = args.output or os.path.join(
        PROJECT_ROOT, "results",
        f"analyze_hits_{args.label_a}_vs_{args.label_b}_{args.tier}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
