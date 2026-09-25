#!/usr/bin/env python3
"""E-R5 (2026-09-03): a SAME-LANGUAGE organic-duplicate
evaluation set, built from the monolingual duplicate clusters that the
cross-lingual set (data/crosslingual_eval) deliberately skipped.

Besides the cross-language pairs, the mined duplicate graph holds clusters
whose members share a language. Retrieval within one language takes language
transfer out of the retention measurement, and the English queries give an
English-only reading as well; one construction serves both.

Source: data/crosslingual_eval/clusters.json (661 transitive clusters over the
754 verified mined pairs; member languages from the mining pipeline's
stopword lang-ID). A cluster is MONOLINGUAL when it has >= 2 members of one
known language and no member of another known language (members with
lang-ID "unknown" stay golds only, as in the cross-lingual set). Every
known-language member of such a cluster is a query; its golds are the other
members. Retrieval runs against the same full 27,817-problem corpus with
mandatory self-masking (scripts/eval_crosslingual.py --eval-dir).

Two flags decide the slices the verdict script reads:
  exact_text_cluster   any supporting pair was mined by the exact_text rule
                       (near-byte-identical reprints; a ceiling check, not a
                       paraphrase test). The PRIMARY slice excludes them.
  clean_of_training    no member of the cluster is a source problem in ANY
                       training pair file of the campaign (ctrl-LLM/D1, the
                       CAS file, D2, D3, D4, the regenerated clean LLM file).
                       The trainer's eval gate removed only the 779 ids that
                       are queries or golds of the CROSS-LINGUAL set, so
                       monolingual-cluster members could be trained on; the
                       PRIMARY slice is restricted to clean clusters, and the
                       leaked ones are reported separately.
Registered readout: scripts/samelang_eval.slurm header (P-R5), evaluated in
code by scripts/samelang_verdict.py.
"""
import json
import os
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XL_DIR = os.path.join(PROJECT, "data", "crosslingual_eval")
OUT_DIR = os.path.join(PROJECT, "data", "samelang_eval")
TRAIN_FILES = {
    "ctrl-LLM (D1)": "data/llm_pairs/pairs.jsonl",
    "CAS file": "data/cas_pairs/pairs.jsonl",
    "D2": "data/llm_pairs_paraphrase/pairs.jsonl",
    "D3": "data/llm_pairs_style/pairs.jsonl",
    "D4": "data/llm_pairs_unrelated/pairs.jsonl",
    "regenerated clean LLM": "data/llm_pairs_cleanfull/pairs.jsonl",
}
CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def sources(rel):
    with open(os.path.join(PROJECT, rel), encoding="utf-8") as f:
        return {json.loads(l)["source_id"] for l in f}


def main():
    with open(os.path.join(XL_DIR, "clusters.json"), encoding="utf-8") as f:
        clusters = json.load(f)["clusters"]
    corpus = {}
    with open(os.path.join(XL_DIR, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["_id"]] = r["text"]
    with open(os.path.join(PROJECT, "anchor_to_corpus_mapping.json"),
              encoding="utf-8") as f:
        retrieve_anchor_ids = set(json.load(f)["exclude_corpus_ids"])
    trained = {k: sources(v) for k, v in TRAIN_FILES.items()}

    queries, qrels, clusters_out = [], [], []
    for c in clusters:
        langs = [m["lang"] for m in c["members"] if m["lang"] != "unknown"]
        if len(langs) < 2 or len(set(langs)) != 1:
            continue                       # multilingual or unknown-only
        lang = langs[0]
        ids = [m["id"] for m in c["members"]]
        lang_of = {m["id"]: m["lang"] for m in c["members"]}
        methods = sorted({p["method"] for p in c["pairs"]})
        exact = "exact_text" in methods
        trained_in = sorted(k for k, s in trained.items() if any(i in s for i in ids))
        min_conf = max((p["confidence"] for p in c["pairs"]),
                       key=lambda x: CONF_RANK[x])
        rec = dict(c)
        rec.update({"lang": lang, "exact_text_cluster": exact,
                    "trained_in": trained_in, "clean_of_training": not trained_in})
        clusters_out.append(rec)
        for q in sorted(ids):
            if lang_of[q] == "unknown":
                continue
            golds = sorted(i for i in ids if i != q)
            direct = [{"other_id": (p["id_b"] if p["id_a"] == q else p["id_a"]),
                       "method": p["method"], "confidence": p["confidence"]}
                      for p in c["pairs"] if q in (p["id_a"], p["id_b"])]
            queries.append({
                "_id": q,
                "text": corpus[q],
                "metadata": {
                    "lang": lang,
                    "cluster_id": c["cluster_id"],
                    "cluster_size": len(ids),
                    "doc_lang": lang,
                    "gold_ids": golds,
                    "gold_langs": [lang_of[g] for g in golds],
                    "n_crosslingual_golds": 0,
                    "n_samelang_golds": sum(lang_of[g] == lang for g in golds),
                    "supporting_pairs": direct,
                    "methods": methods,
                    "min_confidence": min_conf,
                    "exact_text_cluster": exact,
                    "trained_in": trained_in,
                    "clean_of_training": not trained_in,
                    "query_in_retrieve_anchors": q in retrieve_anchor_ids,
                    "n_golds_in_retrieve_anchors": sum(g in retrieve_anchor_ids
                                                       for g in golds),
                    "overlaps_retrieve_anchor": (q in retrieve_anchor_ids or any(
                        g in retrieve_anchor_ids for g in golds)),
                },
            })
            qrels.extend((q, g) for g in golds)

    queries.sort(key=lambda r: r["_id"])
    os.makedirs(os.path.join(OUT_DIR, "qrels"), exist_ok=True)
    # The corpus is the cross-lingual set's file; this directory ships no copy
    # and no symlink (a symlink broke Overleaf's GitHub sync, 2026-09-04).
    # scripts/eval_crosslingual.py falls back to data/crosslingual_eval/corpus.jsonl.
    stale_link = os.path.join(OUT_DIR, "corpus.jsonl")
    if os.path.islink(stale_link):
        os.remove(stale_link)
    with open(os.path.join(OUT_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, g in qrels:
            f.write(f"{q}\t{g}\t1\n")
    with open(os.path.join(OUT_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({"n_clusters": len(clusters_out), "clusters": clusters_out},
                  f, indent=1, ensure_ascii=False)

    def n_q(pred):
        return sum(1 for r in queries if pred(r["metadata"]))

    def n_c(pred):
        return sum(1 for c in clusters_out if pred(c))

    stats = {
        "generated": "2026-09-03",
        "built_by": "scripts/build_samelang_eval.py",
        "n_clusters_monolingual": len(clusters_out),
        "n_queries": len(queries),
        "n_qrel_rows": len(qrels),
        "n_corpus_docs": len(corpus),
        "clusters_exact_text": n_c(lambda c: c["exact_text_cluster"]),
        "clusters_non_exact": n_c(lambda c: not c["exact_text_cluster"]),
        "clusters_non_exact_clean_of_training": n_c(
            lambda c: not c["exact_text_cluster"] and c["clean_of_training"]),
        "clusters_non_exact_leaked": n_c(
            lambda c: not c["exact_text_cluster"] and not c["clean_of_training"]),
        "queries_primary_slice_non_exact_clean": n_q(
            lambda m: not m["exact_text_cluster"] and m["clean_of_training"]),
        "queries_primary_slice_english": n_q(
            lambda m: not m["exact_text_cluster"] and m["clean_of_training"]
            and m["lang"] == "en"),
        "queries_non_exact_leaked": n_q(
            lambda m: not m["exact_text_cluster"] and not m["clean_of_training"]),
        "queries_exact_text": n_q(lambda m: m["exact_text_cluster"]),
        "queries_per_language": dict(Counter(
            r["metadata"]["lang"] for r in queries).most_common()),
        "primary_slice_per_language": dict(Counter(
            r["metadata"]["lang"] for r in queries
            if not r["metadata"]["exact_text_cluster"]
            and r["metadata"]["clean_of_training"]).most_common()),
        "primary_slice_methods": dict(Counter(
            m for r in queries for m in r["metadata"]["methods"]
            if not r["metadata"]["exact_text_cluster"]
            and r["metadata"]["clean_of_training"]).most_common()),
        "leaked_clusters_by_training_file": dict(Counter(
            k for c in clusters_out if not c["exact_text_cluster"]
            for k in c["trained_in"]).most_common()),
        "queries_overlapping_retrieve_anchors": n_q(
            lambda m: m["overlaps_retrieve_anchor"]),
        "training_files_checked": TRAIN_FILES,
    }
    with open(os.path.join(OUT_DIR, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(**stats))
    print(json.dumps(stats, indent=2))
    print(f"[done] wrote {OUT_DIR}", flush=True)


README = """# Same-language organic-duplicate evaluation set (E-R5, 2026-09-03)

Companion to `data/crosslingual_eval/` (the cross-lingual real-duplicate
probe). Built by `scripts/build_samelang_eval.py` from the {n_clusters_monolingual}
MONOLINGUAL duplicate clusters of `data/crosslingual_eval/clusters.json`,
which the cross-lingual set skipped by design. Every known-language member
of such a cluster is a query and the other members are its golds:
{n_queries} queries, {n_qrel_rows} qrels, against the full {n_corpus_docs}-problem
corpus (the cross-lingual set's `data/crosslingual_eval/corpus.jsonl`; this
directory ships no copy, and `scripts/eval_crosslingual.py` falls back to it).
Self-masking is mandatory, exactly as for the cross-lingual set.

Two per-query flags (in `metadata`) define the slices:

* `exact_text_cluster` -- {clusters_exact_text} clusters were mined by the
  exact-text rule (near-byte-identical reprints): a ceiling check, not a
  paraphrase test. The {clusters_non_exact} non-exact clusters (formula-,
  near-text- or answer-mined) are the informative ones.
* `clean_of_training` -- the trainer's eval gate removed only the 779 ids
  that are queries or golds of the CROSS-LINGUAL set, so members of
  monolingual clusters could be trained on. {clusters_non_exact_clean_of_training}
  of the non-exact clusters have no member in ANY training pair file of the
  campaign; {clusters_non_exact_leaked} have at least one (`trained_in` lists the files).

PRIMARY slice = non-exact AND clean of training: {queries_primary_slice_non_exact_clean}
queries ({queries_primary_slice_english} English). Leaked and exact-text
queries are reported separately, never mixed in.

Metric: R@1 of the best-ranked gold (`overall_any_gold`) and of the best
gold in the query's own language (`same_language_gold`), which coincide
except where a cluster carries a member with unknown language. Evaluate
with `scripts/eval_crosslingual.py --eval-dir data/samelang_eval`; the
registered readout is in `scripts/samelang_eval.slurm` and is computed by
`scripts/samelang_verdict.py`.

License: derived from the public MathNet corpus (CC-BY-4.0); likewise CC-BY-4.0.
"""

if __name__ == "__main__":
    main()
