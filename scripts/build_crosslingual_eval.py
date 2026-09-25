#!/usr/bin/env python3
"""
Build the InvarEmbed cross-lingual duplicate-retrieval eval set (Task B).

Packages the verified duplicate pairs mined by scripts/mine_duplicates.py
(results/duplicate_candidates.jsonl, 754 pairs / 1,377 problems / 661
transitive clusters) into a BEIR-style benchmark at data/crosslingual_eval/:

  corpus.jsonl    ALL 27,817 public MathNet corpus problems ({_id, title, text})
                  so retrieval runs against the full realistic corpus.
  queries.jsonl   one row per query problem ({_id, text, metadata}). Queries
                  are the NON-document-language members of every multilingual
                  duplicate cluster (see QUERY SELECTION below).
  qrels/test.tsv  query-id -> every OTHER member of its duplicate cluster
                  (transitively merged), score 1. A query can have >1 gold.
  clusters.json   full provenance: every cluster with member langs and the
                  underlying mined pairs (rule/method + confidence kept).
  leakage_exclude_ids.json
                  every corpus id that appears as a query or gold here.
                  ANY future training data for InvarEmbed must exclude these
                  ids (in addition to anchor_to_corpus_mapping.json's
                  exclude_corpus_ids) or this eval is contaminated.
  build_summary.json  construction stats (also printed).

QUERY SELECTION (cluster-level generalization of the task rule):
  * Merge the 754 pairs into transitive clusters (union-find).
  * Member language = the mining pipeline's stopword-based lang-ID (lang_a/
    lang_b in duplicate_candidates.jsonl; verified conflict-free per id).
  * Document language L_doc of a cluster = 'en' if any member is English,
    else the member language that is MOST FREQUENT corpus-wide (so the query
    side is the rarer language, per the task rule).
  * Queries = members with a KNOWN language != L_doc. Clusters whose known
    languages are all identical (monolingual clusters, incl. en-en) yield no
    queries; members with lang-ID 'unknown' are never queries but remain
    valid golds (they are verified duplicates).
  * Golds of a query = ALL other members of its cluster, any language.
    By construction every query has >=1 gold in a language different from
    its own (the L_doc member(s)).

EVAL-INTEGRITY decisions (documented, not silently applied):
  * Pairs touching MathNet-Retrieve anchors are NOT dropped. The exclusion
    list in anchor_to_corpus_mapping.json protects MathNet-Retrieve (a
    *separate* eval) from TRAIN-side leakage; this set is itself an eval, so
    overlap is a reporting concern, not leakage. Each query carries an
    `overlaps_retrieve_anchor` flag (query or any gold in the exclusion
    list) so results can be sliced. The real leakage risk is the training
    side of *this* eval -> leakage_exclude_ids.json.
  * Self-match: every query problem also exists verbatim in corpus.jsonl
    under the same _id (the corpus is the full 27,817 so other queries'
    golds stay intact). The query's own document is NEVER a gold in qrels,
    and consumers MUST mask the corpus doc whose _id equals the query _id at
    ranking time (scripts/eval_crosslingual.py does this automatically).
    We chose masking over deleting query docs from corpus.jsonl because in
    multi-query clusters a query is another query's gold document.

Usage (CPU, seconds):
  python scripts/build_crosslingual_eval.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_JSONL = os.path.join(PROJECT_ROOT, "results", "duplicate_candidates.jsonl")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "results", "duplicate_mining_summary.json")
MAPPING_JSON = os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json")
PARQUET = os.path.join(PROJECT_ROOT, "data", "mathnet_corpus.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "crosslingual_eval")


def main():
    # ---------------- inputs ----------------
    pairs = [json.loads(l) for l in open(PAIRS_JSONL, encoding="utf-8")]
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        lang_freq = json.load(f)["language_distribution"]  # corpus-wide lang-ID counts
    with open(MAPPING_JSON, encoding="utf-8") as f:
        exclude_ids = set(json.load(f)["exclude_corpus_ids"])

    import duckdb
    rows = duckdb.sql(
        f"SELECT id, problem_markdown, country, competition FROM '{PARQUET}'"
    ).fetchall()
    corpus = {r[0]: {"text": r[1], "country": r[2], "competition": r[3]} for r in rows}
    print(f"[input] {len(pairs)} pairs, corpus {len(corpus)} problems", flush=True)

    # ---------------- member languages (conflict-checked) ----------------
    lang = {}
    for p in pairs:
        for idk, lk in (("id_a", "lang_a"), ("id_b", "lang_b")):
            i, l = p[idk], p[lk]
            assert lang.get(i, l) == l, f"lang conflict for {i}: {lang[i]} vs {l}"
            lang[i] = l
    missing = [i for i in lang if i not in corpus]
    assert not missing, f"{len(missing)} pair ids missing from corpus: {missing[:5]}"

    # ---------------- transitive clusters (union-find) ----------------
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p["id_a"]), find(p["id_b"])
        if ra != rb:
            parent[ra] = rb

    members_of = defaultdict(set)
    for i in lang:
        members_of[find(i)].add(i)
    clusters = sorted(members_of.values(), key=lambda c: min(c))
    pairs_of = defaultdict(list)
    for p in pairs:
        pairs_of[find(p["id_a"])].append(p)

    # ---------------- query selection ----------------
    queries, qrels, clusters_out = [], [], []
    n_mono = 0
    for cl in clusters:
        cid = "cl_" + min(cl)
        cpairs = pairs_of[find(next(iter(cl)))]
        known = {lang[i] for i in cl} - {"unknown"}
        cluster_rec = {
            "cluster_id": cid,
            "members": sorted([{"id": i, "lang": lang[i]} for i in cl],
                              key=lambda m: m["id"]),
            "pairs": [{"id_a": p["id_a"], "id_b": p["id_b"], "method": p["rule"],
                       "confidence": p["confidence"], "cross_language": p["cross_language"]}
                      for p in cpairs],
        }
        if len(known) <= 1:  # monolingual (or unknown-only) cluster -> no queries
            n_mono += 1
            cluster_rec["l_doc"] = None
            clusters_out.append(cluster_rec)
            continue
        l_doc = "en" if "en" in known else max(known, key=lambda l: lang_freq.get(l, 0))
        cluster_rec["l_doc"] = l_doc
        clusters_out.append(cluster_rec)
        for q in sorted(cl):
            if lang[q] == l_doc or lang[q] == "unknown":
                continue
            golds = sorted(cl - {q})
            direct = [{"other_id": (p["id_b"] if p["id_a"] == q else p["id_a"]),
                       "method": p["rule"], "confidence": p["confidence"]}
                      for p in cpairs if q in (p["id_a"], p["id_b"])]
            conf_rank = {"high": 0, "medium": 1, "low": 2}
            queries.append({
                "_id": q,
                "text": corpus[q]["text"],
                "metadata": {
                    "lang": lang[q],
                    "cluster_id": cid,
                    "cluster_size": len(cl),
                    "doc_lang": l_doc,
                    "gold_ids": golds,
                    "gold_langs": [lang[g] for g in golds],
                    "n_crosslingual_golds": sum(lang[g] != lang[q] for g in golds),
                    "supporting_pairs": direct,
                    "methods": sorted({p["rule"] for p in cpairs}),
                    "min_confidence": max((p["confidence"] for p in cpairs),
                                          key=lambda c: conf_rank[c]),
                    "country": corpus[q]["country"],
                    "competition": corpus[q]["competition"],
                    "query_in_retrieve_anchors": q in exclude_ids,
                    "n_golds_in_retrieve_anchors": sum(g in exclude_ids for g in golds),
                    "overlaps_retrieve_anchor": (q in exclude_ids
                                                 or any(g in exclude_ids for g in golds)),
                },
            })
            qrels.extend((q, g) for g in golds)

    queries.sort(key=lambda r: r["_id"])
    used_ids = sorted({r["_id"] for r in queries}
                      | {g for r in queries for g in r["metadata"]["gold_ids"]})

    # ---------------- write ----------------
    os.makedirs(os.path.join(OUT_DIR, "qrels"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for cid_ in sorted(corpus):
            f.write(json.dumps({"_id": cid_, "title": "",
                                "text": corpus[cid_]["text"]}, ensure_ascii=False) + "\n")
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
    with open(os.path.join(OUT_DIR, "leakage_exclude_ids.json"), "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Corpus ids used as queries or golds in the cross-lingual eval. "
                     "InvarEmbed training data MUST exclude all of them (on top of "
                     "anchor_to_corpus_mapping.json exclude_corpus_ids). The "
                     "conservative option additionally excludes every mined-duplicate "
                     "member (all_pair_member_ids)."),
            "n_eval_member_ids": len(used_ids),
            "eval_member_ids": used_ids,
            "n_all_pair_member_ids": len(lang),
            "all_pair_member_ids": sorted(lang),
        }, f, indent=1)

    # ---------------- stats ----------------
    lang_counts = Counter(r["metadata"]["lang"] for r in queries)
    stats = {
        "generated": "2026-07-29",
        "source_pairs": len(pairs),
        "unique_pair_members": len(lang),
        "n_clusters_total": len(clusters),
        "n_clusters_monolingual_skipped": n_mono,
        "n_clusters_multilingual": len(clusters) - n_mono,
        "n_queries": len(queries),
        "n_qrel_rows": len(qrels),
        "n_corpus_docs": len(corpus),
        "queries_per_language": dict(lang_counts.most_common()),
        "queries_with_multiple_golds": sum(
            1 for r in queries if len(r["metadata"]["gold_ids"]) > 1),
        "queries_overlapping_retrieve_anchors": sum(
            r["metadata"]["overlaps_retrieve_anchor"] for r in queries),
        "queries_min_confidence_medium": sum(
            r["metadata"]["min_confidence"] == "medium" for r in queries),
        "doc_lang_distribution": dict(Counter(
            r["metadata"]["doc_lang"] for r in queries).most_common()),
        "n_eval_member_ids": len(used_ids),
        "eval_member_ids_in_retrieve_exclusion": sum(
            1 for i in used_ids if i in exclude_ids),
    }
    with open(os.path.join(OUT_DIR, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"[done] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
