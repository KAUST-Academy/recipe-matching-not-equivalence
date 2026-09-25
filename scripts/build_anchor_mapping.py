"""Build the contamination-exclusion mapping between MathNet-Retrieve anchors and the public corpus.

Usage:  python scripts/build_anchor_mapping.py
Output: anchor_to_corpus_mapping.json

Matches every MathNet-Retrieve anchor query (easy tier; queries are identical
across tiers) against the public corpus by whitespace-normalized, lowercased
exact text. Corpus ids listed in "exclude_corpus_ids" must be excluded from any
retriever training data. Result on 2026-07-29: 8,761/15,000 anchors matched
(58.4%), 8,698 distinct corpus ids to exclude.
"""
import json
import re
import urllib.request

import duckdb

PROJECT = "/ibex/user/habiam0b/MathNet_Follow_Up"
QUERIES_URL = ("https://huggingface.co/datasets/ShadenA/MathNet-Retrieve/"
               "resolve/main/easy/queries.jsonl")
OUT = f"{PROJECT}/anchor_to_corpus_mapping.json"


def norm(t):
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


queries = [json.loads(l) for l in
           urllib.request.urlopen(QUERIES_URL).read().decode().splitlines()]
print("anchor queries:", len(queries))

con = duckdb.connect()
corpus = con.execute(
    f"SELECT id, problem_markdown FROM '{PROJECT}/data/mathnet_corpus.parquet'"
).fetchall()

text2pid = {}
for pid, t in corpus:
    text2pid.setdefault(norm(t), pid)

mapping, unmatched = [], []
for q in queries:
    pid = text2pid.get(norm(q["text"]))
    if pid:
        mapping.append({"anchor_id": q["_id"], "corpus_id": pid})
    else:
        unmatched.append(q["_id"])

json.dump({
    "description": ("Exact-text matches between MathNet-Retrieve anchor queries and the "
                    "public ShadenA/MathNet corpus (config 'all'). Corpus IDs in "
                    "exclude_corpus_ids MUST be excluded from retriever training data. "
                    "Matching: whitespace-normalized, lowercased exact text."),
    "generated": "2026-07-29",
    "n_anchors": len(queries), "n_matched": len(mapping), "n_unmatched": len(unmatched),
    "exclude_corpus_ids": sorted({m["corpus_id"] for m in mapping}),
    "mapping": mapping,
}, open(OUT, "w"), indent=1)
print(f"saved {OUT}: {len(mapping)} matched, {len(unmatched)} unmatched")
