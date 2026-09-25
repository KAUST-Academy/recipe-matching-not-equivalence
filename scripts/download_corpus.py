"""Download the public MathNet corpus (text + metadata, no images) to a local parquet.

Usage:  python scripts/download_corpus.py
Output: data/mathnet_corpus.parquet  (27,817 rows as of 2026-07-29)

Reads the HF datasets-server parquet listing for ShadenA/MathNet (config 'all')
and projects out the image column via duckdb, so only ~24 MB is materialized.
"""
import json
import urllib.request

import duckdb

OUT = "/ibex/user/habiam0b/MathNet_Follow_Up/data/mathnet_corpus.parquet"

d = json.load(urllib.request.urlopen(
    "https://datasets-server.huggingface.co/parquet?dataset=ShadenA%2FMathNet"))
urls = [f["url"] for f in d["parquet_files"] if f["config"] == "all"]
print(f"{len(urls)} parquet shards")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("""
COPY (SELECT id, problem_markdown, solutions_markdown, country, competition,
             topics_flat, language, problem_type, final_answer
      FROM read_parquet($u))
TO '""" + OUT + """' (FORMAT PARQUET)
""", {"u": urls})
n = con.execute(f"SELECT count(*), count(DISTINCT id) FROM '{OUT}'").fetchone()
print("saved rows (total, distinct ids):", n)
