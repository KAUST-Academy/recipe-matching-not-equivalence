#!/usr/bin/env python3
"""
Build D2's pair file restricted to the source problems D3 also has
(round-2 review finding R2-M5).

WHY. The H2 budget control retrained D2 at D3's row COUNT (5,591). That
separates volume from prompt, but not SELECTION: D3's 5,591 rows are the
survivors of the ladder's worst generation-failure rate (1,266 failures), and
its dropped sources are systematically longer and skew proof-only. A
count-matched retrain cannot detect a selection effect of that kind, so the
paper currently states the limit rather than closing it.

This closes it. We intersect D2's and D3's source ids and train D2 on exactly
that intersection, so the two arms share sources as well as count and the only
remaining difference is the rewriter prompt.

Writes data/llm_pairs_paraphrase_d3sources/pairs.jsonl plus a manifest
recording the intersection sizes, so the row count that job trains on is a
measured fact rather than an assumption.

Usage:  python scripts/make_d2_on_d3_sources.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2 = os.path.join(ROOT, "data", "llm_pairs_paraphrase", "pairs.jsonl")
D3 = os.path.join(ROOT, "data", "llm_pairs_style", "pairs.jsonl")
OUTDIR = os.path.join(ROOT, "data", "llm_pairs_paraphrase_d3sources")
OUT = os.path.join(OUTDIR, "pairs.jsonl")
MANIFEST = os.path.join(ROOT, "results", "d2_on_d3_sources_manifest.json")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    os.chdir(ROOT)
    d2, d3 = load(D2), load(D3)
    s2 = {r["source_id"] for r in d2}
    s3 = {r["source_id"] for r in d3}
    both = s2 & s3

    kept = [r for r in d2 if r["source_id"] in both]
    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    doc = {
        "generated": "2026-07-31",
        "purpose": "R2-M5: D2 restricted to the source problems D3 also has, so the "
                   "D2-vs-D3 comparison is matched on SELECTION as well as on row count.",
        "d2_rows": len(d2), "d2_sources": len(s2),
        "d3_rows": len(d3), "d3_sources": len(s3),
        "shared_sources": len(both),
        "d3_only_sources": len(s3 - s2),
        "d2_only_sources": len(s2 - s3),
        "rows_written": len(kept),
        "output": os.path.relpath(OUT, ROOT),
        "note": "Train with --max-rows 0 (no cap): the row count IS the matched "
                "quantity here, and capping it would reintroduce an arbitrary "
                "truncation. Compare against D3, which trains all 5,591 of its own.",
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    for k, v in doc.items():
        if isinstance(v, int):
            print(f"  {k:22s} {v}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)} and the manifest")


if __name__ == "__main__":
    main()
