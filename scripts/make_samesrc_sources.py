#!/usr/bin/env python3
"""Same-source regenerated control, step 2a (2026-09-17): the survivor source list for
the same-source regenerated control.

data/pairs/source_ids_survivors.txt = the source_ids of data/llm_pairs/pairs.jsonl
that the corrected gate (anchor_to_corpus_mapping_v2.json) keeps: 4,101 rows, one
per source. Asserts that none is a cross-lingual eval member and that every one is
a source of the verified arm under the same gate (the 4,749 sources behind
models/cleanfull-cas-6145-s*), so both arms of the control sit on one source list.

  python scripts/make_samesrc_sources.py
"""
import json, os, sys

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(P, "data", "pairs", "source_ids_survivors.txt")


def main():
    v2 = set(json.load(open(f"{P}/anchor_to_corpus_mapping_v2.json"))["exclude_corpus_ids"])
    xl = set(json.load(open(f"{P}/data/crosslingual_eval/leakage_exclude_ids.json"))["eval_member_ids"])
    rows = [json.loads(l) for l in open(f"{P}/data/llm_pairs/pairs.jsonl", encoding="utf-8") if l.strip()]
    surv = [r["source_id"] for r in rows if r["source_id"] not in v2]
    if len(surv) != 4101 or len(set(surv)) != 4101:
        sys.exit(f"FATAL: expected 4,101 survivors with one row each, got {len(surv)} rows / {len(set(surv))} sources")
    if set(surv) & xl:
        sys.exit(f"FATAL: {len(set(surv) & xl)} survivors are cross-lingual eval members")
    cas = {json.loads(l)["source_id"] for l in open(f"{P}/data/cas_pairs/pairs.jsonl", encoding="utf-8") if l.strip()}
    cas_gate = {s for s in cas if s not in v2 and s not in xl}       # the verified arm's sources under the corrected gate
    if not set(surv) <= cas_gate:
        sys.exit(f"FATAL: {len(set(surv) - cas_gate)} survivors are not sources of the gated verified arm")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(sorted(surv)) + "\n")
    print(f"[done] {len(surv)} survivor sources -> {OUT}; all inside the verified arm's "
          f"{len(cas_gate)} gated sources ({len(cas)} before the gate)")


if __name__ == "__main__":
    main()
