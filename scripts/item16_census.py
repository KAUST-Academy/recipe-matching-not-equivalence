#!/usr/bin/env python3
"""Fresh-source CAS census, step 1 (2026-09-17): measure whether a like-for-like verified arm is
buildable instead of asserting it.

The question is whether the SymPy pipeline yields pairs on the 3,000 fresh
sources of the regenerated retrain. This script counts, with the census's own span regex and
results/cas_expr_results.json, which of those sources carry a relational span
at all, checks that every such source was attempted by the full CAS run and
yielded nothing, and writes the id list for a relaxed-budget rerun:

  rel   corpus problems with >= 1 census-relational span      (13,678)
  pool  the 7,585 candidates of scripts/make_clean_sources.py (corpus minus
        the corrected gate, the real-duplicate cluster members and the
        original 7,089-source list)
  used  the first 2,044 rows of data/llm_pairs_cleanfull/new_rows.jsonl, the
        fresh sources the regenerated recipe arm trained on
  gen   all 2,602 sources the regeneration verified

Writes data/pairs/item16_relational_pool.txt (pool & rel, sorted) and
results/item16_census.json. When data/cas_pairs_item16/pairs.jsonl exists
(the rerun of scripts/generate_cas_pairs.py over that list), the yield inside
each set is added: sources with a verified positive, rows, rows / 2,044.

  python scripts/item16_census.py
"""
import json, os, re, sys
from collections import Counter

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\$)\$([^$]+?)\$(?!\$)", re.DOTALL)
NEEDED = 2044          # fresh rows the regenerated recipe arm used
RERUN = os.path.join(P, "data", "cas_pairs_item16", "pairs.jsonl")
RERUN_STATS = os.path.join(P, "results", "cas_pairs_item16_stats.json")


def spans(md):
    return [s for m in MATH_SPAN.finditer(md or "")
            for s in [next(g for g in m.groups() if g is not None).strip()] if s]


def ids(path, key=None):
    if key:
        return {json.loads(l)[key] for l in open(path, encoding="utf-8") if l.strip()}
    return {l.strip() for l in open(path) if l.strip()}


def main():
    import duckdb
    rows = duckdb.connect().execute(
        f"SELECT id, problem_markdown FROM '{P}/data/mathnet_corpus.parquet'").fetchall()
    status = json.load(open(f"{P}/results/cas_expr_results.json"))
    rel = {pid for pid, md in rows
           if any(status.get(s, {}).get("status") == "relational" for s in spans(md))}
    v1 = set(json.load(open(f"{P}/anchor_to_corpus_mapping.json"))["exclude_corpus_ids"])
    v2 = set(json.load(open(f"{P}/anchor_to_corpus_mapping_v2.json"))["exclude_corpus_ids"])
    xl = set(json.load(open(f"{P}/data/crosslingual_eval/leakage_exclude_ids.json"))["all_pair_member_ids"])
    orig = ids(f"{P}/data/phase2/cas_source_ids.txt")
    pool = {pid for pid, _ in rows} - v2 - xl - orig
    new = [json.loads(l)["source_id"] for l in open(f"{P}/data/llm_pairs_cleanfull/new_rows.jsonl")]
    used, gen = set(new[:NEEDED]), set(new)
    cas_sources = ids(f"{P}/data/cas_pairs/pairs.jsonl", "source_id")
    cas_stats = json.load(open(f"{P}/results/cas_pairs_stats.json"))
    eligible = rel - v1                      # the full CAS run's attempted set
    target = pool & rel
    out = {"generated_by": "scripts/item16_census.py", "needed_rows": NEEDED,
           "census": {"corpus": len(rows), "relational": len(rel), "pool": len(pool),
                      "pool_relational": len(target), "used": len(used), "used_relational": len(used & rel),
                      "generated": len(gen), "generated_relational": len(gen & rel),
                      "used_subset_of_generated": used <= gen, "generated_subset_of_pool": gen <= pool},
           "full_cas_run": {"eligible_recomputed": len(eligible),
                            "eligible_in_stats": cas_stats["eligibility"]["n_eligible_census_relational_not_excluded"],
                            "attempted_in_stats": cas_stats["eligibility"]["n_attempted"],
                            "sources_with_positive": len(cas_sources),
                            "pool_relational_all_eligible": target <= eligible,
                            "pool_relational_with_cas_row": len(target & cas_sources),
                            "used_with_cas_row": len(used & cas_sources),
                            "generated_with_cas_row": len(gen & cas_sources),
                            "pool_relational_per_source_rows_in_verified_pool":
                                round(cas_stats["outcomes"]["positive_records_written"] / len(cas_sources), 3)}}
    os.makedirs(f"{P}/data/pairs", exist_ok=True)
    with open(f"{P}/data/pairs/item16_relational_pool.txt", "w") as f:
        f.write("\n".join(sorted(target)) + "\n")
    if os.path.exists(RERUN):
        recs = [json.loads(l) for l in open(RERUN, encoding="utf-8") if l.strip()]
        st = json.load(open(RERUN_STATS)) if os.path.exists(RERUN_STATS) else {}
        by_src = Counter(r["source_id"] for r in recs)
        fams = Counter(r.get("transform_family") for r in recs)
        def yield_in(name, S):
            srcs = {s for s in by_src if s in S}; n_rows = sum(by_src[s] for s in srcs)
            return {"sources": len(S), "sources_with_positive": len(srcs), "rows": n_rows,
                    "rows_over_needed": round(n_rows / NEEDED, 4),
                    "sources_with_positive_pct": round(100 * len(srcs) / max(1, len(S)), 2)}
        out["rerun"] = {"meta": st.get("meta"), "outcomes": st.get("outcomes"),
                        "rows_written": len(recs), "sources_with_positive": len(by_src),
                        "rows_per_yielding_source": round(len(recs) / max(1, len(by_src)), 3),
                        "transform_families": dict(fams),
                        "all_rows_from_listed_pool": set(by_src) <= target,
                        "yield": {"pool_relational": yield_in("pool_relational", target),
                                  "used_2044": yield_in("used", used & rel),
                                  "generated_2602": yield_in("generated", gen & rel)}}
    json.dump(out, open(f"{P}/results/item16_census.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
