#!/usr/bin/env python3
"""Build the E2 source- and negative-matched CAS arm (acceptance-review E2).

Constructs data/cas_pairs/pairs_negmatched.jsonl: a CAS training file matched
to ctrl-LLM's on every training-signal axis the arms differ on --
    rows            6,145 (one per source, like ctrl-LLM; ctrl-CAS's draw
                    clusters 1.33 rows/source over ~4,621 sources)
    sources         ctrl-LLM's EXACT 6,145 source ids
    negatives       exactly 16,169 attachments (2.6312/row, ctrl-LLM's total),
                    reached by REUSING each source's own counterexampled
                    negatives (cycling duplicates), never cross-source ones
    dev split       identical to ctrl-LLM's per seed for free (the trainer
                    splits by source id with the run seed; same source set =>
                    same split -- verified by simulation in the E2 dossier)

Matching rule (fixed before any training run):
  * positive: one CAS record per source, chosen uniformly by the build RNG
    (first-in-file would yield zero 'mirror' positives; seeded choice keeps
    the family mix neutral). A source's records all carry the SAME negative
    set, so this choice affects only the positive text.
  * negatives: let D(s) = the source's distinct counterexampled negatives
    (dedup by text; identical across the source's records). Rows with
    |D(s)| == 0 get none -- 1,797 of the 6,145 sources have no counterexampled
    negative at all, and cross-source filler would not be a same-anchor
    near-miss, so we do not fake it. The 4,348 rows with |D(s)| >= 1 are
    RNG-shuffled; the first 3,125 receive 4 attachments and the rest 3
    (4*3125 + 3*1223 = 16,169 exactly), each row cycling its own shuffled
    D(s). Duplicated attachments are true oversampling under the
    NO_DUPLICATES batch sampler (copies land in different batches).

Residual mismatches, disclosed rather than hidden: the per-row histogram is
not matched (E2 {0:1797, 3:1223, 4:3125} vs ctrl-LLM {0:79, 1:356, 2:1317,
3:4393}); E2's zero-negative rows feed the trainer's "pairs" dataset (1,797
vs 79). Totals, means, rows, sources, steps and the dev split are matched.

Gating: the file is keyed to ctrl-LLM's sources, which the v1 anchor gate and
the 779-id crosslingual gate already pass (intersection 0, asserted below),
so the trainer's default (headline, v1) gates drop nothing.

Deterministic: RNG seed 20260802, fixed here. Certificate with every count
and assertion -> results/e2_negmatched_construction.json.
"""
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_FILE = ROOT / "data/llm_pairs/pairs.jsonl"
CAS_FILE = ROOT / "data/cas_pairs/pairs.jsonl"
OUT_FILE = ROOT / "data/cas_pairs/pairs_negmatched.jsonl"
CERT_FILE = ROOT / "results/e2_negmatched_construction.json"
V1_GATE = ROOT / "anchor_to_corpus_mapping.json"
XLING_GATE = ROOT / "data/crosslingual_eval/leakage_exclude_ids.json"
BUILD_SEED = 20260802

def main():
    rng = random.Random(BUILD_SEED)

    llm_rows = [json.loads(l) for l in open(LLM_FILE)]
    llm_sources = [r["source_id"] for r in llm_rows]
    assert len(llm_sources) == 6145 and len(set(llm_sources)) == 6145
    target_total = sum(len(r.get("negatives") or []) for r in llm_rows)
    assert target_total == 16169, target_total

    cas_by_source = {}
    for line in open(CAS_FILE):
        r = json.loads(line)
        cas_by_source.setdefault(r["source_id"], []).append(r)
    missing = [s for s in llm_sources if s not in cas_by_source]
    assert not missing, f"{len(missing)} ctrl-LLM sources lack a CAS record"

    # Gate hygiene: the sources must already pass both trainer-enforced gates.
    v1_excluded = set(json.load(open(V1_GATE))["exclude_corpus_ids"])
    xling_excluded = set(json.load(open(XLING_GATE))["eval_member_ids"])
    assert not (set(llm_sources) & v1_excluded), "source hits the v1 anchor gate"
    assert not (set(llm_sources) & xling_excluded), "source hits the xling gate"

    # Distinct same-source negatives (identical across a source's records).
    distinct_negs = {}
    for s in llm_sources:
        seen, ordered = set(), []
        for rec in cas_by_source[s]:
            for n in rec.get("negatives") or []:
                if n["text"] not in seen:
                    seen.add(n["text"])
                    ordered.append({"text": n["text"], "edit": n.get("edit")})
        ordered.sort(key=lambda n: n["text"])  # order-independent determinism
        distinct_negs[s] = ordered

    avail = Counter(min(len(distinct_negs[s]), 9) for s in llm_sources)
    eligible = [s for s in llm_sources if distinct_negs[s]]
    zero_rows = [s for s in llm_sources if not distinct_negs[s]]
    n4 = target_total - 3 * len(eligible)
    assert 0 <= n4 <= len(eligible), (n4, len(eligible))
    order = sorted(eligible)
    rng.shuffle(order)
    quota = {s: (4 if i < n4 else 3) for i, s in enumerate(order)}

    out_rows, attach_hist, family_mix = [], Counter(), Counter()
    total_attached = 0
    for s in sorted(llm_sources):
        rec = rng.choice(sorted(cas_by_source[s], key=lambda r: r["positive_text"]))
        family_mix[rec.get("transform_family", "?")] += 1
        negs = []
        if distinct_negs[s]:
            pool = list(distinct_negs[s])
            rng.shuffle(pool)
            k = quota[s]
            negs = [pool[i % len(pool)] for i in range(k)]
        attach_hist[len(negs)] += 1
        total_attached += len(negs)
        out_rows.append({
            "source_id": s,
            "positive_text": rec["positive_text"],
            "transform_family": rec.get("transform_family"),
            "negatives": negs,
            "e2_distinct_negatives": len(distinct_negs[s]),
        })
    assert total_attached == target_total, (total_attached, target_total)
    assert len(out_rows) == 6145

    with open(OUT_FILE, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cert = {
        "purpose": "E2: CAS arm matched to ctrl-LLM on rows, sources, and total negative attachments",
        "build_seed": BUILD_SEED,
        "inputs": {"llm_file": str(LLM_FILE), "cas_file": str(CAS_FILE)},
        "output": str(OUT_FILE),
        "rows": len(out_rows),
        "sources_equal_ctrl_llm": sorted(set(llm_sources)) == sorted({r["source_id"] for r in out_rows}),
        "total_negative_attachments": total_attached,
        "target_from_ctrl_llm": target_total,
        "negatives_per_row_mean": round(total_attached / len(out_rows), 4),
        "attachment_histogram": dict(sorted(attach_hist.items())),
        "ctrl_llm_histogram_for_contrast": {"0": 79, "1": 356, "2": 1317, "3": 4393},
        "distinct_same_source_negatives_available": dict(sorted(avail.items())),
        "rows_with_zero_negatives": len(zero_rows),
        "rows_at_4_attachments": n4,
        "rows_at_3_attachments": len(eligible) - n4,
        "positive_family_mix": dict(family_mix),
        "gate_checks": {"v1_anchor_intersection": 0, "xling_intersection": 0},
        "semantics_note": "every negative is a counterexampled minimal edit of the row's own source problem; reuse is within-source cycling only, never cross-source",
    }
    CERT_FILE.write_text(json.dumps(cert, indent=2) + "\n")
    print(json.dumps(cert, indent=2))

if __name__ == "__main__":
    main()
