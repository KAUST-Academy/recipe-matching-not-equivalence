#!/usr/bin/env python
"""Ingest exported hard-audit answers -> results/hard_audit_stats.json + table.

Joins the annotator's hard_audit_answers.json (exported from
data/hard_audit/audit_sheet.html) with the provenance file
data/hard_audit/audit_items.json (which holds the gold/near-miss permutation),
then computes:

  * gold judged-equivalent rate (Wilson 95% CI)
  * near-miss judged-equivalent (label-error) rate (Wilson 95% CI)
  * human "R@1-analog": gold marked Equivalent AND every judged near-miss
    marked Not equivalent -- overall and by stratum (recipe_hit vs non_hit).
    Two variants: lenient (cannot-judge near-misses ignored) and strict
    (cannot-judge near-miss counts as a failure to distinguish).
    CAVEAT: this is a classification-based analog, not a ranking task --
    the annotator judges each candidate independently instead of ranking a
    117k-doc corpus, so it upper-bounds what ranking-style R@1 a human could
    achieve on these 4 candidates and is not directly comparable to model R@1.
  * disguise-rating distribution.

Only items with all 4 candidates judged count as completed; partial exports
are fine and n is reported everywhere.

Usage:
  python scripts/audit_ingest.py path/to/hard_audit_answers.json \
      [--items data/hard_audit/audit_items.json] [--out results/hard_audit_stats.json]
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGMENTS = ("equiv", "not_equiv", "cannot_judge")


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate_block(k, n):
    lo, hi = wilson_ci(k, n)
    return {
        "k": k,
        "n": n,
        "rate": (k / n) if n else None,
        "wilson_ci95": [lo, hi] if n else None,
    }


def load_answers(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != "hard_audit_answers_v1":
        raise ValueError(f"unexpected schema: {d.get('schema')!r}")
    return d


def compute_stats(items_doc, answers_doc):
    items = {it["item_index"]: it for it in items_doc["items"]}
    answers = answers_doc.get("answers", {})

    completed = []  # (item, answer) with all 4 candidates judged
    n_touched = 0
    join_errors = []
    for k, a in answers.items():
        idx = int(k)
        it = items.get(idx)
        if it is None:
            join_errors.append(f"answer for unknown item_index {idx}")
            continue
        if a.get("anchor_id") and a["anchor_id"] != it["anchor_id"]:
            join_errors.append(
                f"item {idx}: anchor mismatch {a['anchor_id']} != {it['anchor_id']}"
            )
            continue
        cand = a.get("cand", {})
        judged = {L: cand.get(L) for L in "ABCD"}
        if any(v is not None for v in judged.values()) or a.get("disguise") is not None:
            n_touched += 1
        if all(judged[L] in JUDGMENTS for L in "ABCD"):
            completed.append((it, a))
    completed.sort(key=lambda p: p[0]["item_index"])

    # Per-judgment tallies, mapping displayed letters back to roles.
    gold_j = Counter()
    nm_j = Counter()
    per_item = []
    for it, a in completed:
        roles = {L: it["candidates"][L]["role"] for L in "ABCD"}
        gold_letter = next(L for L in "ABCD" if roles[L] == "gold")
        gv = a["cand"][gold_letter]
        gold_j[gv] += 1
        nvs = [a["cand"][L] for L in "ABCD" if L != gold_letter]
        for v in nvs:
            nm_j[v] += 1
        nm_defined = [v for v in nvs if v != "cannot_judge"]
        r1_lenient = (
            gv == "equiv"
            and all(v == "not_equiv" for v in nm_defined)
            and len(nm_defined) > 0
        )
        r1_strict = gv == "equiv" and all(v == "not_equiv" for v in nvs)
        per_item.append(
            {
                "item_index": it["item_index"],
                "anchor_id": it["anchor_id"],
                "stratum": it["stratum"],
                "domain": it["domain"],
                "core_sample": it.get("core_sample", it["item_index"] <= 60),
                "gold_judgment": gv,
                "nm_judgments": nvs,
                "r1_lenient": r1_lenient,
                "r1_strict": r1_strict,
                "gold_defined": gv != "cannot_judge",
                "disguise": a.get("disguise"),
                "note": a.get("note", ""),
            }
        )

    def r1_block(rows, key):
        eligible = [r for r in rows if r["gold_defined"]]
        blk = rate_block(sum(r[key] for r in eligible), len(eligible))
        blk["n_completed"] = len(rows)
        return blk

    def stratum_split(key):
        out = {}
        for s in ("recipe_hit", "non_hit"):
            rows = [r for r in per_item if r["stratum"] == s]
            out[s] = r1_block(rows, key)
        out["overall"] = r1_block(per_item, key)
        return out

    gold_defined = gold_j["equiv"] + gold_j["not_equiv"]
    nm_defined = nm_j["equiv"] + nm_j["not_equiv"]
    disguise = Counter(
        r["disguise"] for r in per_item if isinstance(r["disguise"], int)
    )
    n_rated = sum(disguise.values())

    stats = {
        "source": {
            "answers_exported_at": answers_doc.get("exported_at"),
            "items_generated": items_doc.get("generated"),
            "seed": items_doc.get("seed"),
        },
        "n_items_total": len(items),
        "n_items_touched": n_touched,
        "n_items_completed": len(completed),
        "n_completed_core_sample": sum(1 for r in per_item if r["core_sample"]),
        "n_completed_by_stratum": dict(
            Counter(r["stratum"] for r in per_item)
        ),
        "n_completed_by_domain": dict(Counter(r["domain"] for r in per_item)),
        "join_errors": join_errors,
        "gold_judged_equivalent": {
            **rate_block(gold_j["equiv"], gold_defined),
            "n_cannot_judge": gold_j["cannot_judge"],
            "note": "denominator excludes cannot-judge golds",
        },
        "near_miss_judged_equivalent_label_error": {
            **rate_block(nm_j["equiv"], nm_defined),
            "n_cannot_judge": nm_j["cannot_judge"],
            "note": "denominator excludes cannot-judge near-misses; unit = judgment (3 per item)",
        },
        "human_r1_analog": {
            "caveat": (
                "Classification-based analog, NOT ranking-based: the annotator "
                "judges 4 candidates independently rather than ranking the full "
                "117k-doc corpus. Success = gold judged Equivalent AND every "
                "judged near-miss judged Not equivalent. Items whose gold was "
                "marked cannot-judge are excluded. Not directly comparable to "
                "model R@1 over the corpus."
            ),
            "lenient_cannot_judge_nm_ignored": stratum_split("r1_lenient"),
            "strict_cannot_judge_nm_fails": stratum_split("r1_strict"),
        },
        "disguise_rating": {
            "n_rated": n_rated,
            "counts": {str(v): disguise.get(v, 0) for v in range(1, 6)},
            "mean": (
                sum(v * c for v, c in disguise.items()) / n_rated
                if n_rated
                else None
            ),
        },
        "n_notes_nonempty": sum(1 for r in per_item if r["note"].strip()),
        "per_item": per_item,
    }
    return stats


def fmt_rate(block):
    if block["n"] == 0:
        return "   --            (n=0)"
    lo, hi = block["wilson_ci95"]
    return f"{block['rate']*100:5.1f}%  [{lo*100:5.1f}, {hi*100:5.1f}]  (k={block['k']}, n={block['n']})"


def print_table(s):
    print("=" * 78)
    print("HARD-TIER HUMAN AUDIT")
    print(
        f"completed items: {s['n_items_completed']} / {s['n_items_total']} "
        f"(core-sample completed: {s['n_completed_core_sample']}; "
        f"by stratum: {s['n_completed_by_stratum']})"
    )
    if s["join_errors"]:
        print(f"JOIN ERRORS: {s['join_errors']}")
    print("-" * 78)
    print(f"{'metric':<44s} rate    [Wilson 95% CI]")
    g = s["gold_judged_equivalent"]
    print(f"{'gold judged Equivalent':<44s}{fmt_rate(g)}")
    print(f"{'  gold cannot-judge count':<44s}{g['n_cannot_judge']}")
    nm = s["near_miss_judged_equivalent_label_error"]
    print(f"{'near-miss judged Equivalent (label error)':<44s}{fmt_rate(nm)}")
    print(f"{'  near-miss cannot-judge count':<44s}{nm['n_cannot_judge']}")
    print("-" * 78)
    for variant, label in [
        ("lenient_cannot_judge_nm_ignored", "human R@1-analog (lenient)"),
        ("strict_cannot_judge_nm_fails", "human R@1-analog (strict)"),
    ]:
        blocks = s["human_r1_analog"][variant]
        for key, name in [
            ("overall", "overall"),
            ("recipe_hit", "recipe-hit stratum"),
            ("non_hit", "non-hit stratum"),
        ]:
            print(f"{label + ' - ' + name:<44s}{fmt_rate(blocks[key])}")
    print("  CAVEAT: classification-based analog over 4 candidates, not")
    print("  ranking over the corpus; not directly comparable to model R@1.")
    print("-" * 78)
    d = s["disguise_rating"]
    dist = "  ".join(f"{v}:{d['counts'][str(v)]}" for v in range(1, 6))
    mean = f"{d['mean']:.2f}" if d["mean"] is not None else "--"
    print(f"disguise rating (1..5): {dist}   mean={mean}   n={d['n_rated']}")
    print(f"non-empty notes: {s['n_notes_nonempty']}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("answers", help="hard_audit_answers.json exported from the sheet")
    ap.add_argument("--items", default=str(ROOT / "data/hard_audit/audit_items.json"))
    ap.add_argument("--out", default=str(ROOT / "results/hard_audit_stats.json"))
    args = ap.parse_args()

    with open(args.items) as f:
        items_doc = json.load(f)
    answers_doc = load_answers(args.answers)
    stats = compute_stats(items_doc, answers_doc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print_table(stats)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
