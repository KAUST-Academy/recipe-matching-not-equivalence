#!/usr/bin/env python
"""Unit tests for audit_ingest.compute_stats using synthetic data.

Covers: partial items (excluded), cannot-judge on gold and near-misses,
label-error near-miss, lenient vs strict R@1-analog, stratum split,
disguise distribution, join errors, Wilson CI sanity.

Run: python scripts/test_audit_ingest.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ingest import compute_stats, wilson_ci  # noqa: E402


def make_item(idx, stratum, roles_order, domain="Algebra"):
    """roles_order = roles at letters A,B,C,D in display order."""
    letters = "ABCD"
    return {
        "item_index": idx,
        "anchor_id": f"anc_{idx}",
        "stratum": stratum,
        "domain": domain,
        "core_sample": idx <= 60,
        "candidates": {
            L: {"role": r, "doc_id": f"anc_{idx}::{r}", "text": "t"}
            for L, r in zip(letters, roles_order)
        },
    }


ITEMS_DOC = {
    "generated": "test",
    "seed": 42,
    "items": [
        make_item(1, "recipe_hit", ["gold", "nm0", "nm1", "nm2"]),
        make_item(2, "recipe_hit", ["nm0", "gold", "nm1", "nm2"]),
        make_item(3, "non_hit", ["nm0", "nm1", "gold", "nm2"], domain="Geometry"),
        make_item(4, "non_hit", ["nm0", "nm1", "nm2", "gold"], domain="Geometry"),
        make_item(5, "non_hit", ["gold", "nm0", "nm1", "nm2"]),
        make_item(6, "recipe_hit", ["gold", "nm0", "nm1", "nm2"]),
    ],
}

ANSWERS_DOC = {
    "schema": "hard_audit_answers_v1",
    "exported_at": "2026-07-30T00:00:00Z",
    "answers": {
        # item 1: perfect distinguisher -> r1 lenient+strict hit
        "1": {
            "anchor_id": "anc_1",
            "cand": {"A": "equiv", "B": "not_equiv", "C": "not_equiv", "D": "not_equiv"},
            "disguise": 5,
            "note": "hard one",
        },
        # item 2 (gold at B): one near-miss judged equivalent (label error)
        # -> r1 fails both variants
        "2": {
            "anchor_id": "anc_2",
            "cand": {"A": "equiv", "B": "equiv", "C": "not_equiv", "D": "not_equiv"},
            "disguise": 3,
            "note": "",
        },
        # item 3 (gold at C): gold cannot-judge -> excluded from gold rate
        # denominator and from R@1 eligibility
        "3": {
            "anchor_id": "anc_3",
            "cand": {"A": "not_equiv", "B": "not_equiv", "C": "cannot_judge", "D": "not_equiv"},
            "disguise": None,
            "note": "cannot tell",
        },
        # item 4 (gold at D): gold equiv, one nm cannot-judge, others not
        # -> lenient hit, strict miss
        "4": {
            "anchor_id": "anc_4",
            "cand": {"A": "not_equiv", "B": "cannot_judge", "C": "not_equiv", "D": "equiv"},
            "disguise": 2,
            "note": "",
        },
        # item 5: PARTIAL (only 2 candidates judged) -> not completed
        "5": {
            "anchor_id": "anc_5",
            "cand": {"A": "equiv", "B": "not_equiv", "C": None, "D": None},
            "disguise": 1,
            "note": "wip",
        },
        # unknown item index -> join error
        "99": {
            "anchor_id": "anc_99",
            "cand": {"A": "equiv", "B": "equiv", "C": "equiv", "D": "equiv"},
            "disguise": 1,
            "note": "",
        },
        # item 6: anchor mismatch -> join error, skipped
        "6": {
            "anchor_id": "anc_WRONG",
            "cand": {"A": "equiv", "B": "not_equiv", "C": "not_equiv", "D": "not_equiv"},
            "disguise": 4,
            "note": "",
        },
    },
}


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def main():
    s = compute_stats(ITEMS_DOC, ANSWERS_DOC)

    # completion accounting
    assert s["n_items_total"] == 6
    assert s["n_items_completed"] == 4, s["n_items_completed"]  # 1,2,3,4
    assert s["n_items_touched"] == 5  # 1,2,3,4,5 (99 and 6 are join errors)
    assert len(s["join_errors"]) == 2, s["join_errors"]
    assert s["n_completed_by_stratum"] == {"recipe_hit": 2, "non_hit": 2}
    assert s["n_completed_by_domain"] == {"Algebra": 2, "Geometry": 2}
    assert s["n_completed_core_sample"] == 4

    # gold: items 1,2,4 equiv; item 3 cannot-judge -> 3/3 among defined
    g = s["gold_judged_equivalent"]
    assert g["k"] == 3 and g["n"] == 3 and approx(g["rate"], 1.0)
    assert g["n_cannot_judge"] == 1

    # near-misses: 12 judgments total (4 items x 3). item2 has 1 equiv;
    # item4 has 1 cannot-judge. defined = 11, equiv = 1.
    nm = s["near_miss_judged_equivalent_label_error"]
    assert nm["k"] == 1 and nm["n"] == 11, (nm["k"], nm["n"])
    assert nm["n_cannot_judge"] == 1

    # R@1-analog lenient: eligible = items 1,2,4 (item3 gold cannot-judge).
    # hits: item1 yes, item2 no (nm equiv), item4 yes -> 2/3
    len_all = s["human_r1_analog"]["lenient_cannot_judge_nm_ignored"]["overall"]
    assert len_all["k"] == 2 and len_all["n"] == 3, len_all
    # strict: item4 fails (cannot-judge nm) -> 1/3
    str_all = s["human_r1_analog"]["strict_cannot_judge_nm_fails"]["overall"]
    assert str_all["k"] == 1 and str_all["n"] == 3, str_all
    # stratum split: recipe_hit eligible = items 1,2 -> lenient 1/2;
    # non_hit eligible = item 4 -> lenient 1/1
    rh = s["human_r1_analog"]["lenient_cannot_judge_nm_ignored"]["recipe_hit"]
    nh = s["human_r1_analog"]["lenient_cannot_judge_nm_ignored"]["non_hit"]
    assert rh["k"] == 1 and rh["n"] == 2, rh
    assert nh["k"] == 1 and nh["n"] == 1, nh
    assert nh["n_completed"] == 2  # item 3 completed but not eligible

    # disguise: rated on completed items only -> {5:1, 3:1, 2:1}, n=3
    d = s["disguise_rating"]
    assert d["n_rated"] == 3 and d["counts"] == {"1": 0, "2": 1, "3": 1, "4": 0, "5": 1}
    assert approx(d["mean"], (5 + 3 + 2) / 3)

    assert s["n_notes_nonempty"] == 2  # items 1 and 3

    # Wilson CI sanity: known value for k=8, n=10
    lo, hi = wilson_ci(8, 10)
    assert approx(lo, 0.4901, 5e-4) and approx(hi, 0.9433, 5e-4), (lo, hi)
    lo0, hi0 = wilson_ci(0, 0)
    assert lo0 is None and hi0 is None
    lo1, hi1 = wilson_ci(0, 5)
    assert lo1 == 0.0 and hi1 < 0.5

    # empty answers -> everything n=0, no crash
    s0 = compute_stats(ITEMS_DOC, {"schema": "hard_audit_answers_v1", "answers": {}})
    assert s0["n_items_completed"] == 0
    assert s0["gold_judged_equivalent"]["rate"] is None
    assert s0["human_r1_analog"]["strict_cannot_judge_nm_fails"]["overall"]["n"] == 0

    print("ALL audit_ingest TESTS PASSED (14 assertion groups)")


if __name__ == "__main__":
    main()
