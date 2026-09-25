#!/usr/bin/env python3
"""
Pre-registered verdict for the clean-gate retrain (acceptance-review E1).

THE DECISION RULE BELOW WAS FIXED IN scripts/clean_gate_retrain.slurm's header
AND COMMITTED BEFORE THE JOB WAS SUBMITTED, and is restated here verbatim. It
is evaluated in code so the branch cannot be chosen after seeing the numbers.

  PRIMARY: three-seed mean easy-tier arm gap (ctrl-LLM - ctrl-CAS, R@1), both
  arms gated by anchor_to_corpus_mapping_v2.json at a matched 4,101 rows.

  G1 CONFIRMS  mean easy gap >= 35.0 AND positive in all three seeds.
  G2 PARTIAL   15.0 <= mean easy gap < 35.0, all seeds positive.
  G3 REFUTES   mean easy gap < 15.0, OR any seed non-positive.

  SECONDARY (reported, not gated): hard-tier gap, and the easy-vs-real-data
  difference-in-differences under the clean gate (published: 35.59).

SCOPE WARNING carried into the output: both arms shrink from their published
budgets to 4,101 rows, so these numbers are matched TO EACH OTHER but are NOT
budget-matched to the published 45.33. The comparison isolates the gate, not
the budget.

Usage:  python scripts/clean_gate_verdict.py
Writes: results/clean_gate_stats.json
"""

import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = tuple(range(42, 50))
ROWS = 4101
PUBLISHED = {"easy_gap": 45.33, "hard_gap": 9.26, "did": 35.59, "xling_gap": 9.74}


def _read(path, *keys):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    os.chdir(ROOT)
    per_seed = {"easy": {}, "hard": {}, "xling": {}}
    missing = []

    for seed in SEEDS:
        for tier in ("easy", "hard"):
            vals = {}
            for arm in ("llm", "cas"):
                name = f"ctrl-{arm}-v2gate-{ROWS}-s{seed}"
                v = _read(f"results/eval_{tier}_{name}.json", "overall", "recall@1")
                if v is None:
                    missing.append(f"results/eval_{tier}_{name}.json")
                vals[arm] = v
            if vals["llm"] is not None and vals["cas"] is not None:
                per_seed[tier][seed] = {
                    "llm": vals["llm"], "cas": vals["cas"],
                    "gap": round(vals["llm"] - vals["cas"], 4),
                }
        vals = {}
        for arm in ("llm", "cas"):
            name = f"ctrl-{arm}-v2gate-{ROWS}-s{seed}"
            v = _read(f"results/crosslingual_{name}.json",
                      "strict_crosslingual_gold", "recall@1")
            if v is None:
                missing.append(f"results/crosslingual_{name}.json")
            vals[arm] = v
        if vals["llm"] is not None and vals["cas"] is not None:
            per_seed["xling"][seed] = {
                "llm": vals["llm"], "cas": vals["cas"],
                "gap": round(vals["llm"] - vals["cas"], 4),
            }

    easy = [per_seed["easy"][s]["gap"] for s in SEEDS if s in per_seed["easy"]]
    hard = [per_seed["hard"][s]["gap"] for s in SEEDS if s in per_seed["hard"]]
    xling = [per_seed["xling"][s]["gap"] for s in SEEDS if s in per_seed["xling"]]

    # Record the seeds actually FOUND, not the seeds looked for: the slurm
    # script defaults to three seeds, so stamping list(SEEDS) here would label a
    # three-seed run with the headline's eight-seed list.
    seeds_found = sorted(per_seed["easy"])

    out = {
        "generated": "clean_gate_verdict.py",
        "preregistered_in": "scripts/clean_gate_retrain.slurm header (committed before submission)",
        "gate": "anchor_to_corpus_mapping_v2.json (15,244 excluded corpus ids)",
        "matched_budget_rows": ROWS,
        "seeds": seeds_found,
        "seeds_looked_for": list(SEEDS),
        "per_seed": per_seed,
        "missing_inputs": missing,
        "published_reference": PUBLISHED,
        "scope_warning": (
            "Both arms are gated to 4,101 rows, so these numbers are matched to "
            "each other but NOT budget-matched to the published 45.33 (6,145 "
            "rows). The design isolates the contamination gate, not the budget. "
            "Training-seed spread only; no eval-sampling interval is included."
        ),
    }

    if 3 <= len(easy) < len(SEEDS):
        out["partial_run_warning"] = (
            f"PARTIAL: {len(easy)} of {len(SEEDS)} seeds on disk ({seeds_found}). "
            "The pre-registered branch is evaluated on what is here, but the "
            "paper's 43.11 +/- 0.90 is the EIGHT-seed value; rerun with "
            'SEEDS="42 43 44 45 46 47 48 49" to reproduce it.'
        )

    if len(easy) < 3:
        out["verdict"] = (
            f"INCOMPLETE — only {len(easy)} easy-tier seed pair(s) on disk. The "
            "pre-registered branch needs at least the three seeds it was written for."
        )
        _write(out)
        print(out["verdict"])
        return 1

    mean_easy = st.mean(easy)
    sd_easy = st.stdev(easy) if len(easy) > 1 else 0.0
    all_pos = all(g > 0 for g in easy)

    out["easy_gap_per_seed"] = easy
    out["easy_gap_mean"] = round(mean_easy, 4)
    out["easy_gap_sd"] = round(sd_easy, 4)
    out["easy_gap_positive_in_every_seed"] = all_pos
    if hard:
        out["hard_gap_per_seed"] = hard
        out["hard_gap_mean"] = round(st.mean(hard), 4)
        out["hard_gap_sd"] = round(st.stdev(hard), 4) if len(hard) > 1 else 0.0
    if xling:
        out["xling_gap_per_seed"] = xling
        out["xling_gap_mean"] = round(st.mean(xling), 4)
        out["did_easy_minus_xling"] = round(mean_easy - st.mean(xling), 4)
        out["did_published_reference"] = PUBLISHED["did"]

    # ---- the pre-registered branch, evaluated in code ----
    if not all_pos or mean_easy < 15.0:
        out["verdict_code"] = "G3_REFUTES"
        out["verdict"] = (
            f"G3 REFUTES — {len(easy)}-seed mean easy-tier gap {mean_easy:.2f} "
            f"(per seed {easy}), positive in every seed: {all_pos}. Leakage is a "
            "first-order driver of the headline. The paper's central claim must be "
            "rebuilt on the v2-gated numbers; abstract, intro, gaming and ood revert."
        )
    elif mean_easy < 35.0:
        out["verdict_code"] = "G2_PARTIAL"
        out["verdict"] = (
            f"G2 PARTIAL — {len(easy)}-seed mean easy-tier gap {mean_easy:.2f} +/- "
            f"{sd_easy:.2f} (per seed {easy}), all seeds positive. The effect is real "
            "but materially smaller than the published 45.33. The contamination "
            "section must revise its 1.24-point leakage attribution upward, and the "
            "abstract must carry this v2-gated number alongside the headline."
        )
    else:
        out["verdict_code"] = "G1_CONFIRMS"
        out["verdict"] = (
            f"G1 CONFIRMS — {len(easy)}-seed mean easy-tier gap {mean_easy:.2f} +/- "
            f"{sd_easy:.2f} (per seed {easy}), positive in every seed. A gate that "
            "removes every near-verbatim twin from BOTH arms leaves the gap intact, "
            "so leakage is not a first-order driver. The 1.24-point attribution "
            "stands qualitatively and this is a headline robustness result."
        )

    _write(out)
    print(f"\n  gate     : {out['gate']}")
    print(f"  budget   : {ROWS} rows/arm, seeds {list(SEEDS)}")
    for s in SEEDS:
        e = per_seed["easy"].get(s, {})
        print(f"  seed {s} : llm {e.get('llm')}  cas {e.get('cas')}  gap {e.get('gap')}")
    print(f"\n  easy gap : {mean_easy:.2f} +/- {sd_easy:.2f}   (published, 6145 rows: 45.33)")
    if hard:
        print(f"  hard gap : {st.mean(hard):.2f}                    (published: 9.26)")
    if xling:
        print(f"  DiD      : {out['did_easy_minus_xling']:.2f}                   (published: 35.59)")
    print(f"\nVERDICT: {out['verdict']}\n")
    return 0


def _write(out):
    with open("results/clean_gate_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote results/clean_gate_stats.json")


if __name__ == "__main__":
    sys.exit(main())
