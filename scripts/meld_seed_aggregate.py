#!/usr/bin/env python3
"""
Across-training-seed aggregation of the MELD recipe-matching attack
(round-2 review finding R2-H7), emitted as a citable JSON fact source.

WHY THIS EXISTS
  Every MELD verdict in the draft rested on ONE seed-42 run per arm with an
  interval that resamples MELD's 270 pairs and carries no training-seed
  variance -- the exact inference the paper's own appendix warns is
  "confidently wrong where the seeds disagree". scripts/meld_attack_seeds.slurm
  retrains both arms at seeds 43 and 44; this script turns the three runs into
  the numbers the paper quotes, the same way scripts/dose_seed_aggregate.py
  does for the dose ladder.

PRE-REGISTERED READING (copied verbatim from the SLURM header, which was
written before the seed-43/44 runs were submitted -- and, unlike the dose
aggregator, before their outputs existed at all):
  R1  P-M1 replicates if all three meld9 seeds clear base 10.19 by >= +10.0.
  R2  P-M2's refutation holds if the three-seed mean real-duplicate R@1 stays
      ABOVE base 84.73. If it lands below, P-M2 is NOT refuted, seed 42 was a
      lucky run, and the incentive-inversion scoping must be withdrawn.
  R3  P-M4's genre reading holds if heldout's three-seed mean clears base by
      >= +10.0. The meld9-minus-heldout domain increment is reported with its
      three-seed spread and is NOT called replicated unless all three seeds
      keep its sign.

Reference points (fixed, measured before this job): base MELD pairs-only R@1
10.19, base real-duplicate strict R@1 84.73, ctrl-LLM MELD R@1 18.89,
real-duplicate single-run sd 2.31 (results/ctrl_seed_stats.json, 8 seeds).

Usage:  python scripts/meld_seed_aggregate.py
Output: results/meld_seed_stats.json + a printed table. Missing inputs are
reported as nulls with an explicit `missing` list -- never silently skipped.
"""

import json
import math
import os
import statistics
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

BASE_MELD_R1 = 10.19          # results/meld_atk_base.json, query-prompt condition
BASE_REALDUP_R1 = 84.73       # results/crosslingual_qwen3-0.6b-base.json
CTRL_LLM_MELD_R1 = 18.89      # results/meld_atk_ctrl_llm.json
SD1_REALDUP = 2.3115          # single-run sd, 8-seed ctrl-LLM arm
SEEDS = (42, 43, 44)
ARMS = ("meld9", "heldout")


def _load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _dig(p, *keys):
    d = _load(p)
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def meld_path(arm, seed):
    # seed 42 keeps the original naming from attack_second_benchmark.slurm
    return os.path.join(RESULTS, f"meld_atk_attack_{arm}.json" if seed == 42
                        else f"meld_atk_attack_{arm}_s{seed}.json")


def realdup_path(arm, seed):
    return os.path.join(RESULTS, f"crosslingual_meld-attack-{arm}.json" if seed == 42
                        else f"crosslingual_meld-attack-{arm}-s{seed}.json")


def hard_path(seed):
    return os.path.join(RESULTS, "eval_hard_meld-attack-meld9.json" if seed == 42
                        else f"eval_hard_meld-attack-meld9-s{seed}.json")


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "values": []}
    return {"n": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
            "values": vals}


def main():
    missing = []
    per_seed = {}
    for arm in ARMS:
        per_seed[arm] = {}
        for s in SEEDS:
            mp, rp = meld_path(arm, s), realdup_path(arm, s)
            meld = _dig(mp, "retrieval_pairs_only", "recall@1")
            meld5 = _dig(mp, "retrieval_pairs_only", "recall@5")
            rdup = _dig(rp, "strict_crosslingual_gold", "recall@1")
            for val, path in ((meld, mp), (rdup, rp)):
                if val is None:
                    missing.append(os.path.relpath(path, ROOT))
            per_seed[arm][s] = {"meld_R@1": meld, "meld_R@5": meld5,
                                "realdup_R@1": rdup}

    hard = {}
    for s in SEEDS:
        v = _dig(hard_path(s), "overall", "recall@1")
        if v is None:
            missing.append(os.path.relpath(hard_path(s), ROOT))
        hard[s] = v

    agg = {}
    for arm in ARMS:
        agg[arm] = {
            "meld_R@1": summarize([per_seed[arm][s]["meld_R@1"] for s in SEEDS]),
            "meld_R@5": summarize([per_seed[arm][s]["meld_R@5"] for s in SEEDS]),
            "realdup_R@1": summarize([per_seed[arm][s]["realdup_R@1"] for s in SEEDS]),
        }
    agg["meld9"]["mathnet_hard_R@1"] = summarize([hard[s] for s in SEEDS])

    # ---- the three pre-registered readings ----
    verdicts = {}

    m9 = agg["meld9"]["meld_R@1"]
    gains = [v - BASE_MELD_R1 for v in m9["values"]]
    verdicts["R1_P1_replicates"] = {
        "rule": "all meld9 seeds clear base 10.19 by >= +10.0",
        "per_seed_gain_over_base": [round(g, 4) for g in gains],
        "mean_gain": round(statistics.mean(gains), 4) if gains else None,
        "verdict": ("PENDING — %d of %d seeds present" % (len(gains), len(SEEDS))
                    if len(gains) < len(SEEDS) else
                    "REPLICATED" if all(g >= 10.0 for g in gains)
                    else "NOT REPLICATED"),
    }

    rd = agg["meld9"]["realdup_R@1"]
    if rd["mean"] is None or rd["n"] < len(SEEDS):
        verdicts["R2_P2_refutation_holds"] = {
            "verdict": "PENDING — %d of %d seeds present" % (rd["n"], len(SEEDS))}
    else:
        se = (rd["sd"] / math.sqrt(rd["n"])) if rd["sd"] else SD1_REALDUP / math.sqrt(rd["n"])
        verdicts["R2_P2_refutation_holds"] = {
            "rule": "three-seed mean real-duplicate R@1 stays ABOVE base 84.73",
            "mean": rd["mean"], "sd": rd["sd"], "values": rd["values"],
            "margin_over_base": round(rd["mean"] - BASE_REALDUP_R1, 4),
            "margin_in_se_units": round((rd["mean"] - BASE_REALDUP_R1) / se, 3) if se else None,
            "all_seeds_above_base": all(v > BASE_REALDUP_R1 for v in rd["values"]),
            "verdict": ("P-M2 REFUTATION HOLDS — the attacked model keeps real "
                        "equivalence ability; the incentive inversion is "
                        "MathNet-specific"
                        if rd["mean"] > BASE_REALDUP_R1 else
                        "P-M2 REFUTATION WITHDRAWN — seed 42 was a lucky run; the "
                        "incentive-inversion scoping in ood.tex/limitations.tex "
                        "must be revised"),
        }

    ho = agg["heldout"]["meld_R@1"]
    hg = [v - BASE_MELD_R1 for v in ho["values"]]
    verdicts["R3_P4_genre_reading"] = {
        "rule": "heldout three-seed mean clears base by >= +10.0",
        "per_seed_gain_over_base": [round(g, 4) for g in hg],
        "mean_gain": round(statistics.mean(hg), 4) if hg else None,
        "verdict": ("PENDING — %d of %d seeds present" % (len(hg), len(SEEDS))
                    if len(hg) < len(SEEDS) else
                    "GENRE-LEVEL HOLDS" if statistics.mean(hg) >= 10.0
                    else "GENRE READING WITHDRAWN"),
    }

    # domain increment, paired by seed
    inc = [per_seed["meld9"][s]["meld_R@1"] - per_seed["heldout"][s]["meld_R@1"]
           for s in SEEDS
           if per_seed["meld9"][s]["meld_R@1"] is not None
           and per_seed["heldout"][s]["meld_R@1"] is not None]
    verdicts["domain_increment_meld9_minus_heldout"] = {
        "note": "R@1 only; at seed 42 it was null at R@5 and on the distractor "
                "metric. Not claimed as replicated unless all seeds keep the sign.",
        "per_seed": [round(d, 4) for d in inc],
        "mean": round(statistics.mean(inc), 4) if inc else None,
        "sd": round(statistics.stdev(inc), 4) if len(inc) > 1 else None,
        "signs_all_agree": (len({d > 0 for d in inc}) == 1) if inc else None,
        "verdict": ("REPLICATED" if len(inc) == 3 and len({d > 0 for d in inc}) == 1
                    else "NOT REPLICATED" if len(inc) == 3 else "PENDING"),
    }

    doc = {
        "generated": str(date.today()),
        "script": "scripts/meld_seed_aggregate.py",
        "slurm_job": "scripts/meld_attack_seeds.slurm (seeds 43/44; seed 42 from 49639491)",
        "seeds": list(SEEDS),
        "reference_points": {
            "base_meld_R@1": BASE_MELD_R1,
            "base_realdup_R@1": BASE_REALDUP_R1,
            "ctrl_llm_meld_R@1": CTRL_LLM_MELD_R1,
            "realdup_single_run_sd": SD1_REALDUP,
        },
        "scope_warning": (
            "These are TRAINING-SEED statistics. The paired cluster-bootstrap CIs "
            "in results/meld_compare_*.json resample MELD's 270 pairs and carry no "
            "seed variance; the two uncertainties are additive and neither may be "
            "quoted as the total."),
        "data_provenance": (
            "No data was regenerated: seeds 43/44 train on the same gated pair "
            "files as seed 42, so results/meld_disjointness_*.json still certifies "
            "exactly what these models saw."),
        "per_seed": {a: {str(s): per_seed[a][s] for s in SEEDS} for a in ARMS},
        "aggregate": agg,
        "preregistered_verdicts": verdicts,
        "missing_inputs": sorted(set(missing)),
    }
    out = os.path.join(RESULTS, "meld_seed_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    def cell(d):
        if not d["n"]:
            return "     --     "
        sd = f" ±{d['sd']:5.2f}" if d["sd"] is not None else "       "
        return f"{d['mean']:6.2f}{sd} n={d['n']}"

    print(f"\n{'arm':10s} {'MELD R@1':>20s} {'MELD R@5':>20s} {'real-dup R@1':>20s}")
    for a in ARMS:
        print(f"{a:10s} {cell(agg[a]['meld_R@1']):>20s} "
              f"{cell(agg[a]['meld_R@5']):>20s} {cell(agg[a]['realdup_R@1']):>20s}")
    print(f"{'(base)':10s} {BASE_MELD_R1:>20.2f} {'34.26':>20s} {BASE_REALDUP_R1:>20.2f}")
    if agg["meld9"].get("mathnet_hard_R@1", {}).get("n"):
        print(f"\nmeld9 MathNet hard R@1 (P-M3b): {cell(agg['meld9']['mathnet_hard_R@1'])}")

    print("\n--- pre-registered verdicts ---")
    for k, v in verdicts.items():
        print(f"  {k}: {v.get('verdict')}")
        if "per_seed_gain_over_base" in v:
            print(f"      per-seed gain over base: {v['per_seed_gain_over_base']}")
        if "margin_over_base" in v:
            print(f"      mean {v['mean']} = base {BASE_REALDUP_R1} {v['margin_over_base']:+.2f} "
                  f"({v.get('margin_in_se_units')} se), all seeds above base: "
                  f"{v['all_seeds_above_base']}")
        if "per_seed" in v:
            print(f"      per-seed: {v['per_seed']}  signs agree: {v['signs_all_agree']}")

    if doc["missing_inputs"]:
        print("\nMISSING INPUTS (reported as nulls, not skipped):")
        for m in doc["missing_inputs"]:
            print(f"  {m}")
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
