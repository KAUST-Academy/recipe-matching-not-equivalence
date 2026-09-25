#!/usr/bin/env python3
"""
Paired per-query bootstrap CIs for ADJACENT-RUNG differences of the
prompt-proximity dose-response ladder (review finding M4: the two-regime
centerpiece rested on single seed-42 point estimates; adjacent-rung
differences like D1-D2 hard 1.48 are ~1.7-1.8 sigma of the arm's own
training-seed noise).

WHAT THIS DOES AND DOES NOT MEASURE  (read before quoting any CI):
  * The bootstrap resamples the 15,000 EVALUATION QUERIES with replacement,
    applying ONE shared resample to both models of a pair, so the per-query
    hit correlation is preserved. The resulting CI is an EVAL-SAMPLING CI: it
    answers "would this rung ordering survive a different draw of benchmark
    queries from the same pool?".
  * It does NOT contain training-seed variance. Two models trained on the
    same data with different seeds move by +-0.61 (hard R@1) / +-0.85 (easy
    R@1) for this arm (results/ctrl_seed_stats.json, 8 seeds), i.e. the sd of
    a DIFFERENCE of two independent runs is ~0.86 / ~1.20. Every comparison
    below therefore also reports `n_sigma_seed` = |diff| / that sd, and the
    printed table shows both. A tight eval CI with n_sigma_seed < 2 is NOT
    evidence that a re-trained model would reproduce the ordering; only the
    seed replicas from scripts/dose_seeds.slurm can settle that.
  * Real-duplicate (cross-lingual) ranks are deliberately NOT handled here:
    that eval has 393 queries in 370 mined clusters and needs the
    hierarchical CLUSTER bootstrap already implemented in
    scripts/bootstrap_stats.py (per-query resampling there is invalid).

Zero GPU: consumes the per-query --dump-ranks JSONL files
(results/ranks/{tier}_{tag}.ranks.jsonl, format of eval_retrieve.py:
{qid, gold_rank (exact, 0-based), gold_sim, top10_ids}).
R@k hit indicator = gold_rank < k.

Rung -> dump tag (seed 42 dumps already exist; seeds 43/44 come from
scripts/dose_seeds.slurm):
  D1 exact       ctrl-llm-s{S}         (models/ctrl-llm-6145* -- the D1
                                        reference IS the ctrl-LLM arm)
  D2 paraphrase  dose-paraphrase-s{S}
  D3 style       dose-style-s{S}
  D4 unrelated   dose-unrelated-s{S}
plus the H2 row-count control dose-paraphrase-5591-s42 (D2 retrained with
--max-rows 5591 = D3's actual row count, a NESTED prefix subsample of D2's
own shuffled row order; compared against both D2 and D3).

KNOWN COUPLING (round-2 review finding R2-M2): a single default_rng is
threaded through every comparison, so INSERTING a comparison shifts the
bootstrap stream for every comparison after it and moves their CIs by ~0.01.
That is why adding the H2 control's comparisons in 2026-07-31 changed the
hard D1-D2 and D3-D4 intervals the paper had quoted from an earlier run.
The released results/dose_bootstrap.json and the paper now agree; if you
change the comparison LIST, re-run this script and re-sync the appendix.
Proper fix (not applied, because it would move every published interval
again): give each comparison its own rng seeded from (rng_seed, pair, tier).

Percentile CIs. Two-sided bootstrap p = 2*min(P(diff<=0), P(diff>=0)) with
the +1 small-sample correction, so p is FLOORED at 2/(n_boot+1) and is
reported as "<floor" when it hits the floor (never as 0).

Integrity cross-check: dump-derived R@1 must match the canonical
results/eval_{tier}_*.json within 0.5 pt (same rule as bootstrap_stats.py);
mismatches are listed in `crosscheck_failures` and printed.

Usage (login node, CPU, seconds):
  python scripts/dose_bootstrap.py                     # auto-detect dumps
  python scripts/dose_bootstrap.py --n-boot 20000 --rng-seed 20260730
Output: results/dose_bootstrap.json + a printed table.
"""

import argparse
import json
import os
import statistics
from datetime import date

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS_DIR = os.path.join(PROJECT_ROOT, "results", "ranks")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SEED_STATS = os.path.join(RESULTS_DIR, "ctrl_seed_stats.json")

TIERS = ("easy", "hard")
K_VALUES = (1, 5, 10)
RUNGS = ("D1", "D2", "D3", "D4")
CONTROL_RUNG = "D2r5591"
ADJACENT = (("D1", "D2"), ("D2", "D3"), ("D3", "D4"))
# extra H2 comparisons: the D2-at-5591-rows control vs D2 (row-count effect at
# fixed prompt) and vs D3 (prompt effect at fixed row count)
CONTROL_PAIRS = (("D2", CONTROL_RUNG), (CONTROL_RUNG, "D3"))
VARIANT = {"D2": "paraphrase", "D3": "style", "D4": "unrelated"}


def dump_tag(rung: str, seed: int) -> str:
    if rung == "D1":
        return f"ctrl-llm-s{seed}"
    if rung == CONTROL_RUNG:
        return f"dose-paraphrase-5591-s{seed}"
    return f"dose-{VARIANT[rung]}-s{seed}"


def canonical_eval_paths(rung: str, seed: int, tier: str):
    """Candidate canonical eval JSONs for the cross-check (first hit wins)."""
    if rung == "D1":
        cands = [f"eval_{tier}_ctrl-llm-s{seed}.json"]
        if seed == 42:
            cands.append(f"eval_{tier}_ctrl-llm.json")   # legacy seed-42 name
    elif rung == CONTROL_RUNG:
        cands = [f"eval_{tier}_dose-paraphrase-5591.json"]
    else:
        cands = [f"eval_{tier}_dose-{VARIANT[rung]}-s{seed}.json"]
        if seed == 42:
            cands.append(f"eval_{tier}_dose-{VARIANT[rung]}.json")  # dose job
    return [os.path.join(RESULTS_DIR, c) for c in cands]


def load_ranks(tier: str, tag: str):
    """qid -> gold_rank (int) from a dump file, or (None, path) if absent."""
    path = os.path.join(RANKS_DIR, f"{tier}_{tag}.ranks.jsonl")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None, path
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["qid"]] = row["gold_rank"]
    return out, path


def canonical_r1(rung: str, seed: int, tier: str):
    for p in canonical_eval_paths(rung, seed, tier):
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)["overall"]["recall@1"], p
            except Exception:
                pass
    return None, None


def seed_noise_sd():
    """Training-seed sd of the LLM arm per tier/metric, from the 8 ctrl seeds.

    Returns {(tier, 'recall@k'): sd_of_one_run}. The sd of a DIFFERENCE of two
    independent runs is sqrt(2)x this (used for n_sigma_seed).
    """
    out = {}
    if not os.path.isfile(SEED_STATS):
        return out
    try:
        with open(SEED_STATS, encoding="utf-8") as f:
            stats = json.load(f)
        per_seed = stats["per_seed"]
        for tier in TIERS:
            for k in K_VALUES:
                metric = f"recall@{k}"
                vals = [per_seed[s]["llm"][tier][metric] for s in sorted(per_seed)
                        if tier in per_seed[s].get("llm", {})]
                if len(vals) > 1:
                    out[(tier, metric)] = {
                        "sd_single_run": round(statistics.stdev(vals), 4),
                        "n_seeds": len(vals),
                        "source": "results/ctrl_seed_stats.json per_seed.*.llm "
                                  "(D1 arm, identical data, seed-only variation)",
                    }
    except Exception:
        return out
    return out


def paired_boot(hits_a, hits_b, n_boot, rng):
    """Percentile CI + bootstrap p for mean(hits_a) - mean(hits_b), paired."""
    n = len(hits_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = hits_a[idx].mean(axis=1) - hits_b[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p_le = (np.sum(reps <= 0) + 1) / (n_boot + 1)
    p_ge = (np.sum(reps >= 0) + 1) / (n_boot + 1)
    return float(lo), float(hi), float(min(1.0, 2 * min(p_le, p_ge)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", default="42,43,44",
                    help="training seeds to look for (default 42,43,44)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--rng-seed", type=int, default=20260730)
    ap.add_argument("--output",
                    default=os.path.join(RESULTS_DIR, "dose_bootstrap.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(args.rng_seed)
    p_floor = 2.0 / (args.n_boot + 1)
    noise = seed_noise_sd()

    result = {
        "generated": date.today().isoformat(),
        "script": "scripts/dose_bootstrap.py",
        "design": "paired per-query bootstrap (one shared query resample per "
                  "pair), percentile CIs, R@k hit = gold_rank < k",
        "scope_warning": "CIs are EVAL-SAMPLING only (resampling the 15,000 "
                         "benchmark queries). They contain NO training-seed "
                         "variance; see n_sigma_seed on each comparison and "
                         "seed_noise_reference below.",
        "n_boot": args.n_boot,
        "rng_seed": args.rng_seed,
        "p_boot_floor": round(p_floor, 8),
        "seed_noise_reference": {f"{t}/{m}": v for (t, m), v in noise.items()},
        "comparisons": [],
        "cross_seed": [],
        "point_estimates": {},
        "crosscheck_failures": [],
        "missing_dumps": [],
    }
    missing = set()

    # ---- load every rung/tier/seed dump that exists ----
    ranks = {}   # (tier, rung, seed) -> {qid: gold_rank}
    for tier in TIERS:
        for seed in seeds:
            for rung in RUNGS + (CONTROL_RUNG,):
                if rung == CONTROL_RUNG and seed != 42:
                    continue   # the row-count control is seed-42 only
                tag = dump_tag(rung, seed)
                r, path = load_ranks(tier, tag)
                if r is None:
                    missing.add(os.path.relpath(path, PROJECT_ROOT))
                    continue
                ranks[(tier, rung, seed)] = r
                r1 = 100.0 * float(np.mean([v == 0 for v in r.values()]))
                key = f"{tier}/{rung}/s{seed}"
                entry = {"R@1_from_dump": round(r1, 2), "n_queries": len(r),
                         "dump": os.path.relpath(path, PROJECT_ROOT)}
                canon, cpath = canonical_r1(rung, seed, tier)
                if canon is not None:
                    entry["R@1_canonical"] = canon
                    entry["canonical_file"] = os.path.relpath(cpath, PROJECT_ROOT)
                    if abs(canon - r1) > 0.5:
                        result["crosscheck_failures"].append(
                            f"{key}: dump {r1:.2f} vs canonical {canon:.2f} "
                            f"({cpath})")
                else:
                    entry["R@1_canonical"] = None
                    result["crosscheck_failures"].append(
                        f"{key}: no canonical eval JSON found for cross-check")
                result["point_estimates"][key] = entry

    # ---- paired bootstrap per tier / seed / pair ----
    def pstr(p):
        return f"<{p_floor:.1e}" if p <= p_floor + 1e-12 else f"{p:.5f}"

    print("EVAL-SAMPLING CIs (query resample). n_sig = |diff| / seed-diff sd "
          "(training-seed noise, NOT covered by the CI).")
    print(f"{'tier':5s} {'seed':5s} {'pair':16s} {'R@1 diff':>9s} "
          f"{'95% CI (eval)':>18s} {'p(boot)':>10s} {'n_sig':>6s} {'nq':>6s}")
    per_seed_diffs = {}   # (tier, pair, k) -> {seed: diff}
    for tier in TIERS:
        for seed in seeds:
            for a, b in ADJACENT + CONTROL_PAIRS:
                ra = ranks.get((tier, a, seed))
                rb = ranks.get((tier, b, seed))
                if ra is None or rb is None:
                    continue
                qids = sorted(set(ra) & set(rb))
                if len(qids) != len(ra) or len(qids) != len(rb):
                    print(f"[warn] {tier} s{seed} {a}-{b}: qid sets differ "
                          f"({len(ra)} vs {len(rb)}, {len(qids)} common)")
                for k in K_VALUES:
                    metric = f"recall@{k}"
                    ha = np.array([ra[q] < k for q in qids], dtype=np.float64)
                    hb = np.array([rb[q] < k for q in qids], dtype=np.float64)
                    diff = 100.0 * (ha.mean() - hb.mean())
                    lo, hi, p = paired_boot(ha, hb, args.n_boot, rng)
                    lo, hi = 100.0 * lo, 100.0 * hi
                    sd1 = noise.get((tier, metric), {}).get("sd_single_run")
                    sd_diff = None if sd1 is None else sd1 * (2 ** 0.5)
                    n_sig = None if not sd_diff else abs(diff) / sd_diff
                    comp = {
                        "tier": tier, "seed": seed, "pair": f"{a}-{b}",
                        "metric": f"R@{k}",
                        "diff": round(diff, 2),
                        "ci95_eval": [round(lo, 2), round(hi, 2)],
                        "p_boot_two_sided": round(p, 6),
                        "p_boot_at_floor": bool(p <= p_floor + 1e-12),
                        "n_queries": len(qids),
                        "eval_ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "seed_diff_sd": None if sd_diff is None else round(sd_diff, 3),
                        "n_sigma_seed": None if n_sig is None else round(n_sig, 2),
                        "exceeds_2sigma_seed": None if n_sig is None else bool(n_sig >= 2.0),
                    }
                    result["comparisons"].append(comp)
                    per_seed_diffs.setdefault((tier, f"{a}-{b}", k), {})[seed] = diff
                    if k == 1:
                        print(f"{tier:5s} s{seed:<4d} {a + '-' + b:16s} "
                              f"{diff:9.2f} [{lo:7.2f}, {hi:7.2f}] "
                              f"{pstr(p):>10s} "
                              f"{'  --  ' if n_sig is None else f'{n_sig:6.2f}'} "
                              f"{len(qids):6d}")

    # ---- cross-seed aggregation (only meaningful once 43/44 dumps exist) ----
    print("\nCROSS-SEED replication of each adjacent-rung difference "
          "(this is the variance the eval CI cannot see):")
    print(f"{'tier':5s} {'pair':16s} {'metric':7s} {'n_seeds':>7s} "
          f"{'mean diff':>10s} {'sd':>7s} {'sign-consistent':>16s}")
    for (tier, pair, k), by_seed in sorted(per_seed_diffs.items()):
        vals = [by_seed[s] for s in sorted(by_seed)]
        row = {
            "tier": tier, "pair": pair, "metric": f"R@{k}",
            "n_seeds": len(vals),
            "per_seed_diff": {str(s): round(by_seed[s], 2) for s in sorted(by_seed)},
            "mean_diff": round(statistics.mean(vals), 2),
            "sd_diff": None if len(vals) < 2 else round(statistics.stdev(vals), 2),
            "sign_consistent": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals)),
            "single_seed_only": len(vals) < 2,
        }
        result["cross_seed"].append(row)
        if k == 1:
            sd = "  --  " if row["sd_diff"] is None else f"{row['sd_diff']:7.2f}"
            print(f"{tier:5s} {pair:16s} {'R@1':7s} {len(vals):7d} "
                  f"{row['mean_diff']:10.2f} {sd:>7s} "
                  f"{str(row['sign_consistent']):>16s}"
                  f"{'   (single seed — not yet replicated)' if row['single_seed_only'] else ''}")

    result["missing_dumps"] = sorted(missing)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"\n[done] {len(result['comparisons'])} comparisons "
          f"({len(missing)} dumps still missing) -> {args.output}")
    if missing:
        print("[note] missing dumps (scripts/dose_seeds.slurm produces these):")
        for m in sorted(missing):
            print("   -", m)
    if result["crosscheck_failures"]:
        print("[WARN] canonical cross-check failures:")
        for c in result["crosscheck_failures"]:
            print("   -", c)
    else:
        print("[ok] every loaded dump reproduces its canonical eval R@1 "
              "within 0.5 pt")


if __name__ == "__main__":
    main()
