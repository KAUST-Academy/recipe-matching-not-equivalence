#!/usr/bin/env python3
"""
Seed / backbone statistics for the supervision-controlled experiment
(statistical hardening; see README "Headline results").

Aggregates, across training seeds (default 42/43/44, Qwen3-Embedding-0.6B),
the key cells of the controlled experiment:

  * MathNet-Retrieve benchmark easy/medium/hard R@1  (per arm)
  * real-duplicate (cross-lingual) STRICT R@1        (per arm)

and gives a PAIRED read of the LLM-minus-CAS gap (the circularity signal:
huge on the benchmark's hard tier, ~zero on real duplicates), computed
per-seed then summarized as mean +/- sample std.

File conventions (produced by scripts/train_ctrl_param.slurm):
  results/eval_{tier}_ctrl-{arm}-s{seed}.json
  results/crosslingual_ctrl-{arm}-s{seed}.json
Legacy seed-42 names from train_controlled_exp.slurm / eval_ctrl_ood.slurm
are used as fallback:
  results/eval_{tier}_ctrl-{arm}.json
  results/crosslingual_ctrl-{arm}-6145.json

Non-Qwen backbone replications (single seed) are reported as separate
sections via --backbone-tags (default: e5-s42 = intfloat/multilingual-e5-large,
seed 42), reading results/eval_{tier}_ctrl-{arm}-{tag}.json etc.

Missing files are reported and skipped -- the script is safe to run before
all seeds have finished (it then prints stats over whatever exists).

Usage (login node, seconds):
  python scripts/aggregate_ctrl_stats.py
  python scripts/aggregate_ctrl_stats.py --seeds 42,43,44 --backbone-tags e5-s42

Output: results/ctrl_seed_stats.json + printed table.
"""

import argparse
import json
import math
import os
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")

ARMS = ("cas", "llm")
TIERS = ("easy", "medium", "hard")
KS = ("recall@1", "recall@5", "recall@10")
# the four key cells of the printed table
KEY_METRICS = [("easy", "recall@1"), ("medium", "recall@1"),
               ("hard", "recall@1"), ("xling_strict", "recall@1")]


def _first_existing(cands):
    for c in cands:
        p = os.path.join(RESULTS, c)
        if os.path.exists(p):
            return p
    return None


def load_arm_numbers(arm, tag, legacy_seed42=False):
    """Return ({metric_name: {recall@k: float}}, [missing file descriptions]).

    metric_name in TIERS + ('xling_strict',). tag e.g. 's43' or 'e5-s42'.
    """
    out, missing = {}, []
    for tier in TIERS:
        cands = [f"eval_{tier}_ctrl-{arm}-{tag}.json"]
        if legacy_seed42:
            cands.append(f"eval_{tier}_ctrl-{arm}.json")
        p = _first_existing(cands)
        if p is None:
            missing.append(cands[0])
            continue
        overall = json.load(open(p, encoding="utf-8"))["overall"]
        out[tier] = {k: float(overall[k]) for k in KS}
    cands = [f"crosslingual_ctrl-{arm}-{tag}.json"]
    if legacy_seed42:
        cands.append(f"crosslingual_ctrl-{arm}-6145.json")
    p = _first_existing(cands)
    if p is None:
        missing.append(cands[0])
    else:
        strict = json.load(open(p, encoding="utf-8"))["strict_crosslingual_gold"]
        out["xling_strict"] = {k: float(strict[k]) for k in KS}
    return out, missing


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None, 0
    mu = sum(vals) / n
    if n < 2:
        return mu, None, n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))  # sample std
    return mu, sd, n


def fmt(mu, sd, n):
    if mu is None:
        return "      --      "
    if sd is None:
        return f"{mu:6.2f} (n=1)  "
    return f"{mu:6.2f} ±{sd:5.2f} (n={n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated Qwen3-0.6B seeds (default 42,43,44)")
    ap.add_argument("--backbone-tags", default="e5-s42",
                    help="comma-separated TAGs of non-Qwen backbone runs to "
                         "report as single-run sections ('' to skip)")
    ap.add_argument("--output", default=os.path.join(RESULTS, "ctrl_seed_stats.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    btags = [t.strip() for t in args.backbone_tags.split(",") if t.strip()]

    report = {"generated": str(date.today()),
              "seeds_requested": seeds,
              "note": ("sample std (ddof=1) across seeds; paired gap = "
                       "per-seed (llm - cas), then mean±std. ctrl-llm's "
                       "benchmark numbers are the CIRCULARITY EXHIBIT, "
                       "never capability."),
              "per_seed": {}, "aggregate": {}, "paired_gap_llm_minus_cas": {},
              "backbone_replications": {}, "missing_files": []}

    # ---------------- per-seed numbers ----------------
    per_seed = {}   # seed -> arm -> metric -> {recall@k}
    for seed in seeds:
        per_seed[seed] = {}
        for arm in ARMS:
            nums, missing = load_arm_numbers(arm, f"s{seed}",
                                             legacy_seed42=(seed == 42))
            per_seed[seed][arm] = nums
            report["missing_files"] += [f"seed{seed}/{arm}: {m}" for m in missing]
    report["per_seed"] = {str(s): per_seed[s] for s in seeds}

    # ---------------- aggregates + paired gaps ----------------
    for metric, k in [(m, kk) for m in TIERS + ("xling_strict",) for kk in KS]:
        for arm in ARMS:
            vals = [per_seed[s][arm][metric][k] for s in seeds
                    if metric in per_seed[s][arm]]
            mu, sd, n = mean_std(vals)
            report["aggregate"].setdefault(metric, {}).setdefault(arm, {})[k] = {
                "mean": mu, "std": sd, "n": n, "values": vals}
        gaps = [per_seed[s]["llm"][metric][k] - per_seed[s]["cas"][metric][k]
                for s in seeds
                if metric in per_seed[s]["llm"] and metric in per_seed[s]["cas"]]
        mu, sd, n = mean_std(gaps)
        report["paired_gap_llm_minus_cas"].setdefault(metric, {})[k] = {
            "mean": mu, "std": sd, "n": n, "values": gaps}

    # ---------------- backbone replications (single runs) ----------------
    for tag in btags:
        sec = {}
        for arm in ARMS:
            nums, missing = load_arm_numbers(arm, tag)
            sec[arm] = nums
            report["missing_files"] += [f"{tag}/{arm}: {m}" for m in missing]
        report["backbone_replications"][tag] = sec

    # ---------------- printed table ----------------
    have = [s for s in seeds if any(per_seed[s][a] for a in ARMS)]
    print(f"\n=== Controlled experiment, Qwen3-Embedding-0.6B — seeds with "
          f"data: {have or 'NONE'} (requested {seeds}) ===")
    print(f"{'metric':<22}| {'cas mean±std':<22}| {'llm mean±std':<22}| "
          f"paired gap llm−cas (per-seed values)")
    print("-" * 100)
    for metric, k in KEY_METRICS:
        label = f"{metric} R@1" if metric != "xling_strict" else "real-dup strict R@1"
        cells = []
        for arm in ARMS:
            a = report["aggregate"][metric][arm][k]
            cells.append(fmt(a["mean"], a["std"], a["n"]))
        g = report["paired_gap_llm_minus_cas"][metric][k]
        gap = fmt(g["mean"], g["std"], g["n"]).strip()
        vals = ", ".join(f"{v:+.2f}" for v in g["values"])
        print(f"{label:<22}| {cells[0]:<22}| {cells[1]:<22}| {gap}  [{vals}]")

    for tag, sec in report["backbone_replications"].items():
        if not any(sec.values()):
            print(f"\n--- backbone replication {tag}: no result files yet ---")
            continue
        print(f"\n--- backbone replication {tag} (single run) ---")
        print(f"{'metric':<22}| {'cas':>8} | {'llm':>8} | gap llm−cas")
        for metric, k in KEY_METRICS:
            label = f"{metric} R@1" if metric != "xling_strict" else "real-dup strict R@1"
            c = sec["cas"].get(metric, {}).get(k)
            l = sec["llm"].get(metric, {}).get(k)
            gap = f"{l - c:+.2f}" if (c is not None and l is not None) else "--"
            cs = f"{c:.2f}" if c is not None else "--"
            ls = f"{l:.2f}" if l is not None else "--"
            print(f"{label:<22}| {cs:>8} | {ls:>8} | {gap}")

    if report["missing_files"]:
        print(f"\n[warn] {len(report['missing_files'])} missing result files "
              f"(runs not finished / not submitted):")
        for m in report["missing_files"]:
            print(f"  - {m}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
