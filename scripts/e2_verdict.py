#!/usr/bin/env python3
"""Pre-registered verdict for E2, the source- and negative-matched CAS arm.

The bands below restate scripts/e2_negmatched.slurm's header and were fixed
before the job was submitted; this script evaluates them IN CODE so the
branch cannot be chosen after the numbers are seen.

  PRIMARY READOUT: the three-seed mean easy-tier R@1 of the negmatched arm
  (models/ctrl-cas-negmatched-6145-s{42,43,44}); reference points are the
  eight-seed ctrl-LLM 62.38 +/- 0.85, ctrl-CAS 17.05 +/- 0.79 and phase-1
  pure-CAS 15.92.

  E1 CONFIRMS  mean easy R@1 <= 25.0 AND (62.38 - easy) positive per seed:
               matching rows, sources, negative volume and steps does NOT
               close the arm gap -- the training-signal asymmetry is not a
               first-order driver and the recipe explanation stands.
  E2 PARTIAL   25.0 < mean < 40.0: a material share of the published gap is
               training signal; setup.tex must concede the share explicitly.
  E3 REFUTES   mean >= 40.0: the asymmetry is a first-order driver; the
               controlled-experiment interpretation must be rebuilt.

  SECONDARY (reported, ungated): hard-tier R@1, real-duplicate strict R@1,
  realized rows/negatives/steps from run_config.json.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
NAME = "ctrl-cas-negmatched-6145"
CTRL_LLM_EASY_8SEED = 62.3825

def load(path, *keys):
    try:
        v = json.load(open(path))
        for k in keys:
            v = v[k]
        return float(v)
    except (FileNotFoundError, KeyError):
        return None

def main():
    easy, hard, xling, cfg = {}, {}, {}, {}
    for s in SEEDS:
        tag = f"{NAME}-s{s}"
        e = load(ROOT / f"results/eval_easy_{tag}.json", "overall", "recall@1")
        h = load(ROOT / f"results/eval_hard_{tag}.json", "overall", "recall@1")
        x = load(ROOT / f"results/crosslingual_{tag}.json", "strict_crosslingual_gold", "recall@1")
        # eval_retrieve.py / eval_crosslingual.py store recall@1 in PERCENT
        # (easy ~17.5, hard ~0.4); no rescaling. A <=1 heuristic mangled the
        # hard tier on the first pass (0.41% -> 41.0) and is deliberately gone.
        if e is not None:
            easy[s] = e
        if h is not None:
            hard[s] = h
        if x is not None:
            xling[s] = x
        rc = ROOT / f"models/{tag}/run_config.json"
        if rc.exists():
            d = json.load(open(rc)).get("data_stats", {})
            cfg[s] = {k: d.get(k) for k in
                      ("rows_used", "n_source_ids", "n_train_triplets",
                       "n_train_pairs", "n_dev_source_ids")}

    out = {
        "experiment": "E2 source- and negative-matched CAS arm",
        "construction": "results/e2_negmatched_construction.json",
        "seeds_found": sorted(easy),
        "seeds_looked_for": list(SEEDS),
        "easy_per_seed": easy, "hard_per_seed": hard, "xling_per_seed": xling,
        "run_config_checks": cfg,
        "reference": {"ctrl_llm_easy_8seed": CTRL_LLM_EASY_8SEED,
                      "ctrl_cas_easy_8seed": 17.05, "phase1_cas_easy": 15.92},
    }
    if len(easy) < len(SEEDS):
        out["verdict"] = (f"INCOMPLETE -- only {len(easy)} of {len(SEEDS)} "
                          "easy-tier seeds on disk; no pre-registered branch is taken.")
        print(json.dumps(out, indent=2))
        Path(ROOT / "results/e2_negmatched_stats.json").write_text(json.dumps(out, indent=2) + "\n")
        return 1

    mean = statistics.mean(easy.values())
    sd = statistics.stdev(easy.values())
    gaps = {s: CTRL_LLM_EASY_8SEED - v for s, v in easy.items()}
    out["easy_mean_sd"] = [round(mean, 4), round(sd, 4)]
    out["gap_vs_ctrl_llm_per_seed"] = {s: round(g, 4) for s, g in gaps.items()}
    for label, vals in (("hard", hard), ("xling", xling)):
        if len(vals) == len(SEEDS):
            out[f"{label}_mean_sd"] = [round(statistics.mean(vals.values()), 4),
                                       round(statistics.stdev(vals.values()), 4)]
    if mean <= 25.0 and all(g > 0 for g in gaps.values()):
        out["verdict"] = (f"E1 CONFIRMS -- {len(easy)}-seed mean easy R@1 {mean:.2f} <= 25.0 and the "
                          "gap to ctrl-LLM is positive in every seed: the training-signal asymmetry "
                          "is not a first-order driver of the arm gap.")
    elif mean < 40.0:
        out["verdict"] = (f"E2 PARTIAL -- {len(easy)}-seed mean easy R@1 {mean:.2f} in (25, 40): a "
                          "material share of the published gap is training signal; setup.tex must "
                          "concede the share explicitly.")
    else:
        out["verdict"] = (f"E3 REFUTES -- {len(easy)}-seed mean easy R@1 {mean:.2f} >= 40.0: the "
                          "training-signal asymmetry is a first-order driver; the "
                          "controlled-experiment interpretation must be rebuilt.")
    print(json.dumps(out, indent=2))
    Path(ROOT / "results/e2_negmatched_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
