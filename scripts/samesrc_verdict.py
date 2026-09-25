#!/usr/bin/env python3
"""Same-source regenerated control, step 2e (2026-09-17): registered readout of the
same-source regenerated control, written BEFORE any samesrc number existed; the
verdict branch is computed in code.

Reads results/ranks/{easy,medium,hard,xling}_samesrc-llm-s{42..49}.summary.json
against the SAME files for cleanfull-cas (the verified arm under the corrected
gate at 6,145 rows, not retrained) and writes results/samesrc_verdict.json:

  PRIMARY  eight-seed mean easy-tier gap (samesrc-llm - cleanfull-cas)
           G1 CONFIRMS  gap >= 35.0 and positive in all 8 seeds
           G2 PARTIAL   15.0 <= gap < 35.0, all seeds positive
           G3 REFUTES   gap < 15.0
  CONTROL  the real-duplicate gap (xling strict R@1): nearer the filtered
           column's value (source lists shared: the regenerated column's
           movement was the sources) or nearer the regenerated column's
           value (the movement was the budget). Both references are
           recomputed here from the committed result files.
  Also     per-seed gaps, means and seed SDs (ddof=1) for every tier, and the
           difference-in-differences (easy gap minus xling gap) per seed.
"""
import json, os, sys

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(42, 50))
TIERS = ("easy", "medium", "hard", "xling")


def read_rank(tier, tag, seed):
    p = os.path.join(PROJECT, "results", "ranks", f"{tier}_{tag}-s{seed}.summary.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d["strict_crosslingual_gold"]["recall@1"] if tier == "xling" else d["overall"]["recall@1"]


def read_filtered(tier, arm, seed):
    """The filtered retrain (corrected gate, 4,101 rows): results/eval_<tier>_ctrl-<arm>-v2gate-4101-s<seed>.json."""
    if tier == "xling":
        p = os.path.join(PROJECT, "results", f"crosslingual_ctrl-{arm}-v2gate-4101-s{seed}.json")
        return json.load(open(p))["strict_crosslingual_gold"]["recall@1"]
    p = os.path.join(PROJECT, "results", f"eval_{tier}_ctrl-{arm}-v2gate-4101-s{seed}.json")
    return json.load(open(p))["overall"]["recall@1"]


def stats(llm, cas):
    gaps = [a - b for a, b in zip(llm, cas)]
    return {"llm_mean": round(float(np.mean(llm)), 2), "cas_mean": round(float(np.mean(cas)), 2),
            "gap_mean": round(float(np.mean(gaps)), 2),
            "gap_sd": round(float(np.std(gaps, ddof=1)), 2) if len(gaps) > 1 else None,
            "gap_per_seed": [round(g, 2) for g in gaps], "llm_per_seed": llm, "cas_per_seed": cas,
            "n_seeds": len(gaps)}


def main():
    out = {"generated_by": "scripts/samesrc_verdict.py", "seeds": SEEDS, "tiers": {}, "reference": {}}
    # references recomputed from committed files: the regenerated and the filtered columns of tab:cleangate
    reg = json.load(open(os.path.join(PROJECT, "results", "cleanfull_verdict.json")))["tiers"]
    out["reference"]["regenerated"] = {t: reg[t]["gap_mean"] for t in TIERS if t in reg}
    try:
        out["reference"]["filtered"] = {t: round(float(np.mean([read_filtered(t, "llm", s) - read_filtered(t, "cas", s)
                                                              for s in SEEDS])), 2) for t in ("easy", "hard", "xling")}
    except FileNotFoundError as e:
        out["reference"]["filtered"] = {"missing": str(e)}
    out["reference"]["headline_tab_cleangate"] = {"easy": 45.33, "hard": 9.26, "xling": 9.74, "did": 35.59}
    for r in ("regenerated", "filtered"):
        ref = out["reference"][r]
        if "easy" in ref and "xling" in ref:
            ref["did"] = round(ref["easy"] - ref["xling"], 2)
    missing = []
    for tier in TIERS:
        llm, cas = [], []
        for s in SEEDS:
            try:
                a, b = read_rank(tier, "samesrc-llm", s), read_rank(tier, "cleanfull-cas", s)
            except FileNotFoundError as e:
                missing.append(f"{tier}_s{s}: {os.path.basename(str(e).split(chr(39))[1]) if chr(39) in str(e) else e}")
                continue
            llm.append(a); cas.append(b)
        if llm:
            out["tiers"][tier] = stats(llm, cas)
    if missing:
        out["missing"] = missing
        print(f"[warn] {len(missing)} summaries missing; verdict is provisional", file=sys.stderr)
    easy, xl = out["tiers"].get("easy"), out["tiers"].get("xling")
    if easy and xl and easy["n_seeds"] == xl["n_seeds"]:
        did = [e - x for e, x in zip(easy["gap_per_seed"], xl["gap_per_seed"])]
        out["difference_in_differences"] = {"mean": round(float(np.mean(did)), 2),
                                            "sd": round(float(np.std(did, ddof=1)), 2), "per_seed": [round(d, 2) for d in did]}
    if easy and easy["n_seeds"] == len(SEEDS):
        g, allpos = easy["gap_mean"], all(x > 0 for x in easy["gap_per_seed"])
        if g >= 35.0 and allpos:
            v = "G1 CONFIRMS: the easy-tier gap survives a same-source, fully-clean matched 6,145 budget."
        elif g >= 15.0 and allpos:
            v = "G2 PARTIAL: effect real but materially smaller than published."
        else:
            v = "G3 REFUTES: contamination section must be rewritten."
        out["verdict"] = v
        print(v)
    if xl and xl["n_seeds"] == len(SEEDS) and "xling" in out["reference"].get("filtered", {}):
        f, r, g = out["reference"]["filtered"]["xling"], out["reference"]["regenerated"]["xling"], xl["gap_mean"]
        nearer = "filtered (the regenerated column's real-duplicate movement was the source lists)" if abs(g - f) < abs(g - r) \
            else "regenerated (the movement was the budget, not the sources)"
        out["real_duplicate_control"] = {"samesrc_gap": g, "filtered_ref": f, "regenerated_ref": r,
                                         "distance_to_filtered": round(abs(g - f), 2), "distance_to_regenerated": round(abs(g - r), 2),
                                         "nearer": nearer}
        print(f"real-duplicate gap {g} is nearer the {nearer}")
    if not out["tiers"]:
        raise SystemExit("[samesrc] no results/ranks/*_samesrc-llm-s*.summary.json found: run "
                         "sbatch scripts/samesrc_train.slurm first; results/samesrc_verdict.json left unchanged")
    with open(os.path.join(PROJECT, "results", "samesrc_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("tiers", "difference_in_differences", "reference") if k in out}, indent=1))


if __name__ == "__main__":
    main()
