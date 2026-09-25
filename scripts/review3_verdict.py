#!/usr/bin/env python3
"""2026-09-05: compute every registered readout of E-R10..E-R15
from the result files and write results/review3_verdict.json.

Registrations (all written before any result): the E-R10..E-R13 block of
scripts/factorial_cells.slurm; E-R15's readout in the header of
scripts/calib4b.slurm; E-R14's band as stated below. Bands:
2*sigma_1 = 1.72 hard / 2.40 easy / 6.54 cross-lingual points; for the probe
2*sigma_probe with sigma_probe = sqrt(2) * sd of the verified arm's eight seeds.

Cells and their rank-dump tags (results/ranks/<eval>_<tag>-s<seed>.summary.json):
  recipe arm ctrl-llm (8 seeds), verified arm ctrl-cas (8), reference
  fact-d4casnegs (8), D4 alone dose-unrelated (3), d4llmnegsfull
  fact-d4llmnegsfull (3), BT fact-btcasnegs (3), and the new cells
  fact-d5nonegs / fact-d5casnegs / fact-d1split / fact-d1twojudge /
  fact-bt2casnegs (3 each). Missing files are reported as pending, never
  invented.
"""
import glob
import json
import math
import statistics
import os
import re
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = os.path.join(PROJECT, "results", "ranks")
CELLS = {
    "recipe_arm": "ctrl-llm", "verified_arm": "ctrl-cas", "reference": "fact-d4casnegs",
    "d4_alone": "dose-unrelated", "d4llmnegsfull": "fact-d4llmnegsfull", "bt": "fact-btcasnegs",
    "d5nonegs": "fact-d5nonegs", "d5casnegs": "fact-d5casnegs", "d1split": "fact-d1split",
    "d1twojudge": "fact-d1twojudge", "bt2casnegs": "fact-bt2casnegs", "base": "qwen3-0.6b-base",
    # 2026-09-06: the remaining factorial cells, so the probe listing covers every hard-tier level
    "d4llmnegs_matched": "fact-d4llmnegs", "unrelnegs_matched": "fact-d4unrelnegs",
    "unrelnegs_full": "fact-d4unrelnegsfull", "cas_llmnegs": "fact-casllmnegs",
    "cas_nonegs": "fact-casnonegs",
}
BANDS = {"easy": 2.40, "hard": 1.72, "xling": 6.54}
PAPER = {"recipe_specific_easy": 15.47, "did_vs_reference": 27.24,
         "reference": {"easy": 46.91, "hard": 10.30, "xling": 78.02},
         "recipe_arm": {"easy": 62.38, "hard": 9.42, "xling": 66.26},
         "d4_alone": {"easy": 30.03, "hard": 0.04, "xling": 93.81},
         "d4llmnegsfull_hard": 26.97, "bt_char3": 0.783, "d4_char3": 0.421,
         "published_4b_easy_r1": 14.76}


def summary_value(path, ev):
    d = json.load(open(path, encoding="utf-8"))
    if ev == "xling":
        return d["strict_crosslingual_gold"]["recall@1"]
    if ev in ("casprobe", "casprobepairs"):
        return d["overall"]["recall@1"]
    return d["overall"]["recall@1"]


def seeds_of(tag, ev):
    out = {}
    for p in glob.glob(os.path.join(RANKS, f"{ev}_{tag}-s*.summary.json")):
        m = re.search(r"-s(\d+)\.summary\.json$", p)
        if m:
            try:
                out[int(m.group(1))] = summary_value(p, ev)
            except (KeyError, json.JSONDecodeError):
                pass
    if not out and ev in ("easy", "hard") and tag == "qwen3-0.6b-base":
        p = os.path.join(PROJECT, "results", f"eval_{ev}_qwen3-0.6b-base.json")
        if os.path.exists(p):
            out[0] = json.load(open(p))["overall"]["recall@1"]
    if not out and ev == "xling" and tag == "qwen3-0.6b-base":
        p = os.path.join(PROJECT, "results", "crosslingual_qwen3-0.6b-base.json")
        if os.path.exists(p):
            out[0] = json.load(open(p))["strict_crosslingual_gold"]["recall@1"]
    return dict(sorted(out.items()))


def samelang_primary(tag):
    """same-language PRIMARY-slice R@1 per seed from the rank dumps."""
    qpath = os.path.join(PROJECT, "data", "samelang_eval", "queries.jsonl")
    if not os.path.exists(qpath):
        return {}
    meta = {q["_id"]: q["metadata"] for q in (json.loads(l) for l in open(qpath, encoding="utf-8"))}
    primary = {q for q, m in meta.items() if not m["exact_text_cluster"] and m["clean_of_training"]}
    out = {}
    for p in glob.glob(os.path.join(RANKS, f"samelang_{tag}-s*.ranks.jsonl")):
        seed = int(re.search(r"-s(\d+)\.ranks\.jsonl$", p).group(1))
        hits = [json.loads(l) for l in open(p, encoding="utf-8")]
        sel = [h for h in hits if h["qid"] in primary]
        if sel:
            out[seed] = round(100 * sum(h["same_rank"] == 0 for h in sel) / len(sel), 2)
    return dict(sorted(out.items()))


def probe_r1(tag, prefix):
    out = {}
    for p in glob.glob(os.path.join(RANKS, f"{prefix}_{tag}-s*.ranks.jsonl")):
        seed = int(re.search(r"-s(\d+)\.ranks\.jsonl$", p).group(1))
        hits = [json.loads(l) for l in open(p, encoding="utf-8")]
        if hits:
            out[seed] = round(100 * sum(h["gold_rank"] == 0 for h in hits) / len(hits), 2)
    if not out:
        for p in glob.glob(os.path.join(RANKS, f"{prefix}_{tag}.ranks.jsonl")):
            hits = [json.loads(l) for l in open(p, encoding="utf-8")]
            if hits:
                out[0] = round(100 * sum(h["gold_rank"] == 0 for h in hits) / len(hits), 2)
    return dict(sorted(out.items()))


def stats(vals):
    v = list(vals.values())
    if not v:
        return None
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"n": len(v), "mean": round(mean, 2), "sd": round(sd, 2), "per_seed": {str(k): x for k, x in vals.items()}}


def verdict(diff, band, inside, above, below):
    if diff is None:
        return "PENDING"
    if abs(diff) <= band:
        return inside
    return above if diff > 0 else below


def main():
    cells = {}
    for name, tag in CELLS.items():
        cells[name] = {"tag": tag}
        for ev in ("easy", "medium", "hard", "xling"):
            cells[name][ev] = stats(seeds_of(tag, ev))
        cells[name]["samelang_primary"] = stats(samelang_primary(tag))
        cells[name]["casprobe"] = stats(probe_r1(tag, "casprobe"))
        cells[name]["casprobepairs"] = stats(probe_r1(tag, "casprobepairs"))

    def m(name, ev):
        s = cells[name].get(ev)
        if name == "d4_alone" and ev in PAPER["d4_alone"] and (not s or s["n"] < 3):
            return PAPER["d4_alone"][ev]     # the three-seed dose means (never the seed-42 singles)
        return s["mean"] if s else None

    def d(a, b, ev):
        x, y = m(a, ev), m(b, ev)
        return round(x - y, 2) if x is not None and y is not None else None

    out = {"cells": cells, "bands": BANDS, "paper_numbers": PAPER, "readouts": {}}
    R = out["readouts"]
    # ---- E-R10
    gap5 = d("recipe_arm", "d5casnegs", "easy")
    R["R10-a"] = {"recipe_minus_d5casnegs_easy": gap5, "paper_recipe_specific_share": PAPER["recipe_specific_easy"],
                  "diff": None if gap5 is None else round(gap5 - PAPER["recipe_specific_easy"], 2),
                  "verdict": verdict(None if gap5 is None else gap5 - PAPER["recipe_specific_easy"], BANDS["easy"],
                                     "share does not depend on the recipe-free prompt",
                                     "share LARGER with D5: report as a range",
                                     "share SMALLER with D5: part of 15.47 was D4's prompt; report the range, narrow to the smaller")}
    R["R10-b"] = {"d5casnegs_minus_reference_xling": d("d5casnegs", "reference", "xling"),
                  "d5nonegs_minus_d4_xling": d("d5nonegs", "d4_alone", "xling"),
                  "verdict_reference": verdict(d("d5casnegs", "reference", "xling"), BANDS["xling"],
                                               "D4 not unusually retention-friendly (reference replicates)",
                                               "D5 reference retains MORE: reported",
                                               "D4 was retention-friendly: DiD reported as a range over both references"),
                  "verdict_alone": verdict(d("d5nonegs", "d4_alone", "xling"), BANDS["xling"],
                                           "D5 alone retains like D4", "D5 alone retains more: reported",
                                           "D4 alone was unusually retention-friendly: reported")}
    R["R10-c"] = {"d5casnegs_minus_reference_hard": d("d5casnegs", "reference", "hard"),
                  "verdict": verdict(d("d5casnegs", "reference", "hard"), BANDS["hard"],
                                     "recipe-free rewrites with verified negatives reach the tier: replicates",
                                     "D5 reference higher on hard: reported", "D5 reference lower on hard: reported, D4-specific")}
    if m("d5casnegs", "easy") is not None and m("d5casnegs", "xling") is not None:
        R["R10-did"] = {"did_recipe_vs_d5reference": round((m("recipe_arm", "easy") - m("d5casnegs", "easy"))
                                                           - (m("recipe_arm", "xling") - m("d5casnegs", "xling")), 2),
                        "paper_did_vs_reference": PAPER["did_vs_reference"]}
    # ---- E-R11
    R["R11-a"] = {"d1split_minus_recipe_hard": d("d1split", "recipe_arm", "hard"),
                  "verdict": verdict(d("d1split", "recipe_arm", "hard"), BANDS["hard"],
                                     "one-call pairing does not depress the hard tier",
                                     "shared-phrasing account HOLDS: separate calls lift the hard tier",
                                     "separate calls LOWER the hard tier: reported")}
    R["R11-b"] = {"d1split_minus_d4llmnegsfull_hard": d("d1split", "d4llmnegsfull", "hard"),
                  "within_band": None if d("d1split", "d4llmnegsfull", "hard") is None
                  else abs(d("d1split", "d4llmnegsfull", "hard")) <= BANDS["hard"]}
    R["R11-c"] = {ev: d("d1split", "recipe_arm", ev) for ev in ("easy", "xling", "samelang_primary")}
    # ---- E-R12
    tj = os.path.join(PROJECT, "results", "llm_pairs_twojudge_summary.json")
    R["R12-a"] = json.load(open(tj)) if os.path.exists(tj) else "PENDING"
    R["R12-b"] = {ev: {"d1twojudge_minus_recipe": d("d1twojudge", "recipe_arm", ev),
                       "verdict": verdict(d("d1twojudge", "recipe_arm", ev), BANDS[ev],
                                          "within band: no dependence on the single self-judge",
                                          "two-judge arm HIGHER: reported", "two-judge arm LOWER: reported, disclose single-judge")}
                  for ev in ("easy", "hard", "xling")}
    tjrows = os.path.join(PROJECT, "data", "llm_pairs_twojudge", "pairs.jsonl")
    if os.path.exists(tjrows):
        R["R12-b"]["two_judge_trainer_rows"] = sum(1 for _ in open(tjrows, encoding="utf-8"))
    # ---- E-R13
    b2 = os.path.join(PROJECT, "results", "bt2_pairs_build.json")
    depth = json.load(open(b2))["depth_vs_source_kept_rows"] if os.path.exists(b2) else "PENDING"
    R["R13"] = {"depth_multi_hop": depth, "depth_single_hop_char3": PAPER["bt_char3"], "depth_d4_char3": PAPER["d4_char3"],
                "bt2casnegs_minus_reference_easy": d("bt2casnegs", "reference", "easy"),
                "bt_minus_reference_easy": d("bt", "reference", "easy"),
                "verdict": verdict(d("bt2casnegs", "reference", "easy"), BANDS["easy"],
                                   "deeper non-LLM paraphrase RECOVERS the reference's gain: retire 'LLM authorship'",
                                   "BT2 above the reference: reported",
                                   "attribution SURVIVES at the deeper depth: report reference minus BT2"),
                "xling": d("bt2casnegs", "reference", "xling"), "samelang_primary": d("bt2casnegs", "reference", "samelang_primary")}
    # ---- E-R14
    va = cells["verified_arm"]["casprobe"]
    # sd from the unrounded per-seed values (rounding the sd first gave 0.45 instead of 0.46)
    sd_unrounded = statistics.stdev(va["per_seed"].values()) if va and va["n"] >= 3 else None
    sigma_probe = round(math.sqrt(2) * sd_unrounded, 2) if sd_unrounded is not None else None
    band_p = round(2 * math.sqrt(2) * sd_unrounded, 2) if sd_unrounded is not None else None
    def probe_read(a, b, prefix):
        diff = d(a, b, prefix)
        return {"diff": diff, "verdict": verdict(diff, band_p if band_p else 1e9,
                                                 "within band: no recipe-specific gain in generator-free near-miss discrimination",
                                                 "ABOVE the band: a measured equivalence gain; the hard-tier claim narrows",
                                                 "BELOW the band: reported") if band_p else "PENDING (sigma_probe)"}
    R["R14"] = {"sigma_probe": sigma_probe, "band": band_p,
                "R14-a_recipe_vs_reference_full": probe_read("recipe_arm", "reference", "casprobe"),
                "R14-b_recipe_vs_base_full": probe_read("recipe_arm", "base", "casprobe"),
                "R14-c_verified_vs_base_full": {"diff": d("verified_arm", "base", "casprobe"), "note": "in-family, no direction registered"},
                "R14-d_pairs_only": {"recipe_vs_reference": probe_read("recipe_arm", "reference", "casprobepairs"),
                                     "recipe_vs_base": probe_read("recipe_arm", "base", "casprobepairs"),
                                     "verified_vs_base": d("verified_arm", "base", "casprobepairs")},
                "all_cells_full": {n: m(n, "casprobe") for n in CELLS},
                "all_cells_pairs": {n: m(n, "casprobepairs") for n in CELLS}}
    cb = os.path.join(PROJECT, "results", "cas_probe_build.json")
    if os.path.exists(cb):
        R["R14"]["probe_build"] = {k: v for k, v in json.load(open(cb)).items()
                                   if k in ("n_eligible_held_out", "n_attempted", "n_probe_items", "negatives_per_item", "families", "languages")}
    # ---- E-R15
    c4 = os.path.join(PROJECT, "results", "calib4b_summary.json")
    if os.path.exists(c4):
        cal = json.load(open(c4))
        best = cal["cells"][0] if cal["cells"] else None
        R["R15"] = {"closest": best, "published": PAPER["published_4b_easy_r1"],
                    "gap": None if not best else round(PAPER["published_4b_easy_r1"] - best["recall@1"], 2),
                    "promote": bool(best) and abs(PAPER["published_4b_easy_r1"] - best["recall@1"]) <= 0.5,
                    "n_cells": len(cal["cells"])}
    else:
        R["R15"] = "PENDING"
    with open(os.path.join(PROJECT, "results", "review3_verdict.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(R, indent=1)[:6000])
    print("\n[cells]")
    for n, c in cells.items():
        print(f"  {n:14s} " + "  ".join(f"{ev}={c[ev]['mean']:.2f}({c[ev]['n']})" if c.get(ev) else f"{ev}=--"
                                        for ev in ("easy", "hard", "xling", "samelang_primary", "casprobe", "casprobepairs")))


if __name__ == "__main__":
    main()
