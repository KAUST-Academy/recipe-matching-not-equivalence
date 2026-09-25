#!/usr/bin/env python3
"""
Summary + pre-registered verdicts for scripts/round2_closeout.slurm.

Each block below evaluates a reading fixed in that job's header, written before
the job was submitted and before any of its outputs existed. The point of doing
it in code is that two of the four branches would WEAKEN claims the paper makes,
and it should not be possible to read the numbers first and then decide which
branch we were in.

Usage:  python scripts/round2_closeout_summary.py
Output: results/round2_closeout.json + a printed report.
"""

import json
import os
import statistics
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# reference points, all measured before this job
MELD9_3SEED = 37.16
MELD_PUBLISHED_BEST = 28.9      # MathLeap Table 2, THEIR harness
D2_HARD_3SEED, D2_HARD_SD = 6.76, 1.01
D3_HARD_3SEED = 4.67
D2_5591_S42_HARD = 8.27
D2_6145_S42_HARD = 7.89
SD1_HARD = 0.86


def dig(path, *keys):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    out = {"generated": str(date.today()),
           "script": "scripts/round2_closeout_summary.py",
           "preregistered_in": "scripts/round2_closeout.slurm header"}

    # --- A: MELD's published models in OUR harness ------------------------
    a = {}
    for m in ("octen", "qwen"):
        a[f"MathLeap-{m}-8B_our_harness"] = dig(
            os.path.join(R, f"meld_mathleap_{m}_8b.json"),
            "retrieval_pairs_only", "recall@1")
    best = max([v for v in a.values() if v is not None], default=None)
    if best is None:
        a["verdict"] = "PENDING"
    else:
        a["best_mathleap_in_our_harness"] = best
        a["their_published_best_their_harness"] = MELD_PUBLISHED_BEST
        a["meld9_ours_3seed"] = MELD9_3SEED
        a["our_harness_reads_them_high_by"] = round(best - MELD_PUBLISHED_BEST, 2)
        # A point comparison is not enough: 270 pairs give an MDE of ~3.6 R@1, so
        # a margin of a couple of points is inside the noise. Require the paired
        # cluster bootstrap to exclude zero. (The first version of this function
        # compared point estimates and would have claimed a win it cannot support.)
        cmp_o = dig(os.path.join(R, "meld_compare_attack_meld9_vs_mathleap_octen.json"),
                    "metrics", "recall@1_pairs_only")
        a["paired_meld9_minus_octen"] = cmp_o
        if not cmp_o:
            a["verdict"] = ("POINT ESTIMATES ONLY — run the paired comparison before "
                            "stating any ordering.")
        elif cmp_o["ci95"][0] > 0:
            a["verdict"] = (
                f"IN-HARNESS WIN — meld9 beats MathLeap measured in OUR harness by "
                f"{cmp_o['diff']:+.2f} {cmp_o['ci95']}, CI excluding zero.")
        else:
            a["verdict"] = (
                f"INDISTINGUISHABLE, AND THAT IS THE CLAIM — meld9 and MELD's best "
                f"published model are statistically tied in our harness "
                f"({cmp_o['diff']:+.2f} {cmp_o['ci95']}, CI contains zero). We may NOT "
                f"say we beat it. What is supported, and is the stronger statement "
                f"anyway: a 0.6B model trained on a reconstruction of MELD's procedure "
                f"MATCHES an 8B model purpose-built for the benchmark. Note also that "
                f"our harness reads their model {best - MELD_PUBLISHED_BEST:+.2f} above "
                f"their published figure, which is why the cross-table comparison was "
                f"deleted in the first place.")
    out["A_meld_inharness"] = a

    # --- B: D2 on D3's own sources ----------------------------------------
    b = {"d2_hard_3seed": D2_HARD_3SEED, "d3_hard_3seed": D3_HARD_3SEED}
    h = dig(os.path.join(R, "eval_hard_dose-paraphrase-d3sources.json"),
            "overall", "recall@1")
    b["d2_on_d3_sources_hard"] = h
    b["easy"] = dig(os.path.join(R, "eval_easy_dose-paraphrase-d3sources.json"),
                    "overall", "recall@1")
    b["realdup"] = dig(os.path.join(R, "crosslingual_dose-paraphrase-d3sources.json"),
                       "strict_crosslingual_gold", "recall@1")
    if h is None:
        b["verdict"] = "PENDING"
    else:
        # midpoint between D2's and D3's three-seed means is the decision line
        mid = (D2_HARD_3SEED + D3_HARD_3SEED) / 2
        b["decision_midpoint"] = round(mid, 3)
        b["distance_from_D2_in_sd"] = round(abs(h - D2_HARD_3SEED) / D2_HARD_SD, 2)
        b["verdict"] = (
            f"SELECTION IS NOT DRIVING IT — {h} sits on D2's side of the midpoint "
            f"({mid:.2f}), so restricting D2 to D3's own sources does not move it "
            f"toward D3. The D2->D3 step is the prompt."
            if h > mid else
            f"SELECTION MATTERS — {h} falls to D3's side of the midpoint "
            f"({mid:.2f}). Part of what mechanism.tex attributes to prompt distance "
            f"is which sources survived D3's generation failures, and the paper must "
            f"say so.")
    out["B_source_matched_control"] = b

    # --- C: the H2 control at three seeds ---------------------------------
    c = {"per_seed": {"42": D2_5591_S42_HARD}}
    for s in (43, 44):
        c["per_seed"][str(s)] = dig(
            os.path.join(R, f"eval_hard_dose-paraphrase-5591-s{s}.json"),
            "overall", "recall@1")
    vals = [v for v in c["per_seed"].values() if v is not None]
    if len(vals) < 3:
        c["verdict"] = f"PENDING — {len(vals)} of 3 seeds"
    else:
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        # the row-count effect is D2@5591 minus D2@6145; D2@6145's own three-seed
        # mean is 6.76, but the like-for-like seed-42 pair is what the paper quotes
        c.update({"mean": round(mean, 3), "sd": round(sd, 3),
                  "row_effect_vs_D2_6145_3seed": round(mean - D2_HARD_3SEED, 3),
                  "row_effect_seed42_pair": round(D2_5591_S42_HARD - D2_6145_S42_HARD, 3)})
        se = sd / (len(vals) ** 0.5)
        lo, hi = mean - D2_HARD_3SEED - 2 * se, mean - D2_HARD_3SEED + 2 * se
        c["row_effect_95ci_approx"] = [round(lo, 3), round(hi, 3)]
        eff = mean - D2_HARD_3SEED
        step = D3_HARD_3SEED - D2_HARD_3SEED          # NEGATIVE: D3 scores lower
        # Sign matters and the first version of this function ignored it. The
        # step to explain is a DROP. A positive row effect (fewer rows scoring
        # HIGHER) cannot produce a drop; it makes the prompt effect larger.
        if lo > 0:
            c["verdict"] = (
                f"WRONG SIGN TO EXPLAIN THE STEP — cutting to D3's row count RAISES "
                f"hard R@1 by {eff:+.2f} [{lo:+.2f}, {hi:+.2f}], reliably positive "
                f"across all three seeds. The D2->D3 drop of {step:+.2f} is therefore "
                f"entirely the prompt, and at matched rows the prompt effect is "
                f"{D3_HARD_3SEED - mean:+.2f}, LARGER than the uncontrolled figure.")
        elif hi < 0 and abs(hi) > abs(step) * 0.5:
            c["verdict"] = (
                f"CONFOUNDED — the row cut costs {eff:+.2f} [{lo:+.2f}, {hi:+.2f}], a "
                f"substantial share of the {step:+.2f} step. The paper must attribute it.")
        else:
            c["verdict"] = (
                f"row effect {eff:+.2f} [{lo:+.2f}, {hi:+.2f}] against a step of "
                f"{step:+.2f}; report as a partial contribution.")
        c["prompt_effect_at_matched_rows"] = round(D3_HARD_3SEED - mean, 3)
    out["C_h2_control_seeds"] = c

    # --- D: SABER control rows at the campaign-wide cap -------------------
    d = {}
    for arm in ("llm", "cas"):
        d[f"ctrl_{arm}_cap1024"] = dig(os.path.join(R, f"saber_ctrl_{arm}_cap1024.json"),
                                       "settings", "statement-full", "ndcg@10")
        d[f"ctrl_{arm}_uncapped_released"] = dig(os.path.join(R, f"saber_ctrl_{arm}.json"),
                                                 "settings", "statement-full", "ndcg@10")
    d["base_reference"] = 0.5747
    got = [v for k, v in d.items() if k.endswith("cap1024") and v is not None]
    d["verdict"] = ("PENDING" if len(got) < 2 else
                    "P-S3's 'both below the base' survives at a matched context length."
                    if all(v < 0.5747 for v in got) else
                    "At a matched context length a control arm is NOT below the base; "
                    "ood.tex's P-S3 sentence must be restated.")
    out["D_saber_matched_context"] = d

    with open(os.path.join(R, "round2_closeout.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for k, blk in out.items():
        if not isinstance(blk, dict):
            continue
        print(f"\n--- {k} ---")
        for kk, vv in blk.items():
            if kk != "verdict":
                print(f"    {kk}: {vv}")
        print(f"    VERDICT: {blk.get('verdict')}")
    print("\nwrote results/round2_closeout.json")


if __name__ == "__main__":
    main()
