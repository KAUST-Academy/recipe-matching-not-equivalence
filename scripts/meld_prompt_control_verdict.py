#!/usr/bin/env python3
"""
The MELD prompt control's pre-registered verdict (round-2 finding R2-H6).

The readings below are copied from scripts/meld_prompt_control.slurm's header,
which was written and submitted BEFORE any meld9alt output existed. Evaluating
them in code rather than by eye is the point: the C2 branch weakens a claim the
paper currently makes, and it should not be possible to read the numbers first
and then decide which branch we were in.

  C1 RECIPE-SPECIFIC  alt <= heldout (30.80) AND meld9 - alt >= 3.6
  C2 GENRE-ONLY       |meld9 - alt| < 3.6 and the paired CI contains zero
  C3 GRADIENT         anything else

3.6 is the minimum detectable effect from results/meld_attack_power.json,
computed from off-the-shelf encoders before any attacked model existed.

Usage:  python scripts/meld_prompt_control_verdict.py
Output: results/meld_prompt_control.json + a printed verdict.
"""

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

MELD9_3SEED = 37.16          # results/meld_seed_stats.json
HELDOUT_3SEED = 30.80
MELD9_S42 = 36.67            # like-for-like with the alt arm's single seed
BASE = 10.19
CTRL_LLM = 18.89
MDE = 3.6


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
    alt = dig(os.path.join(R, "meld_atk_attack_meld9alt.json"),
              "retrieval_pairs_only", "recall@1")
    if alt is None:
        print("PENDING — results/meld_atk_attack_meld9alt.json not written yet")
        return

    cmp_m9 = dig(os.path.join(R, "meld_compare_attack_meld9alt_vs_attack_meld9.json"),
                 "metrics", "recall@1_pairs_only")
    diff = cmp_m9["diff"] if cmp_m9 else round(alt - MELD9_S42, 2)
    ci = cmp_m9["ci95"] if cmp_m9 else None
    ci_contains_zero = (ci[0] <= 0 <= ci[1]) if ci else None
    gap_from_meld9 = round(MELD9_S42 - alt, 2)

    if alt <= HELDOUT_3SEED and gap_from_meld9 >= MDE:
        verdict = ("C1 RECIPE-SPECIFIC — the wording of MELD's procedure carries a "
                   "measurable part of the effect. The paper may keep a recipe-level "
                   "reading of the MELD result, scoped to this reconstruction.")
    elif abs(gap_from_meld9) < MDE and (ci_contains_zero is not False):
        verdict = ("C2 GENRE-ONLY — the alt prompt matches MELD's own within the "
                   "detectable floor. DROP every recipe-specific reading of the MELD "
                   "result: it is evidence about cross-dialect equivalence FORMAT, not "
                   "about recipes. ood.tex, discussion.tex and the taxonomy framing must "
                   "be revised accordingly.")
    else:
        verdict = ("C3 GRADIENT — report as a partial effect, as the MathNet ladder's "
                   "partial outcomes were, without claiming either branch.")

    doc = {
        "generated": str(date.today()),
        "script": "scripts/meld_prompt_control_verdict.py",
        "design": ("Same nine MELD domain pairs, same schema, generator, judge, gates, "
                   "row budget and negatives-per-row as the meld9 arm; only the "
                   "authorial framing of the generation prompt differs "
                   "(MELD_GEN_SYSTEM_ALT). This is the control the P-M3a contrast could "
                   "not provide, since that one varies task, sources, budget and "
                   "negatives simultaneously."),
        "preregistered_in": "scripts/meld_prompt_control.slurm header, fixed before the run",
        "mde_floor": MDE,
        "reference_points": {"base": BASE, "ctrl_llm": CTRL_LLM,
                             "meld9_seed42": MELD9_S42, "meld9_3seed": MELD9_3SEED,
                             "heldout_3seed": HELDOUT_3SEED},
        "meld9alt_R@1": alt,
        "gain_over_base": round(alt - BASE, 2),
        "meld9_minus_alt_seed42": gap_from_meld9,
        "paired_bootstrap_alt_minus_meld9": {"diff": diff, "ci95": ci,
                                             "ci_contains_zero": ci_contains_zero},
        "verdict": verdict,
        "realdup_R@1": dig(os.path.join(R, "crosslingual_meld-attack-meld9alt.json"),
                           "strict_crosslingual_gold", "recall@1"),
        "mathnet_hard_R@1": dig(os.path.join(R, "eval_hard_meld-attack-meld9alt.json"),
                                "overall", "recall@1"),
    }
    with open(os.path.join(R, "meld_prompt_control.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"  base                {BASE}")
    print(f"  ctrl-LLM            {CTRL_LLM}")
    print(f"  meld9 (seed 42)     {MELD9_S42}   [three seeds {MELD9_3SEED}]")
    print(f"  heldout (3 seeds)   {HELDOUT_3SEED}")
    print(f"  meld9alt            {alt}   gain over base {doc['gain_over_base']:+.2f}")
    print(f"  meld9 - alt         {gap_from_meld9:+.2f}   (MDE floor {MDE})")
    if ci:
        print(f"  paired alt-meld9    {diff:+.2f} {ci}  contains zero: {ci_contains_zero}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nwrote results/meld_prompt_control.json")


if __name__ == "__main__":
    main()
