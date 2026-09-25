#!/usr/bin/env python3
"""
Pre-registered verdict for the SABER label-style attack.

The readings are copied from scripts/saber_label_attack.slurm's header, which
was written and submitted before either arm existed. They are evaluated here in
code because the L1 branch would force a rewrite of the paper's taxonomy, and
that decision must not depend on reading the numbers first.

  L1  label arm >= base + 0.02  -> SABER does NOT resist; the "organic corpora
                                   resist" cell empties and the paper changes.
  L2  both arms below base      -> the negative holds and strengthens, since it
                                   now covers both surfaces at a matched budget.
  L3  label > doc by >= 0.02 but still below base -> partial exposure.

Usage:  python scripts/saber_label_verdict.py
Output: results/saber_label_attack.json + a printed verdict.
"""

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

BASE = 0.5747              # our base row; SABER publishes 0.575
PUBLISHED_ATTACK = 0.5618  # models/saber-attack-6145, doc channel, 6,145 rows
BAR = 0.02                 # the same bar the original P-S1 used
ROWCAP = 4733


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
    got = {}
    for tag in ("label", "doc"):
        got[tag] = {
            "noprompt": dig(os.path.join(R, f"saber_{tag}_{ROWCAP}_noprompt.json"),
                            "settings", "statement-full", "ndcg@10"),
            "queryprompt": dig(os.path.join(R, f"saber_{tag}_{ROWCAP}.json"),
                               "settings", "statement-full", "ndcg@10"),
            "realdup": dig(os.path.join(R, f"crosslingual_saber-{tag}-{ROWCAP}.json"),
                           "strict_crosslingual_gold", "recall@1"),
        }

    lab = got["label"]["noprompt"]
    doc = got["doc"]["noprompt"]
    if lab is None or doc is None:
        verdict = "PENDING — one or both promptless SABER evaluations missing"
    elif lab >= BASE + BAR:
        verdict = (
            f"L1 THE LABEL SURFACE IS EXPOSED — the label arm scores {lab:.4f} against "
            f"the base's {BASE}, clearing the +{BAR} bar. SABER did not resist "
            f"recipe-matching; it resisted an attack on the wrong surface. The "
            f"taxonomy's 'organic corpora resist' cell is now EMPTY and the abstract, "
            f"intro, discussion, conclusion and limitations must be rewritten: the "
            f"claim narrows to 'a benchmark is exposed on whichever surface its "
            f"construction procedure defines'.")
    elif lab < BASE and doc < BASE:
        verdict = (
            f"L2 THE NEGATIVE HOLDS AND STRENGTHENS — both surfaces stay below the base "
            f"({lab:.4f} label, {doc:.4f} doc, base {BASE}) at a matched {ROWCAP}-row "
            f"budget. The paper may now say SABER resisted attacks on its document AND "
            f"its label surface, which is a materially stronger negative than the one "
            f"currently reported, and the 'label-style attack remains untested' "
            f"qualification comes out.")
    elif lab - doc >= BAR:
        verdict = (
            f"L3 PARTIAL EXPOSURE — the label surface beats the document surface by "
            f"{lab - doc:+.4f} but stays below the base. Report as a partial exposure "
            f"and correct the description of what the published attack tested.")
    else:
        verdict = (f"NO CHANNEL EFFECT — label {lab:.4f} vs doc {doc:.4f}, both below "
                   f"base. Report as L2 with the added note that channel did not matter.")

    doc_out = {
        "generated": str(date.today()),
        "script": "scripts/saber_label_verdict.py",
        "preregistered_in": "scripts/saber_label_attack.slurm header",
        "why": ("The first SABER attack trained on a seeded random draw of 6,145 rows of "
                "data/saber_attack/pairs.jsonl: 4,359 document-template rows and 1,786 "
                "summary rows (29.1%; see scripts/make_saber_channels.py). SABER's "
                "relevance rule is a Jaccard test over LLM-written core-idea summaries, "
                "so the label substrate is the surface that matters, and it had only been "
                "presented as a minority admixture. The channel arms isolate each surface."),
        "reference_points": {"base_noprompt": BASE,
                             "published_attack_noprompt": PUBLISHED_ATTACK,
                             "bar": BAR, "matched_budget": ROWCAP},
        "arms": got,
        "label_minus_base": None if lab is None else round(lab - BASE, 4),
        "doc_minus_base": None if doc is None else round(doc - BASE, 4),
        "label_minus_doc": None if (lab is None or doc is None) else round(lab - doc, 4),
        "verdict": verdict,
    }
    with open(os.path.join(R, "saber_label_attack.json"), "w", encoding="utf-8") as f:
        json.dump(doc_out, f, indent=2)

    print(f"  base (promptless)          {BASE}")
    print(f"  published attack (doc,6145) {PUBLISHED_ATTACK}")
    for tag in ("label", "doc"):
        g = got[tag]
        print(f"  {tag+' arm':26s} noprompt={g['noprompt']}  queryprompt={g['queryprompt']}"
              f"  realdup={g['realdup']}")
    print(f"\n  VERDICT: {verdict}")
    print("\nwrote results/saber_label_attack.json")


if __name__ == "__main__":
    main()
