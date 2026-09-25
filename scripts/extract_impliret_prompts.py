#!/usr/bin/env python3
"""
Extract ImpliRet's published generation prompts from their paper.

These are the attack surface. ImpliRet's documents are synthesised from tuples
by GEMMA-3-27B-IT using prompts printed in the paper's appendix (Figures 2-15),
so a practitioner who reads the paper can reproduce the recipe -- which is the
premise of this whole line of work, now tested outside mathematics.

We extract the prompt text VERBATIM rather than paraphrasing it, for the same
reason the MathNet arm used the Appendix-F template near-verbatim: the claim is
about what a published procedure leaks, so the procedure has to be the one they
published. Each figure's prompt body is the text between the previous figure's
caption and its own.

The arithmetic pipeline is figures 2-7:
    2  STARTING_CONVERSATION_PROMPT   uni-speaker (chat)
    3  STARTING_CONVERSATION_PROMPT   multi-speaker (forum)
    4  CONVERSATION_GENERATION_PROMPT multi-speaker, arithmetic
    5  CONVERSATION_GENERATION_PROMPT uni-speaker,   arithmetic
    6  FEATURE_EXTRACTION_PROMPT      multi-speaker, arithmetic
    7  FEATURE_EXTRACTION_PROMPT      uni-speaker,   arithmetic

Usage:  python scripts/extract_impliret_prompts.py
Output: data/impliret_attack/published_prompts.json (+ a printed summary)
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "papers", "05_crossdomain", "ImpliRet_2506.14407.pdf")
OUTDIR = os.path.join(ROOT, "data", "impliret_attack")
OUT = os.path.join(OUTDIR, "published_prompts.json")

ROLES = {
    2: ("starting_phrases", "unispeaker", None),
    3: ("starting_phrases", "multispeaker", None),
    4: ("conversation", "multispeaker", "arithmetic"),
    5: ("conversation", "unispeaker", "arithmetic"),
    6: ("feature_extraction", "multispeaker", "arithmetic"),
    7: ("feature_extraction", "unispeaker", "arithmetic"),
    8: ("conversation", "multispeaker", "wknow"),
    9: ("conversation", "unispeaker", "wknow"),
    10: ("feature_extraction", "multispeaker", "wknow"),
    11: ("feature_extraction", "unispeaker", "wknow"),
    12: ("conversation", "multispeaker", "temporal"),
    13: ("conversation", "unispeaker", "temporal"),
    14: ("feature_extraction", "multispeaker", "temporal"),
    15: ("feature_extraction", "unispeaker", "temporal"),
}


def main():
    import fitz
    os.chdir(ROOT)
    text = "".join(p.get_text() for p in fitz.open(PDF))

    caps = [(int(m.group(1)), m.start(), m.end())
            for m in re.finditer(r"Figure (\d+): ", text)]
    caps.sort(key=lambda x: x[1])

    prompts = {}
    for i, (num, start, end) in enumerate(caps):
        if num not in ROLES:
            continue
        prev_end = caps[i - 1][2] if i else 0
        body = text[prev_end:start].strip()
        # Every printed prompt begins at "**Task**". Cutting there removes both the
        # tail of the PREVIOUS figure's caption and, for the first extracted figure,
        # the body text that precedes the appendix. Without this the blocks bleed
        # into each other and figure 2 swallows 40k characters of the paper.
        j = body.rfind("**Task**")
        if j > 0:
            body = body[j:].strip()
        # drop the trailing PROMPT-NAME line that sits between the prompt and caption
        lines = body.rstrip().split("\n")
        if lines and re.match(r"^[A-Z_]{6,}_PROMPT\b", lines[-1].strip()):
            body = "\n".join(lines[:-1]).rstrip()
        role, style, cat = ROLES[num]
        key = f"{role}|{style}" + (f"|{cat}" if cat else "")
        prompts[key] = {"figure": num, "role": role, "style": style,
                        "category": cat, "chars": len(body), "text": body}

    os.makedirs(OUTDIR, exist_ok=True)
    doc = {
        "source": "arXiv:2506.14407 appendix figures, extracted verbatim",
        "note": ("Extracted verbatim, not paraphrased: the claim under test is what a "
                 "PUBLISHED procedure leaks, so the attack must use the published text. "
                 "PDF extraction can mangle layout, so each block is stored with its "
                 "character count and must be eyeballed before use -- a silently truncated "
                 "prompt would make the attack a test of something else."),
        "their_generator": "GEMMA-3-27B-IT (we attack with a different model, as for MathNet and MELD)",
        "prompts": prompts,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"extracted {len(prompts)} prompt blocks -> {os.path.relpath(OUT, ROOT)}\n")
    for k, v in sorted(prompts.items(), key=lambda kv: kv[1]["figure"]):
        head = v["text"][:90].replace("\n", " ")
        print(f"  fig{v['figure']:<3} {k:42s} {v['chars']:5d} chars | {head}...")


if __name__ == "__main__":
    main()
