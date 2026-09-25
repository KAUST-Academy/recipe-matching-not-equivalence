#!/usr/bin/env python3
r"""LLM-judged supervision arm for the InvarEmbed controlled experiment (plan Rank 3).

This is the CONTRAST condition: pairs produced the way prior work does it
(RaDeR / ReasonIR / MELD-style, and MathNet-Retrieve itself, Sec 3.3.2 +
Appendix F) — an instruction-tuned LLM rewrites each source problem into
  * claimed-EQUIVALENT variants  (candidate positives), and
  * claimed near-miss variants   (candidate hard negatives),
then a SECOND LLM pass judges equivalence. No symbolic verification anywhere.
The generation prompt is adapted nearly verbatim from the MathNet paper's own
Appendix F variant-generation prompt (paper_updated_text.txt, pages 32-33);
the judge pass mirrors the paper's LLM-as-judge filter (the paper used two
API judges, Gemini-3-flash + GPT-5; here one local judge = the matched-budget
open-model analog, same model as the generator by default).

MATCHED-BUDGET CONSTRAINT (plan Rank 3): both arms must use the same source
problems and the same per-problem counts. Point --source-ids-file at exactly
the file the CAS arm used (or produce one here with --write-source-ids and
point the CAS arm at it). The 8,698 anchor-matched corpus ids in
anchor_to_corpus_mapping.json are ALWAYS excluded, in every mode.

Output: JSONL rows validated by scripts/validate_pairs_schema.py (flat
one-row-per-candidate schema), plus a JSON summary. All rows are written
(judge-rejected ones carry verification.verified=false). Before training,
convert to the grouped trainer format with scripts/convert_llm_pairs.py --
it filters on verified==true and writes data/llm_pairs/pairs.jsonl, the path
train_invarembed.py's docstring names for the LLM arm. (generate_cas_pairs.py
writes the grouped trainer format directly.)

Backends:
  --backend vllm          offline vLLM inference (default; needs the 'vllm'
                          conda env + 1 GPU; see generate_llm_pairs.slurm)
  --backend transformers  plain HF generate() fallback for small-scale tests
  --backend stub          NO model, NO GPU: deterministic fake completions
                          (valid strict-JSON for both passes, with a seeded
                          ~20% judge-rejection rate so verified=false rows
                          exist). Exercises the ENTIRE pipeline -- prompt
                          building, JSON extraction, judging, schema
                          validation, output writing -- for login-node
                          contract tests against convert_llm_pairs.py and
                          the trainer. NEVER train on stub output.

No-GPU modes (run on a login node, mathnet env):
  --dry-run               print the EXACT prompts for 3 sample problems + exit
  --emit-mock PATH        write 3 schema-valid mock rows + validate + exit

PROMPT-VARIANT DOSE (prompt-proximity dose-response; see
scripts/dose_response.slurm for the pre-registered predictions):
--prompt-variant selects the PASS-1 generation prompt only; the judge pass
is byte-identical for every variant.
  exact       (default) the MathNet Appendix-F rewriter prompt -- guaranteed
              byte-identical to this script's original behavior (dose point
              D1 = data/pairs/llm_pairs.jsonl / models/ctrl-llm-6145; never
              regenerate).
  paraphrase  D2: semantic paraphrase of the Appendix-F prompt -- the same
              task and 1-equivalent+N-near-miss structure described in
              different words and section structure.
  style       D3: same GOAL (1 equivalent + near-miss traps) in a genuinely
              different authorial framing: a competition coach assembling a
              worksheet, conversational voice, different constraints, no
              transformation-ideas menu.
  unrelated   D4: equivalence-preserving math augmentation OUTSIDE the
              benchmark's recipe -- restate the problem for an audience at a
              different level, mathematics unchanged. POSITIVES ONLY: this
              variant generates no near-misses (--negatives-per-problem is
              forced to 0), so trainer rows carry zero attached negatives and
              training relies on in-batch negatives under (C)MNRL. The
              asymmetry is inherent to the dose -- the 1-pos-3-neg near-miss
              channel IS part of the recipe being removed -- and must be
              reported alongside any D4 number.
  survey      D5 (2026-09-05; tests whether the reference replicates with a
              second recipe-free restatement prompt whose task framing
              differs from D4's audience change): a survey
              author restating another author's problem for a themed problem
              collection -- own wording, notation and order, level and
              mathematics unchanged. POSITIVES ONLY, own schema
              ("restatements" keyed problem/context/justification), no
              near-miss channel, exactly as D4.
  exact_split D1 written in TWO calls (the recipe arm's positive and
              near-misses are written in one call; regenerating them in
              separate calls tests the shared-phrasing account): the
              Appendix-F prompt split into a positives-only
              prompt (Step 1 + the JSON schema without near_miss_variants) and
              a near-misses-only prompt (Step 2 + the schema without
              equivalent_variants), both derived from GEN_SYSTEM by string
              surgery at import time so the wording stays verbatim; the judge
              pass is unchanged.

REJUDGE MODE (measures how the recipe arm moves under a second, independent
judge): --rejudge-input PATH re-runs ONLY the judge pass
over an existing pairs file with --model as the second judge (a different
vendor and family; no generation), writes every row with
verification.verified = judge1 AND judge2, verification.verified_judge1 and
verification.evidence.second_judge, plus an agreement summary.

Held fixed across variants (comparability/parsing constraints, stated
honestly): generator+judge model, seeds, source ids, per-problem counts,
the judge prompt, and -- for D1-D3 -- the strict-JSON output contract
(identical top-level and per-item keys; only the surrounding instructional
text varies, because the keys are this script's parsing interface). D4 uses
its own schema ("restatements", items keyed problem/audience/justification;
audience is mapped into transformation_tags).

Usage (full run, inside the vllm env on 1x A100):
  python3 generate_llm_pairs.py --source-ids-file data/pairs/source_ids.txt \
      --model Qwen/Qwen3-32B-AWQ --output data/pairs/llm_pairs.jsonl
Dose-response arms (see scripts/dose_response.slurm):
  python3 generate_llm_pairs.py --prompt-variant paraphrase \
      --source-ids-file data/phase2/cas_source_ids.txt \
      --model Qwen/Qwen3-32B-AWQ --output data/pairs/llm_pairs_paraphrase.jsonl
"""
import argparse
import json
import random
from collections import Counter
import re
import sys
import time
from pathlib import Path

ROOT = Path("/ibex/user/habiam0b/MathNet_Follow_Up")
sys.path.insert(0, str(ROOT / "scripts"))
from validate_pairs_schema import validate_row  # noqa: E402  (shared contract)

# ---------------------------------------------------------------------------
# Prompts. GEN_SYSTEM is the MathNet paper's own Appendix F rewriter prompt
# (pages 32-33 of paper_updated_text.txt), lightly adapted: parameterized
# variant counts (the paper's rules block itself overrides the step headers to
# "Exactly 1 equivalent and 3 near-miss variants"), and the original problem
# is supplied in the user turn. JUDGE_SYSTEM is ours (the paper does not print
# its judge prompt); it operationalizes the paper's Invariance definition
# (Table 2: strict equivalence under reformulation) with a strict-JSON verdict.
# ---------------------------------------------------------------------------

GEN_SYSTEM = """You are a rigorous mathematical editor and rewriter.

## Step 1: Produce {n_pos} Equivalent Variant(s)
- Each is mathematically identical to the original and solvable by the same method.
- Make them look substantially different in structure/phrasing.
- Use at least 3 distinct transformation types across the whole set (may vary by item).
- Keep statements concise.
- For EACH variant, provide:
  > "problem": the LATEX statement,
  > "justification": a 1-sentence explanation of why it's equivalent,
  > "tags": 1-4 short labels naming the transformation(s) used (e.g., "Variable rename", "Modular rewrite").

// Some Transformation Ideas (use multiple or your own):
1. Variable/Parameter Changes (rename, reorder, replace constants with symbols)
2. Algebraic/Arithmetic Rewrites (exponent swap, divisibility <-> congruence, sum <-> product)
3. Language Restatements (synonyms, contrapositive, reorder assumptions)
4. Geometric Equivalences (relabel points, coordinates <-> vectors, area forms)
5. Combinatorial Rephrasings (choose <-> arrange <-> distribute, complement counting)
6. Number Theory Tricks (gcd, modular swaps, prime restatements)
7. Structural Shifts (lemma/theorem framing, reverse flow, regrouping)
8. Sequences <-> Functions (recurrence <-> functional form)
9. Ratios/Normalizations (normalized terms, integer ratios)
10. Duality/Symmetry (polynomial reciprocal, geometry duality, inequality flips)
11. Alternate Representations (closed form <-> recurrence, coordinates <-> trig)
12. Constraint Shifts (additive <-> multiplicative, quantifier swaps)
13. Meta-Level Changes (prove vs. counterexample, classification form)

## Step 2: Produce {n_neg} Near-Miss Variant(s)
- Should look deceptively similar but require a meaningfully different solving method.
- Make small, crucial mathematical changes (e.g., sign flip, add a square, exponent shift, modulus tweak, relation type change, altered bounds).
- Ensure each truly alters the strategy (not a trivial edit).
- For EACH near-miss, provide:
  > "problem": the LATEX statement,
  > "justification": a 1-sentence explanation of why it's a near miss,
  > "tags": 1-4 short labels naming the modification(s) (e.g., "Sign flip", "Modulus change", "Inequality direction").

## Step 3: Return Strict JSON
Return only a single valid JSON object using this exact schema:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "equivalent_variants": [
    {{"problem": "...", "justification": "...", "tags": ["...", "..."]}}
  ],
  "near_miss_variants": [
    {{"problem": "...", "justification": "...", "tags": ["..."]}}
  ]
}}
// Rules:
- All math must be in LATEX.
- Exactly {n_pos} equivalent and {n_neg} near-miss variants.
- Each variant MUST include non-empty "tags".
- Output JSON only; no code fences, no extra text.
- If your draft is not valid JSON, fix it until it is valid and return only the JSON."""

GEN_USER = """Original problem:

{problem}"""

JUDGE_SYSTEM = """You are an expert mathematical olympiad editor acting as a strict judge of mathematical equivalence.

You will be given Problem A and Problem B. Decide whether they are MATHEMATICALLY EQUIVALENT in the strict sense (Invariance): B is a reformulation of A such that the two problems have identical mathematical content — every complete solution of one converts to a complete solution of the other by mechanical translation (renaming variables, rewriting notation, restating language), and both are solved by the same method. Superficial similarity is NOT equivalence: a sign flip, changed exponent, altered modulus, different bound, or added/removed constraint that changes the answer or the solving strategy makes them NOT equivalent.

Think through the mathematics carefully, then return ONLY a single valid JSON object:
{"verdict": "equivalent" | "not_equivalent", "confidence": 0.0-1.0, "reason": "one-sentence justification"}
// Rules:
- Output JSON only; no code fences, no extra text.
- "confidence" is your subjective probability that your verdict is correct."""

JUDGE_USER = """Problem A:

{anchor}

Problem B:

{candidate}"""

# ---------------------------------------------------------------------------
# Prompt-variant dose points D2-D4. Each constant below is the
# VERBATIM system prompt used for its dose point -- quote from here, not from
# memory, when documenting the experiment. D1 = GEN_SYSTEM above (Appendix F).
# The strict-JSON blocks of D2/D3 are intentionally byte-identical to D1's
# Step-3 block: the JSON keys are the parsing interface, so the dose
# manipulates only the surrounding instructional text. D4 has its own schema.
# ---------------------------------------------------------------------------

# D2 -- semantic paraphrase of the Appendix-F prompt: same role, same two-part
# task, same transformation ideas (folded into prose), same rules, reworded
# and restructured throughout.
GEN_SYSTEM_PARAPHRASE = """You act as a careful, exacting editor of mathematics problems.

Your assignment has two parts.

### Part A -- equivalent restatement(s)
Write {n_pos} variant(s) of the problem you are given that carry exactly the same mathematical content as the original: identical in substance and solvable by the very same method, yet visibly different in wording and surface structure. Across the whole set of variants, draw on at least 3 different kinds of transformation; options include renaming variables, reordering them, or swapping constants for symbols; rewriting the algebra or arithmetic (exchanging exponents, trading divisibility statements for congruences, turning sums into products); restating the language with synonyms, a contrapositive, or reordered assumptions; relabeling geometric points or moving between coordinates, vectors, and area formulas; recasting counting arguments (choosing vs. arranging vs. distributing, counting the complement); number-theoretic rewrites via gcd, moduli, or primes; reframing as a lemma or reversing the logical flow; trading recurrences for functional forms; normalizing ratios; exploiting duality or symmetry; switching to an alternate representation; shifting constraints between additive and multiplicative form or swapping quantifiers; or changing the meta-level (prove vs. find a counterexample, classification form). You may also invent transformations of your own. Keep every statement concise. Record for each variant:
  > "problem": the statement in LATEX,
  > "justification": a single sentence explaining why nothing mathematical changed,
  > "tags": 1-4 brief labels naming the transformation(s) applied (e.g., "Variable rename", "Modular rewrite").

### Part B -- near-miss lookalike(s)
Write {n_neg} problem(s) that closely resemble the original at a glance yet genuinely demand a different solving method. Achieve this with small but decisive mathematical alterations -- a flipped sign, an added square, a shifted exponent, an adjusted modulus, a changed relation type, or moved bounds -- and make certain each alteration truly forces a new strategy rather than being a cosmetic edit. Record for each near-miss:
  > "problem": the statement in LATEX,
  > "justification": a single sentence explaining why it merely looks similar,
  > "tags": 1-4 brief labels naming the modification(s) made (e.g., "Sign flip", "Modulus change", "Inequality direction").

### Part C -- answer in strict JSON
Return only a single valid JSON object using this exact schema:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "equivalent_variants": [
    {{"problem": "...", "justification": "...", "tags": ["...", "..."]}}
  ],
  "near_miss_variants": [
    {{"problem": "...", "justification": "...", "tags": ["..."]}}
  ]
}}
Constraints: express all mathematics in LATEX; produce exactly {n_pos} equivalent and {n_neg} near-miss variant(s); never leave "tags" empty; emit nothing besides the JSON object -- no code fences, no commentary; and if your draft fails to parse as JSON, repair it until it is valid and return only the JSON."""

GEN_USER_PARAPHRASE = """The problem to work from:

{problem}"""

# D3 -- different authorial style/framing, same goal: a coach writing a
# worksheet, conversational voice, "disguises" and "traps", no transformation
# menu, its own constraints (recognition test, leanness, failed-trap rule).
GEN_SYSTEM_STYLE = """Hi! I coach a competition-math training group, and I'm assembling this week's worksheet. I need your help with one exercise at a time.

Here's the game. My students have already seen the problem I'm about to show you. To test whether they recognize a problem by its mathematics rather than by its looks, the worksheet mixes disguises of it in among traps:

1. {n_pos} disguise(s): the same problem dressed up to look new. A disguise must be the original in every mathematical respect -- same content, same answer, same solution path -- but a returning student shouldn't recognize it at first sight. Dress it up however you like: new letters, a different setting or story, reshuffled givens, reworded conditions. Keep it lean; competition problems don't ramble.

2. {n_neg} trap(s): problems that LOOK like today's problem but are NOT it. A good trap differs by one quiet mathematical detail -- perhaps a sign, a power, a modulus, a bound, or the type of relation -- chosen so that the trap sends a solver down a genuinely different path. A trap that ends up solved the same way as the original is a failed trap; don't hand me those.

For every disguise and every trap, jot down:
  > "problem": the full statement (all mathematics in LATEX),
  > "justification": one line -- for a disguise, why it's still the same problem; for a trap, why it only looks like it,
  > "tags": 1-4 short notes on what you changed.

So my worksheet tool can ingest your work, reply with strict JSON only:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "equivalent_variants": [
    {{"problem": "...", "justification": "...", "tags": ["...", "..."]}}
  ],
  "near_miss_variants": [
    {{"problem": "...", "justification": "...", "tags": ["..."]}}
  ]
}}
House rules: disguises go under "equivalent_variants" and traps under "near_miss_variants", exactly {n_pos} and {n_neg} of them; every "tags" list non-empty; all math in LATEX; nothing outside the JSON object -- no code fences, no chat; and if what you wrote isn't valid JSON, fix it before you answer. Thanks!"""

GEN_USER_STYLE = """Today's problem for the worksheet:

{problem}"""

# D4 -- recipe-UNRELATED equivalence-preserving augmentation: audience-level
# restatement, positives only, own JSON schema. Deliberately shares neither
# the near-miss channel nor the equivalent/near-miss vocabulary of D1-D3.
GEN_SYSTEM_UNRELATED = """You are an editor adapting mathematics problems for textbooks aimed at different audiences.

Rewrite the given problem for a student at a DIFFERENT level of mathematical maturity than its original audience. Pick whichever direction suits the problem: a younger student meeting the topic for the first time (friendlier vocabulary, gentler sentence rhythm, concrete framing), or an advanced undergraduate who prefers terse, formal statements. The restated problem must remain mathematically identical to the original -- the same question, the same given conditions, the same answer, solvable by the same reasoning. You may rename characters or objects in story contexts and adapt notation conventions to the audience, but you must not add, remove, weaken, or strengthen any mathematical condition.

Produce {n_pos} restatement(s). For each, record:
  > "problem": the restated problem (all mathematics in LATEX),
  > "audience": who you wrote it for (e.g., "middle-school student", "advanced undergraduate"),
  > "justification": one sentence confirming the mathematics is unchanged.

Return only a single valid JSON object using this exact schema:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "restatements": [
    {{"problem": "...", "audience": "...", "justification": "..."}}
  ]
}}
// Rules:
- All math must be in LATEX.
- Exactly {n_pos} restatement(s).
- Output JSON only; no code fences, no extra text.
- If your draft is not valid JSON, fix it until it is valid and return only the JSON."""

GEN_USER_UNRELATED = """Problem to adapt:

{problem}"""

# E-R7 (2026-09-04): the D4 audience-adaptation persona
# extended with a "spot the difference" drill, so that near-miss negatives are
# written under wording that shares nothing with Appendix F. Positives are the
# same restatement task as D4 (they are discarded when the cell is built; the
# D4 positives already on disk are reused); only the companions are new. The
# JSON keys deliberately differ from D1-D3's contract ("companions" /
# "what_changed"), as D4's do ("restatements" / "audience").
GEN_SYSTEM_UNRELATED_NEGS = """You are an editor adapting mathematics problems for textbooks aimed at different audiences.

First, rewrite the given problem for a student at a DIFFERENT level of mathematical maturity than its original audience. Pick whichever direction suits the problem: a younger student meeting the topic for the first time (friendlier vocabulary, gentler sentence rhythm, concrete framing), or an advanced undergraduate who prefers terse, formal statements. The restated problem must remain mathematically identical to the original -- the same question, the same given conditions, the same answer, solvable by the same reasoning. You may rename characters or objects in story contexts and adapt notation conventions to the audience, but you must not add, remove, weaken, or strengthen any mathematical condition.

Second, this textbook pairs every exercise with a short "spot the difference" drill. Write {n_neg} companion exercise(s) for that drill. Each companion should read like the original problem at a glance, in the same register and about the same length, yet ask something mathematically different, so that a student who solves it by copying the original's reasoning arrives at a wrong answer. Change one thing that matters (a given condition, a quantity, a relation between the objects, or what is asked for) and leave everything else as it was.

Produce {n_pos} restatement(s) and {n_neg} companion(s). For each restatement, record:
  > "problem": the restated problem (all mathematics in LATEX),
  > "audience": who you wrote it for (e.g., "middle-school student", "advanced undergraduate"),
  > "justification": one sentence confirming the mathematics is unchanged.
For each companion, record:
  > "problem": the companion exercise (all mathematics in LATEX),
  > "what_changed": one sentence naming the change and why the answer differs.

Return only a single valid JSON object using this exact schema:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "restatements": [
    {{"problem": "...", "audience": "...", "justification": "..."}}
  ],
  "companions": [
    {{"problem": "...", "what_changed": "..."}}
  ]
}}
// Rules:
- All math must be in LATEX.
- Exactly {n_pos} restatement(s) and {n_neg} companion(s).
- Output JSON only; no code fences, no extra text.
- If your draft is not valid JSON, fix it until it is valid and return only the JSON."""

GEN_USER_UNRELATED_NEGS = """Problem to adapt:

{problem}"""

# D5 (E-R10, 2026-09-05): a SECOND recipe-free restatement
# prompt whose task framing is not an audience change: a survey author
# restating another author's problem for a themed collection. Positives only,
# own schema keys ("restatements" / "context"), no near-miss channel, so the
# reference cell (D4 positives + verified negatives) can be replicated with
# different recipe-free positives (R5 W2 / Q1).
GEN_SYSTEM_SURVEY = """You are a mathematician writing the problems section of a survey article that collects competition problems on one theme.

Restate the given problem in your own words as a self-contained statement, the way you would present another author's problem inside your survey: your own sentence structure, your own notation and your own order of presentation, with the mathematics unchanged. Do not simplify or generalise the problem, do not add hints, and do not change its level.

Produce {n_pos} restatement(s). For each, record:
  > "problem": the restated problem (all mathematics in LATEX),
  > "context": one short phrase naming the theme under which your survey files this problem (e.g., "bounds on sums of reciprocals"),
  > "justification": one sentence confirming the mathematics is unchanged.

Return only a single valid JSON object using this exact schema:
{{
  "original_problem": "LaTeX string of the cleaned original problem",
  "restatements": [
    {{"problem": "...", "context": "...", "justification": "..."}}
  ]
}}
// Rules:
- All math must be in LATEX.
- Exactly {n_pos} restatement(s).
- Output JSON only; no code fences, no extra text.
- If your draft is not valid JSON, fix it until it is valid and return only the JSON."""

GEN_USER_SURVEY = """Problem to restate for the survey:

{problem}"""


def _split_appendix_f(kind):
    """E-R11: the Appendix-F prompt with one of its two
    generation steps removed, by string surgery on GEN_SYSTEM so that every
    surviving line is verbatim D1. kind = "pos" keeps Step 1 (equivalent
    variants), kind = "neg" keeps Step 2 (near-miss variants); the JSON schema
    and the count rule keep only the surviving key."""
    s = GEN_SYSTEM
    i1, i2, i3 = s.index("## Step 1:"), s.index("## Step 2:"), s.index("## Step 3:")
    head, step1, step2, step3 = s[:i1], s[i1:i2], s[i2:i3], s[i3:]
    pos_block = ('  "equivalent_variants": [\n'
                 '    {{"problem": "...", "justification": "...", "tags": ["...", "..."]}}\n'
                 '  ],\n')
    neg_block = ('  "near_miss_variants": [\n'
                 '    {{"problem": "...", "justification": "...", "tags": ["..."]}}\n'
                 '  ]\n')
    rule = "- Exactly {n_pos} equivalent and {n_neg} near-miss variants.\n"
    assert pos_block in step3 and neg_block in step3 and rule in step3, "GEN_SYSTEM layout changed"
    if kind == "pos":
        body = step1
        schema = step3.replace(neg_block, "").replace(pos_block, pos_block[:-2] + "\n")
        schema = schema.replace(rule, "- Exactly {n_pos} equivalent variant(s).\n")
    else:
        body = step2.replace("## Step 2:", "## Step 1:")
        schema = step3.replace(pos_block, "")
        schema = schema.replace(rule, "- Exactly {n_neg} near-miss variant(s).\n")
    schema = schema.replace("## Step 3:", "## Step 2:")
    out = head + body + schema
    if kind == "pos":
        assert "near_miss_variants" not in out and "equivalent_variants" in out
    else:
        assert "equivalent_variants" not in out and "near_miss_variants" in out
    return out


GEN_SYSTEM_SPLIT_POS = _split_appendix_f("pos")
GEN_SYSTEM_SPLIT_NEG = _split_appendix_f("neg")

# Registry: gen prompts + the JSON keys pass-1 parsing reads per variant.
# neg_key=None => the variant structurally produces no near-miss negatives.
VARIANTS = {
    "exact":      {"gen_system": GEN_SYSTEM, "gen_user": GEN_USER,
                   "pos_key": "equivalent_variants", "neg_key": "near_miss_variants"},
    "paraphrase": {"gen_system": GEN_SYSTEM_PARAPHRASE, "gen_user": GEN_USER_PARAPHRASE,
                   "pos_key": "equivalent_variants", "neg_key": "near_miss_variants"},
    "style":      {"gen_system": GEN_SYSTEM_STYLE, "gen_user": GEN_USER_STYLE,
                   "pos_key": "equivalent_variants", "neg_key": "near_miss_variants"},
    "unrelated":  {"gen_system": GEN_SYSTEM_UNRELATED, "gen_user": GEN_USER_UNRELATED,
                   "pos_key": "restatements", "neg_key": None},
    "unrelated_negs": {"gen_system": GEN_SYSTEM_UNRELATED_NEGS,
                       "gen_user": GEN_USER_UNRELATED_NEGS,
                       "pos_key": "restatements", "neg_key": "companions"},
    "survey":     {"gen_system": GEN_SYSTEM_SURVEY, "gen_user": GEN_USER_SURVEY,
                   "pos_key": "restatements", "neg_key": None},
    # E-R11: two pass-1 calls per problem (positives-only, near-misses-only);
    # "gen_system" is the positives prompt so every existing code path that
    # reads it (dry-run, stub) keeps working; main() uses "gen_system_neg" for
    # the second call.
    "exact_split": {"gen_system": GEN_SYSTEM_SPLIT_POS, "gen_system_neg": GEN_SYSTEM_SPLIT_NEG,
                    "gen_user": GEN_USER, "split": True,
                    "pos_key": "equivalent_variants", "neg_key": "near_miss_variants"},
}
POSITIVES_ONLY_VARIANTS = ("unrelated", "survey")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_source_problems(args):
    """Return list of (id, problem_markdown, meta) after anchor exclusion."""
    import pandas as pd

    df = pd.read_parquet(args.corpus)
    excl = set(json.load(open(args.exclusion_file))["exclude_corpus_ids"])
    n0 = len(df)
    df = df[~df["id"].isin(excl)]
    print(f"[data] corpus {n0} -> {len(df)} after excluding {len(excl)} anchor-matched ids")

    if args.source_ids_file:
        raw = Path(args.source_ids_file).read_text().strip()
        ids = json.loads(raw) if raw.startswith("[") else raw.split()
        ids = [str(i) for i in ids]
        bad = [i for i in ids if i in excl]
        if bad:
            sys.exit(f"[FATAL] source-ids-file contains {len(bad)} EXCLUDED anchor-matched ids "
                     f"(e.g. {bad[:5]}) — the CAS arm must not have used these either. Aborting.")
        have = set(df["id"])
        missing = [i for i in ids if i not in have]
        if missing:
            sys.exit(f"[FATAL] {len(missing)} ids not in corpus (e.g. {missing[:5]})")
        df = df.set_index("id").loc[ids].reset_index()
        print(f"[data] matched-budget mode: {len(df)} problems from {args.source_ids_file}")
    elif args.num_problems:
        df = df.sample(n=min(args.num_problems, len(df)), random_state=args.seed)
        print(f"[data] sampled {len(df)} problems (seed={args.seed}) — no --source-ids-file; "
              f"use --write-source-ids and point the CAS arm at it for matched budget")

    rows = []
    for r in df.itertuples(index=False):
        topics = list(r.topics_flat) if r.topics_flat is not None else []
        rows.append((r.id, r.problem_markdown,
                     {"language": None if str(r.language) == "nan" else r.language,
                      "domain": topics[0].split(">")[0].strip() if topics else None,
                      "competition": r.competition, "problem_type": r.problem_type}))
    return rows


def extract_json(text):
    """Best-effort strict-JSON extraction from a model completion."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # Qwen3 thinking
    text = re.sub(r"```(?:json)?|```", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
                    return None
    return None


# ---------------------------------------------------------------------------
# extract_json above is FROZEN: every corpus under data/ that is cited in the
# paper was parsed by exactly that function, so changing it would silently
# change the provenance of already-published rows.  The two helpers below are
# additive and opt-in; use them for any NEW corpus whose strings carry raw
# LaTeX (see generate_meld_pairs.py).
# ---------------------------------------------------------------------------
def repair_json_escapes(blob):
    r"""Re-escape raw LaTeX backslashes inside JSON string literals.

    A model told to write "natural language and LaTeX" *inside a JSON string*
    routinely emits `"$\mathbb{P}(A)$"` where strict JSON needs `"$\\mathbb…"`.
    That produces two different failures, one loud and one silent:

      * `\m \s \a \l \c \(` ... are not legal JSON escapes, so json.loads
        raises and the ENTIRE call is discarded  (loud: counted as a
        generation JSON failure);
      * `\b \f \n \r \t` ARE legal escapes, so `\frac{1}{2}` parses as
        FORMFEED + "rac{1}{2}", `\times` as TAB + "imes", `\nu` as NEWLINE +
        "u"  (silent: a corrupted statement lands in the training corpus).

    Both are fixed by doubling every backslash that does not begin a legal
    JSON escape, treating a whitespace escape followed by another lowercase
    letter as the start of a LaTeX control word rather than as whitespace
    (`\nu`, `\rho`, `\beta`, `\frac`, `\times` -> literal backslash; a real
    `\n` before a space, quote, digit or capital is preserved).  Round-trip
    safe on already-correct JSON: `\\` and `\"` are passed through untouched.
    """
    out, i, n, in_str = [], 0, len(blob), False
    while i < n:
        ch = blob[i]
        if not in_str:
            in_str = ch == '"'
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = blob[i + 1] if i + 1 < n else ""
        after = blob[i + 2] if i + 2 < n else ""
        keep = (nxt in '"\\/'
                or (nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}",
                                                blob[i + 2:i + 6] or "") is not None)
                or (nxt in "bfnrt" and not after.islower()))
        if keep:
            out.append(ch)
            out.append(nxt)
            i += 2
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _balanced_blob(text):
    """First balanced {...} or [...] span, or None if it never closes.

    Returning None IS the truncation signal: a completion cut off by
    max_tokens can never balance.
    """
    starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
    if not starts:
        return None
    start = min(starts)
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_json_robust(text):
    """extract_json + LaTeX-escape repair + top-level-array support.

    Candidate ladder, repaired forms FIRST so that the silently-corrupting
    `\\frac` -> FORMFEED case is fixed rather than merely parsed; the raw forms
    remain as a fallback so nothing that already parsed can start failing.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # Qwen3 thinking
    text = re.sub(r"```(?:json)?|```", "", text)
    blob = _balanced_blob(text)
    if blob is None:
        return None
    notrail = re.sub(r",\s*([}\]])", r"\1", blob)
    for candidate in (repair_json_escapes(blob), repair_json_escapes(notrail),
                      blob, notrail):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def build_gen_messages(problem, args, which="pos"):
    cfg = VARIANTS[args.prompt_variant]
    system = cfg["gen_system_neg"] if (which == "neg" and cfg.get("split")) else cfg["gen_system"]
    return [{"role": "system", "content": system.format(
                n_pos=args.positives_per_problem, n_neg=args.negatives_per_problem)},
            {"role": "user", "content": cfg["gen_user"].format(problem=problem)}]


def build_judge_messages(anchor, candidate):
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(anchor=anchor, candidate=candidate)}]


# ---------------------------------------------------------------------------
# Backends: each takes a list of chat message-lists, returns list of strings.
# Imported lazily so --dry-run / --emit-mock work on the login node.
# ---------------------------------------------------------------------------


def make_backend(args):
    if args.backend == "stub":
        import zlib

        def run(batches, max_tokens, temperature):
            cfg = VARIANTS[args.prompt_variant]
            outs = []
            for msgs in batches:
                system, user = msgs[0]["content"], msgs[1]["content"]
                if not system.startswith("You are an expert mathematical olympiad editor"):
                    # generation pass (any prompt variant); every variant's
                    # user turn is "<lead-in>:\n\n{problem}"
                    problem = user.split(":", 1)[1].strip()
                    if args.prompt_variant in POSITIVES_ONLY_VARIANTS:
                        pos_items = [
                            {"problem": f"[stub-equivalent {k}] {problem}",
                             "audience": "middle-school student",
                             "justification": "stub: level restatement, mathematics unchanged"}
                            for k in range(args.positives_per_problem)]
                    else:
                        pos_items = [
                            {"problem": f"[stub-equivalent {k}] {problem}",
                             "justification": "stub: mechanical variable rename",
                             "tags": ["Variable rename"]}
                            for k in range(args.positives_per_problem)]
                    obj = {"original_problem": problem, cfg["pos_key"]: pos_items}
                    if cfg["neg_key"]:
                        obj[cfg["neg_key"]] = [
                            {"problem": f"[stub-nearmiss {k}] {problem}",
                             "justification": "stub: sign flip changes the answer",
                             "tags": ["Sign flip"]}
                            for k in range(args.negatives_per_problem)]
                    outs.append(json.dumps(obj, ensure_ascii=False))
                else:  # judge pass
                    cand = user.split("Problem B:", 1)[1].strip()
                    claimed_equiv = cand.startswith("[stub-equivalent")
                    # seeded ~20% verdict flips so verified=false rows exist
                    flip = zlib.crc32(f"{args.seed}:{cand}".encode()) % 5 == 0
                    verdict = ("not_equivalent" if claimed_equiv == flip
                               else "equivalent")
                    outs.append(json.dumps(
                        {"verdict": verdict, "confidence": 0.9,
                         "reason": "stub deterministic verdict"}))
            return outs
        return run

    if args.backend == "gemini":
        # 2026-09-17: the benchmark's own generator family
        # (Gemini 3 Flash) through the Generative Language API, same messages,
        # temperature and token cap as the vLLM path; thinking level low
        # (a structured rewrite needs little); 8 concurrent calls, retries
        # with backoff on 429/5xx; token usage accumulated to results/gemini_usage.json.
        # The key is read from GEMINI_API_KEY / Gemini_API (env or the repo .env) and never logged.
        import concurrent.futures, urllib.request, urllib.error, time as _time, os as _os, sys as _sys, json as _json, threading as _thr
        from pathlib import Path as _Path
        key = _os.environ.get("GEMINI_API_KEY") or _os.environ.get("Gemini_API")
        if not key:
            envp = _Path(__file__).resolve().parent.parent / ".env"
            if envp.exists():
                for line in envp.read_text().splitlines():
                    if line.startswith(("Gemini_API=", "GEMINI_API_KEY=")):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not key:
            _sys.exit("[gemini] no API key: set GEMINI_API_KEY or put Gemini_API=... in .env")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{args.model}:generateContent?key={key}")
        usage = {"model": args.model, "calls": 0, "failures": 0, "prompt_tokens": 0,
                 "output_tokens": 0, "thought_tokens": 0}
        usage_path = _Path(__file__).resolve().parent.parent / "results" / "gemini_usage.json"
        lock = _thr.Lock()

        def one(msgs, max_tokens, temperature):
            body = {"systemInstruction": {"parts": [{"text": msgs[0]["content"]}]},
                    "contents": [{"role": "user", "parts": [{"text": msgs[1]["content"]}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                         "thinkingConfig": {"thinkingLevel": "low"}}}
            delay = 4.0
            for attempt in range(7):
                try:
                    req = urllib.request.Request(url, data=_json.dumps(body).encode("utf-8"),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=180) as r:
                        resp = _json.load(r)
                    cand = (resp.get("candidates") or [{}])[0]
                    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])
                                   if "text" in p and not p.get("thought"))
                    um = resp.get("usageMetadata", {})
                    with lock:
                        usage["calls"] += 1
                        if cand.get("finishReason") == "MAX_TOKENS":
                            usage["truncated"] = usage.get("truncated", 0) + 1
                        usage["prompt_tokens"] += int(um.get("promptTokenCount", 0))
                        usage["output_tokens"] += int(um.get("candidatesTokenCount", 0))
                        usage["thought_tokens"] += int(um.get("thoughtsTokenCount", 0))
                    return text
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")[:300]
                    if e.code == 400 and "thinking" in msg.lower():
                        body["generationConfig"].pop("thinkingConfig", None); continue
                    if e.code in (429, 500, 502, 503, 504):
                        _time.sleep(delay); delay = min(delay * 2, 60); continue
                    print(f"[gemini] HTTP {e.code}: {msg}", file=_sys.stderr); break
                except Exception:
                    _time.sleep(delay); delay = min(delay * 2, 60)
            with lock:
                usage["failures"] += 1
            return ""

        def run(batches, max_tokens, temperature):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                outs = list(ex.map(lambda m: one(m, max_tokens, temperature), batches))
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(usage_path, "w") as fh:
                _json.dump(usage, fh, indent=2)
            print(f"[gemini] {usage['calls']} calls, {usage['failures']} failures, "
                  f"{usage['prompt_tokens']} prompt / {usage['output_tokens']} output / "
                  f"{usage['thought_tokens']} thought tokens so far", flush=True)
            return outs
        return run

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, dtype="auto", seed=args.seed,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=True)

        def run(batches, max_tokens, temperature):
            sp = SamplingParams(temperature=temperature, top_p=0.9 if temperature > 0 else 1.0,
                                max_tokens=max_tokens, seed=args.seed)
            outs = llm.chat(batches, sp, chat_template_kwargs={"enable_thinking": False})
            return [o.outputs[0].text for o in outs]
        return run

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None)

    def run(batches, max_tokens, temperature):
        texts = []
        for i in range(0, len(batches), args.batch_size):
            chunk = batches[i:i + args.batch_size]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for m in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_model_len).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_tokens,
                                 do_sample=temperature > 0, temperature=temperature or None,
                                 pad_token_id=tok.eos_token_id)
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return texts
    return run


# ---------------------------------------------------------------------------
# Row construction (shared schema — see validate_pairs_schema.py)
# ---------------------------------------------------------------------------


def make_row(source_id, anchor, candidate, label, kind_idx, gen_variant, judge, args, meta):
    verdict = (judge or {}).get("verdict")
    want = "equivalent" if label == "positive" else "not_equivalent"
    # exact keeps its historical channel/meta byte-identical (D1 must never
    # change); D2-D4 additionally record their prompt variant in both places.
    if args.prompt_variant != "exact":
        meta = dict(meta, prompt_variant=args.prompt_variant)
    return {
        "pair_id": f"llm-{source_id}-{'pos' if label == 'positive' else 'neg'}{kind_idx}",
        "source_id": source_id,
        "anchor_text": anchor,
        "candidate_text": candidate,
        "label": label,
        "arm": "llm",
        "channel": ("llm_rewrite" if args.prompt_variant == "exact"
                    else f"llm_rewrite_{args.prompt_variant}"),
        "verification": {
            "method": "llm_judge",
            "verified": verdict == want,
            "evidence": {
                "generator_model": args.model, "judge_model": args.judge_model or args.model,
                "generator_claim": want, "judge_verdict": verdict,
                "judge_confidence": (judge or {}).get("confidence"),
                "judge_reason": (judge or {}).get("reason"),
                "generator_justification": gen_variant.get("justification", ""),
            },
        },
        "transformation_tags": [str(t) for t in gen_variant.get("tags", []) if t],
        "meta": dict(meta, generation_seed=args.seed),
    }


def mock_rows(problems, args):
    """3 hand-built rows (no model) exercising every schema branch."""
    (sid, anchor, meta) = problems[0]
    rows = [
        make_row(sid, anchor, anchor + " (restated with $m$ in place of $n$)", "positive", 0,
                 {"justification": "pure variable rename", "tags": ["Variable rename"]},
                 {"verdict": "equivalent", "confidence": 0.97, "reason": "mechanical rename"},
                 args, meta),
        make_row(sid, anchor, anchor + " (with the parity condition flipped to odd)",
                 "hard_negative", 0,
                 {"justification": "parity flip changes the invariant", "tags": ["Parity flip"]},
                 {"verdict": "not_equivalent", "confidence": 0.9, "reason": "different invariant"},
                 args, meta),
        make_row(sid, anchor, anchor + " (claimed-equivalent variant the judge rejected)",
                 "positive", 1,
                 {"justification": "claimed contrapositive", "tags": ["Contrapositive"]},
                 {"verdict": "not_equivalent", "confidence": 0.6, "reason": "quantifier changed"},
                 args, meta),
    ]
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def rejudge(args):
    """E-R12: second, independent judge over an existing pairs file."""
    rows = [json.loads(l) for l in open(args.rejudge_input, encoding="utf-8") if l.strip()]
    print(f"[rejudge] {len(rows)} rows from {args.rejudge_input}; second judge = {args.model}")
    t0 = time.time()
    run = make_backend(args)
    judge_out = run([build_judge_messages(r["anchor_text"], r["candidate_text"]) for r in rows],
                    args.judge_max_tokens, 0.0)
    out_rows, fail = [], 0
    agree = Counter()
    for r, jtext in zip(rows, judge_out):
        j = extract_json(jtext)
        if not j or j.get("verdict") not in ("equivalent", "not_equivalent"):
            fail += 1
            j = None
        want = "equivalent" if r["label"] == "positive" else "not_equivalent"
        v1 = bool(r["verification"].get("verified"))
        v2 = bool(j) and j.get("verdict") == want
        r = json.loads(json.dumps(r))  # deep copy
        r["verification"]["verified_judge1"] = v1
        r["verification"]["verified"] = v1 and v2
        r["verification"]["evidence"]["second_judge"] = {
            "model": args.model, "verdict": (j or {}).get("verdict"),
            "confidence": (j or {}).get("confidence"), "reason": (j or {}).get("reason"),
            "verified": v2}
        agree[(r["label"], v1, v2)] += 1
        out_rows.append(r)
    errs = [e for i, r in enumerate(out_rows) for e in validate_row(r, i)]
    if errs:
        sys.exit("[FATAL] schema self-check failed:\n" + "\n".join(errs[:20]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rate(label, j1=None, j2=None):
        sel = [k for k in agree if k[0] == label and (j1 is None or k[1] == j1)
               and (j2 is None or k[2] == j2)]
        return sum(agree[k] for k in sel)

    summary = {"mode": "rejudge", "input": args.rejudge_input, "second_judge_model": args.model,
               "n_rows": len(out_rows), "n_second_judge_json_failures": fail,
               "runtime_s": round(time.time() - t0, 1)}
    for label in ("positive", "hard_negative"):
        n = rate(label)
        a11 = rate(label, True, True); a00 = rate(label, False, False)
        a10 = rate(label, True, False); a01 = rate(label, False, True)
        po = (a11 + a00) / max(n, 1)
        p1 = (a11 + a10) / max(n, 1); p2 = (a11 + a01) / max(n, 1)
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        summary[label] = {"n": n, "judge1_accept": a11 + a10, "judge2_accept": a11 + a01,
                          "both_accept": a11, "neither": a00, "only_judge1": a10, "only_judge2": a01,
                          "agreement": round(po, 4),
                          "cohen_kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None}
    spath = args.summary or str(out) + ".summary.json"
    json.dump(summary, open(spath, "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print(f"[done] {len(out_rows)} rows -> {out}\n[done] summary -> {spath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(ROOT / "data" / "mathnet_corpus.parquet"))
    ap.add_argument("--exclusion-file", default=str(ROOT / "anchor_to_corpus_mapping.json"))
    ap.add_argument("--source-ids-file", default=None,
                    help="ids (one per line, or a JSON list) — MUST be the same file the "
                         "CAS arm used, for the matched-budget experiment")
    ap.add_argument("--write-source-ids", default=None,
                    help="write the ids actually used to this file (whichever arm runs "
                         "first defines the shared set)")
    ap.add_argument("--num-problems", type=int, default=None,
                    help="seeded subsample size when no --source-ids-file is given")
    ap.add_argument("--positives-per-problem", type=int, default=1)
    ap.add_argument("--negatives-per-problem", type=int, default=3)
    ap.add_argument("--prompt-variant", choices=sorted(VARIANTS), default="exact",
                    help="generation-prompt dose point (dose-response): "
                         "exact = MathNet Appendix-F prompt (byte-identical "
                         "default = D1), paraphrase = D2, style = D3, "
                         "unrelated = D4 (positives only; forces "
                         "--negatives-per-problem 0). The judge pass is "
                         "identical for all variants.")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--judge-model", default=None,
                    help="defaults to --model (single-model generate+judge)")
    ap.add_argument("--backend", choices=["vllm", "transformers", "stub", "gemini"],
                    default="vllm",
                    help="stub = no model/GPU, deterministic fake completions "
                         "for pipeline contract tests (never train on it)")
    ap.add_argument("--batch-size", type=int, default=4, help="transformers backend only")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--gen-temperature", type=float, default=0.7)
    ap.add_argument("--gen-max-tokens", type=int, default=2048)
    ap.add_argument("--judge-max-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "data" / "pairs" / "llm_pairs.jsonl"))
    ap.add_argument("--summary", default=None, help="default: <output>.summary.json")
    ap.add_argument("--rejudge-input", default=None, metavar="PATH",
                    help="E-R12: re-run ONLY the judge pass over this existing pairs "
                         "file with --model as a second judge; writes --output and a summary")
    ap.add_argument("--dry-run", action="store_true",
                    help="print exact prompts for 3 sample problems; no model, no GPU")
    ap.add_argument("--emit-mock", default=None, metavar="PATH",
                    help="write 3 schema-valid mock rows to PATH and validate; no model")
    args = ap.parse_args()
    if args.judge_model and args.judge_model != args.model and not args.rejudge_input:
        sys.exit("[FATAL] separate --judge-model not supported in one process yet: run a "
                 "second judge-only pass instead (--rejudge-input; keeps one model resident per job).")
    if args.rejudge_input:
        return rejudge(args)
    if args.prompt_variant in POSITIVES_ONLY_VARIANTS and args.negatives_per_problem != 0:
        print(f"[variant] '{args.prompt_variant}' generates no near-misses by design -- "
              f"forcing --negatives-per-problem 0 (was {args.negatives_per_problem}); "
              f"trainer rows will carry zero attached negatives (in-batch only)")
        args.negatives_per_problem = 0
    random.seed(args.seed)

    problems = load_source_problems(args)
    if not problems:
        sys.exit("[FATAL] no source problems selected")

    if args.dry_run:
        if args.prompt_variant != "exact":  # exact dry-run output stays byte-identical
            print(f"[dry-run] prompt variant: {args.prompt_variant} "
                  f"(dose point D{1 + sorted(VARIANTS).index(args.prompt_variant)})")
        for sid, text, _meta in problems[:3]:
            g = build_gen_messages(text, args)
            j = build_judge_messages(text, "<VARIANT PRODUCED IN PASS 1 GOES HERE>")
            print(f"\n{'=' * 78}\nPROBLEM {sid}\n{'=' * 78}")
            print(f"\n--- PASS 1 (generation) SYSTEM ---\n{g[0]['content']}")
            print(f"\n--- PASS 1 (generation) USER ---\n{g[1]['content']}")
            print(f"\n--- PASS 2 (judge) SYSTEM ---\n{j[0]['content']}")
            print(f"\n--- PASS 2 (judge) USER ---\n{j[1]['content']}")
        return

    if args.emit_mock:
        rows = mock_rows(problems, args)
        errs = [e for i, r in enumerate(rows) for e in validate_row(r, i)]
        Path(args.emit_mock).parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_mock, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[mock] wrote {len(rows)} rows -> {args.emit_mock}")
        print("[mock] schema check:", "FAILED\n" + "\n".join(errs) if errs else "all rows valid")
        sys.exit(1 if errs else 0)

    if args.write_source_ids:
        Path(args.write_source_ids).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_source_ids).write_text("\n".join(p[0] for p in problems) + "\n")
        print(f"[data] wrote {len(problems)} source ids -> {args.write_source_ids}")

    t0 = time.time()
    run = make_backend(args)

    # ---- pass 1: generation --------------------------------------------------
    vcfg = VARIANTS[args.prompt_variant]
    gen_out = run([build_gen_messages(p[1], args) for p in problems],
                  args.gen_max_tokens, args.gen_temperature)
    if vcfg.get("split"):
        # E-R11: the near-misses come from a SECOND call with the negatives-only
        # prompt; the positives call above never sees the near-miss step.
        gen_out_neg = run([build_gen_messages(p[1], args, which="neg") for p in problems],
                          args.gen_max_tokens, args.gen_temperature)
    parsed, gen_fail = [], 0
    for pi, ((sid, anchor, meta), text) in enumerate(zip(problems, gen_out)):
        obj = extract_json(text)
        if vcfg.get("split"):
            obj_neg = extract_json(gen_out_neg[pi])
            if not obj and not obj_neg:
                gen_fail += 1
                continue
            if not obj or not obj_neg:
                gen_fail += 1
            pos = list((obj or {}).get(vcfg["pos_key"]) or [])[: args.positives_per_problem]
            neg = list((obj_neg or {}).get(vcfg["neg_key"]) or [])[: args.negatives_per_problem]
        else:
            if not obj:
                gen_fail += 1
                continue
            pos = list(obj.get(vcfg["pos_key"]) or [])[: args.positives_per_problem]
            neg = (list(obj.get(vcfg["neg_key"]) or [])[: args.negatives_per_problem]
                   if vcfg["neg_key"] else [])
        for k, v in enumerate(pos):
            if isinstance(v, dict) and str(v.get("problem", "")).strip():
                if args.prompt_variant in ("unrelated", "unrelated_negs") and not v.get("tags"):
                    # D4 items carry "audience" instead of "tags"; map it so
                    # the shared schema (transformation_tags) stays auditable
                    aud = str(v.get("audience") or "").strip()
                    v = dict(v, tags=[f"Audience: {aud}" if aud else "Level restatement"])
                if args.prompt_variant == "survey" and not v.get("tags"):
                    ctx = str(v.get("context") or "").strip()
                    v = dict(v, tags=[f"Survey theme: {ctx}" if ctx else "Survey restatement"])
                parsed.append((sid, anchor, meta, "positive", k, v))
        for k, v in enumerate(neg):
            if isinstance(v, dict) and str(v.get("problem", "")).strip():
                if args.prompt_variant == "unrelated_negs" and not v.get("tags"):
                    # E-R7 companions carry "what_changed" instead of
                    # "tags"/"justification"; map both so the row schema holds
                    wc = str(v.get("what_changed") or "").strip()
                    v = dict(v, tags=["Companion: " + (wc[:60] if wc else "changed condition")],
                             justification=wc)
                parsed.append((sid, anchor, meta, "hard_negative", k, v))
    print(f"[pass1] {len(problems)} problems -> {len(parsed)} claimed variants "
          f"({gen_fail} JSON-parse failures) in {time.time() - t0:.0f}s")

    # ---- pass 2: LLM judge (no symbolic verification — that is the point) ----
    t1 = time.time()
    judge_out = run([build_judge_messages(a, v["problem"]) for (_s, a, _m, _l, _k, v) in parsed],
                    args.judge_max_tokens, 0.0)
    rows, judge_fail = [], 0
    for (sid, anchor, meta, label, k, v), jtext in zip(parsed, judge_out):
        j = extract_json(jtext)
        if not j or j.get("verdict") not in ("equivalent", "not_equivalent"):
            judge_fail += 1
            j = None
        rows.append(make_row(sid, anchor, str(v["problem"]), label, k, v, j, args, meta))
    print(f"[pass2] judged {len(parsed)} pairs ({judge_fail} unparseable verdicts) "
          f"in {time.time() - t1:.0f}s")

    # ---- validate + write ----------------------------------------------------
    errs = [e for i, r in enumerate(rows) for e in validate_row(r, i)]
    if errs:
        sys.exit("[FATAL] schema self-check failed:\n" + "\n".join(errs[:20]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nver = sum(r["verification"]["verified"] for r in rows)
    summary = {
        "arm": "llm", "generator_model": args.model, "judge_model": args.model,
        "n_source_problems": len(problems),
        "positives_per_problem": args.positives_per_problem,
        "negatives_per_problem": args.negatives_per_problem,
        "n_rows_written": len(rows), "n_verified": nver,
        "n_rejected_by_judge": len(rows) - nver,
        "n_generation_json_failures": gen_fail, "n_judge_json_failures": judge_fail,
        "by_label": {lb: {"total": sum(r["label"] == lb for r in rows),
                          "verified": sum(r["label"] == lb and r["verification"]["verified"]
                                          for r in rows)}
                     for lb in ("positive", "hard_negative")},
        "runtime_s": round(time.time() - t0, 1), "seed": args.seed,
        "source_ids_file": args.source_ids_file,
    }
    if args.prompt_variant != "exact":  # keep the default summary byte-compatible
        summary["prompt_variant"] = args.prompt_variant
    spath = args.summary or str(out) + ".summary.json"
    json.dump(summary, open(spath, "w"), indent=2)
    print(f"[done] {len(rows)} rows ({nver} verified) -> {out}\n[done] summary -> {spath}")


if __name__ == "__main__":
    main()
