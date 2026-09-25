#!/usr/bin/env python3
r"""MELD recipe-matching attack: training pairs built with MELD's OWN recipe.

The SECOND synthetic-corpus benchmark.

WHY MELD (target selection, verified against the source, not assumed).
The vulnerability taxonomy currently has one member per cell. Cell A
("synthetic corpus": the retrieval documents are themselves LLM output
produced from/alongside the queries) is occupied only by MathNet-Retrieve.
MELD (MathLeap, arXiv:2606.23959) qualifies as a
second member, and the paper says so in its own words (§3, p.4):

    "We generated MELD by iterating through each of the nine pairs of
     complementary domains, describing the connection between the two
     fields, and prompting Claude Opus 4.7 to generate 30 pairs of
     mathematically equivalent but lexically distinct statements. We then
     manually reviewed these, making modifications to increase dissimilarity
     while preserving equivalence. We then evaluated all statements using
     GPT-5.5 (medium) to check (i) whether both statements were valid,
     (ii) whether they were equivalent, and (iii) whether they could be
     made to sound less similar."

Both sides of every MELD retrieval item are LLM-authored: the query-role
statement, the gold document-role statement, AND the 541 distractors (all
540 pair statements come out of the same generation call; the released
distractors are same-framing textbook statements with no organic provenance
given). Human effort enters as *review and revision* of LLM text, not as a
source corpus. That is precisely cell A, and it is the opposite of SABER's
cell B (organic competition problems with LLM-assigned graded labels), where
the analogous attack FAILED (results/saber_attack_results). So MELD is the
right second data point, and the taxonomy makes a directional, falsifiable
prediction about it. Alternatives considered and rejected: MIRB (Ju & Dong,
cited by MELD §2) mixes organic sources — MathStackExchange duplicates, Lean
premise retrieval — i.e. cell B/C, not a second cell-A member; BRIGHT's math
splits are organic StackExchange/AoPS content (cell C); SABER is already the
cell-B member. Nothing else in papers/ ships an all-LLM math IR corpus.

WHAT THIS SCRIPT BUILDS.
Training pairs manufactured with MELD's own construction recipe, through a
DIFFERENT generator (Qwen3-32B-AWQ vs their Claude Opus 4.7) — the same
design as the MathNet attack (Qwen3-32B against GPT-5-built eval data) and
the SABER attack. Two groups, generated in one pass and trained as two
matched-budget arms:

  group "meld9"    the NINE complementary-domain pairs MELD itself used
                   (their Table 1, quoted below via the released framing
                   strings). Maximum recipe fidelity; the direct analog of
                   "same corpus, disjoint items" in the MathNet attack.
                   This is the PRIMARY attack arm.
  group "heldout"  27 complementary-domain pairs we authored, none of whose
                   subfield names appears in MELD's Table 1 or in the
                   released distractor framings. Same recipe, zero topical
                   overlap. This is the SPECIFICITY arm: if meld9 wins and
                   heldout does not, the exposure is topic-level; if both
                   win, it is the genre of LLM-written cross-dialect
                   equivalence pairs — the same genre-vs-template question
                   the D1-D4 dose panel asks of MathNet-Retrieve.

DISJOINTNESS (mandatory; the attack claim depends on it). MELD has no source
corpus to be disjoint from — the recipe's only inputs are a domain pair and a
prose description of the connection — so disjointness is enforced at the ITEM
level against every released eval item (540 pair statements + 541
distractors = 1,081 texts), with four independent gates in --stage pairs:
  1. normalized-text exact match                       -> drop
  2. word-5-gram containment >= 0.35 (the SABER gate)  -> drop
  3. character-8-gram Jaccard >= 0.60                  -> drop
  4. embedding cosine >= 0.90 under an INDEPENDENT encoder
     (all-mpnet-base-v2 by default: never a base model, never fine-tuned in
     this campaign, so the gate cannot be tuned to the attacked geometry)
Gate 4 is deliberately conservative FOR US: it deletes exactly the training
pairs that would help most, so P-M1 is tested against a handicapped attack.
Counts, thresholds, and the full score distributions land in
results/meld_disjointness.json. The "heldout" group additionally carries a
domain-name gate (no MELD subfield string reused), reported separately.

HONEST FIDELITY LIMITS (report these with any number this script produces):
  * MELD's generation prompt is NOT published — the paper describes the
    procedure in the paragraph quoted above and the released records fix the
    output schema (domain / topic / entry_1{framing,statement} /
    entry_2{framing,statement}). MELD_GEN_SYSTEM below is a faithful
    RECONSTRUCTION of that procedure into that exact schema, not a verbatim
    copy (contrast: the SABER attack used their repo's prompt byte-for-byte).
  * their per-domain-pair connection descriptions are not published either;
    the CONNECTION strings below are ours for both groups.
  * their manual review-and-revise pass is not reproducible; we keep only
    their automated check — the GPT-5.5 three-question validity/equivalence/
    dissimilarity screen, run here through Qwen3-32B-AWQ (MELD_JUDGE_SYSTEM).
    We gate on questions (i) and (ii) and only RECORD (iii), because (iii)
    fed their revision loop, which we do not run.
  * MELD produced 30 pairs per domain pair, one call each; to reach a
    6,145-row training budget we sample --rounds independent calls per
    domain pair at temperature 1.0, passing back the topics already produced
    so later rounds extend rather than repeat. Multiple lexical realizations
    of one topic are allowed (deduplicated on statement text). This is a
    SCALE deviation from a one-shot recipe and must be stated.
  * training negatives are same-framing statements drawn from our own
    generated pool, mirroring the distribution of MELD's released
    distractors (true statements in the partner's framing); MELD never
    describes how its distractors were built.

STAGES (chained by scripts/attack_second_benchmark.slurm):
  --stage domains   CPU, seconds. Writes/validates the domain-pair table and
                    the domain-name disjointness report.
  --stage generate  GPU (vllm env). Pass 1 = MELD's generation recipe,
                    pass 2 = MELD's automated screen, one resident model.
                    -> data/meld_attack/raw_<group>.jsonl
                    (--backend stub = deterministic fake completions for
                    login-node contract tests; NEVER train on stub output)
  --stage pairs     CPU (+ a small encoder for gate 4). Gating, dedup, and
                    trainer-format rows -> data/meld_attack/pairs_<group>.jsonl
                    + results/meld_disjointness.json + results/meld_pairs_summary.json
  --stage power     CPU, minutes, NO attacked model involved. Cluster
                    bootstrap of MELD's own sampling noise from two
                    off-the-shelf encoders -> results/meld_attack_power.json
                    (the minimum detectable effect quoted in the
                    pre-registration).

Usage:
  python scripts/generate_meld_pairs.py --stage domains
  python scripts/generate_meld_pairs.py --stage power
  python scripts/generate_meld_pairs.py --stage generate --backend vllm \
      --model Qwen/Qwen3-32B-AWQ --group meld9 --rounds 24
  python scripts/generate_meld_pairs.py --stage pairs --group meld9
Login-node contract test (no GPU, no model):
  python scripts/generate_meld_pairs.py --stage generate --backend stub \
      --group meld9 --rounds 1 --out data/meld_attack/raw_stub.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/ibex/user/habiam0b/MathNet_Follow_Up")
sys.path.insert(0, str(ROOT / "scripts"))
# Shared JSON parser.  extract_json_robust = extract_json + a LaTeX-escape
# repair pass; MELD statements are explicitly requested "in natural language
# and LaTeX", and raw `\mathbb`/`\sigma` inside a JSON string is what makes a
# whole 30-pair call unparseable (job 49630276).  See repair_json_escapes.
from generate_llm_pairs import extract_json_robust  # noqa: E402

MELD_DIR = ROOT / "data" / "external" / "meld"
ATTACK_DIR = ROOT / "data" / "meld_attack"
RESULTS = ROOT / "results"
MELD_FILES = ("adversarial_theorem_pairs_2.json", "distractors_all.json")

# ---------------------------------------------------------------------------
# The domain-pair table = the recipe's ONLY input.
#
# group "meld9": MELD's own nine complementary-domain pairs. field_1/field_2
# are the framing strings EXACTLY as they appear in the released records
# (adversarial_theorem_pairs_2.json), and `domain` is the released domain
# label for that pair -- both verified against the dataset by --stage domains,
# which aborts if either drifts. Table 1 of the paper names two of them
# slightly differently (algebraic topology / group theory -> released
# "topology" / "algebra"; combinatorics / generating functions -> released
# "discrete math" / "complex analysis"); the released strings win because they
# are what the generator has to reproduce.
# `connection` is OURS -- MELD does not publish its connection descriptions.
# ---------------------------------------------------------------------------
MELD9 = [
    dict(domain="algebra", field_1="vector spaces", field_2="module theory",
         connection="A vector space over a field is exactly a module over that "
                    "field; bases, dimension and linear maps become generating "
                    "sets, free ranks and module homomorphisms."),
    dict(domain="probability", field_1="probability", field_2="measure theory",
         connection="A probability space is a measure space of total mass one; "
                    "random variables, expectation and independence become "
                    "measurable functions, integrals and product measures."),
    dict(domain="foundations", field_1="set theory", field_2="category theory",
         connection="Constructions on sets and functions have universal-property "
                    "descriptions; unions, products and images become colimits, "
                    "limits and factorizations of arrows."),
    dict(domain="algebraic_geometry", field_1="geometry",
         field_2="commutative algebra",
         connection="Geometric objects correspond to rings of functions; points, "
                    "subvarieties and dimension become maximal ideals, prime "
                    "ideals and Krull dimension."),
    dict(domain="algebraic_topology", field_1="topology", field_2="algebra",
         connection="Spaces and continuous maps are studied through the groups "
                    "they induce; loops, coverings and connectivity become "
                    "generators, subgroups and quotients."),
    dict(domain="spectral_graph_theory", field_1="graph theory",
         field_2="linear algebra",
         connection="A graph is encoded by its adjacency or Laplacian matrix; "
                    "walks, connectivity and bipartiteness become matrix powers, "
                    "eigenvalue multiplicities and spectral symmetry."),
    dict(domain="discrete_math", field_1="discrete math",
         field_2="complex analysis",
         connection="Counting sequences are the coefficients of generating "
                    "functions; recurrences, convolutions and asymptotics become "
                    "functional equations, products and singularity analysis."),
    dict(domain="representation_theory", field_1="representation theory",
         field_2="Fourier analysis",
         connection="Decomposing a representation is harmonic analysis on the "
                    "group; irreducibles, characters and multiplicities become "
                    "frequencies, transforms and Plancherel coefficients."),
    dict(domain="algebraic_combinatorics", field_1="symmetric functions",
         field_2="tableaux",
         connection="Symmetric function identities are counted by fillings of "
                    "Young diagrams; Schur expansions, products and "
                    "specializations become tableaux enumerations, insertion "
                    "algorithms and shape statistics."),
]

# group "heldout": authored complementary-domain pairs. Constraint enforced by
# --stage domains: no field name here may be an exact (normalized) match of any
# MELD Table-1 subfield or released distractor framing. Substring collisions
# (e.g. "Riemannian geometry" vs "geometry") are ALLOWED but reported.
HELDOUT = [
    dict(domain="differential_equations",
         field_1="ordinary differential equations", field_2="dynamical systems",
         connection="An equation and the flow it generates are one object seen "
                    "analytically and geometrically; solutions, equilibria and "
                    "stability become orbits, fixed points and attractors."),
    dict(domain="riemannian_geometry",
         field_1="Riemannian geometry", field_2="tensor calculus",
         connection="Invariant statements about curvature, geodesics and "
                    "connections have index expressions; each intrinsic object "
                    "is a tensor field with a transformation rule."),
    dict(domain="lie_theory", field_1="Lie theory", field_2="matrix groups",
         connection="Abstract Lie groups and their algebras are realized as "
                    "closed subgroups of GL_n and spaces of matrices, tied "
                    "together by the matrix exponential."),
    dict(domain="galois_theory", field_1="field theory",
         field_2="Galois correspondence",
         connection="Extensions, degrees and solvability translate into "
                    "automorphism groups, subgroup lattices and normality."),
    dict(domain="p_adic_number_theory", field_1="number theory",
         field_2="p-adic analysis",
         connection="Congruence and divisibility statements over the integers "
                    "are analytic statements about valuations, convergence and "
                    "completeness in Q_p."),
    dict(domain="analytic_number_theory", field_1="arithmetic functions",
         field_2="Dirichlet series",
         connection="Multiplicative functions and their convolutions correspond "
                    "to Euler products and identities between Dirichlet series."),
    dict(domain="nonstandard_analysis", field_1="real analysis",
         field_2="nonstandard analysis",
         connection="Epsilon-delta statements about limits, continuity and "
                    "compactness have equivalent formulations with "
                    "infinitesimals, hyperreals and standard parts."),
    dict(domain="quantum_formalism", field_1="Hilbert space theory",
         field_2="quantum mechanics formalism",
         connection="Self-adjoint operators, spectral decompositions and unit "
                    "vectors are observables, measurement outcomes and states "
                    "in Dirac notation."),
    dict(domain="convex_optimization", field_1="convex analysis",
         field_2="Lagrangian duality",
         connection="Convex sets, subgradients and separating hyperplanes "
                    "correspond to dual problems, multipliers and complementary "
                    "slackness."),
    dict(domain="matroids", field_1="matroid theory", field_2="greedy algorithms",
         connection="The independence axioms are exactly when the greedy "
                    "procedure is optimal; rank, circuits and bases become loop "
                    "invariants, failure cases and terminal solutions."),
    dict(domain="information_theory", field_1="information theory",
         field_2="coding theory",
         connection="Entropy, mutual information and capacity correspond to "
                    "achievable rates, minimum distance and decoding error."),
    dict(domain="statistical_mechanics", field_1="Markov chains",
         field_2="statistical mechanics",
         connection="Transition kernels, stationary distributions and mixing "
                    "correspond to Gibbs measures, equilibrium ensembles and "
                    "relaxation."),
    dict(domain="stochastic_analysis", field_1="stochastic processes",
         field_2="partial differential equations",
         connection="Expectations of functionals of diffusions solve parabolic "
                    "equations; generators, martingales and hitting times become "
                    "differential operators, harmonicity and boundary data."),
    dict(domain="numerical_analysis", field_1="numerical analysis",
         field_2="approximation theory",
         connection="Error bounds for quadrature, interpolation and iteration "
                    "are statements about best approximation, moduli of "
                    "continuity and degree."),
    dict(domain="computability", field_1="computability theory",
         field_2="formal logic",
         connection="Decidability, reductions and halting correspond to "
                    "definability, provability and incompleteness in formal "
                    "systems."),
    dict(domain="model_theory", field_1="model theory",
         field_2="universal algebra",
         connection="Elementary classes, embeddings and compactness correspond "
                    "to varieties, homomorphisms and equational theories."),
    dict(domain="type_theory", field_1="lambda calculus", field_2="type theory",
         connection="Terms, reduction and normalization correspond to proofs, "
                    "cut elimination and propositions under Curry-Howard."),
    dict(domain="automata", field_1="automata theory",
         field_2="semigroup theory",
         connection="Recognizable languages, minimization and pumping correspond "
                    "to finite monoids, syntactic congruences and homomorphic "
                    "images."),
    dict(domain="cryptography", field_1="elliptic curves",
         field_2="public-key cryptography",
         connection="Group laws, torsion and discrete logarithms on curves "
                    "correspond to key exchange, hardness assumptions and "
                    "signature schemes."),
    dict(domain="quadratic_forms", field_1="quadratic forms",
         field_2="integral lattices",
         connection="Equivalence, discriminants and representation numbers of "
                    "forms correspond to isometry, covolume and shortest vectors "
                    "of lattices."),
    dict(domain="knots", field_1="knot theory", field_2="braid theory",
         connection="Knot invariants, diagrammatic moves and closures correspond "
                    "to braid words, Markov moves and group relations."),
    dict(domain="mechanics", field_1="calculus of variations",
         field_2="Hamiltonian mechanics",
         connection="Stationary actions, Euler-Lagrange equations and "
                    "constraints correspond to canonical flows, conjugate "
                    "momenta and conserved quantities."),
    dict(domain="order", field_1="order theory", field_2="lattice theory",
         connection="Partial orders, chains and completeness correspond to "
                    "meets, joins, distributivity and closure operators."),
    dict(domain="constructive_logic", field_1="proof theory",
         field_2="intuitionistic logic",
         connection="Classical derivability, double negation and excluded middle "
                    "correspond to constructive provability, translations and "
                    "realizability."),
    dict(domain="ergodic_theory", field_1="ergodic theory",
         field_2="symbolic dynamics",
         connection="Invariant measures, mixing and recurrence correspond to "
                    "shift spaces, subshifts of finite type and word "
                    "combinatorics."),
    dict(domain="equilibria", field_1="game theory",
         field_2="fixed point theory",
         connection="Existence of equilibria, best-response correspondences and "
                    "minimax values correspond to fixed points, convexity "
                    "hypotheses and continuity."),
    dict(domain="spectral_operator_theory",
         field_1="spectral theory of operators",
         field_2="Sturm-Liouville problems",
         connection="Eigenvalue asymptotics, self-adjointness and completeness "
                    "of eigenfunctions correspond to boundary-value problems, "
                    "oscillation and orthogonal expansions."),
]

GROUPS = {"meld9": MELD9, "heldout": HELDOUT, "meld9alt": MELD9, "meld9altm": MELD9}
# Which system prompt each group generates under. meld9alt reuses MELD9's
# domain specs so that the ONLY difference from meld9 is the prompt.
GROUP_PROMPT = {"meld9": "recipe", "heldout": "recipe", "meld9alt": "alt", "meld9altm": "alt"}
# meld9altm (2026-09-17): the ALT prompt generated with meld9's
# exact procedure (greedy first round, pair target, same round cap), so that the
# 84/16 prompt control is matched on the generation factor as well.

# ---------------------------------------------------------------------------
# PROMPTS.
#
# MELD_GEN_SYSTEM is a RECONSTRUCTION of the procedure the MathLeap paper
# describes (§3, p.4 -- quoted verbatim in this module's docstring) into the
# output schema of the released records. Every clause below traces to their
# sentence: "iterating through each of the nine pairs of complementary
# domains" -> one call per domain pair; "describing the connection between the
# two fields" -> the CONNECTION line in the user turn; "prompting ... to
# generate 30 pairs of mathematically equivalent but lexically distinct
# statements" -> the two bullets on equivalence and lexical distinctness plus
# --pairs-per-call (default 30); "all content belongs to a basic graduate
# curriculum in mathematics" (their §3 justification) -> the curriculum bullet.
# NOTHING here is quoted from an unpublished prompt.
# ---------------------------------------------------------------------------
MELD_GEN_SYSTEM = """You are a research mathematician building an evaluation set of mathematically equivalent but lexically different statement pairs.

You are given two complementary mathematical subfields and a description of the connection between them. Produce exactly {n_pairs} statement pairs. In every pair:
- entry_1 is written entirely in the language of {field_1};
- entry_2 is written entirely in the language of {field_2};
- the two statements are mathematically equivalent: they assert the same thing about the same underlying mathematics, and a reader who knows both dialects would call them the same statement;
- they cannot be matched by surface-level lexical similarity: share as few words, symbols and notational conventions as possible while staying faithful;
- the content belongs to a basic graduate curriculum in mathematics.

Each pair carries a short "topic" label naming the shared idea. All {n_pairs} topics must be distinct from each other.

Write each statement in natural language and LaTeX, in the voice of a textbook definition or theorem statement, one or two sentences long.

Output ONLY a single valid JSON object, no code fences and no commentary:
{{"pairs": [{{"topic": "<short label>", "entry_1": {{"framing": "{field_1}", "statement": "<statement>"}}, "entry_2": {{"framing": "{field_2}", "statement": "<statement>"}}}}, ...]}}"""

# ---------------------------------------------------------------------------
# MELD_GEN_SYSTEM_ALT -- the PROMPT CONTROL (round-2 review finding R2-H6).
#
# The paper's P-M3a contrast (attack vs ctrl-LLM) varies task, source data, budget
# and negative design simultaneously, so it cannot separate "MELD's recipe" from
# "cross-dialect equivalence training in general".  This prompt is the missing
# rung: the SAME nine MELD domain pairs, the SAME output schema, the SAME
# generator, judge, row budget and negatives-per-row -- only the authorial
# framing differs, exactly as D3 differs from D1 in the MathNet ladder.
#
# What is deliberately NOT here: the lexical-distinctness instruction ("share as
# few words, symbols and notational conventions as possible"), which is the
# clause of MELD's procedure most specific to what its benchmark measures; the
# "research mathematician building an evaluation set" persona; and the
# textbook-definition voice specification.  What is held identical: the schema,
# the pair count, the topic-distinctness requirement and the curriculum level,
# so this is a difference in wording and framing, not in output format.
# ---------------------------------------------------------------------------
MELD_GEN_SYSTEM_ALT = """You are preparing translation exercises for a graduate reading course that runs two seminars in parallel.

Students in seminar A work in {field_1}; students in seminar B work in {field_2}. Write exactly {n_pairs} exercises. Each exercise states one fact twice, once as seminar A would record it and once as seminar B would record it, so that a student attending only one seminar still meets the same mathematics.

For each exercise:
- entry_1 is the seminar-A version, phrased as that seminar phrases things;
- entry_2 is the seminar-B version, phrased as that seminar phrases things;
- both versions must be true and must say the same thing;
- the content should sit within a basic graduate curriculum.

Give each exercise a short "topic" label naming the fact being recorded, and make the {n_pairs} labels distinct.

Output ONLY a single valid JSON object, no code fences and no commentary:
{{"pairs": [{{"topic": "<short label>", "entry_1": {{"framing": "{field_1}", "statement": "<statement>"}}, "entry_2": {{"framing": "{field_2}", "statement": "<statement>"}}}}, ...]}}"""

MELD_GEN_USER = """Subfield 1: {field_1}
Subfield 2: {field_2}

Connection between the two fields: {connection}

Produce {n_pairs} pairs."""

# Round >= 2 continuation (our scale mechanism, not part of their recipe).
MELD_GEN_USER_CONT = """Subfield 1: {field_1}
Subfield 2: {field_2}

Connection between the two fields: {connection}

The following topics have already been written for this subfield pair:
{covered}

Produce {n_pairs} further pairs. Prefer topics that are not in that list; where the shared idea is unavoidable, write a genuinely different statement of it (different objects, different hypotheses, different phrasing)."""

# Their automated screen (MathLeap §3, p.4): "evaluated all statements using
# GPT-5.5 (medium) to check (i) whether both statements were valid, (ii)
# whether they were equivalent, and (iii) whether they could be made to sound
# less similar." One JSON object per pair, one field per question. We gate on
# (i) and (ii); (iii) is recorded only, because it fed their manual revision
# loop, which we do not reproduce.
MELD_JUDGE_SYSTEM = """You are an expert mathematician screening candidate items for an evaluation set of mathematically equivalent but lexically different statement pairs.

You will be given Statement 1 (written in one subfield's language) and Statement 2 (written in another's). Answer three questions:
(i)   is each statement, on its own, a valid and correct mathematical statement?
(ii)  are the two statements mathematically equivalent -- do they assert the same thing about the same underlying mathematics?
(iii) could the pair be rewritten to sound LESS similar to each other without breaking the equivalence?

Judge (ii) strictly. Sharing a topic is not equivalence: a changed hypothesis, a dropped finiteness assumption, a different quantifier, or a statement that merely implies the other in one direction makes them NOT equivalent.

Return ONLY a single valid JSON object, no code fences and no commentary:
{"entry_1_valid": true|false, "entry_2_valid": true|false, "equivalent": true|false, "could_be_less_similar": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}"""

MELD_JUDGE_USER = """Statement 1 ({framing_1}):

{statement_1}

Statement 2 ({framing_2}):

{statement_2}"""


# ---------------------------------------------------------------------------
# Text helpers (gates 1-3). WORD_RE / word_grams mirror generate_saber_pairs.py
# so the two attacks' disjointness reports use the same containment statistic.
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-z0-9]+")


def norm_text(s: str) -> str:
    """Lowercase, strip LaTeX spacing/markup noise, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\\(?:textbf|textit|emph|mathbf|mathrm|operatorname|text)\b", " ", s)
    s = re.sub(r"[{}$\\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def word_grams(text: str, n: int = 5) -> set:
    toks = WORD_RE.findall(norm_text(text))
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def char_grams(text: str, n: int = 8) -> set:
    t = norm_text(text).replace(" ", "")
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_meld_eval_texts():
    """All 1,081 released MELD eval texts: 540 pair statements + 541 distractors."""
    missing = [f for f in MELD_FILES if not (MELD_DIR / f).exists()]
    if missing:
        sys.exit(f"[FATAL] missing MELD data {missing} in {MELD_DIR} — run any "
                 f"eval_meld.py invocation once to download it")
    pairs = json.load(open(MELD_DIR / MELD_FILES[0], encoding="utf-8"))["pairs"]
    distr = json.load(open(MELD_DIR / MELD_FILES[1], encoding="utf-8"))
    stmts = []
    for p in pairs:
        stmts.append(p["entry_1"]["statement"])
        stmts.append(p["entry_2"]["statement"])
    dtexts = [s for lst in distr.values() for s in lst]
    if len(pairs) != 270:
        sys.exit(f"[FATAL] expected 270 MELD pairs, got {len(pairs)}")
    return pairs, distr, stmts, dtexts


# ---------------------------------------------------------------------------
# STAGE domains — build/validate the recipe's input table
# ---------------------------------------------------------------------------
def stage_domains(args):
    t0 = time.time()
    pairs, distr, _stmts, _d = load_meld_eval_texts()

    # 1. meld9 must reproduce the released (framing_1, framing_2, domain) triples.
    released = {}
    for p in pairs:
        released[(p["entry_1"]["framing"], p["entry_2"]["framing"])] = p["domain"]
    drift = []
    for d in MELD9:
        key = (d["field_1"], d["field_2"])
        if key not in released:
            drift.append(f"{key} not a released MELD framing pair")
        elif released[key] != d["domain"]:
            drift.append(f"{key} domain {d['domain']!r} != released "
                         f"{released[key]!r}")
    if len(released) != len(MELD9):
        drift.append(f"released has {len(released)} framing pairs, MELD9 has "
                     f"{len(MELD9)}")
    if drift:
        sys.exit("[FATAL] meld9 table drifted from the released dataset:\n  "
                 + "\n  ".join(drift))

    # 2. heldout must not reuse any MELD subfield name (exact, normalized).
    forbidden = {norm_text(f) for pair in released for f in pair}
    forbidden |= {norm_text(k) for k in distr}
    # Table-1 names that the released records rename (paper p.4).
    forbidden |= {norm_text(x) for x in
                  ("algebraic topology", "group theory", "combinatorics",
                   "generating functions", "probability theory")}
    exact_hits, substr_hits = [], []
    for d in HELDOUT:
        for f in (d["field_1"], d["field_2"]):
            nf = norm_text(f)
            if nf in forbidden:
                exact_hits.append(f)
            for bad in forbidden:
                if bad != nf and (f" {bad} " in f" {nf} "):
                    substr_hits.append({"heldout_field": f, "meld_subfield": bad})
    if exact_hits:
        sys.exit("[FATAL] heldout reuses MELD subfield names exactly: "
                 + ", ".join(sorted(set(exact_hits))))
    dup = [f for f, c in Counter(
        [d[k] for d in HELDOUT for k in ("field_1", "field_2")]).items() if c > 1]

    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    table = {g: GROUPS[g] for g in GROUPS}
    (ATTACK_DIR / "domain_pairs.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False))
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "domain-level disjointness for the MELD recipe-matching attack",
        "meld9": {
            "n_domain_pairs": len(MELD9),
            "verified_against": str(MELD_DIR / MELD_FILES[0]),
            "check": "field_1/field_2/domain reproduce the released framing "
                     "pairs and domain labels exactly (job aborts otherwise)",
        },
        "heldout": {
            "n_domain_pairs": len(HELDOUT),
            "gate": "no field name may exactly match (normalized) any MELD "
                    "Table-1 subfield or released distractor framing",
            "n_exact_collisions": 0,
            "substring_collisions": substr_hits,
            "substring_collision_note":
                "ALLOWED and reported: a compound name may contain a MELD "
                "subfield word (e.g. 'Riemannian geometry' vs MELD's "
                "'geometry'). The dialect and the statements are different; "
                "the item-level gates in --stage pairs are what enforce "
                "eval disjointness.",
            "duplicate_field_names_within_heldout": dup,
        },
        "forbidden_subfield_names": sorted(forbidden),
        "output": str(ATTACK_DIR / "domain_pairs.json"),
        "runtime_s": round(time.time() - t0, 2),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "meld_domain_disjointness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[domains] meld9 {len(MELD9)} pairs verified against the released "
          f"dataset; heldout {len(HELDOUT)} pairs, 0 exact name collisions, "
          f"{len(substr_hits)} substring collisions (reported)")
    print(f"[domains] -> {ATTACK_DIR / 'domain_pairs.json'} + "
          f"results/meld_domain_disjointness.json")


# ---------------------------------------------------------------------------
# Backends (mirrors generate_saber_pairs.stage_summaries)
# ---------------------------------------------------------------------------
def make_backend(args):
    if args.backend == "stub":
        import zlib

        def run(batches, max_tokens, temperature):
            outs = []
            for msgs in batches:
                system, user = msgs[0]["content"], msgs[1]["content"]
                if system.startswith("You are an expert mathematician screening"):
                    h = zlib.crc32(user.encode())
                    outs.append(json.dumps({
                        "entry_1_valid": True, "entry_2_valid": True,
                        # seeded ~25% rejection so unverified rows exist
                        "equivalent": (h % 4) != 0,
                        "could_be_less_similar": (h % 3) == 0,
                        "confidence": 0.8, "reason": "stub deterministic verdict"}))
                    continue
                f1 = re.search(r"Subfield 1: (.*)", user).group(1).strip()
                f2 = re.search(r"Subfield 2: (.*)", user).group(1).strip()
                n = int(re.search(r"Produce (\d+)", user).group(1))
                salt = zlib.crc32(user.encode()) % 100000
                items = [{
                    "topic": f"stub topic {salt}-{k}",
                    "entry_1": {"framing": f1,
                                "statement": f"[stub {f1} {salt}-{k}] Let $X$ be an "
                                             f"object of type {salt}-{k}; then $X$ "
                                             f"satisfies property P."},
                    "entry_2": {"framing": f2,
                                "statement": f"[stub {f2} {salt}-{k}] Assume $M$ is a "
                                             f"structure indexed by {salt}-{k}; then "
                                             f"$M$ has attribute Q."},
                } for k in range(n)]
                outs.append(json.dumps({"pairs": items}, ensure_ascii=False))
            return outs
        return run

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="auto", seed=args.seed,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True)

    # PER-CALL SAMPLING SEED (round-2 review finding R2-M26).  Pinning
    # seed=args.seed for every call made sampling reproducible but also made a
    # FAILURE reproducible: the continuation prompt is unchanged for a domain
    # that yielded nothing, so a domain whose completion drifted off-schema once
    # re-emitted the byte-identical completion on every later round -- confirmed
    # by two groups of five md5-identical failure dumps in job 49639491.  We now
    # derive each call's seed from (args.seed, the batch's position in the run),
    # which keeps the whole run reproducible from args.seed while ensuring a
    # retry is actually a retry.
    # PER-ROUND, not per-request. Giving each request in a batch its own seed
    # pushes vLLM onto flashinfer's per-request sampler, which JIT-compiles and
    # needs nvcc -- absent on these nodes, and it took job 49677956 down in 117 s.
    # One seed per batch keeps the exact execution path the meld9 arm generated
    # under (so the two arms stay comparable) while still advancing the seed
    # between rounds, which is what the R2-M26 defect actually needed: a domain
    # whose completion drifted off-schema is now retried under different
    # sampling instead of re-emitting byte-identical output.
    batch_counter = {"n": 0}

    def run(batches, max_tokens, temperature):
        batch_seed = (args.seed * 1_000_003 + batch_counter["n"]) % (2 ** 31 - 1)
        batch_counter["n"] += 1
        sp = SamplingParams(temperature=temperature,
                            top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=max_tokens, seed=batch_seed)
        outs = llm.chat(batches, sp, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outs]
    return run


def build_gen_messages(spec, n_pairs, covered, group="meld9"):
    user_tpl = MELD_GEN_USER_CONT if covered else MELD_GEN_USER
    sys_tpl = (MELD_GEN_SYSTEM_ALT if GROUP_PROMPT.get(group) == "alt"
               else MELD_GEN_SYSTEM)
    return [
        {"role": "system", "content": sys_tpl.format(
            n_pairs=n_pairs, field_1=spec["field_1"], field_2=spec["field_2"])},
        {"role": "user", "content": user_tpl.format(
            field_1=spec["field_1"], field_2=spec["field_2"],
            connection=spec["connection"], n_pairs=n_pairs,
            covered="\n".join(f"- {t}" for t in covered))},
    ]


def dump_gen_failure(call_idx, spec, rnd, text, args):
    """Persist an unparseable completion so the NEXT failure is diagnosable.

    Job 49630276 lost one of two generation calls to a JSON parse failure and
    nothing about the offending text survived: the summary recorded the count,
    the vllm log recorded only throughput, and re-creating it costs a GPU
    allocation.

    HEAD *AND* TAIL (fixed 2026-07-31, round-2 review finding R2-M26).  The
    first version kept only the leading 8 KB, which is exactly the wrong half
    for the question one actually asks of a failed completion -- "was it cut
    off?" -- and it misled our own post-mortem into diagnosing truncation where
    re-parsing later showed schema drift (16 of 25 dumps balance early).  We now
    keep 6 KB from each end with an explicit elision marker, so the closing
    brace, or its absence, is always visible.
    """
    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    path = ATTACK_DIR / f"genfail_{call_idx:03d}.txt"
    HEAD = TAIL = 6144
    if len(text) <= HEAD + TAIL:
        body, elided = text, 0
    else:
        elided = len(text) - HEAD - TAIL
        body = (text[:HEAD]
                + f"\n\n... [{elided} chars elided by dump_gen_failure] ...\n\n"
                + text[-TAIL:])
    header = (f"# UNPARSEABLE GENERATION COMPLETION\n"
              f"# group={args.group} domain={spec['domain']} round={rnd} "
              f"call={call_idx}\n"
              f"# backend={args.backend} model={args.model} "
              f"max_tokens={args.gen_max_tokens} temperature-round="
              f"{'greedy' if rnd == 1 and args.first_round_greedy else args.gen_temperature}\n"
              f"# completion_chars={len(text)} "
              f"(dump keeps first {HEAD} + last {TAIL}; {elided} elided)\n"
              + "-" * 72 + "\n")
    path.write_text(header + body, encoding="utf-8")
    print(f"[gen] JSON parse FAILED: call {call_idx} ({spec['domain']}, round "
          f"{rnd}, {len(text)} chars) -> {path}", flush=True)
    return path


def coerce_pairs(obj):
    """Pull the pair list out of whatever shape the model returned.

    Accepts the requested {"pairs": [...]}, a bare top-level [...] and a
    single-list-valued object under another key; anything else is a failure.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        items = obj.get("pairs")
        if items is None:
            lists = [v for v in obj.values() if isinstance(v, list)]
            items = lists[0] if len(lists) == 1 else None
        return items
    return None


def build_judge_messages(rec):
    return [
        {"role": "system", "content": MELD_JUDGE_SYSTEM},
        {"role": "user", "content": MELD_JUDGE_USER.format(
            framing_1=rec["entry_1"]["framing"],
            statement_1=rec["entry_1"]["statement"],
            framing_2=rec["entry_2"]["framing"],
            statement_2=rec["entry_2"]["statement"])},
    ]


# ---------------------------------------------------------------------------
# STAGE generate — pass 1 (their recipe) + pass 2 (their automated screen)
# ---------------------------------------------------------------------------
def stage_generate(args):
    t0 = time.time()
    specs = GROUPS[args.group]
    if args.max_domain_pairs:
        specs = specs[: args.max_domain_pairs]
    out_path = Path(args.out or (ATTACK_DIR / f"raw_{args.group}.jsonl"))

    if args.dry_run:
        s = specs[0]
        g = build_gen_messages(s, args.pairs_per_call, [], args.group)
        g2 = build_gen_messages(s, args.pairs_per_call, ["Identity map", "Product"],
                                args.group)
        j = build_judge_messages({
            "entry_1": {"framing": s["field_1"], "statement": "<ENTRY 1 FROM PASS 1>"},
            "entry_2": {"framing": s["field_2"], "statement": "<ENTRY 2 FROM PASS 1>"}})
        for title, msgs in (("PASS 1 round 1", g), ("PASS 1 round >= 2", g2),
                            ("PASS 2 (MELD's automated screen)", j)):
            print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
            print(f"--- SYSTEM ---\n{msgs[0]['content']}")
            print(f"\n--- USER ---\n{msgs[1]['content']}")
        return

    run = make_backend(args)
    covered = defaultdict(list)          # domain -> topics produced so far
    records, gen_fail, dup_in_round = [], 0, 0
    n_gen_calls, per_call_pairs = 0, []   # per-call accounting for the smoke gate
    seen_stmt = set()

    def covered_sample(domain, rnd):
        """Topics fed back in the round>=2 continuation prompt.

        A seeded SAMPLE across everything covered so far, not the first N:
        showing only the earliest topics would let the later rounds
        re-generate their own recent output unchecked.
        """
        seen = covered[domain]
        if len(seen) <= args.max_covered_shown:
            return list(seen)
        return random.Random(args.seed * 1000 + rnd).sample(
            seen, args.max_covered_shown)

    for rnd in range(1, args.rounds + 1):
        batches = [build_gen_messages(
            s, args.pairs_per_call,
            covered_sample(s["domain"], rnd) if rnd > 1 else [],
            args.group)
            for s in specs]
        temp = 0.0 if (rnd == 1 and args.first_round_greedy) else args.gen_temperature
        outs = run(batches, args.gen_max_tokens, temp)
        n_round = 0
        for spec, text in zip(specs, outs):
            n_gen_calls += 1
            items = coerce_pairs(extract_json_robust(text))
            if not isinstance(items, list):
                gen_fail += 1
                dump_gen_failure(n_gen_calls, spec, rnd, text, args)
                continue
            n_call = 0
            for it in items[: args.pairs_per_call]:
                if not isinstance(it, dict):
                    continue
                e1, e2 = it.get("entry_1"), it.get("entry_2")
                if not (isinstance(e1, dict) and isinstance(e2, dict)):
                    continue
                s1 = str(e1.get("statement") or "").strip()
                s2 = str(e2.get("statement") or "").strip()
                if len(s1) < args.min_statement_chars or len(s2) < args.min_statement_chars:
                    continue
                key = (norm_text(s1), norm_text(s2))
                if key in seen_stmt or key[0] == key[1]:
                    dup_in_round += 1
                    continue
                seen_stmt.add(key)
                topic = str(it.get("topic") or "").strip() or "(untitled)"
                covered[spec["domain"]].append(topic)
                records.append({
                    "pair_id": f"meldatk-{args.group}-{len(records):06d}",
                    "group": args.group, "domain": spec["domain"],
                    "round": rnd, "topic": topic,
                    "entry_1": {"framing": spec["field_1"], "statement": s1},
                    "entry_2": {"framing": spec["field_2"], "statement": s2},
                    "generator_model": (args.model if args.backend != "stub"
                                        else "STUB (never train on this)"),
                })
                n_round += 1
                n_call += 1
            per_call_pairs.append(n_call)
        print(f"[gen] round {rnd}/{args.rounds}: +{n_round} new pairs "
              f"(total {len(records)}, {dup_in_round} dedup, {gen_fail} JSON "
              f"failures) at {time.time() - t0:.0f}s", flush=True)
        if args.target_pairs and len(records) >= args.target_pairs:
            print(f"[gen] --target-pairs {args.target_pairs} reached; stopping early")
            break

    if not records:
        sys.exit("[FATAL] pass 1 produced no parseable pairs")

    # ---- pass 2: MELD's automated screen -----------------------------------
    t1 = time.time()
    verdicts = run([build_judge_messages(r) for r in records],
                   args.judge_max_tokens, 0.0)
    judge_fail = 0
    for rec, text in zip(records, verdicts):
        j = extract_json_robust(text) or {}
        ok = (isinstance(j.get("entry_1_valid"), bool)
              and isinstance(j.get("entry_2_valid"), bool)
              and isinstance(j.get("equivalent"), bool))
        if not ok:
            judge_fail += 1
            j = {}
        rec["screen"] = {
            "entry_1_valid": j.get("entry_1_valid"),
            "entry_2_valid": j.get("entry_2_valid"),
            "equivalent": j.get("equivalent"),
            "could_be_less_similar": j.get("could_be_less_similar"),
            "confidence": j.get("confidence"),
            "reason": j.get("reason"),
            "judge_model": (args.model if args.backend != "stub"
                            else "STUB (never train on this)"),
            "accepted": bool(j.get("entry_1_valid") and j.get("entry_2_valid")
                             and j.get("equivalent")),
        }
    n_acc = sum(r["screen"]["accepted"] for r in records)
    print(f"[gen] pass 2 screened {len(records)} pairs: {n_acc} accepted, "
          f"{judge_fail} unparseable verdicts, in {time.time() - t1:.0f}s",
          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "group": args.group, "backend": args.backend, "model": args.model,
        "n_domain_pairs": len(specs), "rounds": args.rounds,
        "pairs_per_call": args.pairs_per_call, "seed": args.seed,
        "n_pairs_generated": len(records), "n_accepted_by_screen": n_acc,
        "n_dedup_dropped": dup_in_round,
        # ---- per-CALL health (what the smoke gate reads) -------------------
        # A pair count alone cannot separate "every call parsed and each was
        # a bit short" from "half the calls returned garbage"; the smoke gate
        # in attack_second_benchmark.slurm needs the latter distinction, so
        # the generator reports it instead of making the shell guess.
        "n_generation_calls": n_gen_calls,
        "n_generation_json_failures": gen_fail,
        "gen_json_success_rate": round(
            (n_gen_calls - gen_fail) / max(1, n_gen_calls), 4),
        "min_pairs_per_successful_call": min(per_call_pairs, default=0),
        "mean_pairs_per_successful_call": round(
            sum(per_call_pairs) / max(1, len(per_call_pairs)), 2),
        "n_judge_json_failures": judge_fail,
        "pct_could_be_less_similar": round(
            100 * sum(bool(r["screen"]["could_be_less_similar"]) for r in records)
            / max(1, len(records)), 2),
        "unique_topics": len({(r["domain"], norm_text(r["topic"])) for r in records}),
        "runtime_s": round(time.time() - t0, 1),
    }
    (Path(str(out_path) + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[gen] {len(records)} pairs -> {out_path}\n[gen] summary -> "
          f"{out_path}.summary.json", flush=True)


# ---------------------------------------------------------------------------
# STAGE pairs — item-level disjointness gates + trainer rows
# ---------------------------------------------------------------------------
def stage_pairs(args):
    t0 = time.time()
    raw_path = Path(args.raw or (ATTACK_DIR / f"raw_{args.group}.jsonl"))
    if not raw_path.exists():
        sys.exit(f"[FATAL] {raw_path} missing — run --stage generate first")
    records = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    stub = any("STUB" in str(r.get("generator_model", "")) for r in records)
    if stub and not args.allow_stub:
        sys.exit("[FATAL] input contains STUB generations — pass --allow-stub "
                 "only for contract tests, and NEVER train on the result")
    _pairs, _distr, eval_stmts, eval_distr = load_meld_eval_texts()
    eval_texts = eval_stmts + eval_distr
    print(f"[pairs] {len(records)} generated pairs vs {len(eval_texts)} MELD "
          f"eval texts ({len(eval_stmts)} statements + {len(eval_distr)} "
          f"distractors)", flush=True)

    # ---- gates 1-3: lexical ------------------------------------------------
    eval_norm = {norm_text(t) for t in eval_texts}
    eval_wgrams, gram_index = [], defaultdict(list)
    for i, t in enumerate(eval_texts):
        g = word_grams(t, args.ngram)
        eval_wgrams.append(g)
        for h in g:
            gram_index[h].append(i)
    eval_cgrams = [char_grams(t, args.char_ngram) for t in eval_texts]

    def max_containment(g):
        if not g:
            return 0.0, None
        counts = Counter()
        for h in g:
            for i in gram_index.get(h, ()):
                counts[i] += 1
        if not counts:
            return 0.0, None
        best, n = max(counts.items(), key=lambda kv: kv[1])
        return n / len(g), best

    def max_char_jaccard(t):
        cg = char_grams(t, args.char_ngram)
        best, arg = 0.0, None
        for i, eg in enumerate(eval_cgrams):
            j = jaccard(cg, eg)
            if j > best:
                best, arg = j, i
        return best, arg

    kept, dropped = [], []
    cont_hist, cj_hist = [], []
    for rec in records:
        if not rec.get("screen", {}).get("accepted"):
            dropped.append({"pair_id": rec["pair_id"], "reason": "screen_rejected"})
            continue
        # Score BOTH sides before deciding, so the reported distributions are
        # over complete pairs rather than truncated at the first trip.
        reasons = []
        worst_cont, worst_cj = 0.0, 0.0
        for side in ("entry_1", "entry_2"):
            txt = rec[side]["statement"]
            c, ci = max_containment(word_grams(txt, args.ngram))
            cj, cji = max_char_jaccard(txt)
            worst_cont = max(worst_cont, c)
            worst_cj = max(worst_cj, cj)
            if norm_text(txt) in eval_norm:
                reasons.append(f"exact_match({side})")
            elif c >= args.containment_threshold:
                reasons.append(f"ngram_containment({side},{c:.2f},eval{ci})")
            elif cj >= args.char_jaccard_threshold:
                reasons.append(f"char_jaccard({side},{cj:.2f},eval{cji})")
        cont_hist.append(worst_cont)
        cj_hist.append(worst_cj)
        if reasons:
            dropped.append({"pair_id": rec["pair_id"], "reason": "+".join(reasons)})
        else:
            kept.append(rec)
    print(f"[pairs] lexical gates: {len(kept)} kept / {len(records)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- gate 4: semantic, under an INDEPENDENT encoder ---------------------
    sem_stats = {"enabled": False}
    if not args.no_semantic_gate and kept:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        enc = SentenceTransformer(args.semantic_gate_model, device=args.device)
        E = enc.encode(eval_texts, batch_size=args.batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype("float32")
        cand = [t for r in kept for t in (r["entry_1"]["statement"],
                                          r["entry_2"]["statement"])]
        C = enc.encode(cand, batch_size=args.batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype("float32")
        sims = (C @ E.T).max(axis=1)
        keep2, sims_kept = [], []
        for i, r in enumerate(kept):
            s = float(max(sims[2 * i], sims[2 * i + 1]))
            sims_kept.append(s)
            if s >= args.semantic_threshold:
                dropped.append({"pair_id": r["pair_id"],
                                "reason": f"semantic_cosine({s:.3f})"})
            else:
                keep2.append(r)
        sem_stats = {
            "enabled": True, "model": args.semantic_gate_model,
            "threshold": args.semantic_threshold,
            "n_dropped": len(kept) - len(keep2),
            "max_cosine_vs_eval": {
                "mean": round(float(np.mean(sims_kept)), 4),
                "p95": round(float(np.percentile(sims_kept, 95)), 4),
                "p99": round(float(np.percentile(sims_kept, 99)), 4),
                "max": round(float(np.max(sims_kept)), 4)},
        }
        kept = keep2
        print(f"[pairs] semantic gate ({args.semantic_gate_model} @ "
              f"{args.semantic_threshold}): {len(kept)} kept, "
              f"{sem_stats['n_dropped']} dropped", flush=True)

    if not kept:
        sys.exit("[FATAL] every generated pair was gated out")

    # ---- trainer rows -------------------------------------------------------
    # negatives = same-framing statements from OTHER pairs, mirroring MELD's
    # released distractors (true statements in the partner's framing) and the
    # eval's hardest setting (auc_same_framing_distractors).
    rng = random.Random(args.seed)
    by_framing = defaultdict(list)   # framing -> [(pair_id, statement)]
    for r in kept:
        for side in ("entry_1", "entry_2"):
            by_framing[r[side]["framing"]].append((r["pair_id"], r[side]["statement"]))

    out_rows, neg_counts = [], Counter()
    for r in kept:
        directions = (("entry_1", "entry_2"),)
        if args.both_directions:
            directions += (("entry_2", "entry_1"),)
        for a_side, p_side in directions:
            anchor = r[a_side]["statement"]
            pos = r[p_side]["statement"]
            pool = [t for pid, t in by_framing[r[p_side]["framing"]]
                    if pid != r["pair_id"]]
            negs = rng.sample(pool, min(args.negatives_per_row, len(pool)))
            neg_counts[len(negs)] += 1
            out_rows.append({
                "source_id": r["pair_id"],          # BOTH directions share it
                "anchor_text": anchor,
                "positive_text": pos,
                "negatives": [{"text": t} for t in negs],
                "meta": {"group": r["group"], "domain": r["domain"],
                         "topic": r["topic"], "round": r.get("round"),
                         "direction": f"{a_side}->{p_side}",
                         "anchor_framing": r[a_side]["framing"],
                         "positive_framing": r[p_side]["framing"],
                         "generator_model": r["generator_model"]},
            })
    rng.shuffle(out_rows)

    pairs_path = Path(args.out or (ATTACK_DIR / f"pairs_{args.group}.jsonl"))
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- reports ------------------------------------------------------------
    import numpy as np
    by_reason = Counter(d["reason"].split("(")[0] for d in dropped)
    disj = {
        "date": time.strftime("%Y-%m-%d"),
        "group": args.group,
        "target": "uw-math-ai/MELD-dataset (arXiv:2606.23959)",
        "why_item_level": "MELD's recipe has no source corpus (its inputs are a "
                          "domain pair + a prose connection), so 'sources "
                          "disjoint from eval items' can only be enforced "
                          "item-by-item against the released eval texts.",
        "eval_texts_gated_against": {
            "n_pair_statements": len(eval_stmts),
            "n_distractors": len(eval_distr),
            "n_total": len(eval_texts),
        },
        "gates": [
            "normalized-text exact match -> drop",
            f"word-{args.ngram}-gram containment >= "
            f"{args.containment_threshold} (same statistic as "
            f"results/saber_disjointness.json) -> drop",
            f"character-{args.char_ngram}-gram Jaccard >= "
            f"{args.char_jaccard_threshold} -> drop",
            (f"embedding cosine >= {args.semantic_threshold} under "
             f"{args.semantic_gate_model} (independent encoder) -> drop"
             if sem_stats["enabled"] else "semantic gate DISABLED (--no-semantic-gate)"),
            "pairs rejected by MELD's automated screen -> drop (recipe step, "
            "not a disjointness gate; counted separately as screen_rejected)",
        ],
        "gate_direction_note":
            "gates 1-4 remove the training pairs most similar to eval items, "
            "i.e. they can only WEAKEN the attack. A P-M1 success under these "
            "gates is therefore a lower bound on the exposure.",
        "results": {
            "n_generated": len(records),
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "dropped_by_reason": dict(by_reason),
            "ngram_containment_vs_eval": {
                "mean": round(float(np.mean(cont_hist)), 4) if cont_hist else None,
                "p99": round(float(np.percentile(cont_hist, 99)), 4) if cont_hist else None,
                "max": round(float(np.max(cont_hist)), 4) if cont_hist else None},
            "char_jaccard_vs_eval": {
                "mean": round(float(np.mean(cj_hist)), 4) if cj_hist else None,
                "p99": round(float(np.percentile(cj_hist, 99)), 4) if cj_hist else None,
                "max": round(float(np.max(cj_hist)), 4) if cj_hist else None},
            "semantic_gate": sem_stats,
        },
        "dropped_examples": dropped[:25],
        "runtime_s": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    dpath = RESULTS / f"meld_disjointness_{args.group}.json"
    dpath.write_text(json.dumps(disj, indent=2, ensure_ascii=False))

    summary = {
        "group": args.group,
        "n_pairs_kept": len(kept),
        "n_trainer_rows": len(out_rows),
        "both_directions": args.both_directions,
        "negatives_per_row_requested": args.negatives_per_row,
        "negatives_per_row_hist": dict(neg_counts),
        "avg_negatives_per_row": round(
            sum(k * v for k, v in neg_counts.items()) / max(1, len(out_rows)), 3),
        "rows_by_domain": dict(Counter(r["meta"]["domain"] for r in out_rows)),
        "unique_topics": len({(r["meta"]["domain"], norm_text(r["meta"]["topic"]))
                              for r in out_rows}),
        "contains_stub_generations": stub,
        "pairs_file": str(pairs_path),
        "disjointness_report": str(dpath),
        "runtime_s": round(time.time() - t0, 1),
    }
    spath = RESULTS / f"meld_pairs_summary_{args.group}.json"
    spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[pairs] wrote {len(out_rows)} trainer rows from {len(kept)} pairs "
          f"(avg {summary['avg_negatives_per_row']} negatives/row; the "
          f"controlled experiment's LLM arm ran ~2.63) -> {pairs_path}")
    print(f"[pairs] reports -> {dpath} + {spath}", flush=True)


# ---------------------------------------------------------------------------
# STAGE power — MELD's own sampling noise (NO attacked model involved)
# ---------------------------------------------------------------------------
def stage_power(args):
    """Cluster bootstrap of the metrics the pre-registration uses.

    MELD is small: 270 pairs -> 540 pairs_only queries, and the two queries of
    a pair are NOT independent, so the resampling unit is the PAIR. Two
    off-the-shelf encoders (never trained by us, both already measured in
    results/meld_*.json) supply a realistic paired-difference distribution.
    Output: results/meld_attack_power.json, the source of the MDE quoted in
    the slurm pre-registration.
    """
    t0 = time.time()
    import numpy as np
    from sentence_transformers import SentenceTransformer

    pairs, distr, stmts, _d = load_meld_eval_texts()
    distr_texts, distr_framing = [], []
    for fr, lst in distr.items():
        for s in lst:
            distr_texts.append(s)
            distr_framing.append(fr)
    idx_by_framing = defaultdict(list)
    for i, fr in enumerate(distr_framing):
        idx_by_framing[fr].append(i)
    n = len(pairs)

    def model_stats(name):
        m = SentenceTransformer(name, device=args.device)
        S = m.encode(stmts, batch_size=args.batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype("float32")
        D = m.encode(distr_texts, batch_size=args.batch_size,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype("float32")
        sims = S @ S.T
        np.fill_diagonal(sims, -np.inf)
        gold_col = np.arange(2 * n)
        gold_col = gold_col + 1 - 2 * (gold_col % 2)
        gold = sims[np.arange(2 * n), gold_col]
        ranks = (sims > gold[:, None]).sum(axis=1) + 1        # 540 1-based ranks
        # per-query "positive beats every same-framing distractor" indicator
        above = np.zeros(2 * n, dtype=np.float64)
        for i, p in enumerate(pairs):
            for off, target_framing in ((0, p["entry_2"]["framing"]),
                                        (1, p["entry_1"]["framing"])):
                q = S[2 * i + off]
                partner = S[2 * i + (1 - off)]
                ids = idx_by_framing.get(target_framing, [])
                above[2 * i + off] = float(
                    float(q @ partner) > float((D[ids] @ q).max())) if ids else np.nan
        return ranks, above

    ranks, above = {}, {}
    for name in args.power_models:
        ranks[name], above[name] = model_stats(name)
        print(f"[power] {name}: R@1 {100 * np.mean(ranks[name] <= 1):.2f} "
              f"pct_above {100 * np.nanmean(above[name]):.2f}", flush=True)

    rng = np.random.default_rng(args.seed)
    B = args.bootstrap
    a, b = args.power_models[0], args.power_models[1]
    hit_a, hit_b = (ranks[a] <= 1).astype(float), (ranks[b] <= 1).astype(float)
    boot_a, boot_b, boot_d = [], [], []
    for _ in range(B):
        pick = rng.integers(0, n, size=n)                 # resample PAIRS
        q = np.concatenate([2 * pick, 2 * pick + 1])
        boot_a.append(hit_a[q].mean())
        boot_b.append(hit_b[q].mean())
        boot_d.append(hit_a[q].mean() - hit_b[q].mean())
    boot_a, boot_b, boot_d = map(np.asarray, (boot_a, boot_b, boot_d))
    se_single = float(100 * boot_a.std(ddof=1))
    se_diff = float(100 * boot_d.std(ddof=1))
    disc = int(np.sum(hit_a != hit_b))

    # naive iid SE for the same statistic, to expose the clustering penalty
    p_hat = float(hit_a.mean())
    se_iid = float(100 * np.sqrt(p_hat * (1 - p_hat) / (2 * n)))

    # The paired SE is driven by how many queries the two models DISAGREE on,
    # and an attacked model disagrees with base on far more queries than these
    # two off-the-shelf encoders do. McNemar SE ~ 100*sqrt(d)/n_q, inflated by
    # the observed clustering factor; tabulate it so the pre-registered
    # threshold can be read against the discordance actually observed.
    infl = se_diff / (100 * (disc ** 0.5) / (2 * n)) if disc else 1.0
    mde_curve = {}
    for d in (50, 100, 200, 300, 400, 540):
        se_d = infl * 100 * (d ** 0.5) / (2 * n)
        mde_curve[str(d)] = {"se_paired_diff": round(se_d, 2),
                             "mde_80pct": round(2.802 * se_d, 2)}

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "honest power limit for the MELD recipe-matching attack; "
                   "computed from OFF-THE-SHELF encoders only -- no attacked "
                   "model is trained, evaluated or referenced here",
        "benchmark_size": {"n_pairs": n, "n_pairs_only_queries": 2 * n,
                           "n_distractors": len(distr_texts),
                           "resampling_unit": "pair (the two queries of a pair "
                                              "are not independent)"},
        "reference_models": list(args.power_models),
        "observed_recall@1": {m: round(100 * float(np.mean(ranks[m] <= 1)), 2)
                              for m in args.power_models},
        "bootstrap": {"n_resamples": B, "seed": args.seed},
        "se_recall@1_single_model_cluster_bootstrap": round(se_single, 3),
        "se_recall@1_single_model_iid_naive": round(se_iid, 3),
        "se_recall@1_paired_difference": round(se_diff, 3),
        "n_discordant_queries": disc,
        "mde_recall@1_80pct_power_alpha05": round(2.802 * se_diff, 2),
        "mde_recall@1_2se_rule": round(2 * se_diff, 2),
        "clustering_inflation_vs_mcnemar": round(infl, 3),
        "mde_recall@1_by_discordance": mde_curve,
        "discordance_upper_bound_rule":
            "d <= n_queries * (R@1_treatment + R@1_reference) / 100, because "
            "queries both models miss are concordant. At the R@1 levels MELD "
            "actually shows (base 10.19, ctrl-llm 18.89, best published 28.9) "
            "any realistic pair caps out near d ~ 200-300, i.e. an MDE of "
            "7.8-9.6 points -- which is the row the pre-registered threshold "
            "must clear, not the d=540 row (unreachable: it would need the two "
            "models to disagree on every single query).",
        "interpretation":
            "MELD admits ~540 clustered queries. Differences smaller than the "
            "MDE above are NOT detectable on this benchmark at any budget we "
            "can spend, so a null outcome below that threshold is 'no "
            "evidence', never 'evidence of resistance'. The reference SE here "
            "comes from two encoders that disagree on only "
            f"{disc}/{2 * n} queries; an attacked model disagrees with base on "
            "many more, so read mde_recall@1_by_discordance at the discordance "
            "the run actually shows (that exact number comes out of --stage "
            "compare on the per-query dumps). The pre-registered thresholds in "
            "scripts/attack_second_benchmark.slurm are set above the "
            "worst-case row of that table on purpose.",
        "runtime_s": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "meld_attack_power.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("[power] -> results/meld_attack_power.json", flush=True)


# ---------------------------------------------------------------------------
# STAGE compare — paired cluster bootstrap between two eval_meld per-query dumps
# ---------------------------------------------------------------------------
def stage_compare(args):
    """Turn two --per-query-out dumps into the CI the pre-registration needs.

    Resampling unit = the MELD pair (queries 2k and 2k+1 move together), which
    is the only defensible unit on a 270-pair benchmark. Reports the paired
    difference, its bootstrap CI, the McNemar discordance, and the difference
    expressed in SEs so a verdict can be read off without re-deriving anything.
    """
    t0 = time.time()
    import numpy as np
    if len(args.compare) != 2:
        sys.exit("[FATAL] --compare takes exactly two per-query dump paths "
                 "(treatment first, reference second)")
    A, B = (json.load(open(p, encoding="utf-8")) for p in args.compare)
    if A["n_pairs"] != B["n_pairs"]:
        sys.exit(f"[FATAL] dumps disagree on n_pairs: {A['n_pairs']} vs {B['n_pairs']}")
    n = A["n_pairs"]

    metrics = {
        "recall@1_pairs_only":
            lambda d: (np.asarray(d["pairs_only_gold_rank_1based"]) <= 1).astype(float),
        "recall@5_pairs_only":
            lambda d: (np.asarray(d["pairs_only_gold_rank_1based"]) <= 5).astype(float),
        "pct_above_same_framing_distractors":
            lambda d: np.asarray(d["above_all_same_framing_distractors"], dtype=float),
    }
    rng = np.random.default_rng(args.seed)
    out = {}
    for name, fn in metrics.items():
        a, b = fn(A), fn(B)
        boot = np.empty(args.bootstrap)
        for i in range(args.bootstrap):
            pick = rng.integers(0, n, size=n)
            q = np.concatenate([2 * pick, 2 * pick + 1])
            boot[i] = a[q].mean() - b[q].mean()
        diff = 100 * float(a.mean() - b.mean())
        se = 100 * float(boot.std(ddof=1))
        lo, hi = (100 * float(x) for x in np.percentile(boot, [2.5, 97.5]))
        out[name] = {
            "treatment": round(100 * float(a.mean()), 2),
            "reference": round(100 * float(b.mean()), 2),
            "diff": round(diff, 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "se_paired": round(se, 3),
            "n_sig": round(abs(diff) / se, 2) if se > 0 else None,
            "n_discordant_queries": int(np.sum(a != b)),
        }
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "treatment": {"model": A["model"], "prompt": A.get("query_prompt_name"),
                      "dump": args.compare[0]},
        "reference": {"model": B["model"], "prompt": B.get("query_prompt_name"),
                      "dump": args.compare[1]},
        "prompt_conditions_match": A.get("query_prompt_name") == B.get("query_prompt_name")
                                   and A.get("query_prompt") == B.get("query_prompt"),
        "bootstrap": {"n_resamples": args.bootstrap, "seed": args.seed,
                      "resampling_unit": "MELD pair (270 clusters, 2 queries each)"},
        "metrics": out,
        "runtime_s": round(time.time() - t0, 1),
    }
    if not report["prompt_conditions_match"]:
        report["WARNING"] = ("prompt conditions differ between the two dumps — "
                             "this is the exact mismatch flagged as review "
                             "finding M2 for the SABER P-S1 row; quote a "
                             "same-condition pair instead")
    path = Path(args.out or (RESULTS / "meld_attack_compare.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[compare] -> {path}", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["domains", "generate", "pairs", "power", "compare"])
    ap.add_argument("--compare", nargs="+", default=[],
                    help="stage compare: two eval_meld.py --per-query-out "
                         "dumps, TREATMENT first then REFERENCE")
    ap.add_argument("--group", choices=sorted(GROUPS), default="meld9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="override the stage's output path")
    # generate
    ap.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--rounds", type=int, default=8,
                    help="independent generation calls per domain pair "
                         "(MELD ran exactly 1; >1 is our scale deviation)")
    ap.add_argument("--pairs-per-call", type=int, default=30,
                    help="MELD's own per-call count (paper p.4)")
    ap.add_argument("--target-pairs", type=int, default=0,
                    help="stop early once this many unique pairs exist (0 = all rounds)")
    ap.add_argument("--max-domain-pairs", type=int, default=0, help="cap for smoke runs")
    ap.add_argument("--max-covered-shown", type=int, default=200,
                    help="topics listed back in the round>=2 continuation prompt")
    ap.add_argument("--first-round-greedy", action="store_true",
                    help="temperature 0 for round 1 (reproducible seed round)")
    ap.add_argument("--gen-temperature", type=float, default=1.0)
    # Headroom, not the fix.  Measured from job 49630276 (2 calls, 30 pairs
    # each, prose-only statements): the two completions were ~3.1k and ~4.4k
    # tokens (vllm throughput x elapsed), i.e. ~110-150 tokens/pair, so 8192
    # was never reached and truncation was NOT that job's failure.  12288
    # keeps ~2.8x margin for the LaTeX-heavy domains (probability/measure
    # theory, representation theory) and still fits --max-model-len 16384
    # with the ~350-800 token prompt.
    ap.add_argument("--gen-max-tokens", type=int, default=12288)
    ap.add_argument("--judge-max-tokens", type=int, default=384)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--min-statement-chars", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact prompts and exit; no model, no GPU")
    # pairs
    ap.add_argument("--raw", default=None, help="input raw_<group>.jsonl override")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--containment-threshold", type=float, default=0.35)
    ap.add_argument("--char-ngram", type=int, default=8)
    ap.add_argument("--char-jaccard-threshold", type=float, default=0.60)
    ap.add_argument("--semantic-gate-model",
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--semantic-threshold", type=float, default=0.90)
    ap.add_argument("--no-semantic-gate", action="store_true")
    ap.add_argument("--negatives-per-row", type=int, default=3,
                    help="matches the controlled experiment's LLM arm (~2.63/row)")
    ap.add_argument("--both-directions", dest="both_directions",
                    action="store_true", default=True)
    ap.add_argument("--one-direction", dest="both_directions", action="store_false")
    ap.add_argument("--allow-stub", action="store_true",
                    help="permit STUB generations through --stage pairs "
                         "(contract tests only; never train on the result)")
    # power / shared encoder settings
    ap.add_argument("--power-models", nargs="+",
                    default=["sentence-transformers/all-MiniLM-L6-v2",
                             "sentence-transformers/all-mpnet-base-v2"])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    {"domains": stage_domains, "generate": stage_generate,
     "pairs": stage_pairs, "power": stage_power,
     "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()
