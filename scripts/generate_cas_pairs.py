#!/usr/bin/env python3
r"""
CAS-verified contrastive pair generator for InvarEmbed (Task A core).

For every ELIGIBLE problem in the MathNet corpus (>= 1 census-relational math
span, id NOT in anchor_to_corpus_mapping.json exclude_corpus_ids) generate:

VERIFIED POSITIVES
  (a) rename      -- consistent variable renaming across ALL math spans
                     (word-boundary regex inside math only, collision-free
                     targets, \text{...} blocks protected). Verified by
                     re-parsing the renamed relational span and structurally
                     comparing after back-substitution.
  (b) reformulate -- one algebraic reformulation of a relational span,
                     verified equivalent by the residual-identity test:
                     residual r = lhs - rhs; the transform is built so
                     r_new == k * r_old for an exactly known constant k
                     (move_term/expand/factor: k=1; x2 scale: k=2; /2 scale:
                     k=1/2; k>0 preserves inequality direction). Verification
                     = |r_new - k*r_old| ~ 0 at >= 20 random complex points
                     (complex-safe) + sympy.simplify(r_new - k*r_old) == 0 as
                     a secondary check.
  (c) mirror      -- inequality mirror a<b -> b>a (class flip + side swap),
                     verified with the same residual test at k = -1.

VERIFIED HARD NEGATIVES (minimal edits, verified NON-equivalent)
  neg_exponent    -- integer exponent e -> e+1
  neg_constant    -- integer literal c -> c+1 (all occurrences of the literal)
  neg_opswap      -- one '+' term sign flipped (a+b -> a-b)
  neg_flip        -- inequality direction flipped WITHOUT swapping sides
  Every negative carries a machine-checkable numeric counterexample:
    inequalities: a sampled real point where the two relations have opposite
                  truth values;
    equalities:   a point ON the solution set of one relation (obtained by
                  substituting random rationals and solving for the remaining
                  symbol) where the other relation's residual is bounded away
                  from zero.
  Negatives with no verified counterexample are DISCARDED, never emitted.

Transformed expressions are embedded back into the problem markdown (sympy
latex printing for (b)/(c)/negatives -- note positives and negatives share
that printing style, so style is not a pos/neg cue). A PRINT-FIDELITY GUARD
protects every embedding: latex(rel) is re-parsed and must reproduce rel
exactly (srepr equality) before it may replace the span -- sympy's printer
is silently lossy for some constructs (e.g. log(x, base) prints as \log(x)),
and embedding such output would change the math beyond the verified edit.

Output: data/cas_pairs/pairs.jsonl, one record per verified positive:
  {source_id, domain, language, transform_family, positive_transforms:[...],
   positive_text, span_index, span_original, span_transformed,
   negatives:[{text, edit, span_transformed, counterexample}],
   verification:{method, n_samples, k, max_dev, simplify_zero}}
Stats: results/cas_pairs_stats.json.

Usage:
  python3 scripts/generate_cas_pairs.py --target 2200 --workers 8
  python3 scripts/generate_cas_pairs.py --target 0            # ALL eligible (SLURM)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import sympy
from sympy import Add, Integer, Pow, Rational, simplify, srepr
from sympy.core.function import AppliedUndef
from sympy.core.relational import (Equality, GreaterThan, LessThan,
                                   Relational, StrictGreaterThan,
                                   StrictLessThan, Unequality)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_latex import normalize, parse_single  # noqa: E402

ROOT = Path(os.environ.get("PROJECT", "/ibex/user/habiam0b/MathNet_Follow_Up"))
CORPUS = ROOT / "data" / "mathnet_corpus.parquet"

# identical to the census extractor so census statuses match span-for-span
MATH_SPAN = re.compile(
    r"\$\$(.+?)\$\$"
    r"|\\\[(.+?)\\\]"
    r"|\\\((.+?)\\\)"
    r"|(?<!\$)\$([^$]+?)\$(?!\$)",
    re.DOTALL,
)

SOFT_BUDGET_S = 15.0          # worker soft time budget (hard pebble kill later)
MAX_TRIED_SPANS = 8           # candidate relational spans parsed per problem
N_SAMPLES = 20                # verified samples required per positive
MAX_SREPR = 4000              # skip monster expressions

MIRROR = {StrictLessThan: StrictGreaterThan, LessThan: GreaterThan,
          StrictGreaterThan: StrictLessThan, GreaterThan: LessThan}
FLIP = {StrictLessThan: StrictGreaterThan, LessThan: GreaterThan,
        StrictGreaterThan: StrictLessThan, GreaterThan: LessThan}
INEQ_TYPES = tuple(MIRROR.keys())


def extract_spans(markdown):
    """[(inner_start, inner_end, stripped_text)] for every math span."""
    out = []
    for m in MATH_SPAN.finditer(markdown or ""):
        gi = next(i for i in range(1, 5) if m.group(i) is not None)
        s = m.group(gi).strip()
        if s:
            out.append((m.start(gi), m.end(gi), s))
    return out


# --------------------------------------------------------------------------
# numeric machinery
# --------------------------------------------------------------------------

def _num_eval(expr, point):
    """complex value of expr at point, or None if not a finite number."""
    try:
        v = complex(expr.evalf(subs=point))
    except Exception:
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    if abs(v) > 1e10:
        return None
    return v


def _sample_point(syms, rng, real=False):
    pt = {}
    for s in syms:
        if real:
            v = 0.0
            while abs(v) < 0.05:
                v = rng.uniform(-3.5, 3.5)
            pt[s] = sympy.Float(v)
        else:
            re_, im_ = 0.0, 0.0
            while (re_ * re_ + im_ * im_) < 0.04:
                re_, im_ = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
            pt[s] = sympy.Float(re_) + sympy.Float(im_) * sympy.I
    return pt


def _has_nonnumeric(rel):
    return bool(rel.atoms(AppliedUndef, sympy.Integral, sympy.Sum,
                          sympy.Product, sympy.Derivative, sympy.Limit))


def canonize(e):
    """Recursive rebuild with default evaluation.

    parse_latex returns unevaluated nested trees (Mul(Mul(A,C),Mul(B,C)));
    rebuilding flattens/canonicalizes them so printing is clean and srepr
    comparison is meaningful. Constructor auto-evaluation only (no doit()),
    so Integrals/Sums are not computed.
    """
    if e.is_Atom:
        return e
    args = [canonize(a) for a in e.args]
    try:
        return e.func(*args)
    except Exception:
        return e


def print_roundtrip(rel):
    """latex(rel) if re-parsing it reproduces rel EXACTLY, else None.

    sympy's LaTeX printer is silently lossy for some constructs the parser
    handles fine (e.g. log(x, base) prints as plain \\log(x), dropping the
    base). Any transformed relation whose printed form does not parse back
    to the identical expression must NOT be embedded into problem text --
    the embedded math would differ from the verified relation.
    `rel` must already be canonical (see canonize); the re-parsed tree is
    canonized before comparison.
    """
    s = sympy.latex(rel)
    st, rel2 = parse_single(s)
    if st != "relational" or not isinstance(rel2, Relational):
        return None
    rel2c = canonize(rel2)
    if not isinstance(rel2c, Relational) or srepr(rel2c) != srepr(rel):
        return None
    return s


def verify_residual(rel_old, rel_new, k, rng, n=N_SAMPLES):
    """Verify residual identity r_new == k * r_old at n random points.

    Returns dict(verification metadata) on success, None on failure.
    k must be the exactly-known constant of the transform construction.
    """
    r0 = rel_old.lhs - rel_old.rhs
    r1 = rel_new.lhs - rel_new.rhs
    syms = sorted(r0.free_symbols | r1.free_symbols, key=str)
    kc = complex(k)
    if not syms:
        return None
    got, nontrivial, max_dev = 0, 0, 0.0
    for _ in range(4 * n):
        if got >= n:
            break
        pt = _sample_point(syms, rng)
        v0, v1 = _num_eval(r0, pt), _num_eval(r1, pt)
        if v0 is None or v1 is None:
            continue
        dev = abs(v1 - kc * v0)
        if dev > 1e-6 * (1.0 + abs(v0) + abs(v1)):
            return None
        got += 1
        if abs(v0) > 1e-6:
            nontrivial += 1
        max_dev = max(max_dev, dev)
    if got < n or nontrivial < max(3, n // 4):
        return None
    simplify_zero = None
    diff = r1 - sympy.sympify(k) * r0
    if len(srepr(diff)) < 1500:
        try:
            simplify_zero = bool(simplify(diff) == 0)
        except Exception:
            simplify_zero = None
    if simplify_zero is False:
        # numeric said identical at 20 complex points but simplify found a
        # nonzero canonical form -> distrust the transform, drop it
        return None
    return {"method": "residual_ratio_k+simplify", "n_samples": got,
            "k": str(k), "max_dev": float(f"{max_dev:.3g}"),
            "simplify_zero": simplify_zero}


def _truth_at(rel, point):
    """Truth of an inequality at a real point, None near boundary/invalid."""
    v = _num_eval(rel.lhs - rel.rhs, point)
    if v is None or abs(v.imag) > 1e-9 or abs(v.real) < 1e-9:
        return None
    neg = v.real < 0
    return neg if rel.rel_op in ("<", "<=") else not neg


# --------------------------------------------------------------------------
# positive transforms
# --------------------------------------------------------------------------

def t_move_term(rel, rng):
    lhs, rhs = rel.lhs, rel.rhs
    for src_is_rhs in (True, False):
        side = rhs if src_is_rhs else lhs
        if isinstance(side, Add) and len(side.args) >= 2:
            t = rng.choice(sorted(side.args, key=str))
            new = rel.func(lhs - t, rhs - t)
            if isinstance(new, Relational) and new != rel:
                return new, "move_term", 1
    if not rhs.is_zero:
        new = rel.func(lhs - rhs, sympy.Integer(0))
        if isinstance(new, Relational) and new != rel:
            return new, "move_all_terms", 1
    return None


def t_expand_factor(rel, rng):
    for op, name in ((sympy.expand, "expand"), (sympy.factor, "factor")):
        for attr in ("lhs", "rhs"):
            side = getattr(rel, attr)
            if len(srepr(side)) > 1200:
                continue
            try:
                side2 = op(side)
            except Exception:
                continue
            if srepr(side2) == srepr(side):
                continue
            new = (rel.func(side2, rel.rhs) if attr == "lhs"
                   else rel.func(rel.lhs, side2))
            if isinstance(new, Relational) and srepr(new) != srepr(rel):
                return new, f"{name}_{attr}", 1
    return None


def t_scale(rel, rng):
    if rng.random() < 0.5:
        new = rel.func(2 * rel.lhs, 2 * rel.rhs)
        name, k = "scale_x2", 2
    else:
        new = rel.func(rel.lhs / 2, rel.rhs / 2)
        name, k = "scale_div2", Rational(1, 2)
    if isinstance(new, Relational) and srepr(new) != srepr(rel):
        return new, name, k
    return None


def t_mirror(rel, rng):
    cls = MIRROR.get(type(rel))
    if cls is None:
        return None
    new = cls(rel.rhs, rel.lhs)
    if isinstance(new, Relational):
        return new, "mirror", -1
    return None


# --------------------------------------------------------------------------
# negative edits
# --------------------------------------------------------------------------

def e_exponent(rel, rng):
    pows = [p for p in rel.atoms(Pow)
            if p.exp.is_Integer and 2 <= abs(int(p.exp)) <= 6]
    if not pows:
        return None
    p = rng.choice(sorted(pows, key=str))
    e = int(p.exp)
    new = rel.xreplace({p: Pow(p.base, Integer(e + 1))})
    if isinstance(new, Relational) and srepr(new) != srepr(rel):
        return new, f"exponent:{e}->{e + 1} on {sympy.latex(p.base)[:40]}"
    return None


def e_constant(rel, rng):
    ints = [c for c in rel.atoms(Integer) if 1 <= abs(int(c)) <= 10 ** 6]
    if not ints:
        return None
    c = rng.choice(sorted(ints, key=lambda x: str(x)))
    nc = int(c) + 1 if int(c) + 1 != 0 else int(c) + 2
    new = rel.xreplace({c: Integer(nc)})
    if isinstance(new, Relational) and srepr(new) != srepr(rel):
        return new, f"constant:{int(c)}->{nc} (all occurrences)"
    return None


def e_opswap(rel, rng):
    adds = [a for a in rel.atoms(Add) if len(a.args) >= 2
            and len(srepr(a)) < 1200]
    if not adds:
        return None
    a = rng.choice(sorted(adds, key=str))
    t = rng.choice(sorted(a.args, key=str))
    new = rel.xreplace({a: a - 2 * t})
    if isinstance(new, Relational) and srepr(new) != srepr(rel):
        return new, f"opswap:sign of term {sympy.latex(t)[:40]} flipped"
    return None


def e_flip(rel, rng):
    cls = FLIP.get(type(rel))
    if cls is None:
        return None
    new = cls(rel.lhs, rel.rhs)
    if isinstance(new, Relational) and srepr(new) != srepr(rel):
        return new, f"ineq_flip_no_swap:{rel.rel_op}->{new.rel_op}"
    return None


def _fmt_num(v):
    c = complex(v)
    if abs(c.imag) < 1e-12:
        return f"{c.real:.6g}"
    return f"{c.real:.6g}{c.imag:+.6g}j"


def find_counterexample(rel_orig, rel_bad, rng, deadline):
    """Point where the two relations disagree; None if not found (discard)."""
    if isinstance(rel_orig, INEQ_TYPES):
        syms = sorted((rel_orig.lhs - rel_orig.rhs).free_symbols
                      | (rel_bad.lhs - rel_bad.rhs).free_symbols, key=str)
        if not syms:
            return None
        for _ in range(300):
            if time.time() > deadline:
                return None
            pt = _sample_point(syms, rng, real=True)
            t0, t1 = _truth_at(rel_orig, pt), _truth_at(rel_bad, pt)
            if t0 is None or t1 is None or t0 == t1:
                continue
            return {"point": {str(s): _fmt_num(v) for s, v in pt.items()},
                    "orig_truth": t0, "edited_truth": t1,
                    "method": "real_sample_truth_disagreement"}
        return None
    if not isinstance(rel_orig, Equality):
        return None  # Ne and friends: no cheap sound certificate, skip
    r0 = rel_orig.lhs - rel_orig.rhs
    r1 = rel_bad.lhs - rel_bad.rhs
    all_syms = sorted(r0.free_symbols | r1.free_symbols, key=str)
    if not all_syms:
        return None
    for base, other, tag in ((r0, r1, "orig_holds_edited_fails"),
                             (r1, r0, "edited_holds_orig_fails")):
        bsyms = sorted(base.free_symbols, key=str)
        if not bsyms:
            continue
        for attempt in range(6):
            if time.time() > deadline:
                return None
            solvevar = bsyms[attempt % len(bsyms)]
            subs = {s: Rational(rng.randint(-5, 5), rng.randint(1, 3))
                    for s in all_syms if s != solvevar}
            try:
                uni = base.subs(subs)
                if solvevar not in uni.free_symbols:
                    continue
                sols = sympy.solve(sympy.Eq(uni, 0), solvevar)
            except Exception:
                continue
            for sol in sols[:4]:
                if sol.free_symbols:
                    continue
                point = dict(subs)
                point[solvevar] = sol
                vb, vo = _num_eval(base, point), _num_eval(other, point)
                if vb is None or vo is None:
                    continue
                if abs(vb) < 1e-8 and abs(vo) > 1e-4:
                    return {"point": {str(s): _fmt_num(v)
                                      for s, v in point.items()},
                            "satisfies": tag,
                            "other_residual": float(f"{abs(vo):.4g}"),
                            "method": "solve_one_side_check_other"}
    return None


# --------------------------------------------------------------------------
# renaming
# --------------------------------------------------------------------------

_TEXT_BLOCK = re.compile(r"\\(?:text|textrm|mbox|textit|textbf|mathrm)\s*\{[^{}]*\}")
_LOWER_POOL = "wuvtszrqpmnkjhgdcb"
_UPPER_POOL = "WUVTSZRQPMNKJHGDCB"
_SKIP_VARS = {"a", "A", "I", "i", "e", "o", "O", "l"}


def _standalone(letter):
    return re.compile(r"(?<![A-Za-z\\])" + re.escape(letter) + r"(?![A-Za-z])")


def _sub_protected(span, pattern, repl):
    """Regex-substitute inside a math span with \text{...} blocks protected."""
    blocks = []

    def _mask(m):
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    masked = _TEXT_BLOCK.sub(_mask, span)
    replaced = pattern.sub(repl, masked)
    for i, b in enumerate(blocks):
        replaced = replaced.replace(f"\x00{i}\x00", b)
    return replaced


def try_rename(markdown, spans, chosen_idx, rel, rng):
    """Consistent variable renaming across all math spans.

    Returns (new_markdown, transforms, renamed_chosen_span) or None.
    """
    chosen_span = spans[chosen_idx][2]
    prose = MATH_SPAN.sub(" ", markdown)
    all_math = " \n ".join(s for _, _, s in spans)
    # candidate source vars: single-letter free symbols of the chosen relation
    # (base letter of subscripted symbols counts), present in the chosen span
    cands = []
    for s in sorted(rel.free_symbols, key=str):
        base = s.name.split("_")[0]
        if len(base) == 1 and base.isalpha() and base not in _SKIP_VARS:
            if base not in cands:
                cands.append(base)
    ok_vars = []
    for v in cands:
        if not _standalone(v).search(chosen_span):
            continue
        if _standalone(v).search(prose):
            continue  # referenced in prose outside math: renaming would break text
        in_text_block = any(_standalone(v).search(m.group(0))
                            for _, _, sp in spans
                            for m in _TEXT_BLOCK.finditer(sp))
        if in_text_block:
            continue
        ok_vars.append(v)
    if not ok_vars:
        return None
    rng.shuffle(ok_vars)
    ok_vars = ok_vars[:2]
    used_targets = set()
    mapping = {}
    for v in ok_vars:
        pool = _UPPER_POOL if v.isupper() else _LOWER_POOL
        tgt = None
        for w in pool:
            if w == v or w in used_targets:
                continue
            if _standalone(w).search(all_math) or _standalone(w).search(prose):
                continue
            tgt = w
            break
        if tgt:
            mapping[v] = tgt
            used_targets.add(tgt)
    if not mapping:
        return None
    # apply right-to-left over span offsets
    new_md = markdown
    for start, end, span_text in sorted(spans, reverse=True):
        new_span = span_text
        for v, w in mapping.items():
            new_span = _sub_protected(new_span, _standalone(v), w)
        new_md = new_md[:start] + new_span + new_md[end:]
    # verification: re-parse renamed chosen span, map symbols back, compare
    renamed_chosen = spans[chosen_idx][2]
    for v, w in mapping.items():
        renamed_chosen = _sub_protected(renamed_chosen, _standalone(v), w)
    st, rel2 = parse_single(renamed_chosen)
    if st != "relational" or not isinstance(rel2, Relational):
        return None
    back = {}
    for s in rel2.free_symbols:
        name = s.name
        for v, w in mapping.items():
            name = _standalone(w).sub(v, name)
        if name != s.name:
            back[s] = sympy.Symbol(name)
    # canonize AFTER back-substitution (xreplace does not re-sort args)
    rel_back = canonize(rel2.xreplace(back))
    if srepr(rel_back) != srepr(rel):
        return None
    return new_md, [f"rename:{v}->{w}" for v, w in sorted(mapping.items())], \
        renamed_chosen


# --------------------------------------------------------------------------
# per-problem worker
# --------------------------------------------------------------------------

def _embed(markdown, spans, idx, new_inner):
    start, end, _ = spans[idx]
    return markdown[:start] + new_inner + markdown[end:]


def process_problem(pid, markdown, domain, language, census_rel_flags,
                    soft_budget=SOFT_BUDGET_S, max_tried_spans=MAX_TRIED_SPANS):
    """Generate all verified positives + negatives for one problem.

    census_rel_flags: per-span bool, True if the census parsed it relational.
    soft_budget / max_tried_spans: the worker time budget (s) and the number
    of candidate spans parsed before giving up; the defaults are the values
    every committed pair file was generated with (the relaxed-budget reruns
    raise them via --soft-budget / --max-tried-spans).
    Returns (records, stat_counter_dict).
    """
    t_start = time.time()
    deadline = t_start + soft_budget
    rng = random.Random(int(hashlib.md5(pid.encode()).hexdigest()[:12], 16))
    stats = Counter()
    spans = extract_spans(markdown)
    if not spans:
        return [], {"no_spans": 1}

    # ---- choose the relational span --------------------------------------
    order = [i for i in range(len(spans))
             if i < len(census_rel_flags) and census_rel_flags[i]]
    order += [i for i in range(len(spans)) if i not in order]
    best = None  # (score, idx, rel)
    n_tried = 0
    for i in order:
        if n_tried >= max_tried_spans or time.time() > deadline:
            break
        text = spans[i][2]
        if not (3 <= len(text) <= 350):
            continue
        n_tried += 1
        st, rel = parse_single(text)
        if st != "relational" or not isinstance(rel, Relational):
            continue
        rel = canonize(rel)
        if not isinstance(rel, Relational) or isinstance(rel, Unequality):
            continue  # canonization may collapse trivial relations to bool
        nsym = len(rel.free_symbols)
        if not (1 <= nsym <= 6) or len(srepr(rel)) > MAX_SREPR:
            continue
        score = (2.0 * isinstance(rel, INEQ_TYPES)
                 + min(len(srepr(rel)) / 150.0, 3.0)
                 + 1.0 * bool(rel.atoms(Add) or rel.atoms(Pow))
                 - 2.0 * _has_nonnumeric(rel))
        if best is None or score > best[0]:
            best = (score, i, rel)
    if best is None:
        return [], {"no_relational_span": 1}
    _, idx, rel = best
    span_orig = spans[idx][2]
    numeric_ok = not _has_nonnumeric(rel)

    # ---- negatives (shared by all positive records) ----------------------
    negatives = []
    if numeric_ok:
        for editor, ename in ((e_exponent, "neg_exponent"),
                              (e_constant, "neg_constant"),
                              (e_opswap, "neg_opswap"),
                              (e_flip, "neg_flip")):
            if time.time() > deadline:
                break
            out = editor(rel, rng)
            if out is None:
                continue
            rel_bad, edit_desc = out
            stats[f"{ename}_attempted"] += 1
            bad_latex = print_roundtrip(rel_bad)
            if bad_latex is None:
                continue  # printer would change the math: never embed
            cex = find_counterexample(rel, rel_bad, rng,
                                      min(deadline, time.time() + 4.0))
            if cex is None:
                continue
            negatives.append({
                "text": _embed(markdown, spans, idx, bad_latex),
                "edit": edit_desc,
                "span_transformed": bad_latex,
                "counterexample": cex,
            })
            stats[f"{ename}_verified"] += 1

    # ---- positives -------------------------------------------------------
    records = []

    def add_record(family, transforms, pos_text, span_new, verification):
        records.append({
            "source_id": pid, "domain": domain, "language": language,
            "transform_family": family,
            "positive_transforms": transforms,
            "positive_text": pos_text,
            "span_index": idx,
            "span_original": span_orig,
            "span_transformed": span_new,
            "negatives": negatives,
            "verification": verification,
        })

    # (a) rename
    stats["rename_attempted"] += 1
    ren = try_rename(markdown, spans, idx, rel, rng)
    if ren is not None:
        new_md, transforms, renamed_span = ren
        add_record("rename", transforms, new_md, renamed_span,
                   {"method": "parse_back_structural_equality",
                    "n_samples": 0, "k": None, "max_dev": None,
                    "simplify_zero": None})
        stats["rename_verified"] += 1

    # (b) algebraic reformulation
    if numeric_ok and time.time() < deadline:
        for tf in (t_move_term, t_expand_factor, t_scale):
            if time.time() > deadline:
                break
            out = tf(rel, rng)
            if out is None:
                continue
            rel_new, name, k = out
            stats[f"{name}_attempted"] += 1
            new_latex = print_roundtrip(rel_new)
            if new_latex is None:
                continue
            ver = verify_residual(rel, rel_new, k, rng)
            if ver is None:
                continue
            add_record("reformulate", [name],
                       _embed(markdown, spans, idx, new_latex),
                       new_latex, ver)
            stats[f"{name}_verified"] += 1
            break  # one reformulation per problem

    # (c) inequality mirror
    if numeric_ok and isinstance(rel, INEQ_TYPES) and time.time() < deadline:
        out = t_mirror(rel, rng)
        if out is not None:
            rel_new, name, k = out
            stats["mirror_attempted"] += 1
            new_latex = print_roundtrip(rel_new)
            if new_latex is not None:
                ver = verify_residual(rel, rel_new, k, rng)
                if ver is not None:
                    add_record("mirror", [name],
                               _embed(markdown, spans, idx, new_latex),
                               new_latex, ver)
                    stats["mirror_verified"] += 1

    stats["problems_done"] = 1
    if records:
        stats["problems_with_positive"] = 1
    if negatives:
        stats["problems_with_negative"] = 1
    stats["n_records"] = len(records)
    stats["n_negatives_unique"] = len(negatives)
    stats["worker_seconds_x100"] = int(100 * (time.time() - t_start))
    return records, dict(stats)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def primary_domain(topics):
    if topics is None:
        return "(untagged)"
    tl = list(topics)
    if not tl:
        return "(untagged)"
    return tl[0].split(">")[0].strip() or "(untagged)"


def main():
    import duckdb
    from pebble import ProcessPool

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2200,
                    help="problems to attempt (0 = ALL eligible)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="hard per-problem timeout (s)")
    ap.add_argument("--out", default=str(ROOT / "data/cas_pairs/pairs.jsonl"))
    ap.add_argument("--stats", default=str(ROOT / "results/cas_pairs_stats.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-ids-file", default=None,
                    help="restrict the eligible problems to these ids (one per "
                         "line); the census-relational test and the anchor "
                         "exclusion still apply (relaxed-budget reruns)")
    ap.add_argument("--soft-budget", type=float, default=SOFT_BUDGET_S,
                    help=f"worker soft time budget per problem, s "
                         f"(default {SOFT_BUDGET_S}, the committed runs)")
    ap.add_argument("--max-tried-spans", type=int, default=MAX_TRIED_SPANS,
                    help=f"candidate relational spans parsed per problem "
                         f"(default {MAX_TRIED_SPANS}, the committed runs)")
    args = ap.parse_args()
    only = None
    if args.source_ids_file:
        only = {l.strip() for l in open(args.source_ids_file) if l.strip()}
        print(f"restricting to {len(only)} ids from {args.source_ids_file}")

    t0 = time.time()
    excl = set(json.load(open(ROOT / "anchor_to_corpus_mapping.json"))
               ["exclude_corpus_ids"])
    expr_status = json.load(open(ROOT / "results/cas_expr_results.json"))

    con = duckdb.connect()
    df = con.execute(
        f"SELECT id, problem_markdown, topics_flat, language FROM '{CORPUS}'"
    ).fetchdf()
    print(f"corpus {len(df)} problems, {len(excl)} anchor-matched ids excluded")

    # eligibility: id not excluded AND >=1 census-relational span
    eligible = []          # (pid, markdown, domain, language, rel_flags)
    n_excluded_eligible = 0
    for pid, md, topics, lang in zip(df["id"], df["problem_markdown"],
                                     df["topics_flat"], df["language"]):
        spans = extract_spans(md)
        if not spans:
            continue
        flags = [expr_status.get(s, {}).get("status") == "relational"
                 for _, _, s in spans]
        if not any(flags):
            continue
        if pid in excl:
            n_excluded_eligible += 1
            continue
        if only is not None and pid not in only:
            continue
        lang = None if (isinstance(lang, float) and math.isnan(lang)) else lang
        eligible.append((pid, md, primary_domain(topics), lang, flags))
    dom_counts = Counter(e[2] for e in eligible)
    print(f"eligible after exclusion: {len(eligible)} "
          f"(census-relational but anchor-excluded: {n_excluded_eligible})")
    print(f"per-domain eligible: {dict(dom_counts.most_common())}")

    # stratified sample by primary domain
    rng = random.Random(args.seed)
    if args.target and args.target < len(eligible):
        by_dom = defaultdict(list)
        for e in eligible:
            by_dom[e[2]].append(e)
        total = len(eligible)
        chosen = []
        for d, items in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
            n_d = max(1, round(args.target * len(items) / total))
            rng.shuffle(items)
            chosen.extend(items[:n_d])
        rng.shuffle(chosen)
        chosen = chosen[:args.target]
    else:
        chosen = list(eligible)
    print(f"attempting {len(chosen)} problems "
          f"({dict(Counter(e[2] for e in chosen).most_common())})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.tmp")

    agg = Counter()
    per_domain = defaultdict(Counter)
    per_transform = defaultdict(Counter)
    n_timeout = n_crash = 0
    n_written = 0
    t_sched = time.time()
    with open(tmp_path, "w") as fout, \
            ProcessPool(max_workers=args.workers) as pool:
        futs = [(e, pool.schedule(process_problem,
                                  args=(e[0], e[1], e[2], e[3], e[4],
                                        args.soft_budget, args.max_tried_spans),
                                  timeout=args.timeout)) for e in chosen]
        for i, (e, fut) in enumerate(futs):
            pid, dom = e[0], e[2]
            try:
                records, st = fut.result()
            except TimeoutError:
                n_timeout += 1
                per_domain[dom]["timeout"] += 1
                continue
            except Exception:
                n_crash += 1
                continue
            for k, v in st.items():
                agg[k] += v
                if k.endswith("_attempted") or k.endswith("_verified"):
                    base = k.rsplit("_", 1)
                    per_transform[base[0]][base[1]] += v
            per_domain[dom]["attempted"] += 1
            per_domain[dom]["with_positive"] += int(bool(records))
            per_domain[dom]["records"] += len(records)
            for r in records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_written += 1
            if (i + 1) % 250 == 0:
                el = time.time() - t_sched
                print(f"  {i + 1}/{len(futs)}  {n_written} records  "
                      f"{el:.0f}s  ({(i + 1) / el:.1f} prob/s)")
    tmp_path.rename(out_path)

    wall = time.time() - t0
    done = agg["problems_done"] + n_timeout + n_crash
    stats = {
        "meta": {
            "generated_by": "scripts/generate_cas_pairs.py",
            "target": args.target, "workers": args.workers,
            "per_problem_timeout_s": args.timeout, "seed": args.seed,
            "soft_budget_s": args.soft_budget,
            "max_tried_spans": args.max_tried_spans,
            "source_ids_file": args.source_ids_file,
            "n_source_ids_listed": (len(only) if only is not None else None),
            "wall_s": round(wall, 1),
            "throughput_problems_per_s": round(done / max(1e-9, wall), 2),
            "out": str(out_path),
        },
        "eligibility": {
            "n_corpus": len(df),
            "n_eligible_census_relational_not_excluded": len(eligible),
            "n_census_relational_but_anchor_excluded": n_excluded_eligible,
            "n_attempted": len(chosen),
            "per_domain_eligible": dict(dom_counts.most_common()),
        },
        "outcomes": {
            "problems_processed": agg["problems_done"],
            "problems_timeout": n_timeout,
            "problems_crashed": n_crash,
            "problems_no_relational_span": agg["no_relational_span"],
            "problems_with_any_positive": agg["problems_with_positive"],
            "problems_with_any_negative": agg["problems_with_negative"],
            "positive_records_written": n_written,
            "unique_negatives_generated": agg["n_negatives_unique"],
            "mean_worker_s": round(agg["worker_seconds_x100"] /
                                   max(1, agg["problems_done"]) / 100, 2),
        },
        "per_transform": {
            k: {"attempted": v.get("attempted", 0),
                "verified": v.get("verified", 0),
                "success_rate": round(v.get("verified", 0) /
                                      max(1, v.get("attempted", 0)), 3)}
            for k, v in sorted(per_transform.items())
        },
        "per_domain": {d: dict(c) for d, c in
                       sorted(per_domain.items(),
                              key=lambda kv: -kv[1].get("attempted", 0))},
        "verification": {
            "positives": "residual identity r_new == k*r_old at >=20 random "
                         "complex points with exactly-known k, plus "
                         "simplify(r_new - k*r_old) == 0 secondary check; "
                         "renames verified by parse-back structural equality",
            "negatives": "truth-value counterexample required (real-point "
                         "disagreement for inequalities; on-solution-set "
                         "residual violation for equalities); unverified "
                         "negatives discarded",
        },
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("outcomes", "per_transform")},
                     indent=2))
    print(f"wrote {out_path} ({n_written} records) and {args.stats}")


if __name__ == "__main__":
    main()
