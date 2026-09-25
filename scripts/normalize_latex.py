#!/usr/bin/env python3
r"""
LaTeX normalization pre-pass for the InvarEmbed CAS channel.

The SymPy parse census (results/cas_census.json) put unique-expression parse
coverage at 86.61% and identified a failure taxonomy dominated by:
  - chained relations           (a < b < c   -> TypeError)          ~1,560
  - trailing sentence punct     ("M=N+1 .")
  - \left / \right decorations  (\left| x \right| \leq 1)
  - trailing \quad\text{...} prose clauses
  - operator variants           (\leqslant, \geqslant, \ne)
Its normalizer_recovery_experiment predicted a ~40-line pre-pass + chain
splitter lifts unique-expression coverage from 86.6% to ~91.7%.

This module implements exactly that pre-pass:

  normalize(s)        -> str          conservative, semantics-preserving rules
  split_chained(s)    -> [str]        a<b<c -> ['a<b', 'b<c'] (top level only)
  split_statements(s) -> [str]        break at prose \text{...} separators,
                                      top-level ';' and '\\' row breaks
  try_parse(s, ...)   -> (status, [sympy objects])
                         status in {'relational','expression','fail'};
                         used by generate_cas_pairs.py.

__main__ re-runs every census FAILURE (results/cas_expr_results.json) through
the pipeline with hard pebble timeouts and reports recovered coverage, writing
results/normalizer_recovery.json.

Usage:
    python3 scripts/normalize_latex.py [--workers 8] [--timeout 2.0] [--limit N]
"""
import re

# --------------------------------------------------------------------------
# 1. conservative normalization rules
# --------------------------------------------------------------------------

# spacing macros that carry no semantics for the parser
_SPACING = [
    (re.compile(r"\\[,;!:]"), " "),          # \, \; \! \:
    (re.compile(r"\\(?:quad|qquad)\b"), " "),
    (re.compile(r"(?<!\\)~"), " "),          # non-breaking space (not \~)
    (re.compile(r"\\ "), " "),               # backslash-space
]

_OP_NORMALIZE = [
    (re.compile(r"\\leqslant\b"), r"\\leq"),
    (re.compile(r"\\geqslant\b"), r"\\geq"),
    (re.compile(r"\\leqq\b"), r"\\leq"),
    (re.compile(r"\\geqq\b"), r"\\geq"),
    (re.compile(r"\\ne(?![a-zA-Z])"), r"\\neq"),   # \ne -> \neq (census: \ne parses as n*e)
    (re.compile(r"\\dfrac\b"), r"\\frac"),
    (re.compile(r"\\tfrac\b"), r"\\frac"),
]

# trailing sentence punctuation / row breaks inside math
_TRAIL_PUNCT = re.compile(r"(?:\s|\\\\|[.,;:!?])+$")
# a trailing prose clause:  [,] \quad \text{...}   or  , \text{...}
_TRAIL_TEXT = re.compile(
    r"[,;]?\s*\\(?:text|textrm|mbox|textit|textbf)\s*\{[^{}]*\}\s*$")


def normalize(s: str) -> str:
    """Conservative, semantics-preserving cleanup of one LaTeX math span."""
    t = s.strip()
    # \left. / \right. (invisible delimiters) first, then bare \left/\right
    t = re.sub(r"\\left\s*\.", "", t)
    t = re.sub(r"\\right\s*\.", "", t)
    t = re.sub(r"\\left\s*", "", t)
    t = re.sub(r"\\right\s*", "", t)
    for pat, rep in _OP_NORMALIZE:
        t = pat.sub(rep, t)
    # iteratively strip trailing punctuation and trailing prose clauses
    prev = None
    while prev != t:
        prev = t
        t = _TRAIL_PUNCT.sub("", t)
        t = _TRAIL_TEXT.sub("", t)
    for pat, rep in _SPACING:
        t = pat.sub(rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------------------------------
# 2. top-level tokenization helpers
# --------------------------------------------------------------------------

# relational operators recognized at top level (after normalize())
_REL_OP = re.compile(
    r"\\leq(?![a-zA-Z])|\\geq(?![a-zA-Z])|\\le(?![a-zA-Z])|\\ge(?![a-zA-Z])"
    r"|\\neq(?![a-zA-Z])|<|>|(?<![<>=!\\])=(?![<>=])")

_OPEN, _CLOSE = "({[", ")}]"


def _top_level_matches(s: str, pattern: re.Pattern):
    """Yield matches of `pattern` that sit at brace/paren/bracket depth 0."""
    depth = [0] * (len(s) + 1)
    d = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "{}":
            # \{ and \} count as depth changers too (set braces)
            d += 1 if s[i + 1] == "{" else -1
            depth[i] = depth[i + 1] = d
            i += 2
            continue
        if c in _OPEN:
            d += 1
        elif c in _CLOSE:
            d = max(0, d - 1)
        depth[i] = d
        i += 1
    for m in pattern.finditer(s):
        # operator is top-level if depth at its start position is 0 for
        # closers/openers-adjusted scan: use depth just before the char
        pos = m.start()
        d_here = depth[pos]
        # opening chars increment depth at their own index; a '<' is never
        # an opener so depth[pos] is the surrounding depth
        if d_here == 0:
            yield m


def split_chained(s: str):
    """Split a chained relation 'a R1 b R2 c ...' into pairwise relations.

    Returns [s] unchanged when fewer than two top-level relational operators
    are present. Segments with empty sides are dropped.
    """
    ms = list(_top_level_matches(s, _REL_OP))
    if len(ms) < 2:
        return [s]
    cuts, ops = [], []
    prev_end = 0
    segs = []
    for m in ms:
        segs.append(s[prev_end:m.start()])
        ops.append(m.group(0))
        prev_end = m.end()
    segs.append(s[prev_end:])
    out = []
    for i, op in enumerate(ops):
        a, b = segs[i].strip(), segs[i + 1].strip()
        if a and b:
            out.append(f"{a} {op} {b}")
    return out if out else [s]


# prose separators between independent statements inside one span:
#   \text{ and } , \text{si} , top-level ';', '\\' row breaks
_PROSE_TEXT = re.compile(r"\\(?:text|textrm|mbox)\s*\{([^{}]*)\}")
_ROWBREAK = re.compile(r"\\\\|;")


def split_statements(s: str):
    """Split one span into independent statements at prose separators.

    A \text{...} block whose content has no digits and no backslash (pure
    prose like 'and', 'si', 'where') separates statements. Top-level ';' and
    '\\' row breaks also separate. Returns >=1 non-empty fragments.
    """
    # mark prose \text blocks as separators
    seps = []
    for m in _top_level_matches(s, _PROSE_TEXT):
        content = m.group(1)
        if not re.search(r"[0-9\\]", content) and len(content) < 40:
            seps.append((m.start(), m.end()))
    for m in _top_level_matches(s, _ROWBREAK):
        seps.append((m.start(), m.end()))
    if not seps:
        return [s]
    seps.sort()
    frags, prev = [], 0
    for a, b in seps:
        if a >= prev:
            frags.append(s[prev:a])
            prev = b
    frags.append(s[prev:])
    return [f.strip() for f in frags if f.strip()] or [s]


# --------------------------------------------------------------------------
# 3. parse wrapper used by the pair generator
# --------------------------------------------------------------------------

# Macros parse_latex either rejects or silently mis-parses into bare Symbols
# (e.g. \cdots -> Symbol('cdots'), \forall -> Symbol('forall')). Any fragment
# containing one of these is NOT trustworthy and is rejected outright --
# recovering garbage would poison the verified-pair channel.
_BLACKLIST = re.compile(
    r"\\(?:dots[bcim]?|ldots|cdots|vdots|ddots|forall|exists|nexists"
    r"|in(?![a-zA-Z])|notin|ni(?![a-zA-Z])|subset|subseteq|subsetneq|supset"
    r"|supseteq|cup|cap|setminus|mid(?![a-zA-Z])|nmid|parallel|nparallel|perp"
    r"|equiv|pmod|bmod|mod(?![a-zA-Z])|sim(?![a-zA-Z])|simeq|cong|approx"
    r"|propto|mapsto|to(?![a-zA-Z])|rightarrow|Rightarrow|leftrightarrow"
    r"|leftarrow|Leftarrow|Leftrightarrow|iff|implies|angle|measuredangle"
    r"|overrightarrow|underbrace|overbrace|begin|end|stackrel|xrightarrow"
    r"|hline|vert(?![a-zA-Z])|lfloor|rfloor|lceil|rceil|arg(?![a-zA-Z])"
    r"|therefore|because)\b"
    r"|\.\.\.")


def _classify(obj):
    from sympy.core.relational import Relational
    from sympy.logic.boolalg import And, Or, BooleanTrue, BooleanFalse
    if isinstance(obj, (BooleanTrue, BooleanFalse)):
        # chained '=' collapsing to a bare boolean is a mis-parse
        return "fail"
    if isinstance(obj, Relational):
        return "relational"
    if isinstance(obj, (And, Or)) and all(
            isinstance(a, Relational) for a in obj.args):
        return "relational"
    return "expression"


def parse_single(s: str):
    """normalize + parse; returns (status, obj_or_errmsg). No splitting."""
    import warnings
    warnings.filterwarnings("ignore")
    from sympy.parsing.latex import parse_latex
    t = normalize(s)
    if not t:
        return "fail", "empty after normalize"
    if _BLACKLIST.search(t):
        return "fail", "blacklisted macro (would mis-parse)"
    try:
        obj = parse_latex(t)
    except Exception as e:
        return "fail", f"{type(e).__name__}: {str(e)[:120]}"
    st = _classify(obj)
    if st == "fail":
        return "fail", "parsed to bare Boolean (mis-parse)"
    return st, obj


def try_parse(s: str):
    """Full recovery pipeline for one span.

    Returns (status, objs):
      status 'relational'  -- >=1 relational object recovered
             'expression'  -- parsed but nothing relational
             'fail'        -- nothing parsed
      objs: list of sympy objects (relationals first).
    Order of attempts: direct parse of normalized span; then statement
    split; then chained-relation split of each fragment.
    """
    status, obj = parse_single(s)
    if status != "fail":
        return status, [obj]
    t = normalize(s)
    objs, any_expr = [], []
    for frag in split_statements(t):
        pieces = split_chained(frag)
        for p in ([frag] if pieces == [frag] else pieces):
            st, ob = parse_single(p)
            if st == "relational":
                objs.append(ob)
            elif st == "expression":
                any_expr.append(ob)
        if pieces != [frag] and not objs:
            # chain split produced pieces but none relational: still try them
            pass
    if objs:
        return "relational", objs
    if any_expr:
        return "expression", any_expr
    return "fail", []


# --------------------------------------------------------------------------
# 4. __main__: re-run the census failures, report recovered coverage
# --------------------------------------------------------------------------

def _recover_worker(expr: str):
    """Worker: returns recovery status for one previously-failed expression."""
    status, objs = try_parse(expr)
    return status


def main():
    import argparse
    import json
    import time
    from collections import Counter
    from pathlib import Path

    from pebble import ProcessPool

    ROOT = Path("/ibex/user/habiam0b/MathNet_Follow_Up")

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N failures")
    args = ap.parse_args()

    t0 = time.time()
    expr_results = json.load(open(ROOT / "results" / "cas_expr_results.json"))
    n_unique = len(expr_results)
    base_ok = sum(1 for v in expr_results.values()
                  if v["status"] in ("relational", "expression"))
    base_rel = sum(1 for v in expr_results.values() if v["status"] == "relational")
    failures = [e for e, v in expr_results.items()
                if v["status"] in ("fail", "timeout")]
    if args.limit:
        failures = failures[:args.limit]
    print(f"census: {n_unique} unique exprs, {base_ok} parsed "
          f"({100*base_ok/n_unique:.2f}%), {len(failures)} failures to re-run")

    recovered = {}
    counts = Counter()
    with ProcessPool(max_workers=args.workers) as pool:
        futs = [(e, pool.schedule(_recover_worker, args=(e,),
                                  timeout=args.timeout)) for e in failures]
        for i, (e, fut) in enumerate(futs):
            try:
                st = fut.result()
            except TimeoutError:
                st = "timeout"
            except Exception:
                st = "fail"
            recovered[e] = st
            counts[st] += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(futs)}  ({time.time()-t0:.0f}s)  {dict(counts)}")

    n_rec = counts["relational"] + counts["expression"]
    new_ok = base_ok + n_rec
    new_rel = base_rel + counts["relational"]
    summary = {
        "meta": {
            "n_unique_expressions": n_unique,
            "n_census_failures_rerun": len(failures),
            "workers": args.workers, "timeout_s": args.timeout,
            "runtime_s": round(time.time() - t0, 1),
        },
        "census_baseline": {
            "pct_unique_parsed": round(100 * base_ok / n_unique, 2),
            "n_relational": base_rel,
        },
        "recovery": {
            "recovered_relational": counts["relational"],
            "recovered_expression": counts["expression"],
            "still_fail": counts["fail"] + counts["timeout"],
            "pct_of_failures_recovered": round(100 * n_rec / max(1, len(failures)), 2),
        },
        "after_normalizer": {
            "pct_unique_parsed": round(100 * new_ok / n_unique, 2),
            "n_relational": new_rel,
            "census_prediction": "86.6% -> ~91.7%",
        },
    }
    out = ROOT / "results" / "normalizer_recovery.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
