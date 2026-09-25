#!/usr/bin/env python3
r"""
SymPy parse-coverage census over the MathNet corpus.

Purpose (corpus census for the verified arm):
  Measure what fraction of competition problems a CAS pipeline can actually
  touch, to size the CAS-verified supervision channel. For every problem we
  extract LaTeX math spans ($...$, $$...$$, \(...\), \[...\]), attempt
  sympy.parsing.latex.parse_latex on each *unique* expression (deduplicated
  corpus-wide), and classify results as:
     - 'relational'  : Eq / inequality / chained relations  -> usable for
                       equivalence-transform supervision
     - 'expression'  : parsed, but a bare expression (no relation)
     - 'fail'        : parse error
     - 'timeout'     : exceeded the hard per-expression timeout

Hard timeouts: pebble.ProcessPool with timeout=2s per expression. pebble
kills and respawns the worker process on timeout, so a hung ANTLR parse
cannot stall the run (no signal-based timeouts inside workers).

Outputs:
  results/cas_census.json        - aggregate per-problem / per-domain stats
  results/cas_expr_results.json  - per-unique-expression parse status
Run time: ~5-10 min on a login node with 8 workers (full 27,817 problems).

Usage: python3 cas_census.py [--limit N] [--workers 8] [--timeout 2.0]
"""
import argparse
import json
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
from pebble import ProcessPool

ROOT = Path("/ibex/user/habiam0b/MathNet_Follow_Up")
CORPUS = ROOT / "data" / "mathnet_corpus.parquet"
RESULTS_DIR = ROOT / "results"

# --- math span extraction ---------------------------------------------------
# Order matters: $$...$$ before $...$. Inline $...$ content excludes '$' so a
# stray dollar cannot glue two paragraphs into one bogus span.
MATH_SPAN = re.compile(
    r"\$\$(.+?)\$\$"                      # display $$ ... $$
    r"|\\\[(.+?)\\\]"                     # display \[ ... \]
    r"|\\\((.+?)\\\)"                     # inline  \( ... \)
    r"|(?<!\$)\$([^$]+?)\$(?!\$)",        # inline  $ ... $
    re.DOTALL,
)


def extract_spans(markdown: str):
    """Return list of stripped, non-empty math spans in order of appearance."""
    spans = []
    for m in MATH_SPAN.finditer(markdown or ""):
        s = next(g for g in m.groups() if g is not None).strip()
        if s:
            spans.append(s)
    return spans


# --- worker: parse one LaTeX expression -------------------------------------
def parse_one(latex: str):
    """Parse a single LaTeX string; runs inside a pebble worker process.

    Returns (status, detail):
      status in {'relational', 'expression', 'fail'}
      detail = sympy head name on success, truncated error message on failure.
    ('timeout' is assigned by the parent when pebble kills the worker.)
    """
    warnings.filterwarnings("ignore")
    from sympy.core.relational import Relational
    from sympy.logic.boolalg import And, Or
    from sympy.parsing.latex import parse_latex

    try:
        res = parse_latex(latex)
    except Exception as e:  # LaTeXParsingError and friends
        return "fail", f"{type(e).__name__}: {str(e)[:160]}"
    if isinstance(res, Relational):
        return "relational", type(res).__name__
    # chained relations (a < b < c) can come back as boolean combinations
    if isinstance(res, (And, Or)) and all(isinstance(a, Relational) for a in res.args):
        return "relational", type(res).__name__
    return "expression", type(res).__name__


def top_level_domains(topics):
    """Map topics_flat entries to their top-level prefix (before ' > ')."""
    if topics is None:
        topics = []
    doms = set()
    for t in list(topics):
        doms.add(t.split(">")[0].strip())
    return sorted(doms) if doms else ["(untagged)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only N problems")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=2.0, help="hard per-expression timeout (s)")
    args = ap.parse_args()

    t_start = time.time()
    con = duckdb.connect()
    q = f"SELECT id, problem_markdown, topics_flat FROM '{CORPUS}'"
    if args.limit:
        q += f" LIMIT {args.limit}"
    df = con.execute(q).fetchdf()
    print(f"loaded {len(df)} problems")

    # 1) extract spans, dedupe corpus-wide
    prob_spans = {}   # id -> list of spans
    uniq = {}         # expr -> index (insertion order)
    for pid, pm in zip(df["id"], df["problem_markdown"]):
        spans = extract_spans(pm)
        prob_spans[pid] = spans
        for s in spans:
            if s not in uniq:
                uniq[s] = len(uniq)
    exprs = list(uniq.keys())
    n_spans = sum(len(v) for v in prob_spans.values())
    print(f"{n_spans} spans, {len(exprs)} unique expressions")

    # 2) parse unique expressions with hard per-task timeouts
    status = [None] * len(exprs)   # (status, detail) per unique expr
    done = 0
    with ProcessPool(max_workers=args.workers) as pool:
        futures = [(i, pool.schedule(parse_one, args=(e,), timeout=args.timeout))
                   for i, e in enumerate(exprs)]
        for i, fut in futures:
            try:
                status[i] = fut.result()
            except TimeoutError:
                status[i] = ("timeout", f">{args.timeout}s")
            except Exception as e:  # worker crash etc.
                status[i] = ("fail", f"worker:{type(e).__name__}: {str(e)[:120]}")
            done += 1
            if done % 5000 == 0:
                print(f"  parsed {done}/{len(exprs)}  ({time.time()-t_start:.0f}s)")
    print(f"parsing finished at {time.time()-t_start:.0f}s")

    # 3) per-expression aggregate (over unique expressions and over all spans)
    expr_status = {e: status[uniq[e]] for e in exprs}
    uniq_counts = Counter(s for s, _ in status)
    span_counts = Counter()
    for spans in prob_spans.values():
        for s in spans:
            span_counts[expr_status[s][0]] += 1

    # 4) per-problem stats
    per_problem = {}
    dom_agg = defaultdict(lambda: Counter())
    overall = Counter()
    for pid, topics in zip(df["id"], df["topics_flat"]):
        spans = prob_spans[pid]
        st = [expr_status[s][0] for s in spans]
        n_ok = sum(1 for x in st if x in ("relational", "expression"))
        rec = {
            "n_spans": len(spans),
            "n_parsed": n_ok,
            "n_relational": sum(1 for x in st if x == "relational"),
            "any_parsed": n_ok > 0,
            "all_parsed": len(spans) > 0 and n_ok == len(spans),
            "any_relational": any(x == "relational" for x in st),
        }
        per_problem[pid] = rec
        keys = [k for k in ("any_parsed", "all_parsed", "any_relational") if rec[k]]
        overall["n"] += 1
        overall["has_math"] += bool(spans)
        for k in keys:
            overall[k] += 1
        for d in top_level_domains(topics):
            dom_agg[d]["n"] += 1
            dom_agg[d]["has_math"] += bool(spans)
            for k in keys:
                dom_agg[d][k] += 1

    # 5) failure taxonomy sample (for the qualitative pass)
    fail_examples = [
        {"expr": e[:200], "status": s, "detail": d}
        for e, (s, d) in expr_status.items() if s in ("fail", "timeout")
    ]
    fail_reason_counts = Counter(d.split(":")[0] for _, (s, d) in expr_status.items()
                                 if s in ("fail", "timeout"))

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    n = overall["n"]
    n_math = overall["has_math"]
    parsed_uniq = uniq_counts["relational"] + uniq_counts["expression"]
    summary = {
        "meta": {
            "corpus": str(CORPUS),
            "n_problems": n,
            "n_problems_with_math": n_math,
            "n_spans_total": n_spans,
            "n_unique_expressions": len(exprs),
            "per_expression_timeout_s": args.timeout,
            "workers": args.workers,
            "runtime_s": round(time.time() - t_start, 1),
            "sampling": "full corpus (no subsampling)" if not args.limit else f"LIMIT {args.limit}",
        },
        "unique_expression_status": dict(uniq_counts),
        "span_status": dict(span_counts),
        "expression_level": {
            "pct_unique_parsed": pct(parsed_uniq, len(exprs)),
            "pct_unique_relational_of_parsed": pct(uniq_counts["relational"], parsed_uniq),
            "pct_spans_parsed": pct(span_counts["relational"] + span_counts["expression"], n_spans),
        },
        "problem_level": {
            "pct_any_parsed_of_all": pct(overall["any_parsed"], n),
            "pct_all_parsed_of_all": pct(overall["all_parsed"], n),
            "pct_any_parsed_of_with_math": pct(overall["any_parsed"], n_math),
            "pct_all_parsed_of_with_math": pct(overall["all_parsed"], n_math),
            "pct_any_relational_of_all": pct(overall["any_relational"], n),
            "counts": dict(overall),
        },
        "per_domain": {
            d: {
                "n_problems": c["n"],
                "pct_any_parsed": pct(c["any_parsed"], c["n"]),
                "pct_all_parsed": pct(c["all_parsed"], c["n"]),
                "pct_any_relational": pct(c["any_relational"], c["n"]),
                "counts": dict(c),
            }
            for d, c in sorted(dom_agg.items(), key=lambda kv: -kv[1]["n"])
        },
        "failure_reason_counts": dict(fail_reason_counts.most_common()),
        "failure_examples_sample": fail_examples[:400],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "cas_census.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "cas_expr_results.json", "w") as f:
        json.dump({e: {"status": s, "detail": d} for e, (s, d) in expr_status.items()}, f)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("unique_expression_status", "expression_level", "problem_level")}, indent=2))
    print("per_domain:")
    for d, v in summary["per_domain"].items():
        print(f"  {d:25s} n={v['n_problems']:6d} any={v['pct_any_parsed']:6.2f}% "
              f"all={v['pct_all_parsed']:6.2f}% any_rel={v['pct_any_relational']:6.2f}%")
    print(f"total runtime {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
