#!/usr/bin/env python3
"""
Bootstrap / equivalence statistics for the paper's claims (zero GPU).

Four pre-specified pieces of machinery:

(a) HIERARCHICAL CLUSTER BOOTSTRAP for the real-duplicate (cross-lingual)
    eval: resample duplicate CLUSTERS with replacement (provenance:
    data/crosslingual_eval/clusters.json, 370 clusters; every query row of a
    --dump-ranks file carries its cluster_id), then queries within each
    sampled cluster, nested over training seeds for multi-seed model
    families. Queries inside one duplicate cluster share provenance and are
    NOT independent -- a plain per-query bootstrap would understate variance.
    Yields percentile CIs for strict cross-lingual R@k of any model pair.

(b) PAIRED PER-QUERY BOOTSTRAP for MathNet-Retrieve comparisons (15,000
    queries/tier): the same query resample is applied to both models (the
    three tiers share the identical query set, so one resample serves all
    tiers and preserves cross-tier correlation); training seeds are resampled
    in an outer level for the ctrl arms. Used for V2-vs-RaDeR-gte easy R@1
    (plus a McNemar exact check) and for the ctrl LLM-CAS gap per tier.

(c) TOST EQUIVALENCE TEST for "the two supervision arms are equivalent on
    real duplicates". PRE-SPECIFIED margin = --margin-frac (default 0.25) x
    the observed benchmark easy R@1 gap (LLM-CAS): the arms may differ on
    real data by at most a quarter of what the benchmark shows, otherwise
    equivalence fails. Per-query level: equivalence at alpha=0.05 iff the
    90% bootstrap CI of the real-duplicate strict R@1 gap lies inside
    (-margin, +margin). Seed-summary level: classical two one-sided t-tests.

(d) DIFFERENCE-IN-DIFFERENCES, the paper's actual finding:
        DiD = (benchmark R@1 gap, LLM-CAS) - (real-duplicate strict R@1 gap).
    Primary benchmark tier = easy (the only tier whose R@1 lives on the same
    scale as the real-duplicate eval; the hard-tier exhibit is a RATIO story
    -- 0.15 vs 9.9 R@1 -- and is reported here in raw points as secondary).

Data sources (auto-detected; every analysis degrades gracefully and labels
its level):
  * summary JSONs (level "seed_summary", t-intervals across seeds):
      results/eval_{tier}_ctrl-{arm}-s{seed}.json
      results/crosslingual_ctrl-{arm}-s{seed}.json
    (legacy seed-42 names eval_{tier}_ctrl-{arm}.json /
     crosslingual_ctrl-{arm}-6145.json accepted)
  * per-query rank dumps (level "per_query_nested"; written by
    eval_retrieve.py / eval_crosslingual.py --dump-ranks; regenerate all of
    them with scripts/dump_ranks.slurm, driver: scripts/rebuild_stats.sh):
      results/ranks/{easy,medium,hard}_ctrl-{arm}-s{seed}.ranks.jsonl
      results/ranks/xling_ctrl-{arm}-s{seed}.ranks.jsonl
      results/ranks/easy_qwen3-0.6b-p2-instr.ranks.jsonl
      results/ranks/easy_rader-gte-qwen2-7b.ranks.jsonl
      results/ranks/xling_qwen3-0.6b-p2-instr.ranks.jsonl
      results/ranks/xling_qwen3-0.6b-base.ranks.jsonl

Usage (login node, CPU; seconds in summary mode, ~2-5 min with dumps):
  python scripts/bootstrap_stats.py                       # auto-detect
  python scripts/bootstrap_stats.py --seeds 42,43,44,45,46,47,48,49
  python scripts/bootstrap_stats.py --self-test           # synthetic checks
  python scripts/bootstrap_stats.py --print-seeds         # for rebuild_stats.sh
  python scripts/bootstrap_stats.py --check-dumps         # list missing dumps

Output: results/final_stats.json + a printed paper-ready table.
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import date

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
RANKS_DEFAULT = os.path.join(RESULTS, "ranks")

ARMS = ("cas", "llm")
TIERS = ("easy", "medium", "hard")
KS = (1, 5, 10)
SENTINEL = 10 ** 9          # "gold not ranked" in rank matrices
ALPHA = 0.05                # 95% CIs; TOST at alpha via 90% CI inclusion

# --------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------
try:
    from scipy import stats as _sps

    def t_ppf(q, df):
        return float(_sps.t.ppf(q, df))

    def t_sf(x, df):
        return float(_sps.t.sf(x, df))
except ImportError:                                        # pragma: no cover
    print("[warn] scipy unavailable -- normal approximation for t "
          "(anticonservative at n=3!)", file=sys.stderr)

    def t_ppf(q, df):
        return float(_norm_ppf(q))

    def t_sf(x, df):
        return 0.5 * math.erfc(x / math.sqrt(2))

    def _norm_ppf(q):
        lo, hi = -10, 10
        for _ in range(80):
            mid = (lo + hi) / 2
            if 0.5 * math.erfc(-mid / math.sqrt(2)) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def mean_sd(vals):
    n = len(vals)
    mu = float(np.mean(vals)) if n else None
    sd = float(np.std(vals, ddof=1)) if n >= 2 else None
    return mu, sd, n


def t_ci(vals, level=0.95):
    """(mean, lo, hi) t-interval across independent runs (seeds)."""
    mu, sd, n = mean_sd(vals)
    if mu is None or sd is None:
        return mu, None, None
    half = t_ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)
    return mu, mu - half, mu + half


def pct_ci(reps, level=0.95):
    lo, hi = np.percentile(reps, [100 * (1 - level) / 2,
                                  100 * (1 - (1 - level) / 2)])
    return float(lo), float(hi)


def r2(x):
    return None if x is None else round(float(x), 2)


# --------------------------------------------------------------------------
# file discovery / loading
# --------------------------------------------------------------------------
def _summary_paths(arm, seed):
    """Candidate (bench-by-tier, xling) summary filenames, new name first."""
    bench = {t: [f"eval_{t}_ctrl-{arm}-s{seed}.json"] for t in TIERS}
    xling = [f"crosslingual_ctrl-{arm}-s{seed}.json"]
    if seed == 42:  # legacy names from train_controlled_exp / eval_ctrl_ood
        for t in TIERS:
            bench[t].append(f"eval_{t}_ctrl-{arm}.json")
        xling.append(f"crosslingual_ctrl-{arm}-6145.json")
    return bench, xling


def _first(cands):
    for c in cands:
        p = os.path.join(RESULTS, c)
        if os.path.exists(p):
            return p
    return None


def detect_seeds():
    """Seeds with a COMPLETE summary set (both arms x 3 tiers + xling)."""
    found = set()
    for f in os.listdir(RESULTS):
        m = re.match(r"eval_easy_ctrl-cas-s(\d+)\.json$", f)
        if m:
            found.add(int(m.group(1)))
    if _first(["eval_easy_ctrl-cas.json"]):
        found.add(42)
    complete = []
    for s in sorted(found):
        ok = True
        for arm in ARMS:
            bench, xling = _summary_paths(arm, s)
            ok &= all(_first(bench[t]) for t in TIERS) and bool(_first(xling))
        if ok:
            complete.append(s)
    return complete


def load_summaries(seeds):
    """per_seed[seed][arm] = {'easy'/'medium'/'hard'/'xling_strict':
    {recall@k: float}}"""
    per_seed = {}
    for s in seeds:
        per_seed[s] = {}
        for arm in ARMS:
            bench, xling = _summary_paths(arm, s)
            d = {}
            for t in TIERS:
                p = _first(bench[t])
                d[t] = {k: float(v) for k, v in
                        json.load(open(p, encoding="utf-8"))["overall"].items()}
            p = _first(xling)
            d["xling_strict"] = {
                k: float(v) for k, v in
                json.load(open(p, encoding="utf-8"))
                ["strict_crosslingual_gold"].items()}
            per_seed[s][arm] = d
    return per_seed


def expected_dumps(seeds, ranks_dir):
    """(required_for_ctrl_analyses, optional_single_model) dump paths."""
    req, opt = [], []
    for s in seeds:
        for arm in ARMS:
            req.append(os.path.join(ranks_dir,
                                    f"xling_ctrl-{arm}-s{s}.ranks.jsonl"))
            for t in TIERS:
                req.append(os.path.join(ranks_dir,
                                        f"{t}_ctrl-{arm}-s{s}.ranks.jsonl"))
    for f in ("easy_qwen3-0.6b-p2-instr.ranks.jsonl",
              "easy_rader-gte-qwen2-7b.ranks.jsonl",
              "xling_qwen3-0.6b-p2-instr.ranks.jsonl",
              "xling_qwen3-0.6b-base.ranks.jsonl"):
        opt.append(os.path.join(ranks_dir, f))
    return req, opt


def read_ranks(path, field="gold_rank"):
    """{qid: exact 0-based rank (SENTINEL if null/absent)} (+ row list)."""
    out, rows = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            r = row.get(field)
            out[row["qid"]] = SENTINEL if r is None else int(r)
            rows.append(row)
    return out, rows


def rank_matrix(paths, field, qid_order=None):
    """Stack rank dicts from `paths` into an (S, N) int64 matrix aligned on a
    shared qid order (default: sorted qids of the first file)."""
    dicts = []
    first_rows = None
    for p in paths:
        d, rows = read_ranks(p, field)
        dicts.append(d)
        if first_rows is None:
            first_rows = rows
    if qid_order is None:
        qid_order = sorted(dicts[0])
    M = np.full((len(dicts), len(qid_order)), SENTINEL, dtype=np.int64)
    for i, d in enumerate(dicts):
        if set(d) != set(qid_order):
            raise SystemExit(f"[fatal] qid set mismatch in {paths[i]}")
        M[i] = np.array([d[q] for q in qid_order], dtype=np.int64)
    return M, qid_order, first_rows


def cluster_groups(rows, qid_order):
    """List of np.arrays of query positions, one per duplicate cluster."""
    pos = {q: i for i, q in enumerate(qid_order)}
    by_cl = {}
    for row in rows:
        by_cl.setdefault(row["cluster_id"], []).append(pos[row["qid"]])
    return [np.array(v, dtype=np.int64) for v in by_cl.values()]


# --------------------------------------------------------------------------
# bootstrap cores
# --------------------------------------------------------------------------
def cluster_idx(rng, groups):
    """One hierarchical resample: clusters with replacement, then queries
    within each sampled cluster with replacement."""
    C = len(groups)
    parts = []
    for c in rng.integers(0, C, C):
        g = groups[c]
        parts.append(g if g.size == 1 else g[rng.integers(0, g.size, g.size)])
    return np.concatenate(parts)


def joint_ctrl_bootstrap(bench, xling, groups, n_boot, rng, ks=KS):
    """Joint nested bootstrap for the ctrl arms.

    bench: {tier: {arm: (S, Nb) rank matrix}} (may be {} if bench dumps absent)
    xling: {arm: (S, Nx) strict-rank matrix}  (may be None)
    groups: cluster groups for xling queries.
    Returns replicate dict:
      gap[tier][k]  : benchmark gap LLM-CAS (points)
      xgap[k]       : real-duplicate strict gap LLM-CAS (points)
      did[tier][k]  : gap - xgap (only if both sides present)
    Seed resample is SHARED across all statistics of one replicate; the
    benchmark query resample is shared across tiers (identical query sets).
    """
    tiers = sorted(bench)
    some_bench = next(iter(bench.values()), None)
    S = (some_bench["cas"].shape[0] if some_bench is not None
         else xling["cas"].shape[0])
    Nb = some_bench["cas"].shape[1] if some_bench is not None else 0
    out = {"gap": {t: {k: np.empty(n_boot) for k in ks} for t in tiers},
           "xgap": {k: np.empty(n_boot) for k in ks} if xling else None,
           "did": {t: {k: np.empty(n_boot) for k in ks} for t in tiers}
                  if (xling and bench) else None}
    for b in range(n_boot):
        sidx = rng.integers(0, S, S)
        qidx = rng.integers(0, Nb, Nb) if Nb else None
        xidx = cluster_idx(rng, groups) if xling else None
        if xling is not None:
            xl = xling["llm"][sidx][:, xidx]
            xc = xling["cas"][sidx][:, xidx]
        for t in tiers:
            gl = bench[t]["llm"][sidx][:, qidx]
            gc = bench[t]["cas"][sidx][:, qidx]
            for k in ks:
                g = 100 * ((gl < k).mean() - (gc < k).mean())
                out["gap"][t][k][b] = g
                if out["did"] is not None:
                    out["did"][t][k][b] = g - 100 * ((xl < k).mean()
                                                     - (xc < k).mean())
        if xling is not None:
            for k in ks:
                out["xgap"][k][b] = 100 * ((xl < k).mean() - (xc < k).mean())
    return out


def pair_cluster_bootstrap(RA, RB, groups, n_boot, rng, ks=KS):
    """Cluster bootstrap of A-B strict R@k for two (S, N) rank matrices
    (seeds resampled independently per model family only if both are
    multi-seed and paired -- here A and B share the seed axis, so one shared
    seed resample keeps the pairing)."""
    S = RA.shape[0]
    reps = {k: np.empty(n_boot) for k in ks}
    for b in range(n_boot):
        sidx = rng.integers(0, S, S)
        xidx = cluster_idx(rng, groups)
        a = RA[sidx][:, xidx]
        bb = RB[sidx][:, xidx]
        for k in ks:
            reps[k][b] = 100 * ((a < k).mean() - (bb < k).mean())
    return reps


def paired_query_bootstrap(rA, rB, n_boot, rng, ks=KS):
    """Paired per-query bootstrap for two single-run rank vectors (N,)."""
    N = rA.shape[0]
    reps = {k: np.empty(n_boot) for k in ks}
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        for k in ks:
            reps[k][b] = 100 * ((rA[idx] < k).mean() - (rB[idx] < k).mean())
    return reps


def mcnemar_exact(hitA, hitB):
    """Exact two-sided McNemar p on paired hit indicators."""
    b01 = int(np.sum(hitA & ~hitB))
    b10 = int(np.sum(~hitA & hitB))
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "p": 1.0, "A_only": 0, "B_only": 0}
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(b01, b10), n, 0.5).pvalue)
    except ImportError:                                    # pragma: no cover
        p = None
    return {"n_discordant": n, "A_only": b01, "B_only": b10, "p": p}


def tost_t(vals, margin, alpha=ALPHA):
    """Two one-sided t-tests across seeds: H1 = |true gap| < margin."""
    mu, sd, n = mean_sd(vals)
    if sd is None or n < 2:
        return {"level": "seed_summary", "n": n, "note": "needs >=2 seeds"}
    se = sd / math.sqrt(n)
    p_lower = t_sf((mu + margin) / se, n - 1)   # H0: gap <= -margin
    p_upper = t_sf((margin - mu) / se, n - 1)   # H0: gap >= +margin
    p = max(p_lower, p_upper)
    lo90 = mu - t_ppf(1 - alpha, n - 1) * se
    hi90 = mu + t_ppf(1 - alpha, n - 1) * se
    return {"level": "seed_summary", "n_seeds": n, "mean_gap": r2(mu),
            "se": r2(se), "margin_points": r2(margin),
            "ci90": [r2(lo90), r2(hi90)],
            "p_lower": round(p_lower, 4), "p_upper": round(p_upper, 4),
            "p_tost": round(p, 4), "equivalent_at_0.05": bool(p < alpha)}


def tost_boot(reps, margin, alpha=ALPHA):
    """Bootstrap TOST: equivalence iff the 90% percentile CI of the gap lies
    inside (-margin, +margin)."""
    lo, hi = pct_ci(reps, 1 - 2 * alpha)
    return {"level": "per_query_nested", "margin_points": r2(margin),
            "ci90": [r2(lo), r2(hi)],
            "equivalent_at_0.05": bool(-margin < lo and hi < margin)}


# --------------------------------------------------------------------------
# self-test (synthetic; exercises every bootstrap path without GPU results)
# --------------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(7)
    S, Nb = 3, 2000
    p_cas, p_llm = 0.17, 0.62                       # true bench R@1
    bench = {"easy": {
        "cas": np.where(rng.random((S, Nb)) < p_cas, 0, SENTINEL),
        "llm": np.where(rng.random((S, Nb)) < p_llm, 0, SENTINEL)}}
    sizes = rng.integers(1, 4, 300)
    groups, pos = [], 0
    for m in sizes:
        groups.append(np.arange(pos, pos + m))
        pos += m
    Nx = pos
    x_cas, x_llm = 0.57, 0.66                       # true strict R@1
    xling = {"cas": np.where(rng.random((S, Nx)) < x_cas, 0, SENTINEL),
             "llm": np.where(rng.random((S, Nx)) < x_llm, 0, SENTINEL)}
    out = joint_ctrl_bootstrap(bench, xling, groups, 400, rng, ks=(1,))
    true_gap, true_xgap = 100 * (p_llm - p_cas), 100 * (x_llm - x_cas)
    for name, reps, truth in [("bench gap", out["gap"]["easy"][1], true_gap),
                              ("xling gap", out["xgap"][1], true_xgap),
                              ("DiD", out["did"]["easy"][1],
                               true_gap - true_xgap)]:
        lo, hi = pct_ci(reps)
        assert lo < truth < hi, f"{name}: CI [{lo:.2f},{hi:.2f}] misses {truth}"
        print(f"  [ok] {name}: mean {reps.mean():6.2f}  CI [{lo:6.2f},"
              f"{hi:6.2f}]  truth {truth:6.2f}")
    reps = pair_cluster_bootstrap(xling["llm"], xling["cas"], groups,
                                  300, rng, ks=(1,))[1]
    lo, hi = pct_ci(reps)
    assert lo < true_xgap < hi
    print(f"  [ok] pair cluster bootstrap CI [{lo:.2f},{hi:.2f}]")
    rA = np.where(rng.random(3000) < 0.30, 0, SENTINEL)
    rB = np.where(rng.random(3000) < 0.25, 0, SENTINEL)
    reps = paired_query_bootstrap(rA, rB, 300, rng, ks=(1,))[1]
    lo, hi = pct_ci(reps)
    assert lo < 5.0 < hi
    print(f"  [ok] paired query bootstrap CI [{lo:.2f},{hi:.2f}]")
    mc = mcnemar_exact(rA < 1, rB < 1)
    assert mc["n_discordant"] > 0 and mc["p"] is not None
    print(f"  [ok] mcnemar p={mc['p']:.4f} discordant={mc['n_discordant']}")
    tt = tost_t([3.06, 15.52, 8.66], 11.11)
    assert tt["p_tost"] > 0.05 and not tt["equivalent_at_0.05"]
    tt2 = tost_t([0.5, -0.4, 0.2], 11.11)
    assert tt2["equivalent_at_0.05"]
    tb = tost_boot(np.array([0.0, 1.0, -1.0] * 200), 11.11)
    assert tb["equivalent_at_0.05"]
    print("  [ok] TOST (t + bootstrap) behave as expected")
    print("[self-test] ALL PASS")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", default=None,
                    help="comma-separated ctrl seeds (default: auto-detect "
                         "complete summary sets)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--rng-seed", type=int, default=20260730)
    ap.add_argument("--margin-frac", type=float, default=0.25,
                    help="TOST margin as a fraction of the benchmark easy "
                         "R@1 gap (pre-specified: 0.25)")
    ap.add_argument("--ranks-dir", default=RANKS_DEFAULT)
    ap.add_argument("--xling-pairs",
                    default="qwen3-0.6b-p2-instr:qwen3-0.6b-base",
                    help="extra single-run xling pairs 'A:B[,A2:B2]' read "
                         "from <ranks-dir>/xling_<name>.ranks.jsonl")
    ap.add_argument("--output", default=os.path.join(RESULTS,
                                                     "final_stats.json"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--print-seeds", action="store_true",
                    help="print detected complete seeds and exit")
    ap.add_argument("--check-dumps", action="store_true",
                    help="list expected-but-missing rank dumps and exit "
                         "(exit 1 = required missing, 2 = only optional "
                         "missing, 0 = all present)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else detect_seeds())
    if args.print_seeds:
        print(",".join(str(s) for s in seeds))
        return 0
    req, opt = expected_dumps(seeds, args.ranks_dir)
    if args.check_dumps:
        miss_req = [p for p in req if not os.path.exists(p)]
        miss_opt = [p for p in opt if not os.path.exists(p)]
        for p in miss_req:
            print(f"MISSING required: {p}")
        for p in miss_opt:
            print(f"MISSING optional: {p}")
        if not (miss_req or miss_opt):
            print("all expected dumps present")
        return 1 if miss_req else (2 if miss_opt else 0)

    if not seeds:
        raise SystemExit("[fatal] no complete ctrl seed found in results/")
    print(f"[data] ctrl seeds with complete summaries: {seeds}")
    per_seed = load_summaries(seeds)
    rng = np.random.default_rng(args.rng_seed)
    caveats = []

    # ---------------- seed-summary point estimates & gaps -----------------
    gap_vals = {}          # (metric, k) -> per-seed gap list (points)
    for metric in TIERS + ("xling_strict",):
        for k in KS:
            kk = f"recall@{k}"
            gap_vals[(metric, k)] = [
                per_seed[s]["llm"][metric][kk] - per_seed[s]["cas"][metric][kk]
                for s in seeds]

    easy_gap_point = float(np.mean(gap_vals[("easy", 1)]))
    margin = args.margin_frac * easy_gap_point

    # ---------------- per-query data (if dumps exist) ---------------------
    have_all_req = all(os.path.exists(p) for p in req)
    bench_mat, xling_mat, groups = {}, None, None
    if have_all_req:
        qorder = None
        for t in TIERS:
            bench_mat[t] = {}
            for arm in ARMS:
                paths = [os.path.join(args.ranks_dir,
                                      f"{t}_ctrl-{arm}-s{s}.ranks.jsonl")
                         for s in seeds]
                M, qorder, _ = rank_matrix(paths, "gold_rank", qorder)
                bench_mat[t][arm] = M
        xorder, xrows = None, None
        xling_mat = {}
        for arm in ARMS:
            paths = [os.path.join(args.ranks_dir,
                                  f"xling_ctrl-{arm}-s{s}.ranks.jsonl")
                     for s in seeds]
            M, xorder, rows = rank_matrix(paths, "xling_rank", xorder)
            xling_mat[arm] = M
            xrows = xrows or rows
        groups = cluster_groups(xrows, xorder)
        # cross-check dump-derived R@1 against canonical summaries
        for i, s in enumerate(seeds):
            for arm in ARMS:
                d = 100 * (xling_mat[arm][i] < 1).mean()
                c = per_seed[s][arm]["xling_strict"]["recall@1"]
                if abs(d - c) > 0.5:
                    caveats.append(
                        f"xling dump vs summary mismatch >0.5pt: seed {s} "
                        f"{arm}: {d:.2f} vs {c:.2f} (re-encode "
                        f"nondeterminism?)")
        level = "per_query_nested"
        boot = joint_ctrl_bootstrap(bench_mat, xling_mat, groups,
                                    args.n_boot, rng)
    else:
        level = "seed_summary"
        boot = None
        n_missing = sum(1 for p in req if not os.path.exists(p))
        caveats.append(
            f"{n_missing}/{len(req)} required per-query rank dumps missing "
            f"-> ctrl analyses at seed-summary level (t-intervals across "
            f"n={len(seeds)} seeds; low power). Regenerate dumps with "
            f"scripts/dump_ranks.slurm via scripts/rebuild_stats.sh.")

    result = {
        "generated": str(date.today()),
        "seeds_used": seeds,
        "n_boot": args.n_boot,
        "rng_seed": args.rng_seed,
        "level": level,
        "margin_spec": (f"{args.margin_frac} x benchmark easy R@1 gap "
                        f"(LLM-CAS) point estimate"),
        "margin_points": r2(margin),
        "units": "R@k percentage points; gap = LLM-judged arm minus CAS arm",
    }

    # ---------------- (b) benchmark ctrl gaps -----------------------------
    bench_out = {}
    for t in TIERS:
        sec = {}
        for k in KS:
            vals = gap_vals[(t, k)]
            if boot is not None:
                lo, hi = pct_ci(boot["gap"][t][k])
                sec[f"recall@{k}"] = {
                    "estimate": r2(np.mean(vals)), "ci95": [r2(lo), r2(hi)],
                    "per_seed": [r2(v) for v in vals],
                    "level": "per_query_nested"}
            else:
                mu, lo, hi = t_ci(vals)
                sec[f"recall@{k}"] = {
                    "estimate": r2(mu), "ci95": [r2(lo), r2(hi)],
                    "per_seed": [r2(v) for v in vals],
                    "level": "seed_summary"}
        bench_out[t] = sec
    result["benchmark_ctrl_gap_llm_minus_cas"] = bench_out

    # ---------------- (a) real-duplicate ctrl gap -------------------------
    xg = {}
    for k in KS:
        vals = gap_vals[("xling_strict", k)]
        if boot is not None:
            lo, hi = pct_ci(boot["xgap"][k])
            xg[f"recall@{k}"] = {"estimate": r2(np.mean(vals)),
                                 "ci95": [r2(lo), r2(hi)],
                                 "per_seed": [r2(v) for v in vals],
                                 "level": "per_query_nested"}
        else:
            mu, lo, hi = t_ci(vals)
            xg[f"recall@{k}"] = {"estimate": r2(mu),
                                 "ci95": [r2(lo), r2(hi)],
                                 "per_seed": [r2(v) for v in vals],
                                 "level": "seed_summary"}
    result["real_duplicate_ctrl_gap_llm_minus_cas_strict"] = xg

    # ---------------- (c) TOST ---------------------------------------------
    tost = {"summary_level": tost_t(gap_vals[("xling_strict", 1)], margin)}
    if boot is not None:
        tost["per_query_level"] = tost_boot(boot["xgap"][1], margin)
    result["tost_equivalence_real_duplicates"] = tost

    # ---------------- (d) DiD ----------------------------------------------
    did_out = {"primary_tier": "easy",
               "definition": ("benchmark R@1 gap (LLM-CAS) minus "
                              "real-duplicate strict R@1 gap (LLM-CAS)")}
    for t in TIERS:
        did_vals = [gap_vals[(t, 1)][i] - gap_vals[("xling_strict", 1)][i]
                    for i in range(len(seeds))]
        entry = {"per_seed": [r2(v) for v in did_vals]}
        mu, sd, n = mean_sd(did_vals)
        if boot is not None:
            lo, hi = pct_ci(boot["did"][t][1])
            entry.update({"estimate": r2(np.mean(did_vals)),
                          "ci95": [r2(lo), r2(hi)],
                          "level": "per_query_nested",
                          "p_one_sided_did_gt_0":
                              round(float((boot["did"][t][1] <= 0).mean()), 4)})
        else:
            mu, lo, hi = t_ci(did_vals)
            se = sd / math.sqrt(n) if sd else None
            entry.update({"estimate": r2(mu), "ci95": [r2(lo), r2(hi)],
                          "level": "seed_summary",
                          "p_one_sided_did_gt_0":
                              round(t_sf(mu / se, n - 1), 4) if se else None})
        did_out[t] = entry
    result["difference_in_differences"] = did_out

    # ---------------- (b) V2 vs RaDeR-gte easy R@1 -------------------------
    pV2 = os.path.join(args.ranks_dir, "easy_qwen3-0.6b-p2-instr.ranks.jsonl")
    pRa = os.path.join(args.ranks_dir, "easy_rader-gte-qwen2-7b.ranks.jsonl")
    v2sec = {"models": ["qwen3-0.6b-p2-instr (V2, 0.6B)",
                        "rader-gte-qwen2-7b (7B)"]}
    if os.path.exists(pV2) and os.path.exists(pRa):
        dA, _ = read_ranks(pV2)
        dB, _ = read_ranks(pRa)
        qo = sorted(dA)
        if set(dB) != set(qo):
            raise SystemExit("[fatal] V2/RaDeR dump qid sets differ")
        rA = np.array([dA[q] for q in qo])
        rB = np.array([dB[q] for q in qo])
        reps = paired_query_bootstrap(rA, rB, args.n_boot, rng)
        for k in KS:
            lo, hi = pct_ci(reps[k])
            v2sec[f"diff_recall@{k}"] = {
                "estimate": r2(100 * ((rA < k).mean() - (rB < k).mean())),
                "ci95": [r2(lo), r2(hi)], "level": "per_query_paired"}
        v2sec["mcnemar_recall@1"] = mcnemar_exact(rA < 1, rB < 1)
    else:
        eV2 = _first(["eval_easy_qwen3-0.6b-p2-instr.json"])
        eRa = _first(["eval_easy_rader-gte-qwen2-7b.json"])
        if eV2 and eRa:
            a = json.load(open(eV2))["overall"]
            b = json.load(open(eRa))["overall"]
            for k in KS:
                v2sec[f"diff_recall@{k}"] = {
                    "estimate": r2(a[f"recall@{k}"] - b[f"recall@{k}"]),
                    "ci95": None, "level": "summary_point_only"}
            caveats.append(
                "V2-vs-RaDeR-gte CI unavailable: per-query dumps missing "
                "(single runs -- no seed axis to fall back on); "
                "dump_ranks.slurm regenerates both.")
    result["v2_vs_rader_gte_easy"] = v2sec

    # ---------------- extra single-run xling pairs -------------------------
    pairs_out = {}
    for pair in [p for p in args.xling_pairs.split(",") if p.strip()]:
        a, b = pair.split(":")
        pa = os.path.join(args.ranks_dir, f"xling_{a}.ranks.jsonl")
        pb = os.path.join(args.ranks_dir, f"xling_{b}.ranks.jsonl")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            pairs_out[pair] = {"level": "unavailable (dumps missing)"}
            continue
        dA, rowsA = read_ranks(pa, "xling_rank")
        dB, _ = read_ranks(pb, "xling_rank")
        qo = sorted(dA)
        RA = np.array([[dA[q] for q in qo]])
        RB = np.array([[dB[q] for q in qo]])
        grp = cluster_groups(rowsA, qo)
        reps = pair_cluster_bootstrap(RA, RB, grp, args.n_boot, rng)
        sec = {}
        for k in KS:
            lo, hi = pct_ci(reps[k])
            sec[f"diff_recall@{k}"] = {
                "estimate": r2(100 * ((RA < k).mean() - (RB < k).mean())),
                "ci95": [r2(lo), r2(hi)], "level": "cluster_bootstrap"}
        pairs_out[pair] = sec
    result["xling_single_run_pairs_strict"] = pairs_out

    result["caveats"] = caveats + [
        "seed-summary t-intervals with n=3 seeds are wide by construction; "
        "they harden the claim's direction, not its magnitude",
        "nested bootstrap resamples seeds with replacement; with few seeds "
        "the seed-level variance component is captured only coarsely",
        "cluster bootstrap treats the 661 mined duplicate clusters as the "
        "sampling unit (mining precision ~85-90%, see "
        "data/crosslingual_eval/README.md)",
        "hard-tier DiD in raw points is near zero BY CONSTRUCTION (ctrl-CAS "
        "hard R@1 ~0.15); the hard-tier circularity exhibit is the ~60x "
        "RATIO, the point-scale DiD claim lives on the easy tier",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ---------------- paper-ready table ------------------------------------
    W = 78
    print("\n" + "=" * W)
    print(f"RECIPE-MATCHING STATISTICS  (level: {level}; seeds "
          f"{','.join(map(str, seeds))}; B={args.n_boot})")
    print("=" * W)

    def line(label, est, ci, extra=""):
        ci_s = (f"[{ci[0]:7.2f}, {ci[1]:7.2f}]"
                if ci and ci[0] is not None else "        --        ")
        print(f"{label:<44s} {est:>7.2f}  {ci_s}  {extra}")

    print(f"{'quantity (R@1, points, LLM-CAS)':<44s} {'est':>7s}  "
          f"{'95% CI':^18s}")
    print("-" * W)
    for t in TIERS:
        e = bench_out[t]["recall@1"]
        line(f"benchmark {t} gap", e["estimate"], e["ci95"])
    e = xg["recall@1"]
    line("real-duplicate strict gap", e["estimate"], e["ci95"])
    for t in TIERS:
        e = did_out[t]
        star = "  <-- THE FINDING" if t == "easy" else ""
        line(f"DiD ({t} - real-dup)", e["estimate"], e["ci95"],
             f"p(DiD<=0)={e['p_one_sided_did_gt_0']}{star}")
    print("-" * W)
    ts = tost.get("per_query_level") or tost["summary_level"]
    verdict = ("EQUIVALENT" if ts.get("equivalent_at_0.05")
               else "NOT establishable yet")
    print(f"TOST real-dup equivalence  margin=+/-{margin:.2f}pt "
          f"(={args.margin_frac} x easy gap): {verdict}")
    if "p_tost" in ts:
        print(f"  two one-sided t-tests: p_tost={ts['p_tost']} "
              f"(90% CI {ts['ci90']})")
    if "ci90" in ts and "p_tost" not in ts:
        print(f"  bootstrap 90% CI {ts['ci90']} vs (-{margin:.2f}, "
              f"+{margin:.2f})")
    dv = v2sec.get("diff_recall@1")
    if dv:
        ci_s = f"CI {dv['ci95']}" if dv.get("ci95") else \
            "CI needs per-query dumps (dump_ranks.slurm)"
        print(f"V2 - RaDeR-gte easy R@1: {dv['estimate']:+.2f}  {ci_s}")
    print("=" * W)
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
