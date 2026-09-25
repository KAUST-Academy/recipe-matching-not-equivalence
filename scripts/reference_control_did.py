#!/usr/bin/env python3
"""2026-09-03: the recipe arm against the RECIPE-FREE REFERENCE cell (D4
positives + verified negatives, E-R3 d4casnegs), and the decomposition of the
headline 45.33-point easy-tier gap into the part any LLM-rewrite training buys
and the part only the recipe buys, reported as gaps and a difference-in-differences
over eight seeds.

Contrasts (A - B), each on the easy tier (R@1, 15,000 queries) and on real
duplicates (strict cross-lingual R@1, 393 queries), with the difference-in-
differences DiD = easy gap - real gap:
  recipe - reference     ctrl-llm - fact-d4casnegs   the RECIPE-SPECIFIC share
  reference - verified   fact-d4casnegs - ctrl-cas   deep LLM paraphrase + negatives
  recipe - verified      ctrl-llm - ctrl-cas         the published headline (check)
The first two sum to the third by construction.

Uncertainty: nested bootstrap, B draws. Easy queries are resampled with
replacement (shared across arms, paired); real-duplicate CLUSTERS are
resampled with replacement and every query of a sampled cluster is kept
(the cluster bootstrap of Appendix A); training seeds are resampled
independently per arm (ctrl arms eight, the reference cell however many
seeds have dumps -- three at first, eight once E-R6 lands). Also reported:
the seed-paired values on the seeds every arm shares, and, if same-language
dumps exist (E-R5), the same contrasts on the primary same-language slice.
Writes results/reference_control_did.json.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = os.path.join(PROJECT, "results", "ranks")
ARMS = {"ctrl-llm": "ctrl-llm", "ctrl-cas": "ctrl-cas", "ref": "fact-d4casnegs"}
CONTRASTS = [("recipe - reference", "ctrl-llm", "ref"),
             ("reference - verified", "ref", "ctrl-cas"),
             ("recipe - verified", "ctrl-llm", "ctrl-cas")]


def load(path, field):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["qid"]] = (r[field] == 0, r.get("cluster_id"))
    return out


def matrix(tag, seeds, prefix, field, qorder):
    rows = []
    for s in seeds:
        p = os.path.join(RANKS, f"{prefix}_{tag}-s{s}.ranks.jsonl")
        if not os.path.exists(p):
            continue
        d = load(p, field)
        rows.append([d[q][0] for q in qorder])
    return np.array(rows, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    ap.add_argument("--output", default=os.path.join(PROJECT, "results",
                                                     "reference_control_did.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(0)

    # query orders and cluster groups from the ctrl-llm seed-42 dumps
    e42 = load(os.path.join(RANKS, "easy_ctrl-llm-s42.ranks.jsonl"), "gold_rank")
    x42 = load(os.path.join(RANKS, "xling_ctrl-llm-s42.ranks.jsonl"), "xling_rank")
    eq = sorted(e42)
    xq = sorted(x42)
    groups = defaultdict(list)
    for i, q in enumerate(xq):
        groups[x42[q][1]].append(i)
    xgroups = [np.array(v) for v in groups.values()]

    E = {a: matrix(t, seeds, "easy", "gold_rank", eq) for a, t in ARMS.items()}
    X = {a: matrix(t, seeds, "xling", "xling_rank", xq) for a, t in ARMS.items()}
    have_same = os.path.exists(os.path.join(PROJECT, "data", "samelang_eval", "queries.jsonl"))
    S, sgroups, sq = {}, [], []
    if have_same:
        qs = [json.loads(l) for l in open(os.path.join(PROJECT, "data", "samelang_eval",
                                                       "queries.jsonl"), encoding="utf-8")]
        sq = sorted(q["_id"] for q in qs if not q["metadata"]["exact_text_cluster"]
                    and q["metadata"]["clean_of_training"])
        cl = {q["_id"]: q["metadata"]["cluster_id"] for q in qs}
        g = defaultdict(list)
        for i, q in enumerate(sq):
            g[cl[q]].append(i)
        sgroups = [np.array(v) for v in g.values()]
        for a, t in ARMS.items():
            try:
                S[a] = matrix(t, seeds, "samelang", "same_rank", sq)
            except KeyError:
                pass
        S = {a: m for a, m in S.items() if m.size}
    for a in ARMS:
        print(f"[data] {a}: easy seeds {E[a].shape[0]}, xling seeds {X[a].shape[0]}"
              + (f", samelang seeds {S[a].shape[0]}" if a in S else ""))
        if E[a].shape[0] == 0 or X[a].shape[0] == 0:
            raise SystemExit(f"missing dumps for {a}")

    def pt(M, a):
        return 100 * M[a].mean()

    def boot(A, B, MA_e, MB_e, MA_x, MB_x, MA_s=None, MB_s=None):
        ne, nx = MA_e.shape[1], MA_x.shape[1]
        reps = np.empty((args.boot, 3 if MA_s is None else 4))
        for b in range(args.boot):
            qi = rng.integers(0, ne, ne)
            gi = rng.integers(0, len(xgroups), len(xgroups))
            xi = np.concatenate([xgroups[i] for i in gi])
            sa_e = rng.integers(0, MA_e.shape[0], MA_e.shape[0])
            sb_e = rng.integers(0, MB_e.shape[0], MB_e.shape[0])
            sa_x = rng.integers(0, MA_x.shape[0], MA_x.shape[0])
            sb_x = rng.integers(0, MB_x.shape[0], MB_x.shape[0])
            ge = 100 * (MA_e[sa_e][:, qi].mean() - MB_e[sb_e][:, qi].mean())
            gx = 100 * (MA_x[sa_x][:, xi].mean() - MB_x[sb_x][:, xi].mean())
            reps[b, :3] = (ge, gx, ge - gx)
            if MA_s is not None:
                si = rng.integers(0, len(sgroups), len(sgroups))
                ssi = np.concatenate([sgroups[i] for i in si])
                sa_s = rng.integers(0, MA_s.shape[0], MA_s.shape[0])
                sb_s = rng.integers(0, MB_s.shape[0], MB_s.shape[0])
                reps[b, 3] = ge - 100 * (MA_s[sa_s][:, ssi].mean() - MB_s[sb_s][:, ssi].mean())
        return reps

    def ci(v):
        return [round(float(np.percentile(v, p)), 2) for p in (2.5, 97.5)]

    out = {"generated_by": "scripts/reference_control_did.py", "bootstrap": args.boot,
           "seeds_with_dumps": {a: {"easy": int(E[a].shape[0]), "xling": int(X[a].shape[0]),
                                    "samelang": int(S[a].shape[0]) if a in S else 0}
                                for a in ARMS},
           "levels": {a: {"easy": round(pt(E, a), 2), "real_dup": round(pt(X, a), 2),
                          **({"samelang_primary": round(pt(S, a), 2)} if a in S else {})}
                      for a in ARMS},
           "contrasts": {}}
    for name, A, B in CONTRASTS:
        same = A in S and B in S
        reps = boot(A, B, E[A], E[B], X[A], X[B], S[A] if same else None, S[B] if same else None)
        ge, gx = pt(E, A) - pt(E, B), pt(X, A) - pt(X, B)
        rec = {"easy_gap": round(ge, 2), "easy_gap_ci95": ci(reps[:, 0]),
               "real_dup_gap": round(gx, 2), "real_dup_gap_ci95": ci(reps[:, 1]),
               "did": round(ge - gx, 2), "did_ci95": ci(reps[:, 2]),
               "did_frac_positive": round(float((reps[:, 2] > 0).mean()), 4)}
        if same:
            gs = pt(S, A) - pt(S, B)
            rec.update({"samelang_gap": round(gs, 2),
                        "did_samelang": round(ge - gs, 2), "did_samelang_ci95": ci(reps[:, 3])})
        shared = min(E[A].shape[0], E[B].shape[0], X[A].shape[0], X[B].shape[0])
        rec["seed_paired_first_%d" % shared] = {
            "easy": [round(100 * float(E[A][i].mean() - E[B][i].mean()), 2) for i in range(shared)],
            "real_dup": [round(100 * float(X[A][i].mean() - X[B][i].mean()), 2) for i in range(shared)],
            "did": [round(100 * float((E[A][i].mean() - E[B][i].mean())
                                      - (X[A][i].mean() - X[B][i].mean())), 2)
                    for i in range(shared)]}
        out["contrasts"][name] = rec
        print(f"[{name:22s}] easy {ge:+6.2f} {rec['easy_gap_ci95']}  real {gx:+6.2f} "
              f"{rec['real_dup_gap_ci95']}  DiD {ge - gx:+6.2f} {rec['did_ci95']}"
              + (f"  DiD_same {rec['did_samelang']:+6.2f} {rec['did_samelang_ci95']}" if same else ""))
    a, b, c = (out["contrasts"][n]["did"] for n, _, _ in CONTRASTS)
    out["additivity_check"] = {"recipe-reference + reference-verified": round(a + b, 2),
                               "recipe-verified": c}

    # ---- every tier and cutoff, for Table 1's reference column ---------------
    def tier_matrix(tag, prefix, qorder, k):
        rows = []
        for s in seeds:
            p = os.path.join(RANKS, f"{prefix}_{tag}-s{s}.ranks.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                d = {json.loads(l)["qid"]: json.loads(l)["gold_rank"] for l in f}
            rows.append([d[q] < k for q in qorder])
        return np.array(rows, float)

    out["tiers"] = {}
    for tier in ("easy", "medium", "hard"):
        for k in (1, 5):
            M = {a: tier_matrix(t, tier, eq, k) for a, t in ARMS.items()}
            if any(m.size == 0 for m in M.values()):
                continue
            key = f"{tier}_R@{k}"
            rec = {"levels": {a: round(100 * float(M[a].mean()), 2) for a in ARMS}}
            for name, A, B in CONTRASTS:
                reps = np.empty(args.boot)
                ne = M[A].shape[1]
                for b in range(args.boot):
                    qi = rng.integers(0, ne, ne)
                    sa = rng.integers(0, M[A].shape[0], M[A].shape[0])
                    sb = rng.integers(0, M[B].shape[0], M[B].shape[0])
                    reps[b] = 100 * (M[A][sa][:, qi].mean() - M[B][sb][:, qi].mean())
                g = 100 * float(M[A].mean() - M[B].mean())
                rec[name] = {"gap": round(g, 2), "ci95": ci(reps)}
            out["tiers"][key] = rec
            print(f"[{key:11s}] " + "  ".join(
                f"{n}: {rec[n]['gap']:+6.2f} {rec[n]['ci95']}" for n, _, _ in CONTRASTS))
    # ---- 2026-09-04: the recipe-specific share
    # at MATCHED negatives. The reference carries the verified arm's negatives
    # (1.28 per row); the count-matched LLM-negative cell (E-R6, 1.20 per row) and
    # the full-volume cell (2.64 per row, the recipe arm's own) hold the negatives'
    # provenance fixed instead. Same nested bootstrap; a SEPARATE generator so the
    # contrasts above are unchanged by this addition (their intervals are quoted).
    EXTRA_ARMS = {"llmnegs_matched": "fact-d4llmnegs", "llmnegs_full": "fact-d4llmnegsfull",
                  # E-R7 / E-R8 / E-R9 (2026-09-04); appended AFTER the four contrasts
                  # above so their draws are unchanged; skipped while dumps are missing
                  "unrelnegs_matched": "fact-d4unrelnegs", "unrelnegs_full": "fact-d4unrelnegsfull",
                  "cas_llmnegs": "fact-casllmnegs", "bt_casnegs": "fact-btcasnegs"}
    EXTRA_CONTRASTS = [("recipe - llmnegs_matched", "ctrl-llm", "llmnegs_matched"),
                       ("recipe - llmnegs_full", "ctrl-llm", "llmnegs_full"),
                       ("llmnegs_matched - reference", "llmnegs_matched", "ref"),
                       ("llmnegs_full - reference", "llmnegs_full", "ref"),
                       # R7-a/b: only the PROMPT that wrote the near-misses differs
                       ("unrelnegs_matched - llmnegs_matched", "unrelnegs_matched", "llmnegs_matched"),
                       ("unrelnegs_full - llmnegs_full", "unrelnegs_full", "llmnegs_full"),
                       ("recipe - unrelnegs_matched", "ctrl-llm", "unrelnegs_matched"),
                       ("recipe - unrelnegs_full", "ctrl-llm", "unrelnegs_full"),
                       # R8-a/b: near-copy positives, negatives' provenance swapped
                       ("cas_llmnegs - verified", "cas_llmnegs", "ctrl-cas"),
                       ("recipe - cas_llmnegs", "ctrl-llm", "cas_llmnegs"),
                       # R9: positives' authorship swapped at the reference's negatives
                       ("reference - bt_casnegs", "ref", "bt_casnegs"),
                       ("bt_casnegs - verified", "bt_casnegs", "ctrl-cas"),
                       ("recipe - bt_casnegs", "ctrl-llm", "bt_casnegs")]
    rng = np.random.default_rng(1)
    for a, t in EXTRA_ARMS.items():
        E[a] = matrix(t, seeds, "easy", "gold_rank", eq)
        X[a] = matrix(t, seeds, "xling", "xling_rank", xq)
        if have_same:
            try:
                m = matrix(t, seeds, "samelang", "same_rank", sq)
                if m.size:
                    S[a] = m
            except KeyError:
                pass
    out["matched_negatives"] = {"note": "recipe-specific share with the negatives' provenance held "
                                        "at the recipe arm's own near-misses (E-R6 cells, three seeds); "
                                        "separate RNG stream from the contrasts above",
                                "levels": {}, "contrasts": {}}
    for a in EXTRA_ARMS:
        if E[a].shape[0] == 0 or X[a].shape[0] == 0:
            print(f"[matched-negatives] missing dumps for {a}; skipped")
            continue
        out["matched_negatives"]["levels"][a] = {
            "easy": round(pt(E, a), 2), "real_dup": round(pt(X, a), 2),
            "n_seeds": int(E[a].shape[0]),
            **({"samelang_primary": round(pt(S, a), 2)} if a in S else {})}
    for name, A, B in EXTRA_CONTRASTS:
        if A not in out["matched_negatives"]["levels"] and A in EXTRA_ARMS:
            continue
        if B not in out["matched_negatives"]["levels"] and B in EXTRA_ARMS:
            continue
        same = A in S and B in S
        reps = boot(A, B, E[A], E[B], X[A], X[B], S[A] if same else None, S[B] if same else None)
        ge, gx = pt(E, A) - pt(E, B), pt(X, A) - pt(X, B)
        rec = {"easy_gap": round(ge, 2), "easy_gap_ci95": ci(reps[:, 0]),
               "real_dup_gap": round(gx, 2), "real_dup_gap_ci95": ci(reps[:, 1]),
               "did": round(ge - gx, 2), "did_ci95": ci(reps[:, 2]),
               "did_frac_positive": round(float((reps[:, 2] > 0).mean()), 4)}
        if same:
            gs = pt(S, A) - pt(S, B)
            rec.update({"samelang_gap": round(gs, 2),
                        "did_samelang": round(ge - gs, 2), "did_samelang_ci95": ci(reps[:, 3])})
        shared = min(E[A].shape[0], E[B].shape[0], X[A].shape[0], X[B].shape[0])
        rec["seed_paired_first_%d" % shared] = {
            "easy": [round(100 * float(E[A][i].mean() - E[B][i].mean()), 2) for i in range(shared)],
            "real_dup": [round(100 * float(X[A][i].mean() - X[B][i].mean()), 2) for i in range(shared)]}
        # hard tier R@1 for the same contrast (query + seed bootstrap)
        MA = tier_matrix(ARMS.get(A) or EXTRA_ARMS[A], "hard", eq, 1)
        MB = tier_matrix(ARMS.get(B) or EXTRA_ARMS[B], "hard", eq, 1)
        hreps = np.empty(args.boot)
        for b in range(args.boot):
            qi = rng.integers(0, MA.shape[1], MA.shape[1])
            sa = rng.integers(0, MA.shape[0], MA.shape[0])
            sb = rng.integers(0, MB.shape[0], MB.shape[0])
            hreps[b] = 100 * (MA[sa][:, qi].mean() - MB[sb][:, qi].mean())
        rec["hard_gap"] = round(100 * float(MA.mean() - MB.mean()), 2)
        rec["hard_gap_ci95"] = ci(hreps)
        out["matched_negatives"]["contrasts"][name] = rec
        print(f"[{name:28s}] easy {ge:+6.2f} {rec['easy_gap_ci95']}  real {gx:+6.2f} "
              f"{rec['real_dup_gap_ci95']}  DiD {ge - gx:+6.2f} {rec['did_ci95']}  "
              f"hard {rec['hard_gap']:+6.2f} {rec['hard_gap_ci95']}"
              + (f"  DiD_same {rec['did_samelang']:+6.2f} {rec['did_samelang_ci95']}" if same else ""))

    # ---- E-R10 .. E-R13 (2026-09-06): a THIRD generator so every
    # interval above reproduces byte-for-byte; missing dumps skip a contrast.
    R3_ARMS = {"d5_alone": "fact-d5nonegs", "d5_ref": "fact-d5casnegs", "d1split": "fact-d1split",
               "d1twojudge": "fact-d1twojudge", "bt2_casnegs": "fact-bt2casnegs"}
    R3_CONTRASTS = [("recipe - d5_ref", "ctrl-llm", "d5_ref"),
                    ("d5_ref - verified", "d5_ref", "ctrl-cas"),
                    ("d5_ref - reference", "d5_ref", "ref"),
                    ("d1twojudge - recipe", "d1twojudge", "ctrl-llm"),
                    ("d1split - recipe", "d1split", "ctrl-llm"),
                    ("reference - bt2_casnegs", "ref", "bt2_casnegs"),
                    ("bt2_casnegs - verified", "bt2_casnegs", "ctrl-cas")]
    rng = np.random.default_rng(2)
    for a, t in R3_ARMS.items():
        E[a] = matrix(t, seeds, "easy", "gold_rank", eq)
        X[a] = matrix(t, seeds, "xling", "xling_rank", xq)
        if have_same:
            try:
                m3 = matrix(t, seeds, "samelang", "same_rank", sq)
                if m3.size:
                    S[a] = m3
            except KeyError:
                pass
    out["review3"] = {"note": "E-R10 to E-R13 cells; third RNG stream; three seeds per cell",
                      "levels": {}, "contrasts": {}}
    for a in R3_ARMS:
        if E[a].shape[0] == 0 or X[a].shape[0] == 0:
            print(f"[review3] missing dumps for {a}; skipped")
            continue
        out["review3"]["levels"][a] = {"easy": round(pt(E, a), 2), "real_dup": round(pt(X, a), 2),
                                       "n_seeds": int(E[a].shape[0]),
                                       **({"samelang_primary": round(pt(S, a), 2)} if a in S else {})}
    ALL = {**ARMS, **EXTRA_ARMS, **R3_ARMS}
    for name, A, B in R3_CONTRASTS:
        if (A in R3_ARMS and A not in out["review3"]["levels"]) or (B in R3_ARMS and B not in out["review3"]["levels"]):
            continue
        same = A in S and B in S
        reps = boot(A, B, E[A], E[B], X[A], X[B], S[A] if same else None, S[B] if same else None)
        ge, gx = pt(E, A) - pt(E, B), pt(X, A) - pt(X, B)
        rec = {"easy_gap": round(ge, 2), "easy_gap_ci95": ci(reps[:, 0]),
               "real_dup_gap": round(gx, 2), "real_dup_gap_ci95": ci(reps[:, 1]),
               "did": round(ge - gx, 2), "did_ci95": ci(reps[:, 2]),
               "did_frac_positive": round(float((reps[:, 2] > 0).mean()), 4)}
        if same:
            gs = pt(S, A) - pt(S, B)
            rec.update({"samelang_gap": round(gs, 2),
                        "did_samelang": round(ge - gs, 2), "did_samelang_ci95": ci(reps[:, 3])})
        MA = tier_matrix(ALL[A], "hard", eq, 1)
        MB = tier_matrix(ALL[B], "hard", eq, 1)
        hreps = np.empty(args.boot)
        for b in range(args.boot):
            qi = rng.integers(0, MA.shape[1], MA.shape[1])
            sa = rng.integers(0, MA.shape[0], MA.shape[0])
            sb = rng.integers(0, MB.shape[0], MB.shape[0])
            hreps[b] = 100 * (MA[sa][:, qi].mean() - MB[sb][:, qi].mean())
        rec["hard_gap"] = round(100 * float(MA.mean() - MB.mean()), 2)
        rec["hard_gap_ci95"] = ci(hreps)
        out["review3"]["contrasts"][name] = rec
        print(f"[{name:24s}] easy {ge:+6.2f} {rec['easy_gap_ci95']}  real {gx:+6.2f} "
              f"{rec['real_dup_gap_ci95']}  DiD {ge - gx:+6.2f} {rec['did_ci95']}  "
              f"hard {rec['hard_gap']:+6.2f} {rec['hard_gap_ci95']}"
              + (f"  DiD_same {rec['did_samelang']:+6.2f} {rec['did_samelang_ci95']}" if same else ""))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
