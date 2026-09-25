#!/usr/bin/env python3
"""E-R1: does "both trained arms sit at-or-below
the base on real duplicates" survive on the slices of the cross-lingual eval
that were mined LEAST by surface overlap?

The mining pipeline (scripts/mine_duplicates.py) finds most pairs through
shared rare formula 4-grams, which is exactly the signal a surface-matching
embedder exploits; a reviewer can therefore worry that the eval is biased
against models that moved off surface matching. This script slices the
EXISTING per-query rank dumps (results/ranks/xling_*.ranks.jsonl; no new GPU
work) three ways:

  method      per-query mining provenance from queries.jsonl metadata:
              answer-corroborated (any supporting pair V4_answer_math /
              V5_answer_tag), text-mined (any exact_text / V1_text), else
              formula-only (V2_rare_math).
  rcont       recomputed query<->gold rare-formula-4-gram containment
              (the miner's own statistic: IDF mass of shared rare grams /
              (shared + min-side exclusive)), max over cross-lingual golds,
              median split low/high.
  mathdens    fraction of query characters inside $...$ math, median split.

Readout per slice: strict cross-lingual R@1 for base / ctrl-CAS / ctrl-LLM
(8 seeds), with a nested bootstrap CI (queries + seeds, paired on queries)
for each arm-minus-base gap. Writes results/xling_slice_analysis.json.
"""
import argparse, json, math, os, re, sys
from collections import Counter

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from mine_duplicates import math_sig  # noqa: E402  (char 4-grams, stripped LaTeX)

RANKS = os.path.join(PROJECT, "results", "ranks")
EVAL = os.path.join(PROJECT, "data", "crosslingual_eval")
MATH_SPAN_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)

ANSWER_METHODS = {"V4_answer_math", "V5_answer_tag"}
TEXT_METHODS = {"exact_text", "V1_text"}


def load_dump(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["qid"]] = r["xling_rank"]
    return out


def method_group(meta):
    methods = set(meta.get("methods") or [])
    if not methods:
        for p in meta.get("supporting_pairs") or []:
            methods.add(p.get("method"))
    if methods & ANSWER_METHODS:
        return "answer_corroborated"
    if methods & TEXT_METHODS:
        return "text_mined"
    return "formula_only"


def math_char_frac(text):
    inside = sum(len(m.group(0)) for m in MATH_SPAN_RE.finditer(text or ""))
    return inside / max(1, len(text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--rare-df-max", type=int, default=20,
                    help="corpus df ceiling for a formula 4-gram to count as "
                         "rare (mine_duplicates.py's rshared convention)")
    ap.add_argument("--output",
                    default=os.path.join(PROJECT, "results",
                                         "xling_slice_analysis.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(0)

    # ---- corpus texts + rare-gram document frequencies --------------------
    corpus = pd.read_parquet(os.path.join(PROJECT, "data",
                                          "mathnet_corpus.parquet"))
    text_of = dict(zip(corpus["id"], corpus["problem_markdown"]))
    print(f"[df] computing formula-4-gram document frequencies over "
          f"{len(corpus)} docs ...", flush=True)
    df = Counter()
    sigs = {}
    for cid, t in text_of.items():
        s = math_sig(t)
        sigs[cid] = s
        df.update(s)
    n_docs = len(text_of)
    idf = {g: math.log(n_docs / d) for g, d in df.items()
           if 2 <= d <= args.rare_df_max}
    print(f"[df] {len(idf)} rare grams (2 <= df <= {args.rare_df_max})",
          flush=True)

    def rare_mass(grams):
        return sum(idf.get(g, 0.0) for g in grams)

    # ---- per-query slice keys --------------------------------------------
    queries = [json.loads(l) for l in
               open(os.path.join(EVAL, "queries.jsonl"), encoding="utf-8")]
    qinfo = {}
    for q in queries:
        meta = q["metadata"]
        qid = q["_id"]
        qsig = sigs.get(qid, math_sig(q["text"]))
        best_rcont, best_rshared = 0.0, 0.0
        for g, gl in zip(meta["gold_ids"], meta["gold_langs"]):
            if gl == meta["lang"]:
                continue
            gsig = sigs.get(g)
            if gsig is None:
                continue
            shared = rare_mass(qsig & gsig)
            excl_min = min(rare_mass(qsig - gsig), rare_mass(gsig - qsig))
            denom = shared + excl_min
            if denom > 0 and shared / denom > best_rcont:
                best_rcont = shared / denom
            best_rshared = max(best_rshared, shared)
        qinfo[qid] = {
            "method": method_group(meta),
            "rcont": best_rcont,
            "rshared": best_rshared,
            "mathdens": math_char_frac(q["text"]),
        }

    # ---- rank dumps -------------------------------------------------------
    base = load_dump(os.path.join(RANKS, "xling_qwen3-0.6b-base.ranks.jsonl"))
    model_seeds = {"ctrl-cas": seeds, "ctrl-llm": seeds,
                   "dose-paraphrase": [42, 43, 44],
                   "dose-style": [42, 43, 44],
                   "dose-unrelated": [42, 43, 44]}
    arms = {}
    for arm, ss in model_seeds.items():
        try:
            arms[arm] = {s: load_dump(os.path.join(
                RANKS, f"xling_{arm}-s{s}.ranks.jsonl")) for s in ss}
        except FileNotFoundError:
            print(f"[warn] dumps missing for {arm}; skipped", flush=True)
    qids = [q["_id"] for q in queries if q["_id"] in base]
    print(f"[data] {len(qids)} queries with dumped ranks", flush=True)

    rc = np.array([qinfo[q]["rcont"] for q in qids])
    rs = np.array([qinfo[q]["rshared"] for q in qids])
    md = np.array([qinfo[q]["mathdens"] for q in qids])
    md_med = float(np.median(md))
    # rcont saturates at 1.0 for ~80% of queries (verified duplicates share
    # nearly all their rare formula mass), so a median split degenerates;
    # slice on sub-maximal containment and on quartiles of the ABSOLUTE
    # shared rare-IDF mass instead.
    rs_q25, rs_q75 = (float(np.percentile(rs, 25)),
                      float(np.percentile(rs, 75)))

    slices = {
        "all": np.ones(len(qids), bool),
        "method=answer_corroborated": np.array(
            [qinfo[q]["method"] == "answer_corroborated" for q in qids]),
        "method=text_mined": np.array(
            [qinfo[q]["method"] == "text_mined" for q in qids]),
        "method=formula_only": np.array(
            [qinfo[q]["method"] == "formula_only" for q in qids]),
        "rcont<1.0": rc < 1.0, "rcont<0.9": rc < 0.9,
        "rcont=1.0": rc >= 1.0,
        "rshared=bottom_quartile": rs <= rs_q25,
        "rshared=top_quartile": rs >= rs_q75,
        "mathdens=low": md <= md_med, "mathdens=high": md > md_med,
    }

    base_hit = np.array([base[q] == 0 for q in qids], float)
    arm_hit = {a: np.stack([[arms[a][s][q] == 0 for q in qids]
                            for s in model_seeds[a]]).astype(float)
               for a in arms}

    def boot_gap(mask, arm):
        idx = np.flatnonzero(mask)
        gaps = np.empty(args.boot)
        n_s = arm_hit[arm].shape[0]
        for b in range(args.boot):
            qi = rng.choice(idx, len(idx), replace=True)
            si = rng.integers(0, n_s, n_s)
            gaps[b] = arm_hit[arm][si][:, qi].mean() - base_hit[qi].mean()
        return [round(100 * float(np.percentile(gaps, p)), 2)
                for p in (2.5, 97.5)]

    out = {"generated_by": "scripts/xling_slice_analysis.py",
           "metric": "strict cross-lingual R@1 (xling_rank==0), percent",
           "seeds": seeds, "n_queries": len(qids),
           "rare_df_max": args.rare_df_max, "bootstrap": args.boot,
           "cut_points": {"mathdens_median": round(md_med, 4),
                          "rshared_q25": round(rs_q25, 2),
                          "rshared_q75": round(rs_q75, 2)},
           "slices": {}}
    for name, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"n_queries": n,
               "base": round(100 * float(base_hit[mask].mean()), 2)}
        for a in arms:
            per_seed = arm_hit[a][:, mask].mean(axis=1)
            row[a] = {"mean": round(100 * float(per_seed.mean()), 2),
                      "per_seed_min": round(100 * float(per_seed.min()), 2),
                      "per_seed_max": round(100 * float(per_seed.max()), 2),
                      "gap_vs_base": round(100 * float(
                          per_seed.mean() - base_hit[mask].mean()), 2),
                      "gap_vs_base_ci95": boot_gap(mask, a)}
        out["slices"][name] = row
        cells = "  ".join(f"{a}={row[a]['mean']:.2f}({row[a]['gap_vs_base']:+.2f})"
                          for a in arms)
        print(f"[slice] {name:26s} n={n:3d} base={row['base']:6.2f}  {cells}",
              flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
