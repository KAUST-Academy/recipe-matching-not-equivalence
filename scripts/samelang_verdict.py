#!/usr/bin/env python3
"""E-R5 verdict (written before any same-language number existed; bands in
scripts/samelang_eval.slurm). Reads results/ranks/samelang_<tag>.ranks.jsonl
for every model, slices by the flags of data/samelang_eval/queries.jsonl and
writes results/samelang_verdict.json.

Slices (queries):
  primary        non-exact cluster AND clean of every training pair file
  primary_en     primary, English queries only
  leaked         non-exact, at least one cluster member in a training file
  exact          exact-text clusters (near-byte-identical reprints; ceiling)
  all            everything
Readout per arm and slice: same-language R@1 (same_rank == 0), mean over
seeds; gap vs the untrained base with a 95% nested bootstrap over duplicate
CLUSTERS (queries within cluster kept together) and training seeds.
P-R5: DiD_same = easy-tier arm gap (per-seed values from
results/ctrl_seed_stats.json, seeds resampled) - same-language gap on the
primary slice (clusters x seeds resampled).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = os.path.join(PROJECT, "results", "ranks")
EVAL = os.path.join(PROJECT, "data", "samelang_eval")
ARMS = {                      # arm -> regex over tags; seeds captured as sNN
    "base": r"^qwen3-0\.6b-base$",
    "base+instr": r"^qwen3-0\.6b-base-instr$",
    "ctrl-cas": r"^ctrl-cas-s\d+$",
    "ctrl-llm": r"^ctrl-llm-s\d+$",
    "D2": r"^dose-paraphrase-s\d+$",
    "D3": r"^dose-style-s\d+$",
    "D4": r"^dose-unrelated-s\d+$",
    "d4casnegs": r"^fact-d4casnegs-s\d+$",
    "casnonegs": r"^fact-casnonegs-s\d+$",
    "d4llmnegs": r"^fact-d4llmnegs-s\d+$",
    "d4llmnegsfull": r"^fact-d4llmnegsfull-s\d+$",
    "negmatched-cas": r"^ctrl-cas-negmatched-6145-s\d+$",
    "v2gate-cas": r"^ctrl-cas-v2gate-4101-s\d+$",
    "v2gate-llm": r"^ctrl-llm-v2gate-4101-s\d+$",
    "cleanfull-cas": r"^cleanfull-cas-s\d+$",
    "cleanfull-llm": r"^cleanfull-llm-s\d+$",
    "V1": r"^qwen3-0\.6b-p2-mixed(-s\d+)?$",
    "V2": r"^qwen3-0\.6b-p2-instr(-s\d+)?$",
    "phase1-pureCAS": r"^qwen3-0\.6b-cas-cmnrl$",
    "soup0.3": r"^soup-a0\.3$", "soup0.5": r"^soup-a0\.5$", "soup0.7": r"^soup-a0\.7$",
    "meld9": r"^meld-attack-meld9(-s\d+)?$",
    "meld-heldout": r"^meld-attack-heldout(-s\d+)?$",
    "saber-mixed": r"^saber-attack$",
    "saber-doc": r"^saber-doc-4733(-s\d+)?$",
    "saber-summary": r"^saber-label-4733(-s\d+)?$",
    "4B-base": r"^qwen3-embedding-4b$",
    "4B-ctrl-cas": r"^ctrl-cas-4b$", "4B-ctrl-llm": r"^ctrl-llm-4b$",
}
# E-R7 / E-R8 / E-R9 (2026-09-04). Scored AFTER the arms above
# and the P-R5 verdict, under a SECOND random generator, so every interval of the
# arms above reproduces byte-for-byte (the paper quotes them); missing dumps skip.
NEW_ARMS = {
    "d4unrelnegs": r"^fact-d4unrelnegs-s\d+$",
    "d4unrelnegsfull": r"^fact-d4unrelnegsfull-s\d+$",
    "casllmnegs": r"^fact-casllmnegs-s\d+$",
    "btcasnegs": r"^fact-btcasnegs-s\d+$",
}
PASS_BAR = 22.67   # half of the 45.33 easy-tier gap ("most")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--output", default=os.path.join(PROJECT, "results",
                                                     "samelang_verdict.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    queries = [json.loads(l) for l in open(os.path.join(EVAL, "queries.jsonl"),
                                           encoding="utf-8")]
    qids = [q["_id"] for q in queries]
    meta = {q["_id"]: q["metadata"] for q in queries}
    slices = {
        "primary": [q for q in qids if not meta[q]["exact_text_cluster"]
                    and meta[q]["clean_of_training"]],
        "primary_en": [q for q in qids if not meta[q]["exact_text_cluster"]
                       and meta[q]["clean_of_training"] and meta[q]["lang"] == "en"],
        "leaked": [q for q in qids if not meta[q]["exact_text_cluster"]
                   and not meta[q]["clean_of_training"]],
        "exact": [q for q in qids if meta[q]["exact_text_cluster"]],
        "all": list(qids),
    }

    dumps = {}
    for p in glob.glob(os.path.join(RANKS, "samelang_*.ranks.jsonl")):
        tag = os.path.basename(p)[len("samelang_"):-len(".ranks.jsonl")]
        d = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[r["qid"]] = r["same_rank"]
        if len(d) == len(qids):
            dumps[tag] = d
    print(f"[data] {len(dumps)} model dumps, {len(qids)} queries")

    hits = {}                                  # arm -> (n_seeds, n_q) 0/1
    seeds_of = {}
    for arm, rx in ARMS.items():
        tags = sorted(t for t in dumps if re.match(rx, t))
        if not tags:
            continue
        hits[arm] = np.stack([[dumps[t][q] == 0 for q in qids] for t in tags]).astype(float)
        seeds_of[arm] = tags
    if "base" not in hits:
        raise SystemExit("base dump missing; run the driver first")

    def idx_of(qs):
        pos = {q: i for i, q in enumerate(qids)}
        return np.array([pos[q] for q in qs])

    def cluster_groups(qs):
        g = defaultdict(list)
        for q in qs:
            g[meta[q]["cluster_id"]].append(q)
        return [idx_of(v) for v in g.values()]

    def boot_gap(arm, other, qs):
        """arm - other on slice qs; clusters x seeds resampled; other='base'
        or another arm."""
        groups = cluster_groups(qs)
        ha, ho = hits[arm], hits[other]
        out = np.empty(args.boot)
        for b in range(args.boot):
            gi = rng.integers(0, len(groups), len(groups))
            qi = np.concatenate([groups[i] for i in gi])
            sa = rng.integers(0, ha.shape[0], ha.shape[0])
            so = rng.integers(0, ho.shape[0], ho.shape[0])
            out[b] = ha[sa][:, qi].mean() - ho[so][:, qi].mean()
        return out

    out = {"generated_by": "scripts/samelang_verdict.py", "bootstrap": args.boot,
           "slices": {k: len(v) for k, v in slices.items()},
           "seeds_of": seeds_of, "results": {}}
    for sname, qs in slices.items():
        ii = idx_of(qs)
        row = {"n_queries": len(qs),
               "n_clusters": len({meta[q]["cluster_id"] for q in qs})}
        base_mean = 100 * hits["base"][:, ii].mean()
        row["base"] = round(float(base_mean), 2)
        for arm in hits:
            if arm == "base":
                continue
            per_seed = 100 * hits[arm][:, ii].mean(axis=1)
            reps = 100 * boot_gap(arm, "base", qs)
            row[arm] = {"mean": round(float(per_seed.mean()), 2),
                        "n_seeds": int(len(per_seed)),
                        "per_seed": [round(float(x), 2) for x in per_seed],
                        "gap_vs_base": round(float(per_seed.mean() - base_mean), 2),
                        "gap_vs_base_ci95": [round(float(np.percentile(reps, p)), 2)
                                             for p in (2.5, 97.5)]}
        if "ctrl-cas" in hits and "ctrl-llm" in hits:
            reps = 100 * boot_gap("ctrl-llm", "ctrl-cas", qs)
            row["gap_llm_minus_cas"] = {
                "mean": round(float(row["ctrl-llm"]["mean"] - row["ctrl-cas"]["mean"]), 2),
                "ci95": [round(float(np.percentile(reps, p)), 2) for p in (2.5, 97.5)]}
        if "d4casnegs" in hits and "ctrl-llm" in hits:
            reps = 100 * boot_gap("ctrl-llm", "d4casnegs", qs)
            row["gap_llm_minus_d4casnegs"] = {
                "mean": round(float(row["ctrl-llm"]["mean"] - row["d4casnegs"]["mean"]), 2),
                "ci95": [round(float(np.percentile(reps, p)), 2) for p in (2.5, 97.5)]}
        out["results"][sname] = row
        cells = "  ".join(f"{a}={row[a]['mean']:.2f}({row[a]['gap_vs_base']:+.2f})"
                          for a in ("ctrl-cas", "ctrl-llm", "D4", "d4casnegs") if a in row)
        print(f"[slice] {sname:11s} n={len(qs):3d} base={base_mean:6.2f}  {cells}")

    # ---- P-R5 ---------------------------------------------------------------
    verdict = {"registered_in": "scripts/samelang_eval.slurm", "pass_bar": PASS_BAR}
    if "ctrl-cas" in hits and "ctrl-llm" in hits:
        cs = json.load(open(os.path.join(PROJECT, "results", "ctrl_seed_stats.json")))
        easy_llm = np.array([cs["per_seed"][s]["llm"]["easy"]["recall@1"]
                             for s in cs["per_seed"]])
        easy_cas = np.array([cs["per_seed"][s]["cas"]["easy"]["recall@1"]
                             for s in cs["per_seed"]])
        easy_gap = float(easy_llm.mean() - easy_cas.mean())
        qs = slices["primary"]
        same_reps = 100 * boot_gap("ctrl-llm", "ctrl-cas", qs)
        easy_reps = np.array([easy_llm[rng.integers(0, 8, 8)].mean()
                              - easy_cas[rng.integers(0, 8, 8)].mean()
                              for _ in range(args.boot)])
        did_reps = easy_reps - same_reps
        did = easy_gap - out["results"]["primary"]["gap_llm_minus_cas"]["mean"]
        lo, hi = (float(np.percentile(did_reps, p)) for p in (2.5, 97.5))
        if did >= PASS_BAR and lo > 0:
            code = "PASS"
        elif did > 0:
            code = "PARTIAL"
        else:
            code = "REFUTED"
        verdict.update({"easy_gap": round(easy_gap, 2),
                        "samelang_gap_primary": out["results"]["primary"]["gap_llm_minus_cas"],
                        "did_same": round(did, 2), "did_same_ci95": [round(lo, 2), round(hi, 2)],
                        "share_not_transferring_pct": round(100 * did / easy_gap, 1),
                        "verdict": code})
        print(f"[P-R5] easy gap {easy_gap:.2f} - same-language gap "
              f"{out['results']['primary']['gap_llm_minus_cas']['mean']:.2f} = DiD {did:.2f} "
              f"[{lo:.2f}, {hi:.2f}] -> {code}")
    out["P-R5"] = verdict

    # ---- E-R7 .. E-R9 cells (second generator; see NEW_ARMS) ---------------
    rng = np.random.default_rng(1)
    new_hits = {}
    for arm, rx in NEW_ARMS.items():
        tags = sorted(t for t in dumps if re.match(rx, t))
        if not tags:
            continue
        new_hits[arm] = np.stack([[dumps[t][q] == 0 for q in qids] for t in tags]).astype(float)
        seeds_of[arm] = tags
    hits.update(new_hits)
    out["review2_cells"] = {"note": "E-R7/E-R8/E-R9 cells; separate RNG stream", "results": {}}
    for sname, qs in slices.items():
        ii = idx_of(qs)
        base_mean = 100 * hits["base"][:, ii].mean()
        row = {}
        for arm in new_hits:
            per_seed = 100 * hits[arm][:, ii].mean(axis=1)
            reps = 100 * boot_gap(arm, "base", qs)
            rec = {"mean": round(float(per_seed.mean()), 2), "n_seeds": int(len(per_seed)),
                   "per_seed": [round(float(x), 2) for x in per_seed],
                   "gap_vs_base": round(float(per_seed.mean() - base_mean), 2),
                   "gap_vs_base_ci95": [round(float(np.percentile(reps, p)), 2) for p in (2.5, 97.5)]}
            for other in ("ctrl-cas", "ctrl-llm", "d4casnegs", "d4llmnegs", "d4llmnegsfull"):
                if other in hits:
                    r2 = 100 * boot_gap(arm, other, qs)
                    rec[f"gap_vs_{other}"] = round(float(per_seed.mean() - 100 * hits[other][:, ii].mean()), 2)
                    rec[f"gap_vs_{other}_ci95"] = [round(float(np.percentile(r2, p)), 2) for p in (2.5, 97.5)]
            row[arm] = rec
        out["review2_cells"]["results"][sname] = row
        if row:
            print(f"[review2 {sname:11s}] " + "  ".join(f"{a}={r['mean']:.2f}({r['gap_vs_base']:+.2f})" for a, r in row.items()))

    # ---- E-R10 .. E-R13 cells (THIRD generator) ------------------------------
    R3_ARMS = {"d5nonegs": r"^fact-d5nonegs-s\d+$", "d5casnegs": r"^fact-d5casnegs-s\d+$",
               "d1split": r"^fact-d1split-s\d+$", "d1twojudge": r"^fact-d1twojudge-s\d+$",
               "bt2casnegs": r"^fact-bt2casnegs-s\d+$"}
    rng = np.random.default_rng(2)
    r3_hits = {}
    for arm, rx in R3_ARMS.items():
        tags = sorted(t for t in dumps if re.match(rx, t))
        if not tags:
            continue
        r3_hits[arm] = np.stack([[dumps[t][q] == 0 for q in qids] for t in tags]).astype(float)
        seeds_of[arm] = tags
    hits.update(r3_hits)
    out["review3_cells"] = {"note": "E-R10/E-R11/E-R12/E-R13 cells; third RNG stream", "results": {}}
    for sname, qs in slices.items():
        ii = idx_of(qs)
        base_mean = 100 * hits["base"][:, ii].mean()
        row = {}
        for arm in r3_hits:
            per_seed = 100 * hits[arm][:, ii].mean(axis=1)
            reps = 100 * boot_gap(arm, "base", qs)
            rec = {"mean": round(float(per_seed.mean()), 2), "n_seeds": int(len(per_seed)),
                   "per_seed": [round(float(x), 2) for x in per_seed],
                   "gap_vs_base": round(float(per_seed.mean() - base_mean), 2),
                   "gap_vs_base_ci95": [round(float(np.percentile(reps, p)), 2) for p in (2.5, 97.5)]}
            for other in ("ctrl-cas", "ctrl-llm", "d4casnegs", "D4", "btcasnegs"):
                if other in hits:
                    r2 = 100 * boot_gap(arm, other, qs)
                    rec[f"gap_vs_{other}"] = round(float(per_seed.mean() - 100 * hits[other][:, ii].mean()), 2)
                    rec[f"gap_vs_{other}_ci95"] = [round(float(np.percentile(r2, p)), 2) for p in (2.5, 97.5)]
            row[arm] = rec
        out["review3_cells"]["results"][sname] = row
        if row:
            print(f"[review3 {sname:11s}] " + "  ".join(f"{a}={r['mean']:.2f}({r['gap_vs_base']:+.2f})" for a, r in row.items()))

    # ---- table-fill cells (FOURTH generator; E-T1 .. E-T7, 2026-09-08) --------
    # Added after every interval above was quoted; a separate RNG stream keeps
    # those intervals byte-identical. No directional prediction registered.
    T_ARMS = {"casunrelnegs": r"^fact-casunrelnegs-s\d+$", "btnonegs": r"^fact-btnonegs-s\d+$",
              "btunrelnegs": r"^fact-btunrelnegs-s\d+$", "btllmnegs": r"^fact-btllmnegs-s\d+$",
              "d1nonegs": r"^fact-d1nonegs-s\d+$", "d1casnegs": r"^fact-d1casnegs-s\d+$",
              "d1unrelnegs": r"^fact-d1unrelnegs-s\d+$"}
    rng = np.random.default_rng(3)
    t_hits = {}
    for arm, rx in T_ARMS.items():
        tags = sorted(t for t in dumps if re.match(rx, t))
        if not tags:
            continue
        t_hits[arm] = np.stack([[dumps[t][q] == 0 for q in qids] for t in tags]).astype(float)
        seeds_of[arm] = tags
    hits.update(t_hits)
    out["tablefill_cells"] = {"note": "E-T1..E-T7 cells; fourth RNG stream", "results": {}}
    for sname, qs in slices.items():
        ii = idx_of(qs)
        base_mean = 100 * hits["base"][:, ii].mean()
        row = {}
        for arm in t_hits:
            per_seed = 100 * hits[arm][:, ii].mean(axis=1)
            reps = 100 * boot_gap(arm, "base", qs)
            rec = {"mean": round(float(per_seed.mean()), 2), "n_seeds": int(len(per_seed)),
                   "per_seed": [round(float(x), 2) for x in per_seed],
                   "gap_vs_base": round(float(per_seed.mean() - base_mean), 2),
                   "gap_vs_base_ci95": [round(float(np.percentile(reps, p)), 2) for p in (2.5, 97.5)]}
            for other in ("ctrl-cas", "ctrl-llm", "d4casnegs"):
                if other in hits:
                    r2 = 100 * boot_gap(arm, other, qs)
                    rec[f"gap_vs_{other}"] = round(float(per_seed.mean() - 100 * hits[other][:, ii].mean()), 2)
                    rec[f"gap_vs_{other}_ci95"] = [round(float(np.percentile(r2, p)), 2) for p in (2.5, 97.5)]
            row[arm] = rec
        out["tablefill_cells"]["results"][sname] = row
        if row:
            print(f"[tablefill {sname:11s}] " + "  ".join(f"{a}={r['mean']:.2f}({r['gap_vs_base']:+.2f})" for a, r in row.items()))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
