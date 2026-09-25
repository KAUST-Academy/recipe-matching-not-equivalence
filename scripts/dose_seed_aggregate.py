#!/usr/bin/env python3
"""
Across-training-seed aggregation of the prompt-proximity dose-response ladder
(review findings M4 + H2), emitted as a citable JSON fact source.

WHY THIS EXISTS
  scripts/dose_seeds.slurm stage 3 PRINTS a seed table into the job log. The
  paper's fact-source contract requires every number in the
  draft to be readable from a results/*.json file, not scraped from a log, so
  this script recomputes the same table into results/dose_seed_stats.json and
  adds the inference the log does not do.

  It is deliberately separate from scripts/dose_bootstrap.py, which resamples
  EVALUATION QUERIES and therefore carries no training-seed variance. This
  script resamples nothing: it reports the across-seed spread directly.

DECISION RULE  --- FIXED BEFORE ANY SEED-43/44 DOSE NUMBER WAS READ ---
  Written 2026-07-30 while job 49627530 was still training its last plan item.
  HONEST ORDERING (corrected 2026-07-31 after round-2 review finding R2-M1;
  the earlier version of this paragraph claimed more than the repository can
  support, and said "committed before the first run"; in fact this script was
  committed five minutes after it first ran):
    * PRE-DATA, verifiable: the noise model (sd_diff 1.20 easy / 0.86 hard from
      the eight-seed ctrl-LLM arm), the 2-sigma bar, and the naming of the two
      differences to re-examine were all fixed in scripts/dose_seeds.slurm,
      written before those runs were submitted.
    * POST-DATA: the sign-agreement clause and this script were written after
      the seed-43/44 eval JSONs were on disk. The authors had not opened them,
      but that is an assertion, not an artifact -- file mtimes are settable and
      do not survive a clone.
  The mitigation we can actually offer is uniform application: the rule runs
  over all eighteen pair x tier cells, not only the two flagged ones, so it
  cannot have been shaped to a particular cell. (The H2 control's evals did
  not exist at all when this was written.)

  Two orderings in tab:dose were flagged as single-seed claims because the
  seed-42 difference was under 2 sigma of training-seed noise:
        hard D1-D2   +1.48   (1.7 sigma)
        easy D2-D3   -2.18   (1.8 sigma)
  With three seeds per rung (42/43/44) each gets a paired-by-seed difference
  d_s = rung_A(s) - rung_B(s), s in {42,43,44}. Verdict on mean(d):

    REPLICATED            all three d_s carry the seed-42 sign
                          AND |mean(d)| > 2 * se_prior
    DIRECTIONAL           all three d_s carry the seed-42 sign
                          BUT |mean(d)| <= 2 * se_prior
    REFUTED               mean(d) flips the seed-42 sign
    INCONCLUSIVE          signs disagree without a sign flip in the mean

  se_prior = sd_diff_prior / sqrt(3), where sd_diff_prior is the run-to-run sd
  of a DIFFERENCE of two independent trainings taken from the eight-seed
  ctrl-LLM arm (results/ctrl_seed_stats.json): 1.20 easy / 0.86 hard R@1.
  The PRIOR sd is primary and the observed 3-seed sd is reported alongside as
  a check, because an sd estimated from three points is itself so noisy that
  making it the gate would mostly test luck. A paired t (df=2) is reported for
  completeness and should not be read as the headline: at n=3 it needs
  |t| > 4.303 for p < 0.05 and is badly underpowered.

  Any rung difference the paper quotes is reported here for ALL six adjacent
  pairs and both tiers, not only the two flagged ones, so a reader can see the
  rule was not applied selectively.

H2 ROW-COUNT CONTROL
  D3 trained on 5,591 rows against D1/D2/D4's 6,145 (a 9.0% deficit), so the
  D2->D3 drop conflates prompt distance with data volume. The control retrains
  D2 at seed 42 with --max-rows 5591 -- a strict PREFIX of the same shuffled
  row order the original D2 saw (train_invarembed.py shuffles with --seed
  before truncating), i.e. a nested subsample rather than a new random draw.
    row-count effect = D2@5591 - D2@6145
    prompt effect    = D3@5591 - D2@5591
  Verdict threshold is the same sd_diff_prior (0.86 hard / 1.20 easy): a
  row-count effect inside 1 sd of run-to-run noise makes H2 a reporting fix;
  outside it, the paper must attribute part of the drop to data volume.

Usage (login node, CPU, < 1 s):
  python scripts/dose_seed_aggregate.py
  python scripts/dose_seed_aggregate.py --seeds 42,43,44
Output: results/dose_seed_stats.json + a printed table. Missing inputs are
reported as nulls with an explicit `missing` list -- never silently skipped.
"""

import argparse
import json
import math
import os
import statistics
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")

# sd of a DIFFERENCE of two independent trainings, from the 8-seed ctrl-LLM
# arm (results/ctrl_seed_stats.json): sd_diff = sqrt(2) * sd_single.
# Derived at runtime by _sd_diff_prior() below; these are the values the draft
# already quotes and are asserted against the derivation so a silent change to
# ctrl_seed_stats.json cannot quietly move the paper's sigma flags.
SD_DIFF_PRIOR_DRAFT = {"easy": 1.20, "hard": 0.86}
SD_DIFF_PRIOR = dict(SD_DIFF_PRIOR_DRAFT)   # replaced in main()

# The seed-42 signs the flagged claims were made with, so a sign flip is
# detected against the ORIGINAL claim rather than against the new mean.
FLAGGED = {("D1", "D2", "hard"): "+", ("D2", "D3", "easy"): "-"}

RUNGS = ["D1", "D2", "D3", "D4"]
RUNG_LABEL = {
    "D1": "D1 exact template (= ctrl-LLM)",
    "D2": "D2 paraphrased template",
    "D3": "D3 different-style prompt",
    "D4": "D4 recipe-unrelated restatement",
}


def _sd_diff_prior():
    """Run-to-run sd of a DIFFERENCE of two independent trainings, per tier.

    Taken from the eight-seed ctrl-LLM arm: sd_diff = sqrt(2) * sd_single.
    Returns (priors, provenance). Falls back to the draft's hardcoded values
    only if ctrl_seed_stats.json is unreadable, and says so in provenance.
    """
    stats = _load(os.path.join(RESULTS, "ctrl_seed_stats.json"))
    per_seed = (stats or {}).get("per_seed")
    if not per_seed:
        return dict(SD_DIFF_PRIOR_DRAFT), {
            "source": "FALLBACK — results/ctrl_seed_stats.json unreadable; "
            "using the values already quoted in the draft",
            "n_seeds": None,
        }
    key = {"easy": ("easy", "recall@1"), "hard": ("hard", "recall@1"),
           "realdup": ("xling_strict", "recall@1")}
    priors, single = {}, {}
    for tier, (blk, metric) in key.items():
        vals = [d["llm"][blk][metric] for d in per_seed.values()
                if "llm" in d and blk in d["llm"]]
        if len(vals) > 1:
            sd1 = statistics.stdev(vals)
            single[tier] = round(sd1, 4)
            priors[tier] = round(math.sqrt(2) * sd1, 4)
    prov = {
        "source": "results/ctrl_seed_stats.json, ctrl-LLM arm",
        "n_seeds": len(per_seed),
        "sd_single_run": single,
        "formula": "sd_diff = sqrt(2) * sd_single_run",
        "draft_quoted": SD_DIFF_PRIOR_DRAFT,
        "matches_draft_to_2dp": {
            t: (t in priors and abs(priors[t] - v) < 0.005)
            for t, v in SD_DIFF_PRIOR_DRAFT.items()
        },
    }
    for t, v in SD_DIFF_PRIOR_DRAFT.items():
        priors.setdefault(t, v)
    return priors, prov


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _dig(path, *keys):
    d = _load(path)
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def files_for(rung, seed):
    """(easy, hard, realdup) canonical eval paths. Seed 42 has no suffix."""
    if rung == "D1":
        stem = "ctrl-llm" if seed == 42 else f"ctrl-llm-s{seed}"
        xling = "ctrl-llm-6145" if seed == 42 else f"ctrl-llm-s{seed}"
    else:
        variant = {"D2": "paraphrase", "D3": "style", "D4": "unrelated"}[rung]
        stem = f"dose-{variant}" if seed == 42 else f"dose-{variant}-s{seed}"
        xling = stem
    return (
        os.path.join(RESULTS, f"eval_easy_{stem}.json"),
        os.path.join(RESULTS, f"eval_hard_{stem}.json"),
        os.path.join(RESULTS, f"crosslingual_{xling}.json"),
    )


def read_point(rung, seed, missing):
    fe, fh, fx = files_for(rung, seed)
    out = {}
    for tier, path, keys in (
        ("easy", fe, ("overall", "recall@1")),
        ("hard", fh, ("overall", "recall@1")),
        ("realdup", fx, ("strict_crosslingual_gold", "recall@1")),
    ):
        val = _dig(path, *keys)
        if val is None:
            missing.append(os.path.relpath(path, PROJECT_ROOT))
        out[tier] = val
    return out


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "values": []}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        "values": vals,
    }


def paired_diff(points, a, b, tier, seeds):
    """rung a minus rung b, paired by training seed."""
    ds, used = [], []
    for s in seeds:
        va, vb = points[(a, s)][tier], points[(b, s)][tier]
        if va is None or vb is None:
            continue
        ds.append(va - vb)
        used.append(s)
    if not ds:
        return {"pair": f"{a}-{b}", "tier": tier, "n_seeds": 0, "verdict": "no data"}

    n = len(ds)
    mean = statistics.mean(ds)
    sd_obs = statistics.stdev(ds) if n > 1 else None
    sd_prior = SD_DIFF_PRIOR.get(tier)

    rec = {
        "pair": f"{a}-{b}",
        "tier": tier,
        "n_seeds": n,
        "seeds_used": used,
        "per_seed_diff": [round(d, 4) for d in ds],
        "mean_diff": round(mean, 4),
        "sd_diff_observed": round(sd_obs, 4) if sd_obs is not None else None,
        "sd_diff_prior_1run": sd_prior,
        "signs_all_agree": len({d > 0 for d in ds}) == 1,
    }

    if sd_prior:
        se_prior = sd_prior / math.sqrt(n)
        rec["se_prior_mean"] = round(se_prior, 4)
        rec["n_sigma_prior"] = round(abs(mean) / se_prior, 3)
        rec["n_sigma_prior_single_run"] = round(abs(mean) / sd_prior, 3)
    if sd_obs and sd_obs > 0:
        se_obs = sd_obs / math.sqrt(n)
        rec["se_observed_mean"] = round(se_obs, 4)
        rec["n_sigma_observed"] = round(abs(mean) / se_obs, 3)
        rec["t_paired_df{}".format(n - 1)] = round(mean / se_obs, 3)

    key = (a, b, tier)
    if key in FLAGGED:
        want_pos = FLAGGED[key] == "+"
        seed42_sign_kept = [(d > 0) == want_pos for d in ds]
        rec["flagged_in_draft"] = True
        rec["original_seed42_sign"] = FLAGGED[key]
        rec["per_seed_keeps_original_sign"] = seed42_sign_kept
        if not all(seed42_sign_kept) and ((mean > 0) != want_pos):
            rec["verdict"] = "REFUTED"
        elif not all(seed42_sign_kept):
            rec["verdict"] = "INCONCLUSIVE"
        elif sd_prior and abs(mean) > 2 * (sd_prior / math.sqrt(n)):
            rec["verdict"] = "REPLICATED"
        else:
            rec["verdict"] = "DIRECTIONAL"
    else:
        rec["flagged_in_draft"] = False
        if sd_prior and abs(mean) > 2 * (sd_prior / math.sqrt(n)) and rec["signs_all_agree"]:
            rec["verdict"] = "REPLICATED"
        elif rec["signs_all_agree"]:
            rec["verdict"] = "DIRECTIONAL"
        else:
            rec["verdict"] = "INCONCLUSIVE"
    return rec


def h2_control(missing):
    """D2 retrained at D3's exact row count (seed 42)."""
    out = {
        "description": "D2 (paraphrase prompt) retrained at seed 42 with "
        "--max-rows 5591 = D3's actual row count; a nested PREFIX subsample "
        "of D2's own shuffled row order, not a fresh draw.",
        "threshold_note": "|row-count effect| < sd_diff_prior (0.86 hard / "
        "1.20 easy) => inside run-to-run noise => H2 is a reporting fix.",
    }
    for tier, d2f, d3f, ctlf in (
        (
            "hard",
            "eval_hard_dose-paraphrase.json",
            "eval_hard_dose-style.json",
            "eval_hard_dose-paraphrase-5591.json",
        ),
        (
            "easy",
            "eval_easy_dose-paraphrase.json",
            "eval_easy_dose-style.json",
            "eval_easy_dose-paraphrase-5591.json",
        ),
    ):
        d2 = _dig(os.path.join(RESULTS, d2f), "overall", "recall@1")
        d3 = _dig(os.path.join(RESULTS, d3f), "overall", "recall@1")
        ctl = _dig(os.path.join(RESULTS, ctlf), "overall", "recall@1")
        blk = {"D2_at_6145": d2, "D2_at_5591_control": ctl, "D3_at_5591": d3}
        if None in (d2, d3, ctl):
            if ctl is None:
                missing.append(f"results/{ctlf}")
            blk["verdict"] = "PENDING — control eval has not run yet"
        else:
            row_eff, prompt_eff, tot = ctl - d2, d3 - ctl, d3 - d2
            blk["row_count_effect_D2at5591_minus_D2at6145"] = round(row_eff, 4)
            blk["prompt_effect_D3_minus_D2at5591"] = round(prompt_eff, 4)
            blk["total_D2_to_D3"] = round(tot, 4)
            blk["row_count_share_of_total_pct"] = (
                None if abs(tot) < 1e-9 else round(100 * row_eff / tot, 1)
            )
            sd = SD_DIFF_PRIOR[tier]
            blk["row_effect_in_sd_units"] = round(abs(row_eff) / sd, 3)
            blk["verdict"] = (
                "ROW COUNT INSIDE SEED NOISE — the D2->D3 drop is the PROMPT; "
                "H2 is a reporting fix only"
                if abs(row_eff) < sd
                else "ROW COUNT EXCEEDS SEED NOISE — H2's confound is REAL; the "
                "paper must attribute part of the D2->D3 drop to data volume"
            )
        out[tier] = blk
    return out


def comparability_audit(points_files):
    """Prove the per-seed eval JSONs are numerically comparable.

    A seed effect and an eval-configuration change are indistinguishable in the
    output numbers, so before averaging across seeds we diff the harness config
    recorded in every eval JSON. The seed-42 runs predate two fields the M5/M6
    review fixes added, which shows up here as a difference; `self_masking` is
    inert on this benchmark only if no query's own document is in the corpus,
    so we check that rather than assume it.
    """
    CFG = ("benchmark", "n_corpus", "n_queries_evaluated", "corpus_subsampled",
           "query_prompt_name", "query_prompt", "doc_prompt_name", "pooling",
           "append_eos", "model_dtype", "max_seq_length", "trust_remote_code",
           "comparable_to_paper", "self_masking")
    by_tier, notes, inert, blocking = {}, [], [], []
    for (rung, seed), (fe, fh, _fx) in points_files.items():
        for tier, path in (("easy", fe), ("hard", fh)):
            d = _load(path)
            if not d:
                continue
            by_tier.setdefault(tier, []).append(
                (f"{rung}/s{seed}", {k: d.get(k) for k in CFG}, d))

    for tier, rows in sorted(by_tier.items()):
        ref_name, ref, _ = rows[0]
        for name, cfg, d in rows[1:]:
            for k in CFG:
                if cfg[k] == ref[k]:
                    continue
                if k == "self_masking":
                    # inert iff nothing was ever actually masked
                    masked = d.get("n_queries_self_masked")
                    own = d.get("n_queries_with_own_doc_in_corpus")
                    if (masked in (0, None)) and (own in (0, None)):
                        inert.append(
                            f"{tier} {name}: self_masking={cfg[k]} vs {ref[k]} "
                            f"at {ref_name}, but n_queries_with_own_doc_in_corpus"
                            f"={own} and n_queries_self_masked={masked} — the "
                            f"mask set is empty, so the flag cannot move a number")
                    else:
                        blocking.append(
                            f"{tier} {name}: self_masking={cfg[k]} AND {masked} "
                            f"queries were actually masked — NOT comparable to "
                            f"{ref_name}")
                else:
                    blocking.append(
                        f"{tier} {name}: {k}={cfg[k]!r} vs {ref[k]!r} at {ref_name}")
    notes.append(
        "The seed-42 eval JSONs predate the self_masking field (added by the "
        "2026-07-30 M6 fix); absence of the field is not absence of "
        "comparability, which is why the emptiness of the mask set is checked "
        "directly.")
    return {
        "verdict": "COMPARABLE" if not blocking else "NOT COMPARABLE — see blocking",
        "inert_differences": inert,
        "blocking_differences": blocking,
        "fields_checked": list(CFG),
        "notes": notes,
    }


def what_trained():
    """run_config.json data_stats for every model in the ladder."""
    out = {}
    for m in (
        "ctrl-llm-6145",
        "ctrl-llm-6145-s43",
        "ctrl-llm-6145-s44",
        "dose-paraphrase-6145",
        "dose-paraphrase-6145-s43",
        "dose-paraphrase-6145-s44",
        "dose-style-6145",
        "dose-style-6145-s43",
        "dose-style-6145-s44",
        "dose-unrelated-6145",
        "dose-unrelated-6145-s43",
        "dose-unrelated-6145-s44",
        "dose-paraphrase-5591",
    ):
        ds = _dig(os.path.join(PROJECT_ROOT, "models", m, "run_config.json"), "data_stats")
        if ds is None:
            continue
        out[m] = {
            k: ds.get(k)
            for k in ("input_rows", "rows_used", "n_source_ids",
                      "n_train_triplets", "n_train_pairs")
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--output", default=os.path.join(RESULTS, "dose_seed_stats.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    global SD_DIFF_PRIOR
    SD_DIFF_PRIOR, prior_prov = _sd_diff_prior()

    missing = []
    points = {(r, s): read_point(r, s, missing) for r in RUNGS for s in seeds}
    points_files = {(r, s): files_for(r, s) for r in RUNGS for s in seeds}

    per_rung = {}
    for r in RUNGS:
        per_rung[r] = {
            "label": RUNG_LABEL[r],
            "per_seed": {str(s): points[(r, s)] for s in seeds},
        }
        for tier in ("easy", "hard", "realdup"):
            per_rung[r][tier] = summarize([points[(r, s)][tier] for s in seeds])

    # ALL six ordered rung pairs, not only the three adjacent ones: the draft's
    # easy-tier reading ("the exact template is not the best easy-tier gamer")
    # rests on D1-D3, which is not adjacent, and reporting every pair removes
    # any question of which comparisons were chosen after seeing the data.
    diffs = []
    for i, a in enumerate(RUNGS):
        for b in RUNGS[i + 1:]:
            for tier in ("easy", "hard", "realdup"):
                d = paired_diff(points, a, b, tier, seeds)
                d["adjacent"] = RUNGS.index(b) - RUNGS.index(a) == 1
                diffs.append(d)

    doc = {
        "generated": str(date.today()),
        "script": "scripts/dose_seed_aggregate.py",
        "slurm_job": "49627530 (scripts/dose_seeds.slurm)",
        "seeds": seeds,
        "decision_rule": (
            "Fixed before any seed-43/44 dose number was read (see module "
            "docstring). REPLICATED = all seeds keep the seed-42 sign AND "
            "|mean| > 2*se_prior, se_prior = sd_diff_prior/sqrt(n) with "
            "sd_diff_prior = 1.20 easy / 0.86 hard from the 8-seed ctrl-LLM arm."
        ),
        "what_this_does_not_measure": (
            "No resampling of any kind. Eval-query sampling error is a separate "
            "artifact (results/dose_bootstrap.json); these two uncertainties are "
            "additive and must not be quoted as if either were the total."
        ),
        "sd_diff_prior": SD_DIFF_PRIOR,
        "sd_diff_prior_provenance": prior_prov,
        "per_rung": per_rung,
        "adjacent_rung_differences": diffs,
        "comparability_audit": comparability_audit(points_files),
        "h2_row_count_control": h2_control(missing),
        "what_actually_trained": what_trained(),
        "missing_inputs": sorted(set(missing)),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    # ---- printed table ----
    def cell(d):
        if not d["n"]:
            return "     --      "
        sd = f" ±{d['sd']:4.2f}" if d["sd"] is not None else "      "
        return f"{d['mean']:6.2f}{sd} n={d['n']}"

    print(f"\n{'rung':34s} {'easy R@1':>18s} {'hard R@1':>18s} {'realdup R@1':>18s}")
    for r in RUNGS:
        print(f"{RUNG_LABEL[r]:34s} {cell(per_rung[r]['easy']):>18s} "
              f"{cell(per_rung[r]['hard']):>18s} {cell(per_rung[r]['realdup']):>18s}")

    print(f"\n{'adjacent difference':22s} {'mean':>8s} {'per-seed':>26s} "
          f"{'sigma_prior':>12s} {'verdict':>14s}")
    for d in diffs:
        if not d.get("n_seeds"):
            continue
        ps = ",".join(f"{x:+.2f}" for x in d["per_seed_diff"])
        sig = d.get("n_sigma_prior")
        flag = " *" if d.get("flagged_in_draft") else ""
        adj = " " if d.get("adjacent") else "."
        print(f"{adj}{d['pair']+' '+d['tier']:21s} {d['mean_diff']:+8.2f} {ps:>26s} "
              f"{(f'{sig:.2f}' if sig else '--'):>12s} {d['verdict']+flag:>14s}")
    print("  (* = flagged as a single-seed ordering in the current draft;"
          " leading '.' = non-adjacent rungs)")

    ca = doc["comparability_audit"]
    print(f"\n--- eval-config comparability: {ca['verdict']} ---")
    for b in ca["blocking_differences"]:
        print(f"  BLOCKING: {b}")
    if ca["inert_differences"]:
        print(f"  {len(ca['inert_differences'])} inert difference(s), e.g.:")
        print(f"    {ca['inert_differences'][0]}")

    h2 = doc["h2_row_count_control"]
    print("\n--- H2 row-count control ---")
    for tier in ("hard", "easy"):
        b = h2[tier]
        print(f"  [{tier}] D2@6145={b['D2_at_6145']}  D2@5591={b['D2_at_5591_control']}"
              f"  D3@5591={b['D3_at_5591']}")
        print(f"        {b['verdict']}")

    if doc["missing_inputs"]:
        print("\nMISSING INPUTS (reported as nulls, not skipped):")
        for m in doc["missing_inputs"]:
            print(f"  {m}")
    print(f"\nwrote {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
