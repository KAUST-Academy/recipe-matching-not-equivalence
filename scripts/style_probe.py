#!/usr/bin/env python3
"""Recipe-sensitivity audit, part 2: training-free recipe-style detector.

Demonstrates the recipe fingerprint directly:

  --stage fit      (CPU, ~2 min)  Train a logistic-regression bag-of-ngrams
      classifier separating Appendix-F-recipe LLM rewrites from organic
      problems. Training data is PAIR-MATCHED to kill topic/language
      confounds: synthetic = candidate_text, organic = anchor_text of the
      SAME flat rows in data/pairs/llm_pairs.jsonl (our Qwen3-32B run of the
      MathNet paper's own Appendix-F prompt). Group-split by source_id.
      Outputs: held-out AUC, results/style_probe_model.joblib,
      results/style_probe_features.json (top recipe n-grams -- also the
      template-phrase list for analyze_hits.py).

  --stage score    (CPU, ~2 min)  Score every MathNet-Retrieve corpus doc +
      query with the detector. The benchmark's own ids label the transfer
      test: ::eq::/::nm:: docs are GPT-5-generated, ::orig docs and all
      queries are organic. A high AUC here means a detector trained on OUR
      generator (Qwen3-32B) recognizes the benchmark's generator (GPT-5)
      purely from the shared prompt recipe -- cross-generator transfer of the
      fingerprint. Outputs: results/style_scores_retrieve.npz,
      results/style_probe_transfer.json.

  --stage correlate  (CPU if embedding caches exist; else run
      scripts/style_probe.slurm first)  For each model given as
      --model-emb LABEL=DOCS_NPZ,QUERIES_NPZ (globs ok; .npz written by
      eval_retrieve.py --emb-cache-dir), compute per-query Spearman
      correlation between query-doc cosine similarity and the doc's
      synthetic-style score (all docs; synthetic-only docs; plus mean style
      score of each query's top-10). If ctrl-llm's similarities track the
      style score more than base / ctrl-cas, retrieval is literally steered
      by the recipe fingerprint. Paired per-query deltas + Wilcoxon tests.
      Output: results/style_probe_correlations.json.

Artifact stripping (default ON, 2026-07-30, internal audit finding H1): organic
anchors carry two trivially-format-distinguishing artifacts that the
rewrites never reproduce -- a leading "Problem:" OCR boilerplate header
(63.9% of anchors vs 0.02% of rewrites; LR coef -31.6, 3.4x the
next-largest feature) and markdown ![](attached_image_N.png) figure
placeholders (4.0% vs 0.2%). Neither appears anywhere in the
MathNet-Retrieve transfer corpus (0.00% of 117,088 docs and 15,000
queries), so they inflate the in-generator holdout AUC while *hurting*
transfer. --strip-artifacts (default) removes them from every text before
fit AND score; --no-strip-artifacts reproduces the original v1 behavior.
The fitted bundle records the flag and stage score follows the bundle.
Measured 2026-07-30 (--train-synthetic positives, seed 42):

                              stripped   v1 (unstripped)
  holdout AUC (grouped)         0.8802        0.9690
  eq-vs-orig transfer AUC       0.8042        0.7139   <- stripping HELPS
  eq vs disjoint-orig transfer  0.8072        0.7319
  D1/D2/D3/D4 panel AUC   .880/.876/.865/.676   .969/.966/.964/.896
  ctrl-llm minus ctrl-cas rho   +0.1895       +0.1313  (94.2% / 93.6% of q)

stage fit also emits a D1-D4 dose-rung specificity panel (--panel, default
ON): AUC of each rung's judge-verified positives against the held-out
organic anchors, using the group-split holdout detector. The panel is the
specificity control: every rung (including D4, whose prompt never mentions
the recipe) is detected far above chance, so the probe fires on
LLM-rewrite GENRE, with a recipe increment on top (0.880 D1 vs 0.676 D4).

stage score also reports a disjoint-organics transfer split
(--transfer-disjoint, default ON): 636/1,668 (38.1%) of the transfer test's
organic orig docs and 2,118/15,000 (14.1%) of its queries are the same
problems as probe-training anchors, so the headline transfer AUC is also
reported against organics with no problem-level overlap with training.

Stages can be combined: --stage fit,score. --out-summary merges every
stage's numbers into one JSON across invocations (and derives a
stripped-vs-v1 'headline' block); --stage '' refreshes only that block.

Usage (login node, CPU; ~12 s fit+score, ~4 min correlate):
  # 1. provenance column (reproduces the released v1 numbers exactly)
  python scripts/style_probe.py --stage fit,score --no-strip-artifacts \
      --out-summary /tmp/unstripped_v1.json
  # 2. canonical stripped run
  python scripts/style_probe.py --stage fit,score \
      --out-summary results/style_probe_stripped.json \
      --compare-summary /tmp/unstripped_v1.json
  # 3. tracking table (3 models; overwrites style_probe_correlations.json)
  python scripts/style_probe.py --stage correlate \
      --out-summary results/style_probe_stripped.json \
      --model-emb 'base-0.6b=.emb_cache/docs_Qwen__Qwen3-Embedding-0.6B_*.npz,.emb_cache/queries_Qwen__Qwen3-Embedding-0.6B_*.npz' \
      --model-emb 'ctrl-llm-6145=.emb_cache/docs_models__ctrl-llm-6145__final_*.npz,.emb_cache/queries_models__ctrl-llm-6145__final_*.npz' \
      --model-emb 'ctrl-cas-6145=.emb_cache/docs_models__ctrl-cas-6145__final_*.npz,.emb_cache/queries_models__ctrl-cas-6145__final_*.npz'

CAUTION: a stripped run overwrites style_probe_{features,transfer}.json,
style_probe_model.joblib and style_scores_retrieve.npz, which are inputs to
scripts/analyze_hits.py. The v1 copies are preserved as
results/style_probe_*_v1_artifact.* (+ style_scores_retrieve_v1_artifact
.npz); re-run analyze_hits.py against the stripped artifacts before quoting
hit-accounting numbers alongside stripped detector numbers.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAT_PAIRS = os.path.join(PROJECT_ROOT, "data", "pairs", "llm_pairs.jsonl")
RETRIEVE_DIR = os.path.join(PROJECT_ROOT, "data", "retrieve", "hard")
RESULTS = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS, "style_probe_model.joblib")
FEATURES_PATH = os.path.join(RESULTS, "style_probe_features.json")
SCORES_PATH = os.path.join(RESULTS, "style_scores_retrieve.npz")

# D1-D4 dose ladder flat pair files (candidate_text + judge label per row)
DOSE_PANEL = [
    ("D1_appendixF", os.path.join(PROJECT_ROOT, "data", "pairs",
                                  "llm_pairs.jsonl")),
    ("D2_paraphrase", os.path.join(PROJECT_ROOT, "data", "pairs",
                                   "llm_pairs_paraphrase.jsonl")),
    ("D3_style", os.path.join(PROJECT_ROOT, "data", "pairs",
                              "llm_pairs_style.jsonl")),
    ("D4_unrelated", os.path.join(PROJECT_ROOT, "data", "pairs",
                                  "llm_pairs_unrelated.jsonl")),
]

# Format artifacts of the organic OCR pipeline (internal audit finding H1): a
# leading "Problem:" boilerplate header and markdown image placeholders.
# Both are ~absent from LLM rewrites and 100% absent from the
# MathNet-Retrieve transfer corpus, so a detector must not rely on them.
# Measured prevalence (2026-07-30, data/pairs/llm_pairs.jsonl, printed by
# --stage fit as [audit]): "Problem:" header 63.88% of the 6,414 organic
# anchors vs 0.02% of the 6,414 rewrites; attached_image placeholders 4.01%
# vs 0.16%; both 0.00% of the 117,088 MathNet-Retrieve docs and 15,000
# queries. Audited-but-NOT-stripped (no coefficient dominance, <1%
# prevalence): "Proposed by" attributions (0.72% of anchors, 0.00% of
# rewrites, 0.02% of corpus docs); no [asy] blocks, "Solution:" headers or
# non-attached_image markdown images exist in either arm.
_ARTIFACT_HEADER_RE = re.compile(r"^\s*Problem:\s*")
_ARTIFACT_HEADER_LINE_RE = re.compile(r"^[ \t]*Problem:[ \t]*", re.MULTILINE)
_ARTIFACT_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(attached_image_\d+\.png\)")
_ARTIFACT_IMG_BARE_RE = re.compile(r"attached_image_\d+\.png")
_ARTIFACT_AUDIT_RE = {
    "problem_header_leading": _ARTIFACT_HEADER_RE,
    "problem_header_any_line": _ARTIFACT_HEADER_LINE_RE,
    "attached_image_placeholder": _ARTIFACT_IMG_BARE_RE,
    "proposed_by_attribution_NOT_stripped": re.compile(r"Proposed by",
                                                       re.IGNORECASE),
}


def strip_artifacts(text):
    text = _ARTIFACT_HEADER_RE.sub("", text)
    text = _ARTIFACT_HEADER_LINE_RE.sub("", text)
    text = _ARTIFACT_IMG_MD_RE.sub(" ", text)
    text = _ARTIFACT_IMG_BARE_RE.sub(" ", text)
    return text


def artifact_prevalence(texts):
    """Fraction of texts carrying each audited format artifact."""
    n = max(len(texts), 1)
    return {k: round(sum(1 for t in texts if rx.search(t)) / n, 4)
            for k, rx in _ARTIFACT_AUDIT_RE.items()}


def _fast_auc(pos, neg):
    """Mann-Whitney AUC, tie-corrected."""
    from scipy.stats import rankdata
    n1, n0 = len(pos), len(neg)
    if not n1 or not n0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def panel_bootstrap(neg_by_src, pos_by_src, sources, n_boot, seed,
                    ref="D1_appendixF"):
    """Source-clustered paired bootstrap over the held-out sources: one
    resample of source ids per replicate feeds BOTH the organic negatives and
    every rung's positives, so the per-rung AUCs and the D1-minus-rung
    differences are paired and their CIs are comparable."""
    rng = np.random.default_rng(seed)
    labels = list(pos_by_src)
    src = np.array(sources)
    aucs = {l: np.empty(n_boot) for l in labels}
    diffs = {l: np.empty(n_boot) for l in labels}
    for b in range(n_boot):
        samp = rng.choice(src, size=len(src), replace=True)
        neg = np.array([neg_by_src[s] for s in samp if s in neg_by_src])
        cur = {}
        for l in labels:
            pos = np.array([pos_by_src[l][s] for s in samp
                            if s in pos_by_src[l]])
            cur[l] = _fast_auc(pos, neg)
        for l in labels:
            aucs[l][b] = cur[l]
            diffs[l][b] = cur[ref] - cur[l]
    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]
    return {
        "method": f"{n_boot} replicates, resampling the held-out source ids "
                  "with replacement (clusters organic negatives with each "
                  "rung's positives); percentile 95% CIs",
        "auc_ci95": {l: ci(aucs[l]) for l in labels},
        f"{ref}_minus_rung_ci95": {l: ci(diffs[l]) for l in labels
                                   if l != ref},
        f"{ref}_minus_rung_pct_replicates_positive": {
            l: round(100 * float((diffs[l] > 0).mean()), 2)
            for l in labels if l != ref},
    }


def prep(text, strip):
    return strip_artifacts(text) if strip else text


def norm_for_overlap(text):
    """Aggressive normalization used to test whether a transfer-eval organic
    doc is the same underlying problem as a probe-training anchor."""
    return " ".join(strip_artifacts(text).split()).lower()


def load_jsonl_map(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["_id"]] = row["text"]
    return out


# --------------------------------------------------------------------- fit
def stage_fit(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import joblib

    synth, organic = [], {}          # [(source_id, text)], {source_id: text}
    raw_synth, raw_organic = [], {}  # pre-strip copies, for the audit only
    with open(FLAT_PAIRS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if args.train_synthetic == "positives" and r["label"] != "positive":
                continue
            synth.append((r["source_id"],
                          prep(r["candidate_text"], args.strip_artifacts)))
            organic.setdefault(r["source_id"],
                               prep(r["anchor_text"], args.strip_artifacts))
            raw_synth.append(r["candidate_text"])
            raw_organic.setdefault(r["source_id"], r["anchor_text"])
    organic = list(organic.items())
    audit = {
        "note": "fraction of texts carrying each audited format artifact, "
                "BEFORE stripping (review finding H1)",
        "organic_anchors": artifact_prevalence(list(raw_organic.values())),
        "llm_rewrites": artifact_prevalence(raw_synth),
    }
    print(f"[audit] organic anchors {audit['organic_anchors']}", flush=True)
    print(f"[audit] llm rewrites    {audit['llm_rewrites']}", flush=True)
    sources = sorted({s for s, _ in synth} | {s for s, _ in organic})
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    n_test = max(1, int(0.2 * len(sources)))
    test_src = set(sources[:n_test])
    print(f"[fit] synthetic={len(synth)} organic={len(organic)} "
          f"sources={len(sources)} (test sources={len(test_src)}) "
          f"train_synthetic={args.train_synthetic} "
          f"strip_artifacts={args.strip_artifacts}", flush=True)

    def split(rows):
        tr = [(t, s) for s, t in rows if s not in test_src]
        te = [(t, s) for s, t in rows if s in test_src]
        return tr, te

    syn_tr, syn_te = split(synth)
    org_tr, org_te = split(organic)
    X_tr = [t for t, _ in syn_tr] + [t for t, _ in org_tr]
    y_tr = [1] * len(syn_tr) + [0] * len(org_tr)
    X_te = [t for t, _ in syn_te] + [t for t, _ in org_te]
    y_te = [1] * len(syn_te) + [0] * len(org_te)

    def build():
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                              lowercase=True)
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced", solver="liblinear")
        return vec, clf

    t0 = time.time()
    vec, clf = build()
    clf.fit(vec.fit_transform(X_tr), y_tr)
    auc = roc_auc_score(y_te, clf.decision_function(vec.transform(X_te)))
    print(f"[fit] held-out AUC (grouped by source_id) = {auc:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # D1-D4 dose-rung specificity panel against the held-out organic anchors,
    # scored with the group-split holdout detector (never saw test sources)
    panel = None
    if args.panel:
        neg_texts = [t for t, _ in org_te]
        neg_scores = clf.decision_function(vec.transform(neg_texts))
        panel = {"negatives": f"held-out organic anchors (n={len(neg_texts)}, "
                              "20% source split, never in probe training)",
                 "detector": "group-split holdout detector "
                             f"(strip_artifacts={args.strip_artifacts})",
                 "rungs": {}}
        neg_by_src = {s: sc for (_, s), sc in zip(org_te, neg_scores)}
        pos_by_src = {}
        for label, path in DOSE_PANEL:
            pos_all, pos_held, held_src = [], [], []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["label"] != "positive":
                        continue
                    t = prep(r["candidate_text"], args.strip_artifacts)
                    pos_all.append(t)
                    if r["source_id"] in test_src:
                        pos_held.append(t)
                        held_src.append(r["source_id"])
            s_all = clf.decision_function(vec.transform(pos_all))
            s_held = (clf.decision_function(vec.transform(pos_held))
                      if pos_held else np.array([]))
            # one positive per source in every rung; keep the last on the rare
            # duplicate so the bootstrap map stays 1:1 with the negatives
            pos_by_src[label] = dict(zip(held_src, s_held))
            def _auc(pos_s):
                if not len(pos_s):
                    return None
                y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_scores))]
                return round(float(roc_auc_score(
                    y, np.r_[pos_s, neg_scores])), 4)
            panel["rungs"][label] = {
                "auc_heldout_sources_vs_heldout_anchors": _auc(s_held),
                "n_pos_heldout_sources": len(pos_held),
                "auc_all_positives_vs_heldout_anchors": _auc(s_all),
                "n_pos_all": len(pos_all),
                "mean_score_all_positives": round(float(s_all.mean()), 4),
                "mean_score_heldout_sources": (
                    round(float(s_held.mean()), 4) if len(s_held) else None),
            }
            print(f"[panel] {label}: heldout-src AUC="
                  f"{panel['rungs'][label]['auc_heldout_sources_vs_heldout_anchors']} "
                  f"(n={len(pos_held)}), all-pos AUC="
                  f"{panel['rungs'][label]['auc_all_positives_vs_heldout_anchors']} "
                  f"(n={len(pos_all)})", flush=True)
        d1 = panel["rungs"]["D1_appendixF"][
            "auc_heldout_sources_vs_heldout_anchors"]
        panel["recipe_increment_vs_d1"] = {
            k: (None if panel["rungs"][k][
                    "auc_heldout_sources_vs_heldout_anchors"] is None
                else round(d1 - panel["rungs"][k][
                    "auc_heldout_sources_vs_heldout_anchors"], 4))
            for k in ("D2_paraphrase", "D3_style", "D4_unrelated")}
        panel["interpretation"] = (
            "every rung -- including D4, whose prompt never mentions the "
            "recipe -- is detected far above chance, so the probe fires on "
            "LLM-rewrite GENRE; the D1-minus-rung gaps are the recipe "
            "increment on top of that genre floor")
        print(f"[panel] recipe increment vs D1: "
              f"{panel['recipe_increment_vs_d1']}", flush=True)
        if args.panel_bootstrap:
            t1 = time.time()
            panel["bootstrap"] = panel_bootstrap(
                neg_by_src, pos_by_src, sorted(test_src),
                args.panel_bootstrap, args.seed)
            print(f"[panel] bootstrap CIs "
                  f"({args.panel_bootstrap} reps, {time.time() - t1:.0f}s): "
                  f"{panel['bootstrap']['auc_ci95']}", flush=True)
            print(f"[panel] D1-minus-rung CIs: "
                  f"{panel['bootstrap']['D1_appendixF_minus_rung_ci95']}",
                  flush=True)

    # refit on everything for the deployed scorer
    X_all = [t for _, t in synth] + [t for _, t in organic]
    y_all = [1] * len(synth) + [0] * len(organic)
    vec_f, clf_f = build()
    clf_f.fit(vec_f.fit_transform(X_all), y_all)
    joblib.dump({"vectorizer": vec_f, "classifier": clf_f,
                 "holdout_auc": float(auc),
                 "train_synthetic": args.train_synthetic,
                 "strip_artifacts": args.strip_artifacts,
                 "seed": args.seed}, MODEL_PATH)
    print(f"[fit] saved detector to {MODEL_PATH}", flush=True)

    names = np.array(vec_f.get_feature_names_out())
    coef = clf_f.coef_[0]
    top_syn = np.argsort(-coef)[:args.top_ngrams]
    top_org = np.argsort(coef)[:args.top_ngrams]
    feats = {
        "description": "recipe-fingerprint n-grams: top logistic-regression "
                       "features separating Appendix-F LLM rewrites "
                       "(synthetic) from organic problems; pair-matched "
                       "training data (anchor_text vs candidate_text)",
        "holdout_auc_grouped": round(float(auc), 4),
        "n_train_synthetic": len(synth),
        "n_train_organic": len(organic),
        "train_synthetic": args.train_synthetic,
        "strip_artifacts": args.strip_artifacts,
        "artifact_audit": audit,
        "dose_panel": panel,
        "top_synthetic_ngrams": [
            {"ngram": str(names[i]), "coef": round(float(coef[i]), 4)}
            for i in top_syn],
        "top_organic_ngrams": [
            {"ngram": str(names[i]), "coef": round(float(coef[i]), 4)}
            for i in top_org],
    }
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2, ensure_ascii=False)
    print(f"[fit] top synthetic n-grams: "
          f"{[d['ngram'] for d in feats['top_synthetic_ngrams'][:15]]}",
          flush=True)
    print(f"[fit] wrote {FEATURES_PATH}", flush=True)
    return {"holdout_auc_grouped": round(float(auc), 4),
            "n_train_synthetic": len(synth),
            "n_train_organic": len(organic),
            "n_sources": len(sources),
            "n_test_sources": len(test_src),
            "n_heldout_organic_anchors": len(org_te),
            "n_heldout_synthetic": len(syn_te),
            "strip_artifacts": args.strip_artifacts,
            "artifact_audit": audit,
            "dose_panel": panel,
            "top_organic_ngrams_head": [
                {"ngram": str(names[i]), "coef": round(float(coef[i]), 4)}
                for i in top_org[:8]],
            "top_synthetic_ngrams_head": [
                {"ngram": str(names[i]), "coef": round(float(coef[i]), 4)}
                for i in top_syn[:8]]}


# ------------------------------------------------------------------- score
def stage_score(args):
    from sklearn.metrics import roc_auc_score
    import joblib

    bundle = joblib.load(MODEL_PATH)
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    # the scorer must see text preprocessed exactly like its training data
    strip = bool(bundle.get("strip_artifacts", False))
    corpus = load_jsonl_map(os.path.join(RETRIEVE_DIR, "corpus.jsonl"))
    queries = load_jsonl_map(os.path.join(RETRIEVE_DIR, "queries.jsonl"))
    doc_ids = list(corpus)
    t0 = time.time()
    doc_scores = clf.decision_function(
        vec.transform([prep(corpus[i], strip) for i in doc_ids]))
    query_scores = clf.decision_function(
        vec.transform([prep(queries[q], strip) for q in queries]))
    print(f"[score] scored {len(doc_ids)} docs + {len(queries)} queries "
          f"(strip_artifacts={strip}, {time.time() - t0:.0f}s)", flush=True)

    kind = np.array(["eq" if "::eq::" in i else
                     "nm" if "::nm::" in i else "orig" for i in doc_ids])
    eq_tier = np.array([i.split("::eq::")[1] if "::eq::" in i else ""
                        for i in doc_ids])
    synth_mask = kind != "orig"
    s = np.asarray(doc_scores)
    q = np.asarray(query_scores)

    def auc(pos, neg):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return round(float(roc_auc_score(y, np.r_[pos, neg])), 4)

    transfer = {
        "detector": "logistic regression on word 1-2-grams, trained on our "
                    "Qwen3-32B Appendix-F rewrites vs their own source "
                    "problems (pair-matched)",
        "transfer_target": "MathNet-Retrieve corpus (GPT-5-generated eq/nm "
                           "docs vs organic orig docs + query texts)",
        "n_docs": {"eq": int((kind == "eq").sum()),
                   "nm": int((kind == "nm").sum()),
                   "orig": int((kind == "orig").sum()),
                   "queries": len(q)},
        "auc_synthetic_vs_orig_docs": auc(s[synth_mask], s[~synth_mask]),
        "auc_synthetic_vs_orig_plus_queries": auc(
            s[synth_mask], np.r_[s[~synth_mask], q]),
        "auc_eq_vs_orig_docs": auc(s[kind == "eq"], s[~synth_mask]),
        "auc_nm_vs_orig_docs": auc(s[kind == "nm"], s[~synth_mask]),
        "auc_synthetic_docs_vs_queries": auc(s[synth_mask], q),
        "auc_eq_tier_vs_orig_docs": {
            t: auc(s[eq_tier == t], s[~synth_mask])
            for t in ("easy", "medium", "hard")},
        "mean_score_eq_tier": {
            t: round(float(s[eq_tier == t].mean()), 4)
            for t in ("easy", "medium", "hard")},
        "mean_score": {"eq": round(float(s[kind == "eq"].mean()), 4),
                       "nm": round(float(s[kind == "nm"].mean()), 4),
                       "orig": round(float(s[~synth_mask].mean()), 4),
                       "queries": round(float(q.mean()), 4)},
        "detector_holdout_auc": bundle.get("holdout_auc"),
        "detector_strip_artifacts": strip,
    }

    # ---- disjoint-organics transfer split (review LOW finding): part of the
    # transfer test's organic negatives ARE probe training anchors (the same
    # underlying problems), which flatters the transfer AUC. Recompute using
    # only organic docs/queries whose normalized text never appears among the
    # training anchors.
    if args.transfer_disjoint:
        train_norm = set()
        with open(FLAT_PAIRS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if (args.train_synthetic == "positives"
                        and r["label"] != "positive"):
                    continue
                train_norm.add(norm_for_overlap(r["anchor_text"]))
        orig_ids = [i for i, k in zip(doc_ids, kind) if k == "orig"]
        orig_seen = np.array([norm_for_overlap(corpus[i]) in train_norm
                              for i in orig_ids])
        q_seen = np.array([norm_for_overlap(queries[k]) in train_norm
                           for k in queries])
        s_orig = s[~synth_mask]
        s_orig_dj = s_orig[~orig_seen]
        q_dj = q[~q_seen]
        neg_dj = np.r_[s_orig_dj, q_dj]
        transfer["disjoint_organic_negatives"] = {
            "note": "organic negatives whose whitespace/case-normalized, "
                    "artifact-stripped text is NOT among the probe's training "
                    "anchors -- i.e. a strictly generator-transfer test with "
                    "no problem-level overlap with probe training",
            "n_orig_docs_in_probe_training": int(orig_seen.sum()),
            "pct_orig_docs_in_probe_training":
                round(100 * float(orig_seen.mean()), 2),
            "n_queries_in_probe_training": int(q_seen.sum()),
            "pct_queries_in_probe_training":
                round(100 * float(q_seen.mean()), 2),
            "n_orig_docs_disjoint": int(len(s_orig_dj)),
            "n_queries_disjoint": int(len(q_dj)),
            "auc_eq_vs_disjoint_orig_docs": auc(s[kind == "eq"], s_orig_dj),
            "auc_nm_vs_disjoint_orig_docs": auc(s[kind == "nm"], s_orig_dj),
            "auc_synthetic_vs_disjoint_orig_docs": auc(s[synth_mask],
                                                       s_orig_dj),
            "auc_eq_vs_disjoint_orig_plus_queries": auc(s[kind == "eq"],
                                                        neg_dj),
            "auc_synthetic_vs_disjoint_orig_plus_queries": auc(s[synth_mask],
                                                               neg_dj),
            "auc_eq_tier_vs_disjoint_orig_docs": {
                t: auc(s[eq_tier == t], s_orig_dj)
                for t in ("easy", "medium", "hard")},
            "mean_score_disjoint_orig_docs": round(float(s_orig_dj.mean()), 4),
            "mean_score_overlapping_orig_docs": (
                round(float(s_orig[orig_seen].mean()), 4)
                if orig_seen.any() else None),
        }
        print(f"[score] disjoint-organics: "
              f"{int(orig_seen.sum())}/{len(orig_ids)} orig docs and "
              f"{int(q_seen.sum())}/{len(q_seen)} queries were probe-training "
              "anchors; eq-vs-orig AUC "
              f"{transfer['auc_eq_vs_orig_docs']} -> "
              f"{transfer['disjoint_organic_negatives']['auc_eq_vs_disjoint_orig_docs']}",
              flush=True)
    np.savez(SCORES_PATH, doc_ids=np.array(doc_ids),
             doc_scores=s.astype(np.float32),
             doc_is_synthetic=synth_mask,
             query_ids=np.array(list(queries)),
             query_scores=q.astype(np.float32),
             strip_artifacts=np.array(strip))
    with open(os.path.join(RESULTS, "style_probe_transfer.json"), "w",
              encoding="utf-8") as f:
        json.dump(transfer, f, indent=2)
    print(json.dumps(transfer, indent=2), flush=True)
    print(f"[score] wrote {SCORES_PATH} and style_probe_transfer.json",
          flush=True)
    return transfer


# --------------------------------------------------------------- correlate
def resolve_glob(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        sys.exit(f"[error] no file matches {pattern!r}")
    if len(hits) > 1:
        print(f"[warn] {len(hits)} files match {pattern!r}; using newest "
              f"{hits[-1]}", flush=True)
    return hits[-1]


def load_npz(pattern):
    path = resolve_glob(pattern)
    z = np.load(path, allow_pickle=False)
    return [str(x) for x in z["ids"]], z["emb"].astype(np.float32), path


def spearman_rows(sims, style_rank_c, denom_style):
    """Row-wise Spearman(sims_row, style) given centered style ranks."""
    from scipy.stats import rankdata
    r = rankdata(sims, axis=1).astype(np.float32)
    r -= r.mean(axis=1, keepdims=True)
    num = r @ style_rank_c
    den = np.sqrt((r * r).sum(axis=1)) * denom_style
    return num / np.maximum(den, 1e-12)


def stage_correlate(args):
    from scipy.stats import rankdata, wilcoxon

    if not os.path.exists(SCORES_PATH):
        sys.exit("[error] run --stage score first (needs "
                 f"{SCORES_PATH})")
    z = np.load(SCORES_PATH, allow_pickle=False)
    doc_ids = [str(x) for x in z["doc_ids"]]
    style = z["doc_scores"].astype(np.float64)
    synth_mask = z["doc_is_synthetic"]
    doc_pos = {d: i for i, d in enumerate(doc_ids)}

    specs = []
    for spec in args.model_emb:
        label, rest = spec.split("=", 1)
        d_pat, q_pat = rest.split(",", 1)
        specs.append((label, d_pat, q_pat))
    if not specs:
        sys.exit("[error] --stage correlate needs at least one --model-emb")

    # one shared query sample across models
    rng = random.Random(args.seed)
    shared_qids = None
    loaded = []
    for label, d_pat, q_pat in specs:
        d_ids, d_emb, d_path = load_npz(d_pat)
        q_ids, q_emb, q_path = load_npz(q_pat)
        if set(d_ids) != set(doc_ids):
            sys.exit(f"[error] doc ids of {label} do not match the scored "
                     "corpus")
        if d_ids != doc_ids:
            pos = {c: i for i, c in enumerate(d_ids)}
            d_emb = d_emb[[pos[c] for c in doc_ids]]
            print(f"[correlate] reordered {label} doc embeddings to scored "
                  "corpus order", flush=True)
        loaded.append((label, d_emb, dict(zip(q_ids, range(len(q_ids)))),
                       q_emb, d_path, q_path))
        shared_qids = (set(q_ids) if shared_qids is None
                       else shared_qids & set(q_ids))
    shared_qids = sorted(shared_qids)
    n_q = min(args.n_queries, len(shared_qids)) if args.n_queries else len(shared_qids)
    sample_qids = rng.sample(shared_qids, n_q)
    print(f"[correlate] {len(specs)} models, {n_q} shared sampled queries, "
          f"{len(doc_ids)} docs", flush=True)

    # style ranks (full corpus + synthetic-only), centered
    def centered_rank(v):
        r = rankdata(v).astype(np.float32)
        r -= r.mean()
        return r, float(np.sqrt((r * r).sum()))

    style_rank_c, style_den = centered_rank(style)
    style_syn = style[synth_mask]
    style_syn_rank_c, style_syn_den = centered_rank(style_syn)
    corpus_mean_style = float(style.mean())
    # scale-free companion to mean_top10_style_score: the decision function is
    # an unbounded log-odds score, so a ratio of two means ("2.1x style
    # enriched") is not interpretable and flips sign with the intercept. The
    # corpus percentile of a doc's style score is; corpus base rate = 50.0.
    style_pct = 100.0 * rankdata(style) / len(style)

    per_model = {}
    per_query_rho = {}
    for label, d_emb, q_index, q_emb, d_path, q_path in loaded:
        qe = q_emb[[q_index[q] for q in sample_qids]]
        rho_all = np.zeros(n_q); rho_syn = np.zeros(n_q)
        top10_style = np.zeros(n_q); top10_synfrac = np.zeros(n_q)
        top10_pct = np.zeros(n_q)
        t0 = time.time()
        chunk = 256
        d_emb_syn = d_emb[synth_mask]
        for s0 in range(0, n_q, chunk):
            sims = qe[s0:s0 + chunk] @ d_emb.T
            rho_all[s0:s0 + chunk] = spearman_rows(sims, style_rank_c, style_den)
            rho_syn[s0:s0 + chunk] = spearman_rows(
                qe[s0:s0 + chunk] @ d_emb_syn.T, style_syn_rank_c, style_syn_den)
            top = np.argpartition(-sims, 9, axis=1)[:, :10]
            top10_style[s0:s0 + chunk] = style[top].mean(axis=1)
            top10_pct[s0:s0 + chunk] = style_pct[top].mean(axis=1)
            top10_synfrac[s0:s0 + chunk] = synth_mask[top].mean(axis=1)
        per_query_rho[label] = rho_all
        per_model[label] = {
            "doc_emb": d_path, "query_emb": q_path,
            "mean_perquery_spearman_sim_vs_style_alldocs":
                round(float(rho_all.mean()), 4),
            "mean_perquery_spearman_sim_vs_style_syntheticdocs":
                round(float(rho_syn.mean()), 4),
            "mean_top10_style_score": round(float(top10_style.mean()), 4),
            "corpus_mean_style_score": round(corpus_mean_style, 4),
            "mean_top10_style_corpus_percentile":
                round(float(top10_pct.mean()), 2),
            "mean_top10_synthetic_fraction":
                round(float(top10_synfrac.mean()), 4),
        }
        print(f"[correlate] {label}: rho_all={rho_all.mean():.4f} "
              f"rho_syn={rho_syn.mean():.4f} "
              f"top10_style={top10_style.mean():.3f} "
              f"top10_style_pctile={top10_pct.mean():.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    pairwise = []
    labels = [l for l, *_ in loaded]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            d = per_query_rho[labels[i]] - per_query_rho[labels[j]]
            try:
                w = wilcoxon(d)
                p = float(f"{w.pvalue:.3g}")
            except ValueError:
                p = None
            entry = {
                "pair": f"{labels[i]} - {labels[j]}",
                "mean_perquery_delta_rho": round(float(d.mean()), 4),
                "wilcoxon_p": p,
                "pct_queries_higher": round(100 * float((d > 0).mean()), 2),
            }
            if p == 0.0:
                # scipy returns exactly 0 on float64 underflow; the honest
                # reportable statement is an upper bound, not "p = 0"
                entry["wilcoxon_p"] = 0.0
                entry["wilcoxon_p_reportable"] = "< 1e-308 (float64 underflow)"
            pairwise.append(entry)

    result = {
        "audit": "recipe-sensitivity audit -- does similarity track the "
                 "recipe-style score?",
        "n_queries_sampled": n_q,
        "seed": args.seed,
        "per_model": per_model,
        "pairwise_perquery_rho_deltas": pairwise,
        "detector_strip_artifacts": bool(z["strip_artifacts"]) if
                                    "strip_artifacts" in z else None,
        "note": "rho_syntheticdocs is the within-synthetic gradient (immune "
                "to the synthetic-vs-organic base-rate confound); "
                "top10_synthetic_fraction base rate = fraction of synthetic "
                "docs in the corpus = "
                f"{round(float(synth_mask.mean()), 4)}; "
                "mean_top10_style_corpus_percentile base rate = 50.0 (use it "
                "instead of a ratio of mean decision-function scores, which "
                "is not scale-meaningful)",
    }
    out = os.path.join(RESULTS, "style_probe_correlations.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] wrote {out}", flush=True)
    return result


V1_CORR_PATH = os.path.join(RESULTS, "style_probe_correlations_v1_artifact.json")


def headline(merged):
    """Derive the stripped-vs-v1 comparison block from the merged summary
    (pure function of the JSON already written; no recomputation)."""
    h = {"note": "H1 fix: artifact-stripped detector vs the v1 "
                 "artifact-inflated one. 'v1' columns come from "
                 "unstripped_reference (--no-strip-artifacts rerun, which "
                 "reproduces the released numbers exactly) and, for the "
                 "tracking row, results/style_probe_correlations_v1_artifact"
                 ".json."}
    ref = merged.get("unstripped_reference", {})
    fit, rfit = merged.get("fit"), ref.get("fit")
    tr, rtr = merged.get("transfer"), ref.get("transfer")
    if fit and rfit:
        h["holdout_auc_grouped"] = {"stripped": fit["holdout_auc_grouped"],
                                    "v1": rfit["holdout_auc_grouped"]}
        p, rp = fit.get("dose_panel"), rfit.get("dose_panel")
        if p and rp:
            boot = p.get("bootstrap", {})
            h["dose_panel_auc_heldout_sources_vs_heldout_anchors"] = {
                k: {"stripped": p["rungs"][k][
                        "auc_heldout_sources_vs_heldout_anchors"],
                    "stripped_ci95": boot.get("auc_ci95", {}).get(k),
                    "v1": rp["rungs"][k][
                        "auc_heldout_sources_vs_heldout_anchors"],
                    "v1_ci95": rp.get("bootstrap", {}).get(
                        "auc_ci95", {}).get(k)}
                for k in p["rungs"]}
            if boot:
                h["dose_panel_D1_minus_rung_ci95"] = boot[
                    "D1_appendixF_minus_rung_ci95"]
    if tr and rtr:
        h["transfer_auc_eq_vs_orig_docs"] = {
            "stripped": tr["auc_eq_vs_orig_docs"],
            "v1": rtr["auc_eq_vs_orig_docs"]}
        h["transfer_auc_eq_vs_disjoint_orig_docs"] = {
            "stripped": tr["disjoint_organic_negatives"][
                "auc_eq_vs_disjoint_orig_docs"],
            "v1": rtr["disjoint_organic_negatives"][
                "auc_eq_vs_disjoint_orig_docs"]}
        h["transfer_auc_nm_vs_orig_docs"] = {
            "stripped": tr["auc_nm_vs_orig_docs"],
            "v1": rtr["auc_nm_vs_orig_docs"]}
    corr = merged.get("correlate")
    if corr and os.path.exists(V1_CORR_PATH):
        with open(V1_CORR_PATH, encoding="utf-8") as f:
            v1c = json.load(f)
        key = "mean_perquery_spearman_sim_vs_style_alldocs"
        h["tracking_mean_perquery_rho"] = {
            lab: {"stripped": corr["per_model"][lab][key],
                  "v1": v1c["per_model"].get(lab, {}).get(key)}
            for lab in corr["per_model"]}
        pair = "ctrl-llm-6145 - ctrl-cas-6145"
        new_d = next((x for x in corr["pairwise_perquery_rho_deltas"]
                      if x["pair"] == pair), {})
        old_d = next((x for x in v1c["pairwise_perquery_rho_deltas"]
                      if x["pair"] == pair), {})
        h["tracking_llm_minus_cas"] = {
            "stripped": {k: new_d.get(k) for k in
                         ("mean_perquery_delta_rho", "pct_queries_higher",
                          "wilcoxon_p", "wilcoxon_p_reportable")},
            "v1": {k: old_d.get(k) for k in
                   ("mean_perquery_delta_rho", "pct_queries_higher",
                    "wilcoxon_p")}}
        h["top10_style_corpus_percentile_stripped"] = {
            lab: corr["per_model"][lab].get(
                "mean_top10_style_corpus_percentile")
            for lab in corr["per_model"]}
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="fit,score",
                    help="comma-joined subset of fit,score,correlate; empty "
                         "string with --out-summary just refreshes the merged "
                         "summary's derived 'headline' block (no compute)")
    ap.add_argument("--train-synthetic", choices=["all", "positives"],
                    default="positives",
                    help="which recipe outputs count as synthetic exemplars. "
                         "Default 'positives' (full rewrites -- the cleaner "
                         "style instrument; measured 2026-07-30 UNSTRIPPED: "
                         "holdout AUC .969, eq-vs-orig transfer .714 vs "
                         ".949/.575 for 'all', whose near-miss minimal edits "
                         "dilute the signal; stripped 'positives' = "
                         ".880/.804)")
    ap.add_argument("--top-ngrams", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-artifacts", dest="strip_artifacts",
                    action="store_true", default=True,
                    help="strip the organic-pipeline format artifacts (leading "
                         "'Problem:' header, ![](attached_image_N.png) "
                         "placeholders) from every text before fit/score. "
                         "DEFAULT ON (review finding H1). Measured "
                         "2026-07-30: holdout AUC .880, eq-vs-orig transfer "
                         ".804")
    ap.add_argument("--no-strip-artifacts", dest="strip_artifacts",
                    action="store_false",
                    help="reproduce the original v1 behavior (holdout .969, "
                         "eq transfer .714) -- artifact-inflated, kept only "
                         "for provenance")
    ap.add_argument("--panel", dest="panel", action="store_true", default=True,
                    help="emit the D1-D4 dose-rung specificity panel in "
                         "--stage fit (default ON)")
    ap.add_argument("--no-panel", dest="panel", action="store_false")
    ap.add_argument("--panel-bootstrap", type=int, default=2000,
                    metavar="N",
                    help="source-clustered paired bootstrap replicates for "
                         "the D1-D4 panel's 95%% CIs (0 = off; default 2000, "
                         "~30 s)")
    ap.add_argument("--transfer-disjoint", dest="transfer_disjoint",
                    action="store_true", default=True,
                    help="in --stage score, also report transfer AUCs against "
                         "ONLY the organic docs/queries whose problem text is "
                         "absent from the probe's training anchors (LOW "
                         "finding: 38.1%% of orig docs were training "
                         "negatives). Default ON")
    ap.add_argument("--no-transfer-disjoint", dest="transfer_disjoint",
                    action="store_false")
    ap.add_argument("--out-summary", default=None,
                    metavar="PATH",
                    help="write a merged one-file summary of every stage run "
                         "(fit AUC + artifact audit + D1-D4 panel + transfer) "
                         "to PATH, e.g. results/style_probe_stripped.json")
    ap.add_argument("--compare-summary", default=None, metavar="PATH",
                    help="embed the JSON at PATH under 'unstripped_reference' "
                         "in --out-summary (used to carry the "
                         "--no-strip-artifacts column alongside the stripped "
                         "one)")
    ap.add_argument("--n-queries", type=int, default=2000,
                    help="query sample size for --stage correlate (0 = all)")
    ap.add_argument("--model-emb", action="append", default=[],
                    metavar="LABEL=DOCS_NPZ,QUERIES_NPZ",
                    help="embedding caches for --stage correlate (repeatable; "
                         "globs ok)")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    summary = {"id": "style_probe_stripped" if args.strip_artifacts
                     else "style_probe_unstripped_v1",
               "date": "2026-07-30",
               "strip_artifacts": args.strip_artifacts,
               "train_synthetic": args.train_synthetic,
               "seed": args.seed,
               "stages_run": stages}
    for s in stages:
        if s == "fit":
            summary["fit"] = stage_fit(args)
        elif s == "score":
            summary["transfer"] = stage_score(args)
        elif s == "correlate":
            summary["correlate"] = stage_correlate(args)
        else:
            sys.exit(f"[error] unknown stage {s!r}")
    if args.out_summary:
        if args.compare_summary:
            with open(args.compare_summary, encoding="utf-8") as f:
                summary["unstripped_reference"] = json.load(f)
        # merge across invocations so `fit,score` then `correlate` (which needs
        # different arguments) accumulate into one summary file
        merged = {}
        if os.path.exists(args.out_summary):
            with open(args.out_summary, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("strip_artifacts") == args.strip_artifacts:
                merged = prev
                summary["stages_run"] = sorted(
                    set(prev.get("stages_run", [])) | set(stages))
            else:
                print(f"[warn] {args.out_summary} was written with "
                      f"strip_artifacts={prev.get('strip_artifacts')}; "
                      "overwriting rather than merging", flush=True)
        merged.update(summary)
        merged["headline"] = headline(merged)
        with open(args.out_summary, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"[done] wrote {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
