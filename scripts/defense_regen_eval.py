#!/usr/bin/env python3
"""Defense experiment: can regenerating a benchmark's golds make it more robust?

Rebuilds a slice of MathNet-Retrieve's easy tier with regenerated gold
documents at three graded prompt distances from the published recipe, then
scores the EXISTING checkpoints (no retraining) on each rebuilt eval. If
prompt-diversifying the eval's own generation flattens the recipe-matching
advantage, benchmark builders have a cheap, measurable mitigation.

PRE-REGISTERED PREDICTIONS (written 2026-08-19, before any generation ran;
scored against the paired D1-minus-ctrl-CAS R@1 gap on the shared slice):
  P-D1  On the exact-template regenerated eval the gap stays large: the
        attack survives regeneration under the same recipe (>= half its
        original-slice value).
  P-D2  The gap decreases monotonically as the eval's regeneration prompt
        moves away from the template: exact > paraphrase > style.
  P-D3  On the mixed eval (per-query prompt drawn round-robin from the
        three) the gap lies strictly between the exact and style endpoints.
  Falsifier for P-D2/P-D3: gap(style) >= gap(exact) on the shared slice.
Secondary observable, not registered as a prediction: D1-minus-D3 on the
style-regenerated eval, where D3's own training prompt now matches the eval.

Subcommands:
  generate  (GPU; vllm env)  regenerate one gold per sampled query per
            prompt variant (exact / paraphrase / style), judged by the same
            single-LLM judge as the training arms; writes a manifest.
  build     (login node)     from the manifest, write BEIR-style eval dirs
            orig / exact / paraphrase / style / mixed under
            data/defense_eval/, restricted to the shared slice of queries
            whose regenerations were judge-verified under ALL variants.
            The corpus keeps all 117,088 docs; only the sampled golds' text
            is replaced, so distractors and pool size are unchanged.

Contract test (no GPU):  python scripts/defense_regen_eval.py generate \
    --backend stub --n-queries 40 && python scripts/defense_regen_eval.py build
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_llm_pairs as G  # noqa: E402  (prompt builders, backends, JSON parse)

ROOT = Path(__file__).resolve().parent.parent
TIER_DIR = ROOT / "data" / "retrieve" / "easy"
OUT_ROOT = ROOT / "data" / "defense_eval"
MANIFEST = OUT_ROOT / "regen_manifest.json"
VARIANTS = ("exact", "paraphrase", "style")


def load_tier():
    queries = {}
    with open(TIER_DIR / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row
    qrels = {}
    with open(TIER_DIR / "qrels" / "test.tsv") as f:
        next(f)  # header
        for line in f:
            qid, did, _score = line.rstrip("\n").split("\t")
            qrels[qid] = did
    return queries, qrels


def sample_qids(queries, qrels, n, seed):
    pool = sorted(q for q in queries if q in qrels)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def gen_args(ns):
    """Namespace shaped like generate_llm_pairs's argparse output, so its
    build_gen_messages / make_backend / extract_json_robust work unchanged.
    positives/negatives per problem MUST stay 1/3: that keeps the generation
    prompt byte-identical to the one the training arms used."""
    a = argparse.Namespace(
        prompt_variant="exact", positives_per_problem=1, negatives_per_problem=3,
        model=ns.model, judge_model=None, backend=ns.backend,
        batch_size=8, max_model_len=ns.max_model_len,
        gpu_memory_utilization=0.92, gen_temperature=0.7,
        gen_max_tokens=getattr(ns, "gen_max_tokens", None) or 2048, judge_max_tokens=384, seed=ns.seed,
    )
    return a


def cmd_generate(ns):
    queries, qrels = load_tier()
    qids = sample_qids(queries, qrels, ns.n_queries, ns.seed)
    print(f"[defense] sampled {len(qids)} easy-tier queries (seed {ns.seed})")
    args = gen_args(ns)
    run = G.make_backend(args)
    if getattr(ns, "judge_backend", None) and ns.judge_backend != ns.backend:
        jargs = gen_args(ns); jargs.backend = ns.judge_backend; jargs.model = ns.judge_model
        run_judge = G.make_backend(jargs)
    else:
        run_judge = run
    variants = tuple(ns.variants.split(",")) if getattr(ns, "variants", None) else VARIANTS

    prev = None
    if getattr(ns, "only_missing", False) and MANIFEST.exists():
        prev = json.load(open(MANIFEST))
        qids = [q for q in qids if q in prev["queries"]]
    manifest = {"config": {"n_queries": ns.n_queries, "seed": ns.seed,
                           "model": ns.model, "backend": ns.backend,
                           "judge_backend": getattr(ns, "judge_backend", None) or ns.backend,
                           "judge_model": getattr(ns, "judge_model", None) or ns.model,
                           "tier": "easy", "variants": list(variants),
                           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                "queries": {q: {} for q in qids}}

    for variant in variants:
        args.prompt_variant = variant
        t0 = time.time()
        todo = qids
        if prev is not None:
            todo = [q for q in qids if not prev["queries"][q].get(variant, {}).get("text")]
            for q in qids:
                manifest["queries"][q][variant] = dict(prev["queries"][q].get(variant, {"text": "", "verified": False}))
            print(f"[defense:{variant}] only-missing: regenerating {len(todo)} of {len(qids)}", flush=True)
        batches = [G.build_gen_messages(queries[q]["text"], args) for q in todo]
        outs = run(batches, args.gen_max_tokens, args.gen_temperature)
        qids_this = todo
        cfg = G.VARIANTS[variant]
        cands, parse_fail = {}, 0
        for q, text in zip(qids_this, outs):
            obj = G.extract_json_robust(text)
            pos = (obj or {}).get(cfg["pos_key"]) or []
            item = pos[0] if pos and isinstance(pos[0], dict) else None
            cand = str(item.get("problem", "")).strip() if item else ""
            if cand:
                cands[q] = cand
            else:
                parse_fail += 1
        judge_qids = sorted(cands)
        jouts = run_judge([G.build_judge_messages(queries[q]["text"], cands[q])
                           for q in judge_qids], args.judge_max_tokens, 0.0)
        n_ok = 0
        for q, jtext in zip(judge_qids, jouts):
            j = G.extract_json(jtext)
            ok = bool(j) and j.get("verdict") == "equivalent"
            n_ok += ok
            manifest["queries"][q][variant] = {"text": cands[q], "verified": ok}
        for q in qids:
            manifest["queries"][q].setdefault(
                variant, {"text": "", "verified": False})
        print(f"[defense:{variant}] {len(qids)} queries -> {len(cands)} parsed "
              f"({parse_fail} parse failures), {n_ok} judge-verified, "
              f"{time.time() - t0:.0f}s")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[defense] manifest -> {MANIFEST}")


def cmd_judge(ns):
    """Re-judge every candidate in the manifest with the given backend (used
    after a Gemini generation run whose judge was a stub): sets 'verified'."""
    manifest = json.load(open(MANIFEST))
    queries, _ = load_tier()
    args = gen_args(ns)
    run = G.make_backend(args)
    variants = manifest["config"].get("variants", list(VARIANTS))
    for variant in variants:
        items = [(q, v[variant]["text"]) for q, v in manifest["queries"].items()
                 if v.get(variant, {}).get("text")]
        jouts = run([G.build_judge_messages(queries[q]["text"], cand) for q, cand in items],
                    args.judge_max_tokens, 0.0)
        n_ok = 0
        for (q, _cand), jtext in zip(items, jouts):
            j = G.extract_json(jtext)
            ok = bool(j) and j.get("verdict") == "equivalent"
            manifest["queries"][q][variant]["verified"] = ok
            n_ok += ok
        print(f"[defense:judge:{variant}] {len(items)} candidates, {n_ok} judge-verified", flush=True)
    manifest["config"]["judge_backend"] = ns.backend
    manifest["config"]["judge_model"] = ns.model
    manifest["config"]["judged_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[defense:judge] manifest updated -> {MANIFEST}")


def cmd_build(ns):
    manifest = json.load(open(MANIFEST))
    queries, qrels = load_tier()
    shared = sorted(q for q, v in manifest["queries"].items()
                    if all(v.get(x, {}).get("verified") for x in VARIANTS))
    print(f"[defense] shared slice: {len(shared)} / {len(manifest['queries'])} "
          f"queries verified under all {len(VARIANTS)} variants")
    if not shared:
        sys.exit("[FATAL] empty shared slice")

    def mixed_variant(qid):
        return VARIANTS[int(hashlib.sha1(qid.encode()).hexdigest(), 16) % len(VARIANTS)]

    replacements = {}  # eval_name -> {gold_doc_id: new_text}
    for name in ("orig",) + VARIANTS + ("mixed",):
        rep = {}
        if name != "orig":
            for q in shared:
                v = name if name != "mixed" else mixed_variant(q)
                rep[qrels[q]] = manifest["queries"][q][v]["text"]
        replacements[name] = rep

    shared_set = set(shared)
    for name, rep in replacements.items():
        d = OUT_ROOT / name
        (d / "qrels").mkdir(parents=True, exist_ok=True)
        n_swapped = 0
        with open(TIER_DIR / "corpus.jsonl") as src, open(d / "corpus.jsonl", "w") as dst:
            for line in src:
                row = json.loads(line)
                if row["_id"] in rep:
                    row["text"] = rep[row["_id"]]
                    n_swapped += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(d / "queries.jsonl", "w") as f:
            for q in shared:
                f.write(json.dumps(queries[q], ensure_ascii=False) + "\n")
        with open(d / "qrels" / "test.tsv", "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q in shared:
                f.write(f"{q}\t{qrels[q]}\t1\n")
        print(f"[defense:build] {name}: {len(shared)} queries, "
              f"{n_swapped} golds replaced -> {d}")

    summary = {"shared_slice": len(shared), "eval_dirs": sorted(replacements),
               "mixed_assignment_counts": {
                   v: sum(mixed_variant(q) == v for q in shared) for v in VARIANTS}}
    json.dump(summary, open(OUT_ROOT / "build_summary.json", "w"), indent=2)
    print(f"[defense:build] summary -> {OUT_ROOT / 'build_summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n-queries", type=int, default=1700)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    g.add_argument("--backend", choices=["vllm", "transformers", "stub", "gemini"], default="vllm")
    g.add_argument("--judge-backend", choices=["vllm", "transformers", "stub", "gemini"], default=None,
                   help="judge with a different backend (e.g. --backend gemini --judge-backend vllm)")
    g.add_argument("--judge-model", default="Qwen/Qwen3-32B-AWQ")
    g.add_argument("--variants", default=None, help="comma list to restrict (smoke tests)")
    g.add_argument("--only-missing", action="store_true", help="regenerate only entries with empty text in the existing manifest")
    g.add_argument("--gen-max-tokens", type=int, default=None)
    g.add_argument("--max-model-len", type=int, default=8192)
    b = sub.add_parser("build")
    j = sub.add_parser("judge")
    j.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    j.add_argument("--backend", choices=["vllm", "transformers", "stub"], default="vllm")
    j.add_argument("--seed", type=int, default=42)
    j.add_argument("--max-model-len", type=int, default=8192)
    for p in (g, b, j):
        p.add_argument("--out-root", default=None, help="eval dirs + manifest root (default data/defense_eval)")
    ns = ap.parse_args()
    if ns.out_root:
        global OUT_ROOT, MANIFEST
        OUT_ROOT = Path(ns.out_root); MANIFEST = OUT_ROOT / "regen_manifest.json"
    if ns.cmd == "generate":
        cmd_generate(ns)
    elif ns.cmd == "judge":
        cmd_judge(ns)
    else:
        cmd_build(ns)


if __name__ == "__main__":
    main()
