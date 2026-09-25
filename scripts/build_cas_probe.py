#!/usr/bin/env python3
"""E-R14 (2026-09-05): a near-miss-bearing,
generator-free equivalence probe.

The goal is a near-miss-bearing probe that no generator wrote, even a small
one: held-out CAS pairs with counterexampled negatives on problems absent from
every training file. This script builds exactly that, with the
verified arm's own machinery (scripts/generate_cas_pairs.py: computer-algebra-
verified positives, minimal-edit negatives that carry a numeric counterexample)
on the BENCHMARK'S OWN QUERY PROBLEMS that every training file of the campaign
excludes: the anchors of anchor_to_corpus_mapping_v2.json (a query mapped to
its corpus twins) are kept only when NONE of a query's twins occurs in any
data/*/pairs.jsonl source_id and none belongs to a mined duplicate cluster of
either retention probe. (The corpus problems outside the anchor set that carry
a relational span were all attempted by the verified arm's generator already;
the ones it left are the ones it could not transform, so they cannot seed a
probe.) Using the queries themselves also makes the probe comparable query by
query with the benchmark's tiers. No LLM writes or filters anything.

Per source problem ONE record is kept (the first transform family the
generator verified, in its own order rename / reformulate / mirror) with up to
--max-negatives of its distinct counterexampled negatives; sources with no
verified negative are dropped (the probe must carry near-misses).

Two BEIR-style evaluation directories (scripts/eval_crosslingual.py --eval-dir):
  data/casprobe_eval        the 27,817-problem public corpus of the retention
                            probes PLUS every probe positive and negative as new
                            documents (ids cp-pos-<sid>, cp-neg-<sid>-<k>);
                            queries = the source problems (self-masked as in the
                            retention probes); gold = the source's positive.
                            This mirrors the benchmark's hard tier: the gold is
                            an equivalent rewrite ranked against its own
                            minimal-edit near-misses inside a realistic corpus.
  data/casprobe_pairs_eval  only the probe documents (all positives and
                            negatives of all sources): the near-miss-only
                            ranking, the MELD pairs-only shape.

Readouts (E-R14) were registered before the probe was scored; their band is
restated in scripts/review3_verdict.py. Writes
data/cas_probe/probe.jsonl and results/cas_probe_build.json.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))


def load_gen():
    spec = importlib.util.spec_from_file_location(
        "gencas", os.path.join(PROJECT, "scripts", "generate_cas_pairs.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["gencas"] = m
    spec.loader.exec_module(m)
    return m


def held_out_ids():
    used = set()
    files = sorted(glob.glob(os.path.join(PROJECT, "data", "*", "pairs.jsonl")))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("source_id") or r.get("anchor_id")
                if s:
                    used.add(str(s))
    anchors = set(map(str, json.load(open(os.path.join(PROJECT, "anchor_to_corpus_mapping.json")))
                      ["exclude_corpus_ids"]))
    xl = json.load(open(os.path.join(PROJECT, "data", "crosslingual_eval", "leakage_exclude_ids.json")))
    xl_ids = set(map(str, xl["all_pair_member_ids"]))
    sl = json.load(open(os.path.join(PROJECT, "data", "samelang_eval", "clusters.json")))
    sl_ids = {str(m["id"]) for c in sl["clusters"] for m in c["members"]}
    v2 = json.load(open(os.path.join(PROJECT, "anchor_to_corpus_mapping_v2.json")))
    a2c = {a: [str(c) for c in cs] for a, cs in v2["anchor_to_corpus_ids"].items()}
    return used, anchors, xl_ids, sl_ids, a2c, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500, help="source problems to attempt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-negatives", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "data", "cas_probe"))
    ap.add_argument("--stats", default=os.path.join(PROJECT, "results", "cas_probe_build.json"))
    ap.add_argument("--eval-dir", default=os.path.join(PROJECT, "data", "casprobe_eval"))
    ap.add_argument("--pairs-eval-dir", default=os.path.join(PROJECT, "data", "casprobe_pairs_eval"))
    args = ap.parse_args()
    t0 = time.time()
    G = load_gen()
    import pandas as pd
    from pebble import ProcessPool

    used, anchors, xl_ids, sl_ids, a2c, files = held_out_ids()
    df = pd.read_parquet(os.path.join(PROJECT, "data", "mathnet_corpus.parquet"))
    expr_status = json.load(open(os.path.join(PROJECT, "results", "cas_expr_results.json")))
    row_of = {str(pid): (md, topics, lang) for pid, md, topics, lang in
              zip(df["id"], df["problem_markdown"], df["topics_flat"], df["language"])}
    # candidate pool: benchmark queries none of whose corpus twins any training
    # file or duplicate cluster contains; one corpus problem per query
    pool, why = [], Counter()
    anchor_of = {}
    for aid, cids in sorted(a2c.items()):
        if any(c in used for c in cids):
            why["twin_in_training"] += 1
            continue
        if any(c in xl_ids or c in sl_ids for c in cids):
            why["twin_in_duplicate_cluster"] += 1
            continue
        cid = next((c for c in cids if c in row_of), None)
        if cid is None:
            why["no_corpus_row"] += 1
            continue
        if cid in anchor_of:
            why["corpus_row_shared_by_queries"] += 1
            continue
        anchor_of[cid] = aid
        pool.append(cid)
    eligible = []
    for pid in pool:
        md, topics, lang = row_of[pid]
        spans = G.extract_spans(md)
        if not spans:
            why["no_math_span"] += 1
            continue
        flags = [expr_status.get(s, {}).get("status") == "relational" for _, _, s in spans]
        if not any(flags):
            why["no_relational_span"] += 1
            continue
        lang = None if (isinstance(lang, float) and math.isnan(lang)) else lang
        eligible.append((pid, md, G.primary_domain(topics), lang, flags))
    excluded = used | anchors | xl_ids | sl_ids
    print(f"[probe] corpus {len(df)}; v2 anchors {len(a2c)}; clean anchor pool {len(pool)} "
          f"({dict(why)}); training sources {len(used)}, xling {len(xl_ids)}, samelang {len(sl_ids)}; "
          f"eligible (relational span) {len(eligible)}", flush=True)
    rng = random.Random(args.seed)
    if args.target and args.target < len(eligible):
        by_dom = defaultdict(list)
        for e in eligible:
            by_dom[e[2]].append(e)
        chosen = []
        for d, items in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
            n_d = max(1, round(args.target * len(items) / len(eligible)))
            rng.shuffle(items)
            chosen.extend(items[:n_d])
        rng.shuffle(chosen)
        chosen = chosen[:args.target]
    else:
        chosen = list(eligible)
    print(f"[probe] attempting {len(chosen)} problems", flush=True)

    probe, agg = [], Counter()
    n_timeout = n_crash = 0
    with ProcessPool(max_workers=args.workers) as pool:
        futs = [(e, pool.schedule(G.process_problem, args=(e[0], e[1], e[2], e[3], e[4]),
                                  timeout=args.timeout)) for e in chosen]
        for i, (e, fut) in enumerate(futs):
            try:
                records, st = fut.result()
            except TimeoutError:
                n_timeout += 1
                continue
            except Exception:
                n_crash += 1
                continue
            agg["processed"] += 1
            rec = next((r for r in records if r.get("negatives")), None)
            if rec is None:
                agg["no_negative" if records else "no_positive"] += 1
                continue
            seen, negs = set(), []
            for ng in rec["negatives"]:
                t = str(ng.get("text", "")).strip()
                if t and t not in seen and t != rec["positive_text"]:
                    seen.add(t)
                    negs.append({"text": t, "edit": ng.get("edit"),
                                 "counterexample": ng.get("counterexample")})
                if len(negs) >= args.max_negatives:
                    break
            if not negs:
                agg["no_negative"] += 1
                continue
            probe.append({"source_id": e[0], "anchor_id": anchor_of[e[0]], "domain": e[2], "language": e[3],
                          "source_text": e[1], "positive_text": rec["positive_text"],
                          "transform_family": rec["transform_family"],
                          "positive_transforms": rec["positive_transforms"],
                          "negatives": negs, "verification": rec["verification"]})
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(futs)} attempted, {len(probe)} probe items, "
                      f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "probe.jsonl"), "w", encoding="utf-8") as f:
        for r in probe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- eval directories ------------------------------------------------
    base_corpus = os.path.join(PROJECT, "data", "crosslingual_eval", "corpus.jsonl")
    base_ids = set()
    for d, full in ((args.eval_dir, True), (args.pairs_eval_dir, False)):
        os.makedirs(os.path.join(d, "qrels"), exist_ok=True)
        with open(os.path.join(d, "corpus.jsonl"), "w", encoding="utf-8") as out:
            if full:
                with open(base_corpus, encoding="utf-8") as fh:
                    for line in fh:
                        out.write(line if line.endswith("\n") else line + "\n")
                        base_ids.add(json.loads(line)["_id"])
            for r in probe:
                s = r["source_id"]
                out.write(json.dumps({"_id": f"cp-pos-{s}", "title": "", "text": r["positive_text"]},
                                     ensure_ascii=False) + "\n")
                for k, ng in enumerate(r["negatives"]):
                    out.write(json.dumps({"_id": f"cp-neg-{s}-{k}", "title": "", "text": ng["text"]},
                                         ensure_ascii=False) + "\n")
        with open(os.path.join(d, "queries.jsonl"), "w", encoding="utf-8") as out, \
                open(os.path.join(d, "qrels", "test.tsv"), "w") as q:
            q.write("query-id\tcorpus-id\tscore\n")
            for r in probe:
                s = r["source_id"]
                lang = r["language"] or "unknown"
                out.write(json.dumps({"_id": s, "text": r["source_text"], "metadata": {
                    "lang": lang, "cluster_id": s, "cluster_size": 1, "anchor_id": r["anchor_id"],
                    "gold_ids": [f"cp-pos-{s}"], "gold_langs": [lang],
                    "n_negatives": len(r["negatives"]), "transform_family": r["transform_family"],
                    "domain": r["domain"], "min_confidence": "cas_verified",
                    "overlaps_retrieve_anchor": False}}, ensure_ascii=False) + "\n")
                q.write(f"{s}\tcp-pos-{s}\t1\n")
    missing_in_base = sum(r["source_id"] not in base_ids for r in probe)
    stats = {"generated_by": "scripts/build_cas_probe.py", "seed": args.seed,
             "held_out_definition": "benchmark query (anchor_to_corpus_mapping_v2.json) none of whose corpus "
                                    "twins is a source_id of any data/*/pairs.jsonl or a member of a "
                                    "cross-language or same-language duplicate cluster; one corpus row per query",
             "pool_exclusions": dict(why),
             "training_pair_files": [os.path.relpath(f, PROJECT) for f in files],
             "n_excluded_ids": len(excluded), "n_eligible_held_out": len(eligible),
             "n_attempted": len(chosen), "n_timeout": n_timeout, "n_crashed": n_crash,
             "outcomes": dict(agg), "n_probe_items": len(probe),
             "queries_missing_from_base_corpus": missing_in_base,
             "negatives_per_item": round(sum(len(r["negatives"]) for r in probe) / max(1, len(probe)), 3),
             "families": dict(Counter(r["transform_family"] for r in probe)),
             "edits": dict(Counter(ng["edit"] for r in probe for ng in r["negatives"])),
             "languages": dict(Counter(r["language"] or "unknown" for r in probe).most_common()),
             "domains": dict(Counter(r["domain"] for r in probe).most_common()),
             "eval_dirs": {"full_corpus": os.path.relpath(args.eval_dir, PROJECT),
                           "pairs_only": os.path.relpath(args.pairs_eval_dir, PROJECT)},
             "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(args.stats), exist_ok=True)
    json.dump(stats, open(args.stats, "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "training_pair_files"}, indent=1))


if __name__ == "__main__":
    main()
