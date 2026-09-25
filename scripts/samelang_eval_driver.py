#!/usr/bin/env python3
"""E-R5 driver: score every trained model of the campaign on the same-language
organic-duplicate set (data/samelang_eval) under the SAME encoding
convention its cross-lingual run used.

  manifest   writes results/samelang_models.tsv from the existing cross-lingual
             result files (results/crosslingual_*.json and
             results/ranks/xling_*.summary.json): one row per model, with the
             model path and the query/doc prompt fields recorded in that file,
             so no encoding convention is re-decided here. Tags follow the
             rank-dump naming (ctrl-llm-s42, dose-unrelated-s42, ...).
  run        iterates the manifest and calls scripts/eval_crosslingual.py
             --eval-dir data/samelang_eval with --dump-ranks, skipping tags
             whose dump already exists (re-submittable).

Excluded from the manifest (not part of the figure or the verdict): other
backbones (mpnet, e5, bge), smoke tests, the non-math arms, and the two
D2 budget/selection controls (5591, d3sources).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, "results")
MANIFEST = os.path.join(RESULTS, "samelang_models.tsv")
EXCLUDE = re.compile(r"mpnet|bge|e5|smoke|nonmath|5591|d3sources|minilm", re.I)


def norm_tag(stem):
    """crosslingual_<stem>.json -> rank-dump style tag."""
    t = stem
    t = re.sub(r"-6145(?=(-s\d+)?$)", "", t)          # ctrl-cas-6145 -> ctrl-cas
    if re.match(r"^(ctrl-(cas|llm)|dose-[a-z]+)$", t):
        t += "-s42"                                     # seed-42 singles
    return t


def manifest():
    rows = {}

    def add(tag, d):
        if EXCLUDE.search(tag) or tag in rows:
            return
        model = d["model"]
        if not (model.startswith("models/") or model.startswith("Qwen/")):
            return
        rows[tag] = {"model": model,
                     "query_prompt_name": d.get("query_prompt_name") or "",
                     "query_prompt": d.get("query_prompt") or "",
                     "doc_prompt": d.get("doc_prompt") or ""}

    for p in sorted(glob.glob(os.path.join(RESULTS, "ranks", "xling_*.summary.json"))):
        tag = os.path.basename(p)[len("xling_"):-len(".summary.json")]
        with open(p, encoding="utf-8") as f:
            add(tag, json.load(f))
    for p in sorted(glob.glob(os.path.join(RESULTS, "crosslingual_*.json"))):
        stem = os.path.basename(p)[len("crosslingual_"):-len(".json")]
        with open(p, encoding="utf-8") as f:
            add(norm_tag(stem), json.load(f))
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("tag\tmodel\tquery_prompt_name\tquery_prompt\tdoc_prompt\n")
        for tag in sorted(rows):
            r = rows[tag]
            # prompts may contain newlines (Qwen3 "Instruct: ...\nQuery:"); escape
            # them so one model stays one TSV line (bug found on job 51261605).
            esc = lambda s: s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            f.write("\t".join([tag, r["model"], r["query_prompt_name"],
                               esc(r["query_prompt"]), esc(r["doc_prompt"])]) + "\n")
    print(f"[manifest] {len(rows)} models -> {MANIFEST}")
    for tag in sorted(rows):
        r = rows[tag]
        print(f"  {tag:34s} {r['model']:44s} pn={r['query_prompt_name']!r} "
              f"qp={r['query_prompt'][:30]!r} dp={r['doc_prompt'][:20]!r}")


def run(only=None):
    py = sys.executable
    unesc = lambda s: s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    with open(MANIFEST, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for l in f:
            if not l.strip():
                continue
            parts = l.rstrip("\n").split("\t")
            parts += [""] * (len(header) - len(parts))
            rows.append({k: unesc(v) for k, v in zip(header, parts)})
    os.chdir(PROJECT)
    for r in rows:
        tag = r["tag"]
        if only and tag not in only:
            continue
        dump = os.path.join("results", "ranks", f"samelang_{tag}.ranks.jsonl")
        if os.path.exists(dump) and os.path.getsize(dump) > 0:
            print(f"=== skip samelang {tag} (exists)", flush=True)
            continue
        model = r["model"]
        if model.startswith("models/") and not os.path.isdir(model):
            print(f"=== MISSING model dir for {tag}: {model}", flush=True)
            continue
        cmd = [py, "scripts/eval_crosslingual.py", "--eval-dir", "data/samelang_eval",
               "--model", model, "--device", "cuda", "--model-dtype", "bfloat16",
               "--max-seq-length", "1024",
               "--batch-size", "32" if re.search(r"4b", model, re.I) else "64",
               "--dump-ranks", dump,
               "--output", os.path.join("results", "ranks", f"samelang_{tag}.summary.json")]
        if r["query_prompt_name"]:
            cmd += ["--query-prompt-name", r["query_prompt_name"]]
        if r["query_prompt"]:
            cmd += ["--query-prompt", r["query_prompt"]]
        if r["doc_prompt"]:
            cmd += ["--doc-prompt", r["doc_prompt"]]
        print(f"=== eval samelang {tag}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"=== FAILED ({rc}) samelang {tag}", flush=True)
    print("== samelang driver finished ==", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "run"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    manifest() if a.mode == "manifest" else run(a.only)
