#!/usr/bin/env python3
"""E-R14 driver: score every campaign model on the generator-free near-miss
probe (data/casprobe_eval, full corpus; data/casprobe_pairs_eval, near-miss
only), under the encoding convention recorded in results/samelang_models.tsv
(the E-R5 manifest: one row per model with its query/doc prompt fields), plus
any review-3 cell model found under models/fact-*/ that the manifest predates.

  python scripts/casprobe_eval_driver.py run [--only TAG ...]
Dumps: results/ranks/casprobe_<tag>.ranks.jsonl and casprobepairs_<tag>.ranks.jsonl,
summaries beside them. Re-submittable (skips existing dumps).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PROJECT, "results", "samelang_models.tsv")
NEW_CELLS = ("d5nonegs", "d5casnegs", "d1split", "d1twojudge", "bt2casnegs",
             # 2026-09-06: cells the E-R5 manifest predates (reference seeds 47-49,
             # the E-R6 full cell's seeds 43-44, the E-R7/E-R8/E-R9 cells)
             "d4casnegs", "d4llmnegs", "d4llmnegsfull", "d4unrelnegs", "d4unrelnegsfull",
             "casllmnegs", "btcasnegs")


def manifest_rows():
    unesc = lambda s: s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for l in f:
            if not l.strip():
                continue
            parts = l.rstrip("\n").split("\t")
            parts += [""] * (len(header) - len(parts))
            rows.append({k: unesc(v) for k, v in zip(header, parts)})
    have = {r["tag"] for r in rows}
    for cell in NEW_CELLS:
        for d in sorted(glob.glob(os.path.join(PROJECT, "models", f"fact-{cell}-6145-s*"))):
            seed = d.rsplit("-s", 1)[1]
            tag = f"fact-{cell}-s{seed}"
            if tag not in have and os.path.isdir(os.path.join(d, "final")):
                rows.append({"tag": tag, "model": os.path.relpath(os.path.join(d, "final"), PROJECT),
                             "query_prompt_name": "query", "query_prompt": "", "doc_prompt": ""})
    return rows


def run(only=None):
    py = sys.executable
    os.chdir(PROJECT)
    for r in manifest_rows():
        tag = r["tag"]
        if only and tag not in only:
            continue
        model = r["model"]
        if model.startswith("models/") and not os.path.isdir(model):
            print(f"=== MISSING model dir for {tag}: {model}", flush=True)
            continue
        for prefix, eval_dir in (("casprobe", "data/casprobe_eval"),
                                 ("casprobepairs", "data/casprobe_pairs_eval")):
            dump = os.path.join("results", "ranks", f"{prefix}_{tag}.ranks.jsonl")
            if os.path.exists(dump) and os.path.getsize(dump) > 0:
                print(f"=== skip {prefix} {tag} (exists)", flush=True)
                continue
            cmd = [py, "scripts/eval_crosslingual.py", "--eval-dir", eval_dir,
                   "--model", model, "--device", "cuda", "--model-dtype", "bfloat16",
                   "--max-seq-length", "1024",
                   "--batch-size", "32" if re.search(r"4b", model, re.I) else "64",
                   "--dump-ranks", dump,
                   "--output", os.path.join("results", "ranks", f"{prefix}_{tag}.summary.json")]
            if r["query_prompt_name"]:
                cmd += ["--query-prompt-name", r["query_prompt_name"]]
            if r["query_prompt"]:
                cmd += ["--query-prompt", r["query_prompt"]]
            if r["doc_prompt"]:
                cmd += ["--doc-prompt", r["doc_prompt"]]
            print(f"=== eval {prefix} {tag}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"=== FAILED ({rc}) {prefix} {tag}", flush=True)
    print("== casprobe driver finished ==", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "list"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.mode == "list":
        for r in manifest_rows():
            print(r["tag"], r["model"])
    else:
        run(a.only)
