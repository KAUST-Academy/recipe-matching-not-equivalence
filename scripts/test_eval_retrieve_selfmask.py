#!/usr/bin/env python3
"""Self-contained regression tests for scripts/eval_retrieve.py (CPU, ~40 s).

Covers the two defects fixed on 2026-07-30:

  1. SELF-MASKING in --data-dir mode (M6). A local BEIR-style corpus may
     contain the query problem itself under the query's own _id (this is the
     case for data/crosslingual_eval). Without masking, that document is
     retrieved at rank 1 and the real gold is pushed to rank 2, producing the
     signature R@1 ~ 0 / R@5 ~ 100 that invalidated
     results/eval_crosslingual_rader-qwen25-7b.json.INVALID.

  2. figure6_separation ALIGNMENT (M5). Positives used to be appended for
     every query with a gold doc, while the near-miss aggregates were appended
     only for queries that actually have nm docs, so any query with a gold but
     no near-misses shifted all later positives against another query's
     negatives (7 such queries on the full hard tier; 13,872 of 14,993 pairs).
     The fixture below puts the nm-less query in the middle so the shift is
     observable: the buggy code averages the positives of {q1, q_no_nm}, the
     fixed code averages {q1, q2}.

Run:  python scripts/test_eval_retrieve_selfmask.py        (plain asserts)
      pytest -q scripts/test_eval_retrieve_selfmask.py     (also works)

Needs the cached sentence-transformers/all-MiniLM-L6-v2 (HF_HOME=.hf_cache)
and no GPU / no network.
"""

import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(PROJECT_ROOT, "scripts", "eval_retrieve.py")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# (_id, text) -- ids follow the MathNet-Retrieve scheme so the nm:: lookup works
QUERIES = [
    ("q1", "Find all positive integers n such that n^2 + 1 is a prime number."),
    ("q_no_nm", "Let ABC be a triangle with incenter I. Prove that AI bisects angle BAC."),
    ("q2", "Compute the sum of the first 100 positive even integers."),
]
CORPUS = [
    # q1: own doc (self-hit trap), gold rewrite, three near-misses
    ("q1", QUERIES[0][1]),
    ("q1::eq::easy", "Determine every positive integer n for which n^2 + 1 is prime."),
    ("q1::nm::0", "Find all positive integers n such that n^2 - 1 is a prime number."),
    ("q1::nm::1", "Find all positive integers n such that n^3 + 1 is a prime number."),
    ("q1::nm::2", "Find all positive integers n such that 2^n + 1 is a prime number."),
    # q_no_nm: own doc + an almost verbatim gold, but NO nm docs
    ("q_no_nm", QUERIES[1][1]),
    ("q_no_nm::eq::easy",
     "Let ABC be a triangle with incenter I. Show that AI bisects the angle BAC."),
    # q2: own doc, gold rewrite, two near-misses
    ("q2", QUERIES[2][1]),
    ("q2::eq::easy", "What is the total of the first one hundred even positive integers?"),
    ("q2::nm::0", "Compute the sum of the first 100 positive odd integers."),
    ("q2::nm::1", "Compute the product of the first 100 positive even integers."),
    # filler documents so the corpus is larger than the top-k window
    ("f::orig::0", "Prove that the square root of two is irrational."),
    ("f::orig::1", "A fair coin is tossed ten times; find the probability of exactly five heads."),
    ("f::orig::2", "Evaluate the integral of x times exp(-x) from zero to infinity."),
    ("f::orig::3", "Show that every group of prime order is cyclic."),
]
GOLD = {"q1": "q1::eq::easy", "q_no_nm": "q_no_nm::eq::easy", "q2": "q2::eq::easy"}


def _write_fixture(root):
    os.makedirs(os.path.join(root, "qrels"), exist_ok=True)
    with open(os.path.join(root, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for cid, text in CORPUS:
            f.write(json.dumps({"_id": cid, "text": text}) + "\n")
    with open(os.path.join(root, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, text in QUERIES:
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    with open(os.path.join(root, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, cid in GOLD.items():
            f.write(f"{qid}\t{cid}\t1\n")


def _run(data_dir, out_path, extra=()):
    cmd = [sys.executable, EVAL, "--data-dir", data_dir, "--model", MODEL,
           "--device", "cpu", "--batch-size", "8", "--output", out_path, *extra]
    env = dict(os.environ, HF_HOME=os.path.join(PROJECT_ROOT, ".hf_cache"),
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise AssertionError(f"eval_retrieve.py failed:\n{proc.stdout[-3000:]}\n"
                             f"{proc.stderr[-3000:]}")
    with open(out_path, encoding="utf-8") as f:
        return json.load(f), proc.stdout


def _expected_pos_sims():
    """Direct cosine similarities query->gold, computed independently."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    q = dict(QUERIES)
    c = dict(CORPUS)
    ids = list(GOLD)
    qe = m.encode([q[i] for i in ids], normalize_embeddings=True)
    de = m.encode([c[GOLD[i]] for i in ids], normalize_embeddings=True)
    return {i: float((qe[k] * de[k]).sum()) for k, i in enumerate(ids)}


def test_eval_retrieve_selfmask_and_alignment():
    os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "synthetic_eval")
        _write_fixture(data_dir)

        masked_ranks = os.path.join(tmp, "masked.ranks.jsonl")
        unmasked_ranks = os.path.join(tmp, "unmasked.ranks.jsonl")
        masked, log = _run(data_dir, os.path.join(tmp, "masked.json"),
                           extra=("--dump-ranks", masked_ranks))
        unmasked, _ = _run(data_dir, os.path.join(tmp, "unmasked.json"),
                           extra=("--no-self-mask", "--dump-ranks", unmasked_ranks))

        def rows(path):
            with open(path, encoding="utf-8") as f:
                return [json.loads(l) for l in f]

        # --- 1. self-masking ------------------------------------------------
        assert masked["self_masking"] is True
        assert masked["n_queries_with_own_doc_in_corpus"] == 3, masked
        assert masked["n_queries_self_masked"] == 3, masked
        assert masked["n_queries_own_doc_is_relevant"] == 0, masked
        assert "masked to -inf" in log
        # masked: a query's own document is unreachable, so it never appears in
        # its own ranking at all
        for r in rows(masked_ranks):
            assert r["qid"] not in r["top10_ids"], r
        # unmasked: the self-hit takes rank 1 for every query and displaces the
        # gold to rank 2 -- the R@1 ~ 0 / R@5 ~ 100 signature of the .INVALID file
        for r in rows(unmasked_ranks):
            assert r["top10_ids"][0] == r["qid"], r
            assert r["gold_rank"] >= 1, r      # never rank 0: the self-hit owns it
        assert unmasked["self_masking"] is False
        assert unmasked["overall"]["recall@1"] == 0.0, unmasked["overall"]
        assert unmasked["overall"]["recall@5"] == 100.0, unmasked["overall"]
        assert masked["overall"]["recall@1"] > unmasked["overall"]["recall@1"]

        # --- 2. figure6 alignment ------------------------------------------
        sep = masked["figure6_separation"]
        assert sep["n_queries_with_pos_and_neg"] == 2, sep
        assert sep["n_queries_gold_but_no_near_miss"] == 1, sep
        exp = _expected_pos_sims()
        want = round((exp["q1"] + exp["q2"]) / 2, 4)          # fixed behaviour
        buggy = round((exp["q1"] + exp["q_no_nm"]) / 2, 4)     # shifted pairing
        got = sep["mean_positive_sim"]
        assert abs(got - want) < 2e-3, f"mean_positive_sim {got} != aligned {want}"
        assert abs(want - buggy) > 1e-2, "fixture no longer separates the two cases"
        assert abs(got - buggy) > 1e-2, f"mean_positive_sim {got} matches buggy {buggy}"
    print("[ok] self-masking + figure6 alignment regression tests passed")


if __name__ == "__main__":
    test_eval_retrieve_selfmask_and_alignment()
