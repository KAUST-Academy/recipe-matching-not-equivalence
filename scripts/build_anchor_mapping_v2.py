#!/usr/bin/env python3
"""Build the *v2* contamination-exclusion mapping between MathNet-Retrieve
anchors and the public MathNet corpus.

Why v2 exists
-------------
`scripts/build_anchor_mapping.py` (v1) matched anchors to corpus rows by
whitespace-normalized, lowercased **exact** text and kept only the **first**
corpus id per normalized text (`text2pid.setdefault`).  Two consequences,
both confirmed by the internal audit of 2026-07-30 (finding C1):

  1. 12,498 of the 27,817 corpus rows carry a literal ``"Problem:\\n\\n"``
     header that the benchmark queries do not.  Every one of those rows was
     invisible to an exact-text match, so ~6,140 anchors with a byte-identical
     corpus twin were never excluded (e.g. anchor ``usa_2017_6372fd`` <->
     corpus id ``0jyn``).
  2. `setdefault` kept one id per text, so corpus-internal duplicates of an
     excluded anchor stayed trainable (v1: 8,698 ids for 8,761 matched
     anchors; all-ids exact matching gives 8,810).

v2 rules (a match by any rule excludes **every** corpus id sharing the
matched normalized key)
-----------------------------------------------------------------------
  R1_exact_v1     whitespace-collapsed, lowercased exact text (= v1's rule,
                  but excluding *all* ids per key rather than the first).
  R2_boilerplate  additionally strips leading "Problem:"/"Exercise 3."-style
                  boilerplate headers and image placeholders
                  (``![...](...)``, ``attached_image_*.png``).
  R3_aggressive   additionally NFKC-normalizes, de-backslashes LaTeX commands
                  (``\\frac`` -> ``frac``) and deletes every non-alphanumeric
                  character -- i.e. `mine_duplicates.norm_text`, the same
                  aggressive key the duplicate miner hashes on.
  R4_crosslang    for anchors still unmatched: candidate generation over rare
                  LaTeX-formula 4-grams + word 3-shingles, accepted only by
                  `mine_duplicates.verify`'s *high*-confidence rules
                  (V1_text / V2_rare_math).  Catches translated reprints.
  R5_mined_partner  unions in the mined-duplicate partners (from
                  `results/duplicate_candidates.jsonl`) of every corpus id
                  excluded by R1-R4: the project's own verified reprints of
                  anchor problems.

Output: anchor_to_corpus_mapping_v2.json -- v1's schema (description,
generated, n_anchors, n_matched, n_unmatched, exclude_corpus_ids, mapping)
plus "v2_method", "rule_counts", "anchor_to_corpus_ids" and "delta_vs_v1".
`exclude_corpus_ids` is the drop-in replacement for the v1 list consumed by
train_invarembed.py / mine_duplicates.py.

Runtime ~3 min, CPU only, single process.
Usage: python scripts/build_anchor_mapping_v2.py [--no-crosslang]
"""

import argparse
import collections
import json
import math
import os
import re
import sys
import unicodedata

import duckdb

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import mine_duplicates as md  # noqa: E402  (norm/verify helpers reused verbatim)

PARQUET = f"{PROJECT}/data/mathnet_corpus.parquet"
QUERIES_LOCAL = f"{PROJECT}/data/retrieve/easy/queries.jsonl"
QUERIES_URL = ("https://huggingface.co/datasets/ShadenA/MathNet-Retrieve/"
               "resolve/main/easy/queries.jsonl")
DUPES = f"{PROJECT}/results/duplicate_candidates.jsonl"
V1_MAP = f"{PROJECT}/anchor_to_corpus_mapping.json"
OUT = f"{PROJECT}/anchor_to_corpus_mapping_v2.json"

# --------------------------------------------------------------------------
# normalization ladder
# --------------------------------------------------------------------------
# Leading problem/exercise headers.  Applied repeatedly so "## Problem 3."
# and "Problem:\n\nProblem 1." both collapse.  Anchored at string start only:
# never touches mid-document text.
BOILER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\*{0,2}"
    r"(?:problem|exercise|task|question|problema|probleme|problème|aufgabe|"
    r"zadanie|ejercicio|esercizio)"
    r"(?:\s*(?:no\.?|n[°º])?\s*\d{1,3})?\*{0,2}\s*[:.\)\-–]?\s*",
    re.I)
IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
IMG_BARE_RE = re.compile(r"\S*attached_image\S*")
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
NONALNUM_RE = re.compile(r"[^0-9a-zÀ-ɏͰ-ϿЀ-ӿ一-鿿]+")


def norm_v1(t):
    """v1's key: whitespace-collapsed, lowercased."""
    return re.sub(r"\s+", " ", (t or "")).strip().lower()


def strip_boilerplate(t):
    t = t or ""
    prev = None
    while prev != t:
        prev = t
        t = BOILER_RE.sub("", t, count=1)
    return t


def strip_images(t):
    return IMG_BARE_RE.sub(" ", IMG_MD_RE.sub(" ", t or ""))


def norm_boiler(t):
    """R2: v1 key after boilerplate-header and image-placeholder removal."""
    return re.sub(r"\s+", " ", strip_images(strip_boilerplate(t))).strip().lower()


def norm_aggressive(t):
    """R3: mine_duplicates.norm_text, plus the boilerplate/bare-image strip."""
    t = strip_images(strip_boilerplate(t))
    t = unicodedata.normalize("NFKC", t).lower()
    t = LATEX_CMD_RE.sub(lambda m: m.group(0)[1:], t)
    return NONALNUM_RE.sub("", t)


RULES = [("R1_exact_v1", norm_v1),
         ("R2_boilerplate", norm_boiler),
         ("R3_aggressive", norm_aggressive)]
MIN_KEY_LEN = 12   # guard against degenerate keys collapsing to near-nothing


def load_queries():
    if os.path.exists(QUERIES_LOCAL):
        qs = [json.loads(l) for l in open(QUERIES_LOCAL, encoding="utf-8")
              if l.strip()]
        src = QUERIES_LOCAL
    else:                                            # pragma: no cover
        import urllib.request
        qs = [json.loads(l) for l in
              urllib.request.urlopen(QUERIES_URL).read().decode().splitlines()
              if l.strip()]
        src = QUERIES_URL
    return qs, src


# --------------------------------------------------------------------------
# R4: cross-language / reprint search for anchors no key rule matched
# --------------------------------------------------------------------------
def crosslang_matches(unmatched, corpus_rows, topk=25):
    """Return {anchor_id: [(corpus_id, rule, evidence), ...]} using
    mine_duplicates' high-confidence verification only."""
    crow = {}
    for pid, text in corpus_rows:
        crow[pid] = {"id": pid, "text": text, "msig": md.math_sig(text),
                     "fp": md.num_fingerprint(text), "sh": None,
                     "nans": "", "tag": None, "year": None}
    gram_df = collections.Counter()
    for r in crow.values():
        for g in r["msig"]:
            gram_df[g] += 1
    n = max(1, len(crow))
    idf = collections.defaultdict(lambda: math.log(n))
    idf.update({g: math.log(n / c) for g, c in gram_df.items()})

    inv_rare = collections.defaultdict(list)
    for r in crow.values():
        for g in r["msig"]:
            if gram_df[g] <= md.RARE_DF:
                inv_rare[g].append(r["id"])
    inv_sh = collections.defaultdict(list)
    for r in crow.values():
        for h in md.shingles(r["text"]):
            inv_sh[h].append(r["id"])

    answer_df = collections.Counter()   # no answers for benchmark queries
    out = {}
    for q in unmatched:
        ar = {"id": q["_id"], "text": q["text"],
              "msig": md.math_sig(q["text"]),
              "fp": md.num_fingerprint(q["text"]), "sh": None,
              "nans": "", "tag": None, "year": None}
        score = collections.Counter()
        for g in ar["msig"]:
            if gram_df[g] <= md.RARE_DF:
                for pid in inv_rare.get(g, ()):
                    score[pid] += idf[g]
        for h in md.shingles(q["text"]):
            for pid in inv_sh.get(h, ()):
                score[pid] += 0.3
        for pid, _ in score.most_common(topk):
            v = md.verify(ar, crow[pid], answer_df, gram_df, idf)
            if v and v[1] == "high":
                out.setdefault(q["_id"], []).append((pid, v[0], v[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-crosslang", action="store_true",
                    help="skip R4 (rare-formula cross-language search)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    queries, qsrc = load_queries()
    print(f"anchor queries: {len(queries)} from {qsrc}")
    corpus = duckdb.sql(
        f"SELECT id, problem_markdown FROM '{PARQUET}'").fetchall()
    corpus = [(pid, t or "") for pid, t in corpus]
    print(f"corpus rows: {len(corpus)}")

    # per-rule inverted indexes: normalized key -> ALL corpus ids
    indexes = []
    for name, fn in RULES:
        idx = collections.defaultdict(list)
        for pid, text in corpus:
            k = fn(text)
            if len(k) >= MIN_KEY_LEN:
                idx[k].append(pid)
        indexes.append((name, fn, idx))
        multi = sum(1 for v in idx.values() if len(v) > 1)
        print(f"  index {name}: {len(idx)} keys ({multi} keys with >1 corpus id)")

    anchor_ids = collections.defaultdict(set)     # anchor_id -> corpus ids
    anchor_rule = {}                              # anchor_id -> first rule hit
    pair_rule = {}                                # (anchor_id, pid) -> rule
    rule_anchor_hits = collections.Counter()      # anchors matched, by rule
    rule_new_anchors = collections.Counter()      # anchors first matched by rule
    rule_ids = collections.defaultdict(set)       # ids contributed, by rule

    for q in queries:
        qid, text = q["_id"], q["text"]
        for name, fn, idx in indexes:
            k = fn(text)
            if len(k) < MIN_KEY_LEN:
                continue
            hits = idx.get(k)
            if hits:
                rule_anchor_hits[name] += 1
                if qid not in anchor_rule:
                    anchor_rule[qid] = name
                    rule_new_anchors[name] += 1
                anchor_ids[qid].update(hits)
                rule_ids[name].update(hits)
                for pid in hits:
                    pair_rule.setdefault((qid, pid), name)

    unmatched = [q for q in queries if q["_id"] not in anchor_ids]
    print(f"after key rules: {len(anchor_ids)} matched, "
          f"{len(unmatched)} unmatched")

    cl_evidence = {}
    if unmatched and not args.no_crosslang:
        print(f"R4: searching cross-language/reprint twins for "
              f"{len(unmatched)} residual anchors ...")
        cl = crosslang_matches(unmatched, corpus)
        for qid, lst in cl.items():
            anchor_rule.setdefault(qid, "R4_crosslang")
            rule_new_anchors["R4_crosslang"] += 1
            rule_anchor_hits["R4_crosslang"] += 1
            for pid, rule, ev in lst:
                anchor_ids[qid].add(pid)
                rule_ids["R4_crosslang"].add(pid)
                pair_rule.setdefault((qid, pid), "R4_crosslang")
            cl_evidence[qid] = [{"corpus_id": p, "verify_rule": r,
                                 "evidence": e} for p, r, e in lst]
        print(f"R4 accepted {len(cl)} anchors "
              f"({sum(len(v) for v in cl.values())} corpus ids)")
        unmatched = [q for q in queries if q["_id"] not in anchor_ids]

    # ---- R5: mined-duplicate partners of every excluded id ----
    partners = collections.defaultdict(set)
    n_dupe_pairs = 0
    if os.path.exists(DUPES):
        for line in open(DUPES, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            a, b = d["id_a"], d["id_b"]
            partners[a].add(b)
            partners[b].add(a)
            n_dupe_pairs += 1
    else:                                            # pragma: no cover
        print(f"WARNING: {DUPES} missing -- R5 inactive")
    matched_ids = set().union(*anchor_ids.values()) if anchor_ids else set()
    added_partners = set()
    for qid, ids in list(anchor_ids.items()):
        extra = set()
        for pid in list(ids):
            extra |= partners.get(pid, set())
        extra -= ids
        if extra:
            anchor_ids[qid] |= extra
            added_partners |= extra
            for pid in extra:
                pair_rule.setdefault((qid, pid), "R5_mined_partner")
    rule_ids["R5_mined_partner"] = added_partners - matched_ids
    print(f"R5: {n_dupe_pairs} mined pairs -> "
          f"{len(rule_ids['R5_mined_partner'])} additional corpus ids")

    exclude = sorted(set().union(*anchor_ids.values())) if anchor_ids else []
    v1 = json.load(open(V1_MAP, encoding="utf-8"))
    v1_ids = set(v1["exclude_corpus_ids"])

    # one entry per (anchor, corpus id) pair; "rule" is the rule that
    # contributed THAT pair, "anchor_rule" the anchor's strongest (first) rule
    mapping = [{"anchor_id": qid, "corpus_id": pid,
                "rule": pair_rule.get((qid, pid)),
                "anchor_rule": anchor_rule.get(qid)}
               for qid in sorted(anchor_ids) for pid in sorted(anchor_ids[qid])]
    pairs_by_rule = collections.Counter(m["rule"] for m in mapping)

    out = {
        "description": (
            "v2 contamination-exclusion mapping between MathNet-Retrieve "
            "anchor queries (easy tier; queries are identical across tiers) "
            "and the public ShadenA/MathNet corpus (config 'all'). Corpus IDs "
            "in exclude_corpus_ids MUST be excluded from retriever training "
            "data. Supersedes anchor_to_corpus_mapping.json (v1), whose "
            "exact-text/first-id-only matching missed every corpus row "
            "carrying a 'Problem:' header (12,498 rows) and every "
            "corpus-internal duplicate of a matched row."),
        "generated": "2026-07-30",
        "supersedes": "anchor_to_corpus_mapping.json",
        "queries_source": qsrc,
        "corpus": PARQUET,
        "n_anchors": len(queries),
        "n_matched": len(anchor_ids),
        "n_unmatched": len(unmatched),
        "n_pairs": len(mapping),
        "exclude_corpus_ids": exclude,
        "mapping": mapping,
        "anchor_to_corpus_ids": {q: sorted(v) for q, v in
                                 sorted(anchor_ids.items())},
        "unmatched_anchor_ids": sorted(q["_id"] for q in unmatched),
        "v2_method": {
            "rules": [
                {"id": "R1_exact_v1",
                 "desc": ("whitespace-collapsed lowercased exact text (v1's "
                          "rule) but excluding ALL corpus ids per key")},
                {"id": "R2_boilerplate",
                 "desc": ("R1 after stripping leading problem/exercise "
                          "headers and image placeholders (markdown images, "
                          "attached_image_*.png)")},
                {"id": "R3_aggressive",
                 "desc": ("mine_duplicates.norm_text: NFKC, lowercase, LaTeX "
                          "commands de-backslashed, all non-alphanumerics "
                          "deleted (after the R2 strips)")},
                {"id": "R4_crosslang",
                 "desc": ("residual anchors only: rare-LaTeX-4-gram + word-"
                          "3-shingle candidate generation verified by "
                          "mine_duplicates.verify high-confidence rules "
                          "(V1_text / V2_rare_math)"),
                 "skipped": bool(args.no_crosslang)},
                {"id": "R5_mined_partner",
                 "desc": ("union of mined-duplicate partners "
                          "(results/duplicate_candidates.jsonl) of every "
                          "corpus id excluded by R1-R4")},
            ],
            "min_normalized_key_len": MIN_KEY_LEN,
            "all_ids_per_key": True,
            "mined_duplicate_pairs_loaded": n_dupe_pairs,
        },
        "rule_counts": {
            "anchor_corpus_pairs_by_rule": dict(pairs_by_rule),
            "anchors_matched_by_rule_any": dict(rule_anchor_hits),
            "anchors_first_matched_by_rule": dict(rule_new_anchors),
            "corpus_ids_contributed_by_rule": {k: len(v) for k, v
                                               in rule_ids.items()},
            "corpus_ids_new_vs_v1_by_rule": {
                k: len(v - v1_ids) for k, v in rule_ids.items()},
        },
        "delta_vs_v1": {
            "v1_n_matched": v1["n_matched"],
            "v1_n_exclude_corpus_ids": len(v1_ids),
            "v2_n_matched": len(anchor_ids),
            "v2_n_exclude_corpus_ids": len(exclude),
            "anchors_newly_matched": len(anchor_ids) - v1["n_matched"],
            "corpus_ids_added": len(set(exclude) - v1_ids),
            "corpus_ids_dropped": len(v1_ids - set(exclude)),
        },
        "crosslang_evidence": cl_evidence,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in
                      ("n_anchors", "n_matched", "n_unmatched", "n_pairs",
                       "rule_counts", "delta_vs_v1")}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
