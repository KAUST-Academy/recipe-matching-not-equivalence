#!/usr/bin/env python3
r"""SABER-Math recipe-matching attack: training pairs built with THEIR recipe.

The MathNet attack showed that a 0.6B
embedder trained on pairs made with the benchmark's own generation prompt
(via a DIFFERENT generator) games the benchmark without gaining any real
equivalence ability. This script builds the closest analog for SABER-Math
(arXiv:2606.29894): training pairs manufactured with SABER's own construction
recipe, via Qwen3-32B-AWQ (their pipeline used GPT-OSS-120B — a different
generator, so any transfer is again carried by the RECIPE, not the model).

SABER's recipe (paper §3 + official code, reference copies in
data/saber/upstream_ref/):
  1. extract a <=30-word imperative "core idea" summary of each solution
     (annotate/ideas/prompts.py::extract_core_idea_query — used VERBATIM
     here, typos included);
  2. mark a problem pair summary-relevant when the Jaccard similarity of
     their tokenized summaries exceeds tau_summ ~ 0.211 (their exact
     tokenizer [a-zA-Z0-9^_]+ + their exact stop-word list, loaded from
     their stop_words.py; similarities/compute/jaccard.py);
  3. (not replicated: the MathWorld topic-ontology BMA signal and the
     GPT-OSS-120B Swiss tournament — too heavy. FIDELITY NOTE: the summary
     signal drives 2/3 of each query's candidates and their Fig. 4 shows
     summary-selected candidates rank at least as high as topic-selected
     ones under the final tournament ranking, so the summary channel is the
     dominant learnable surface of the recipe.)

TRAINING CHANNELS (grouped trainer rows for train_invarembed.py):
  saber_pair      anchor  = problem statement of x
                  positive= "Problem: {statement_y}\n\nSolution: {solution_y}"
                            (the benchmark's EXACT statement-full document
                            template) for a partner y with J(x,y) >= tau
                  negatives = same template for problems in the mid-band
                            0 < J < neg-band-max (deceptively related but
                            below the recipe's relevance bar)
  saber_summary   anchor  = problem statement of x
                  positive= x's own generated core-idea summary
                  negatives = core-idea summaries of J=0 problems
                  (the "query <-> their-style solution-summary" light analog)

DISJOINTNESS (--stage filter): sources are the matched-budget list
data/pairs/source_ids.txt (already excludes all MathNet-Retrieve anchors and
cross-lingual-eval members). We additionally DROP any source whose problem
statement overlaps a SABER eval text: word-5-gram containment >= 0.35
against (a) the 1,000 query problems [mandatory gate] and (b) all 71,117
candidate-document problems [strict gate], plus a language-independent
numeric-fingerprint gate against the queries (>=5 numeric literals with
Jaccard >= 0.6). Cross-language duplicates (our non-EN problems vs their
English-translated corpus) can evade n-gram matching for DOCUMENTS; the
numeric gate mitigates this for QUERIES, where disjointness is load-bearing.
Every count lands in results/saber_disjointness.json.

STAGES (chained by scripts/saber_attack.slurm):
  --stage filter     CPU, minutes. source ids + SABER parquets ->
                     data/saber_attack/source_ids.txt + disjointness report
  --stage summaries  GPU (vllm env). Their idea-extraction prompt through
                     Qwen3-32B-AWQ -> data/saber_attack/summaries.jsonl
                     (--backend stub = deterministic fake summaries for
                     login-node contract tests; NEVER train on stub output)
  --stage pairs      CPU, minutes. summaries -> their Jaccard mining ->
                     data/saber_attack/pairs.jsonl (trainer format)
                     + results/saber_pairs_summary.json

Usage:
  python scripts/generate_saber_pairs.py --stage filter
  python scripts/generate_saber_pairs.py --stage summaries --backend vllm \
      --model Qwen/Qwen3-32B-AWQ
  python scripts/generate_saber_pairs.py --stage pairs
"""
import argparse
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/ibex/user/habiam0b/MathNet_Follow_Up")
sys.path.insert(0, str(ROOT / "scripts"))
from generate_llm_pairs import extract_json  # noqa: E402 (shared JSON parser)

UPSTREAM = ROOT / "data" / "saber" / "upstream_ref"
ATTACK_DIR = ROOT / "data" / "saber_attack"
RESULTS = ROOT / "results"

# --------------------------------------------------------------------------
# Their exact summary-tokenization (similarities/compute/jaccard.py).
# STOP_WORDS is loaded from the byte-identical copy of their stop_words.py.
# --------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9^_]+")


def load_their_stop_words() -> frozenset:
    spec = importlib.util.spec_from_file_location(
        "saber_stop_words", UPSTREAM / "stop_words.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STOP_WORDS


def saber_tokenize(text: str, stopwords) -> set:
    return {t for t in TOKEN_PATTERN.findall(text.lower())
            if t not in stopwords}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# Their idea-extraction prompt, VERBATIM from
# build_benchmark/annotate/ideas/prompts.py (typos preserved on purpose:
# "anotator", "untrial", "thoughs", "sinlge", "supporingIdeas").
# --------------------------------------------------------------------------
def extract_core_idea_query(statement: str, solution: str) -> str:
    return (
        """# INSTRUCTION
You are an expert mathematical anotator tasked with identifying the *core idea* - the central mathematical insight, from a math problem.

# GOALS
1. Identify what makes the solution work conceptually, not how to carry it out. Capture the untrial step or idea that is the greatest hint for the solution.
2. Never include any multi-step reasoning, equations or numeric computations. Don't include any anotations that are in the solution, but not in the original problem statement.
3. Never try to solve the problem on your own and don't include your reasoning or thoughs.
4. Output a sinlge valid JSON object matching the schema below.
5. Structure the ideas imperatively so they look like you are giving a hint to someone.
5. If the problem seems too easy, straightforward or you can't identify a core idea, store its value as 'null' and set the 'noCoreIdea' to 'true'

# SCHEMA
```json
{
    "noCoreIdea": <true|false>,
    "coreIdea": "<string - one short sentence (up to 30 words) naming the main insight to the problem>",
    "supporingIdeas: ["<strings - 0-3 short techniques phrases>"],
    "keywords": ["<strings - 1-2 word phrases summarizing the ideas, theorems, etc. in the solution"],
    "confidence": <0.0-1.0>
}

Here are the problem statement and solution:

Statement: {"""
        + statement
        + """}

Solution: {"""
        + solution
        + """}"""
    )


# The benchmark's EXACT statement-full document template (benchmark.py).
def doc_template(problem: str, solution: str) -> str:
    return f"Problem: {problem}\n\nSolution: {solution}"


# --------------------------------------------------------------------------
# Shared data loading
# --------------------------------------------------------------------------
def load_sources(args):
    """[(id, problem, first_solution_or_None)] for the matched-budget list."""
    import pandas as pd
    ids = [i for i in Path(args.source_ids_file).read_text().split() if i]
    df = pd.read_parquet(args.corpus,
                         columns=["id", "problem_markdown", "solutions_markdown"])
    df = df[df["id"].isin(set(ids))].set_index("id").loc[ids].reset_index()
    out = []
    for r in df.itertuples(index=False):
        sols = list(r.solutions_markdown) if r.solutions_markdown is not None else []
        sol = (sols[0].strip() if sols and sols[0] and sols[0].strip() else None)
        out.append((r.id, r.problem_markdown, sol))
    print(f"[data] {len(out)} matched-budget sources "
          f"({sum(1 for _, _, s in out if s)} with a solution)", flush=True)
    return out


def load_saber_problems():
    from eval_saber import load_saber
    q_df, d_df = load_saber()
    return list(q_df["problem"]), list(d_df["problem"])


# --------------------------------------------------------------------------
# STAGE filter — disjointness gate + report
# --------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def word_grams(text: str, n: int = 5) -> set:
    toks = WORD_RE.findall(text.lower())
    return {hash(tuple(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def stage_filter(args):
    t0 = time.time()
    sources = load_sources(args)
    saber_q, saber_d = load_saber_problems()

    src_grams = {sid: word_grams(p) for sid, p, _ in sources}

    def build_index(texts):
        idx = defaultdict(list)
        gram_sets = []
        for i, t in enumerate(texts):
            g = word_grams(t)
            gram_sets.append(g)
            for h in g:
                idx[h].append(i)
        return idx, gram_sets

    def max_containment(grams, idx):
        """max over eval texts of |src ∩ eval| / |src|."""
        if not grams:
            return 0.0, None
        counts = Counter()
        for h in grams:
            for i in idx.get(h, ()):
                counts[i] += 1
        if not counts:
            return 0.0, None
        best, n = max(counts.items(), key=lambda kv: kv[1])
        return n / len(grams), best

    print("[filter] indexing SABER query + document problems...", flush=True)
    q_idx, _ = build_index(saber_q)
    d_idx, _ = build_index(saber_d)
    q_nums = [set(NUM_RE.findall(t)) for t in saber_q]

    kept, dropped = [], []
    cont_q_hist, cont_d_hist = [], []
    for sid, prob, sol in sources:
        g = src_grams[sid]
        cq, cq_i = max_containment(g, q_idx)
        cd, cd_i = max_containment(g, d_idx)
        cont_q_hist.append(cq)
        cont_d_hist.append(cd)
        reason = None
        if cq >= args.containment_threshold:
            reason = f"query_overlap({cq:.2f},q{cq_i})"
        elif cd >= args.containment_threshold:
            reason = f"doc_overlap({cd:.2f},d{cd_i})"
        else:
            nums = set(NUM_RE.findall(prob))
            if len(nums) >= 5:
                for qi, qn in enumerate(q_nums):
                    if len(qn) >= 5 and jaccard(nums, qn) >= args.numeric_threshold:
                        reason = f"query_numeric_fingerprint(q{qi})"
                        break
        if reason:
            dropped.append({"source_id": sid, "reason": reason})
        elif sol is None:
            dropped.append({"source_id": sid, "reason": "no_solution"})
        else:
            kept.append(sid)

    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    (ATTACK_DIR / "source_ids.txt").write_text("\n".join(kept) + "\n")

    import numpy as np
    by_reason = Counter(d["reason"].split("(")[0] for d in dropped)
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "method": {
            "gates": [
                "word-5-gram containment >= "
                f"{args.containment_threshold} vs the 1,000 SABER query "
                "problems (MANDATORY: eval-query disjointness)",
                "word-5-gram containment >= "
                f"{args.containment_threshold} vs all 71,117 SABER candidate-"
                "document problems (strict extra gate)",
                f"numeric-fingerprint Jaccard >= {args.numeric_threshold} "
                "(>=5 numeric literals) vs query problems (language-"
                "independent secondary gate)",
                "sources without any solution dropped (summary prompt "
                "needs Statement + Solution)",
            ],
            "caveat": "n-gram matching is language-bound: a non-EN source "
                      "problem that appears English-translated in SABER's "
                      "corpus can evade the DOCUMENT gate; the numeric gate "
                      "partially covers this for the QUERY gate, which is "
                      "the one the attack's claim depends on.",
        },
        "inputs": {
            "source_ids_file": str(args.source_ids_file),
            "n_sources_in": len(sources),
        },
        "results": {
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "dropped_by_reason": dict(by_reason),
            "containment_vs_queries": {
                "mean": round(float(np.mean(cont_q_hist)), 4),
                "p99": round(float(np.percentile(cont_q_hist, 99)), 4),
                "max": round(float(np.max(cont_q_hist)), 4),
            },
            "containment_vs_documents": {
                "mean": round(float(np.mean(cont_d_hist)), 4),
                "p99": round(float(np.percentile(cont_d_hist, 99)), 4),
                "max": round(float(np.max(cont_d_hist)), 4),
            },
        },
        "dropped_examples": dropped[:25],
        "output": str(ATTACK_DIR / "source_ids.txt"),
        "runtime_s": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "saber_disjointness.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[filter] kept {len(kept)} / {len(sources)} sources; dropped "
          f"{dict(by_reason)}; report -> results/saber_disjointness.json",
          flush=True)


# --------------------------------------------------------------------------
# STAGE summaries — their prompt through Qwen3-32B-AWQ (or stub)
# --------------------------------------------------------------------------
def stage_summaries(args):
    t0 = time.time()
    src_file = ATTACK_DIR / "source_ids.txt"
    if not src_file.exists():
        sys.exit("[FATAL] run --stage filter first (data/saber_attack/"
                 "source_ids.txt missing)")
    args.source_ids_file = str(src_file)
    sources = [s for s in load_sources(args) if s[2] is not None]
    if args.max_problems:
        sources = sources[: args.max_problems]
        print(f"[summaries] capped to {len(sources)} problems", flush=True)

    # Char budget so no prompt exceeds --max-model-len (their GPT-OSS pipeline
    # had 131K context and needed no truncation; this only clips outliers).
    def clip(t, n):
        return t if len(t) <= n else t[:n] + " [...truncated]"
    prompts = [[{"role": "user",
                 "content": extract_core_idea_query(clip(p, 6000),
                                                    clip(s, 12000))}]
               for _sid, p, s in sources]

    if args.backend == "stub":
        def run(batches):
            outs = []
            for msgs in batches:
                stmt = msgs[0]["content"].split("Statement: {", 1)[1]
                toks = sorted(saber_tokenize(stmt[:400],
                                             load_their_stop_words()))[:8]
                outs.append(json.dumps({
                    "noCoreIdea": False,
                    "coreIdea": "Use " + " ".join(toks[:6]) + " to reduce the problem.",
                    "supporingIdeas": ["stub technique"],
                    "keywords": toks[:2],
                    "confidence": 0.5}))
            return outs
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="auto", seed=args.seed,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=True)

        def run(batches):
            sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                seed=args.seed)
            outs = llm.chat(batches, sp,
                            chat_template_kwargs={"enable_thinking": False})
            return [o.outputs[0].text for o in outs]

    outs = run(prompts)
    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_no_idea = n_fail = 0
    with open(args.summaries_file, "w") as f:
        for (sid, _p, _s), text in zip(sources, outs):
            obj = extract_json(text)
            if not obj:
                n_fail += 1
                continue
            core = obj.get("coreIdea")
            no_idea = bool(obj.get("noCoreIdea")) or not core \
                or not str(core).strip()
            if no_idea:
                n_no_idea += 1
                continue
            n_ok += 1
            f.write(json.dumps({
                "source_id": sid,
                "core_idea": str(core).strip(),
                "supporting_ideas": obj.get("supporingIdeas")
                or obj.get("supportingIdeas") or [],
                "keywords": obj.get("keywords") or [],
                "confidence": obj.get("confidence"),
                "generator_model": args.model if args.backend != "stub"
                else "STUB (never train on this)",
            }, ensure_ascii=False) + "\n")
    print(f"[summaries] {len(sources)} problems -> {n_ok} summaries "
          f"({n_no_idea} noCoreIdea/empty, {n_fail} JSON failures) in "
          f"{time.time() - t0:.0f}s -> {args.summaries_file}", flush=True)
    if n_ok == 0:
        sys.exit("[FATAL] no summaries produced")


# --------------------------------------------------------------------------
# STAGE pairs — their Jaccard relevance logic -> trainer rows
# --------------------------------------------------------------------------
def stage_pairs(args):
    t0 = time.time()
    rng = random.Random(args.seed)
    stop = load_their_stop_words()

    rows = [json.loads(l) for l in open(args.summaries_file)]
    if any("STUB" in r.get("generator_model", "") for r in rows):
        print("[pairs] WARNING: stub summaries detected — contract test "
              "only, never train on the resulting pairs", flush=True)
    args.source_ids_file = str(ATTACK_DIR / "source_ids.txt")
    src = {sid: (p, s) for sid, p, s in load_sources(args) if s is not None}
    rows = [r for r in rows if r["source_id"] in src]
    ids = [r["source_id"] for r in rows]
    tok_sets = [saber_tokenize(r["core_idea"], stop) for r in rows]
    print(f"[pairs] {len(rows)} summaries; median tokens/summary = "
          f"{sorted(len(t) for t in tok_sets)[len(tok_sets) // 2]}", flush=True)

    # inverted index over summary tokens -> candidate pairs sharing >=2 tokens
    inv = defaultdict(list)
    for i, ts in enumerate(tok_sets):
        for t in ts:
            inv[t].append(i)
    pos_pairs = defaultdict(list)     # i -> [(j, J)]
    neg_band = defaultdict(list)      # i -> [j] with 0 < J < neg_band_max
    j_hist = Counter()
    for i, ts in enumerate(tok_sets):
        if not ts:
            continue
        counts = Counter()
        for t in ts:
            for j in inv[t]:
                if j > i:
                    counts[j] += 1
        for j, shared in counts.items():
            J = shared / (len(ts) + len(tok_sets[j]) - shared)
            j_hist[round(J, 1)] += 1
            if J >= args.tau:
                pos_pairs[i].append((j, J))
                pos_pairs[j].append((i, J))
            elif J < args.neg_band_max:
                if len(neg_band[i]) < 50:
                    neg_band[i].append(j)
                if len(neg_band[j]) < 50:
                    neg_band[j].append(i)

    n_pos_total = sum(len(v) for v in pos_pairs.values()) // 2
    print(f"[pairs] {n_pos_total} unordered pairs with J >= {args.tau} "
          f"({len(pos_pairs)} sources have >=1 partner)", flush=True)

    all_idx = list(range(len(rows)))
    out_rows = []

    # channel saber_pair
    for i, partners in sorted(pos_pairs.items()):
        partners = sorted(partners, key=lambda x: -x[1])[: args.max_pos_per_source]
        partner_set = {j for j, _ in pos_pairs[i]}
        for j, J in partners:
            sid_i, sid_j = ids[i], ids[j]
            p_j, s_j = src[sid_j]
            neg_pool = neg_band[i] or [n for n in rng.sample(all_idx,
                                       min(20, len(all_idx)))
                                       if n != i and n not in partner_set]
            negs = rng.sample(neg_pool, min(2, len(neg_pool)))
            out_rows.append({
                "source_id": sid_i,
                "positive_text": doc_template(p_j, s_j),
                "negatives": [{"text": doc_template(*src[ids[n]])}
                              for n in negs],
                "channel": "saber_pair",
                "jaccard": round(J, 4),
            })

    # channel saber_summary
    for i, r in enumerate(rows):
        cands = [j for j in rng.sample(all_idx, min(200, len(all_idx)))
                 if j != i and jaccard(tok_sets[i], tok_sets[j])
                 < args.neg_band_max]
        if not cands:  # degenerate corpora (e.g. stub): any other summary
            cands = [j for j in rng.sample(all_idx, min(20, len(all_idx)))
                     if j != i]
        negs = cands[:2]
        out_rows.append({
            "source_id": r["source_id"],
            "positive_text": r["core_idea"],
            "negatives": [{"text": rows[j]["core_idea"]} for j in negs],
            "channel": "saber_summary",
        })

    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.pairs_file, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_channel = Counter(r["channel"] for r in out_rows)
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "recipe_provenance": {
            "summary_prompt": "annotate/ideas/prompts.py::"
                              "extract_core_idea_query, verbatim",
            "tokenizer": "similarities/compute/jaccard.py ([a-zA-Z0-9^_]+ "
                         "lowercase, their stop_words.py)",
            "tau_summ": args.tau,
            "doc_template": "benchmark.py transform() statement-full, verbatim",
            "not_replicated": "MathWorld ontology BMA signal + Swiss "
                              "tournament (see module docstring for the "
                              "fidelity argument)",
        },
        "n_summaries": len(rows),
        "n_pos_pairs_at_tau": n_pos_total,
        "jaccard_histogram_0.1bins": {str(k): v for k, v
                                      in sorted(j_hist.items())},
        "rows_by_channel": dict(by_channel),
        "n_rows_total": len(out_rows),
        "negatives_per_row_hist": dict(Counter(len(r["negatives"])
                                               for r in out_rows)),
        "seed": args.seed,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(RESULTS / "saber_pairs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pairs] wrote {len(out_rows)} trainer rows ({dict(by_channel)}) "
          f"-> {args.pairs_file}\n[pairs] summary -> "
          f"results/saber_pairs_summary.json", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["filter", "summaries", "pairs"])
    ap.add_argument("--corpus", default=str(ROOT / "data" / "mathnet_corpus.parquet"))
    ap.add_argument("--source-ids-file",
                    default=str(ROOT / "data" / "pairs" / "source_ids.txt"),
                    help="matched-budget list (stage filter input)")
    ap.add_argument("--summaries-file",
                    default=str(ATTACK_DIR / "summaries.jsonl"))
    ap.add_argument("--pairs-file",
                    default=str(ATTACK_DIR / "pairs.jsonl"))
    # filter
    ap.add_argument("--containment-threshold", type=float, default=0.35)
    ap.add_argument("--numeric-threshold", type=float, default=0.6)
    # summaries
    ap.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--max-problems", type=int, default=0,
                    help="cap for smoke runs (0 = all)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    # pairs
    ap.add_argument("--tau", type=float, default=0.211,
                    help="their tau_summ (paper §3.4)")
    ap.add_argument("--neg-band-max", type=float, default=0.08)
    ap.add_argument("--max-pos-per-source", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    {"filter": stage_filter,
     "summaries": stage_summaries,
     "pairs": stage_pairs}[args.stage](args)


if __name__ == "__main__":
    main()
