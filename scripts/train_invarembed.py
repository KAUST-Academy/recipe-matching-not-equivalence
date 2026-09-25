#!/usr/bin/env python3
"""
InvarEmbed contrastive training pipeline (MathNet follow-up).

Fine-tunes an embedding model on verified math-equivalence pairs:
(anchor = original problem, positive = verified-equivalent rewriting,
negatives = verified-NON-equivalent minimal edits), with sentence-transformers
5.x SentenceTransformerTrainer.

DATA FORMAT (--train-file, jsonl, one object per line):
    {
      "source_id":    "<id into data/mathnet_corpus.parquet>",
      "positive_text": "<verified-equivalent rewriting of the problem>",
      "negatives":    [{"text": "<non-equivalent minimal edit>", ...}, ...]
    }
  * the ANCHOR text is looked up in mathnet_corpus.parquet (problem_markdown)
    by source_id -- UNLESS the row carries an optional "anchor_text" field,
    which overrides the corpus lookup (used by phase-2 cross-transform rows
    where the anchor is itself a transformed positive; contamination gates
    still key on source_id either way);
  * `negatives` may be empty/missing -> the row becomes a plain (anchor,
    positive) pair usable by the in-batch-negative losses (mnrl/cmnrl);
  * negative entries may also be plain strings instead of {"text": ...}.

PROMPTS: --query-prompt-name uses a prompt registered in the model config
(e.g. 'query' for Qwen3-Embedding). --query-prompt-text passes a RAW
instruction string instead (works for any model / custom instructions;
mutually exclusive with --query-prompt-name). Either is applied to the
ANCHOR column only; mirror it at eval time via eval_retrieve.py's
--query-prompt-name / --query-prompt respectively.
--doc-prompt-text additionally applies a RAW prefix to the POSITIVE and
NEGATIVE columns (the "document" side) -- required by symmetric-prefix
models like intfloat/multilingual-e5-* ("query: " / "passage: "); mirror at
eval time with eval_retrieve.py --doc-prompt / eval_crosslingual.py
--doc-prompt. NOTE: the dev TripletEvaluator encodes raw texts without any
prompt (same pre-existing behavior as the Qwen arm) -- it is only used for
best-checkpoint selection, never reported.

CONTAMINATION GATE: rows whose source_id is in
anchor_to_corpus_mapping.json["exclude_corpus_ids"] (8,698 corpus problems
that match MathNet-Retrieve eval anchors) are ALWAYS dropped, with a count
printed. Additionally, rows whose source_id is in
data/crosslingual_eval/leakage_exclude_ids.json["eval_member_ids"] (779
query/gold problems of the cross-lingual duplicate eval) are dropped when
that file exists, so training cannot contaminate the cross-lingual eval
either. There is deliberately no flag to disable any of this.

============================================================================
SUPERVISION-CONTROLLED EXPERIMENT HOOK (the paper's headline claim)
============================================================================
The verified arm and the LLM-judged arm are trained with IDENTICAL commands
except for --train-file:

  verified arm:    --train-file data/cas_pairs/pairs.jsonl
  LLM-judged arm:  --train-file data/llm_pairs/pairs.jsonl

Matched data budget = same number of input ROWS. Set
  --max-rows N        with N = min(#rows(cas_pairs), #rows(llm_pairs))
on BOTH arms (rows are shuffled with --seed before truncation, so the cap is
a uniform subsample). Everything else -- model, loss, seed, epochs, lr,
batch size, dev fraction -- must stay identical between arms. The run's
resolved config + data statistics are written to <output-dir>/run_config.json
so the matched-budget claim is auditable.
============================================================================

SPLITS: a held-out dev split is carved out BY source_id (--dev-frac, default
0.05) -- all rows of one source problem land on the same side, never split by
row. A TripletEvaluator (anchor/positive/negative cosine accuracy) runs on
the dev triplets every --eval-steps, and the best checkpoint by
eval_dev_cosine_accuracy is loaded at the end.

LOSSES (--loss):
  cmnrl   CachedMultipleNegativesRankingLoss (default) -- big effective
          batch (--batch-size) on one GPU via gradient caching with
          --mini-batch-size sized forward chunks; uses in-batch negatives
          PLUS the explicit hard negative column.
  mnrl    plain MultipleNegativesRankingLoss (memory-bound batch).
  triplet TripletLoss on explicit (anchor, positive, negative) triplets only
          (rows without negatives are dropped for this loss).

LoRA (--lora, for the 4B phase-2 backbone): `--lora new` injects a fresh
adapter (r/alpha/dropout/target-modules flags); `--lora <path>` loads an
existing peft adapter and continues training it. The saved output contains
the adapter + a reference to the base model; SentenceTransformer(<output>)
reloads it transparently.

USAGE
  Smoke test (CPU, tiny model, ~1 min; see scripts/make_smoke_pairs.py):
    python scripts/train_invarembed.py \
        --train-file data/smoke_pairs/pairs.jsonl \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --loss mnrl --batch-size 8 --max-steps 10 --eval-steps 5 \
        --dev-frac 0.2 --output-dir models/smoke_mnrl

  Phase-1 (1x A100, Qwen3-Embedding-0.6B full FT -- see train_invarembed.slurm):
    python scripts/train_invarembed.py \
        --train-file data/cas_pairs/pairs.jsonl \
        --model Qwen/Qwen3-Embedding-0.6B \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 \
        --epochs 3 --lr 2e-5 --eval-steps 100 --max-seq-length 1024 \
        --bf16 --query-prompt-name query \
        --output-dir models/qwen3-0.6b-cas-cmnrl --eval-after

After training the final model is saved to <output-dir>/final, reloaded once
as a sanity check, and (with --eval-after) evaluated on the MathNet-Retrieve
easy tier via scripts/eval_retrieve.py.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_JSON = os.path.join(PROJECT_ROOT, "anchor_to_corpus_mapping.json")
MATHNET_PARQUET = os.path.join(PROJECT_ROOT, "data", "mathnet_corpus.parquet")
XLING_LEAKAGE_JSON = os.path.join(PROJECT_ROOT, "data", "crosslingual_eval",
                                  "leakage_exclude_ids.json")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_corpus_texts() -> dict:
    """id -> problem_markdown for the full MathNet corpus."""
    import duckdb
    rows = duckdb.sql(
        f"SELECT id, problem_markdown FROM '{MATHNET_PARQUET}'").fetchall()
    return {cid: text for cid, text in rows if text and text.strip()}


def load_exclude_ids(mapping_json: str = None) -> tuple:
    """(eval-anchor ids, cross-lingual-eval query/gold ids) to drop.

    mapping_json defaults to the v1 exact-text gate the published campaign ran
    under, so omitting it reproduces every existing run byte-for-byte. Pass
    anchor_to_corpus_mapping_v2.json to train under the corrected gate.
    """
    path = mapping_json or MAPPING_JSON
    with open(path, encoding="utf-8") as f:
        anchor = set(json.load(f)["exclude_corpus_ids"])
    print(f"[data] anchor gate: {os.path.basename(path)} "
          f"({len(anchor)} excluded corpus ids)", flush=True)
    xling = set()
    if os.path.exists(XLING_LEAKAGE_JSON):
        with open(XLING_LEAKAGE_JSON, encoding="utf-8") as f:
            xling = set(json.load(f)["eval_member_ids"])
    else:
        print(f"[data] WARNING: {XLING_LEAKAGE_JSON} not found -- "
              "cross-lingual-eval contamination gate inactive", flush=True)
    return anchor, xling


def load_pairs(path: str) -> list:
    """Parse the pairs jsonl into [{source_id, positive_text, negatives:[str]}]."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            negs = []
            for n in row.get("negatives") or []:
                text = n["text"] if isinstance(n, dict) else n
                if text and text.strip():
                    negs.append(text.strip())
            pos = (row.get("positive_text") or "").strip()
            if not pos:
                print(f"[data] WARNING line {ln}: empty positive_text, dropped")
                continue
            rows.append({"source_id": row["source_id"],
                         "positive_text": pos, "negatives": negs,
                         "anchor_text":
                             (row.get("anchor_text") or "").strip() or None})
    return rows


def build_datasets(args):
    """Return (train_dict, eval_dict, dev_triplets, stats).

    train/eval dicts map {"triplets": Dataset, "pairs": Dataset} (either key
    may be absent). dev_triplets = (anchors, positives, negatives) lists for
    the TripletEvaluator.
    """
    from datasets import Dataset

    corpus = load_corpus_texts()
    exclude_anchor, exclude_xling = load_exclude_ids(
        getattr(args, "anchor_mapping", None))
    raw = load_pairs(args.train_file)
    stats = {"input_rows": len(raw)}

    kept, n_excluded, n_xling, n_missing = [], 0, 0, 0
    for r in raw:
        if r["source_id"] in exclude_anchor:
            n_excluded += 1
            continue
        if r["source_id"] in exclude_xling:
            n_xling += 1
            continue
        anchor = r.get("anchor_text") or corpus.get(r["source_id"])
        if not anchor:
            n_missing += 1
            continue
        r["anchor"] = anchor
        kept.append(r)
    stats["rows_dropped_eval_anchor_overlap"] = n_excluded
    stats["rows_dropped_crosslingual_eval_overlap"] = n_xling
    stats["rows_dropped_source_id_not_in_corpus"] = n_missing
    print(f"[data] {len(raw)} rows read; dropped {n_excluded} eval-anchor "
          f"overlaps + {n_xling} cross-lingual-eval overlaps (contamination "
          f"gates) + {n_missing} unknown source_ids", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.max_rows and args.max_rows < len(kept):
        kept = kept[:args.max_rows]
        print(f"[data] matched-budget cap: truncated to {len(kept)} rows "
              f"(--max-rows, seed={args.seed})", flush=True)
    stats["rows_used"] = len(kept)
    if not kept:
        sys.exit("[data] FATAL: no usable rows after filtering")

    # ---- dev split by source_id (never by row) ----
    source_ids = sorted({r["source_id"] for r in kept})
    rng.shuffle(source_ids)
    n_dev = max(1, int(round(args.dev_frac * len(source_ids)))) if args.dev_frac > 0 else 0
    dev_ids = set(source_ids[:n_dev])
    train_rows = [r for r in kept if r["source_id"] not in dev_ids]
    dev_rows = [r for r in kept if r["source_id"] in dev_ids]
    stats.update(n_source_ids=len(source_ids), n_dev_source_ids=len(dev_ids),
                 n_train_rows=len(train_rows), n_dev_rows=len(dev_rows))

    def explode(rows):
        """rows -> (triplet dict, pair dict) column format."""
        trip = {"anchor": [], "positive": [], "negative": []}
        pair = {"anchor": [], "positive": []}
        for r in rows:
            negs = r["negatives"][:args.max_negatives] if args.max_negatives \
                else r["negatives"]
            if negs:
                for n in negs:
                    trip["anchor"].append(r["anchor"])
                    trip["positive"].append(r["positive_text"])
                    trip["negative"].append(n)
            else:
                pair["anchor"].append(r["anchor"])
                pair["positive"].append(r["positive_text"])
        return trip, pair

    tr_trip, tr_pair = explode(train_rows)
    dv_trip, dv_pair = explode(dev_rows)
    stats.update(n_train_triplets=len(tr_trip["anchor"]),
                 n_train_pairs=len(tr_pair["anchor"]),
                 n_dev_triplets=len(dv_trip["anchor"]),
                 n_dev_pairs=len(dv_pair["anchor"]))
    neg_counts = Counter(len(r["negatives"]) for r in kept)
    stats["negatives_per_row_histogram"] = {str(k): v for k, v
                                            in sorted(neg_counts.items())}

    use_pairs = args.loss in ("mnrl", "cmnrl")
    if not use_pairs and (tr_pair["anchor"] or dv_pair["anchor"]):
        print(f"[data] WARNING: loss={args.loss} cannot use rows without "
              f"negatives -- dropping {len(tr_pair['anchor'])} train + "
              f"{len(dv_pair['anchor'])} dev pair-only rows", flush=True)

    def to_dict(trip, pair):
        out = {}
        if trip["anchor"]:
            out["triplets"] = Dataset.from_dict(trip)
        if use_pairs and pair["anchor"]:
            out["pairs"] = Dataset.from_dict(pair)
        return out

    train_dict = to_dict(tr_trip, tr_pair)
    eval_dict = to_dict(dv_trip, dv_pair)
    if not train_dict:
        sys.exit("[data] FATAL: no training examples for this loss")

    # dev triplets for the TripletEvaluator (cap for speed)
    n_ev = min(len(dv_trip["anchor"]), args.dev_max_triplets)
    dev_triplets = (dv_trip["anchor"][:n_ev], dv_trip["positive"][:n_ev],
                    dv_trip["negative"][:n_ev])
    print(f"[data] train: {stats['n_train_triplets']} triplets + "
          f"{stats['n_train_pairs']} pairs | dev: {stats['n_dev_triplets']} "
          f"triplets + {stats['n_dev_pairs']} pairs "
          f"({len(dev_ids)}/{len(source_ids)} source_ids held out)", flush=True)
    return train_dict, eval_dict, dev_triplets, stats


# ---------------------------------------------------------------------------
# model / loss
# ---------------------------------------------------------------------------

def build_model(args):
    import torch
    from sentence_transformers import SentenceTransformer
    model_kwargs = {}
    if args.model_dtype and args.model_dtype != "float32":
        model_kwargs["torch_dtype"] = getattr(torch, args.model_dtype)
    model = SentenceTransformer(args.model, device=args.device,
                                model_kwargs=model_kwargs or None)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    if args.lora:
        from peft import LoraConfig, TaskType
        if args.lora == "new":
            target = args.lora_target_modules.split(",") \
                if args.lora_target_modules != "all-linear" \
                else args.lora_target_modules
            cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=target)
            model.add_adapter(cfg)
            print(f"[model] injected fresh LoRA adapter r={args.lora_r} "
                  f"alpha={args.lora_alpha} targets={target}", flush=True)
        else:
            model.load_adapter(args.lora, is_trainable=True)
            print(f"[model] loaded existing LoRA adapter from {args.lora}",
                  flush=True)
    print(f"[model] {args.model} device={model.device} "
          f"max_seq_length={model.max_seq_length}", flush=True)
    return model


def build_loss(args, model):
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:                       # older sentence-transformers
        from sentence_transformers import losses
    if args.loss == "cmnrl":
        return losses.CachedMultipleNegativesRankingLoss(
            model, scale=args.scale, mini_batch_size=args.mini_batch_size,
            gather_across_devices=args.gather_across_devices)
    if args.loss == "mnrl":
        return losses.MultipleNegativesRankingLoss(model, scale=args.scale)
    if args.loss == "triplet":
        return losses.TripletLoss(model, triplet_margin=args.triplet_margin)
    raise ValueError(args.loss)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="InvarEmbed contrastive training (see module docstring)")
    # data
    ap.add_argument("--train-file", required=True,
                    help="pairs jsonl -- THE ONLY FLAG THAT CHANGES between "
                         "the verified and LLM-judged supervision arms")
    ap.add_argument("--anchor-mapping", default=None,
                    help="path to the anchor->corpus exclusion mapping. "
                         "Default: anchor_to_corpus_mapping.json (the v1 "
                         "exact-text gate the published campaign ran under). "
                         "Pass anchor_to_corpus_mapping_v2.json for the "
                         "corrected near-verbatim-twin gate.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="matched-budget cap: shuffle (seed) then keep N rows "
                         "(0 = all). Set to min(rows of both arms) for the "
                         "supervision-controlled experiment.")
    ap.add_argument("--max-negatives", type=int, default=0,
                    help="cap negatives used per row (0 = all)")
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="fraction of source_ids held out for dev (0 disables)")
    ap.add_argument("--dev-max-triplets", type=int, default=2000,
                    help="cap dev triplets fed to the TripletEvaluator")
    # model
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default=None, help="cpu / cuda (default auto)")
    ap.add_argument("--model-dtype", default=None,
                    choices=[None, "float32", "float16", "bfloat16"])
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="tokenizer truncation override (0 = model default; "
                         "set 1024 for Qwen3-Embedding to match the eval config)")
    ap.add_argument("--query-prompt-name", default=None,
                    help="model prompt applied to the ANCHOR column during "
                         "training (e.g. 'query' for Qwen3-Embedding); mirrors "
                         "the asymmetric prompting used at eval time")
    ap.add_argument("--query-prompt-text", default=None,
                    help="RAW instruction string applied to the ANCHOR column "
                         "(alternative to --query-prompt-name for custom / "
                         "task-specific instructions; mutually exclusive with "
                         "it). Mirror at eval time with eval_retrieve.py "
                         "--query-prompt.")
    ap.add_argument("--doc-prompt-text", default=None,
                    help="RAW prefix applied to the POSITIVE and NEGATIVE "
                         "columns (document side), e.g. 'passage: ' for "
                         "multilingual-e5 models. Mirror at eval time with "
                         "eval_retrieve.py / eval_crosslingual.py "
                         "--doc-prompt.")
    # LoRA
    ap.add_argument("--lora", nargs="?", const="new", default=None,
                    metavar="PATH",
                    help="enable LoRA: bare --lora injects a new adapter, "
                         "--lora <path> resumes an existing peft adapter")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default="all-linear",
                    help="'all-linear' or comma-separated module names")
    # loss / optimization
    ap.add_argument("--loss", choices=["cmnrl", "mnrl", "triplet"],
                    default="cmnrl")
    ap.add_argument("--scale", type=float, default=20.0,
                    help="InfoNCE temperature scale for (c)mnrl")
    ap.add_argument("--mini-batch-size", type=int, default=16,
                    help="cmnrl gradient-cache chunk (memory knob; does not "
                         "change the objective)")
    ap.add_argument("--triplet-margin", type=float, default=5.0)
    ap.add_argument("--gather-across-devices", action="store_true",
                    help="share in-batch negatives across GPUs (DDP runs)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="hard step cap, overrides --epochs (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="per-device train batch (with cmnrl this is the "
                         "EFFECTIVE contrastive batch)")
    ap.add_argument("--eval-steps", type=int, default=100,
                    help="evaluate + checkpoint every N steps (0 disables)")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # output / post-train eval
    ap.add_argument("--output-dir", required=True,
                    help="e.g. models/qwen3-0.6b-cas-cmnrl")
    ap.add_argument("--eval-after", action="store_true",
                    help="run scripts/eval_retrieve.py on the trained model")
    ap.add_argument("--eval-tier", default="easy",
                    choices=["easy", "medium", "hard"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--eval-max-queries", type=int, default=0,
                    help="subsample eval queries (plumbing tests only)")
    ap.add_argument("--eval-max-corpus", type=int, default=0,
                    help="subsample eval corpus (NOT paper-comparable)")
    args = ap.parse_args()
    t0 = time.time()

    from sentence_transformers import (SentenceTransformer,
                                       SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    try:
        from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:
        from sentence_transformers.evaluation import TripletEvaluator
        from sentence_transformers.training_args import BatchSamplers

    train_dict, eval_dict, dev_triplets, stats = build_datasets(args)
    model = build_model(args)
    loss_obj = build_loss(args, model)
    # multi-dataset form throughout: one loss entry per dataset name
    loss = {name: loss_obj for name in set(train_dict) | set(eval_dict)}

    if args.query_prompt_name and args.query_prompt_text:
        sys.exit("[model] FATAL: --query-prompt-name and --query-prompt-text "
                 "are mutually exclusive")
    prompts = None
    if args.query_prompt_name:
        if args.query_prompt_name not in (model.prompts or {}):
            sys.exit(f"[model] FATAL: prompt {args.query_prompt_name!r} not in "
                     f"model prompts {sorted((model.prompts or {}))}")
        prompts = {"anchor": model.prompts[args.query_prompt_name]}
        print(f"[train] anchor-column prompt: "
              f"{prompts['anchor']!r}", flush=True)
    elif args.query_prompt_text:
        prompts = {"anchor": args.query_prompt_text}
        print(f"[train] anchor-column prompt (custom text): "
              f"{prompts['anchor']!r}", flush=True)
    if args.doc_prompt_text:
        prompts = dict(prompts or {})
        prompts["positive"] = args.doc_prompt_text
        prompts["negative"] = args.doc_prompt_text
        print(f"[train] positive/negative-column prompt (custom text): "
              f"{args.doc_prompt_text!r}", flush=True)

    do_eval = bool(eval_dict) and args.eval_steps > 0
    evaluator = None
    if do_eval and dev_triplets[0]:
        evaluator = TripletEvaluator(*dev_triplets, name="dev",
                                     batch_size=args.eval_batch_size)

    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts=prompts,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        save_strategy="steps" if do_eval else "no",
        save_steps=args.eval_steps if do_eval else None,
        save_total_limit=2,
        load_best_model_at_end=bool(do_eval and evaluator),
        metric_for_best_model="eval_dev_cosine_accuracy"
            if (do_eval and evaluator) else None,
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs,
        train_dataset=train_dict,
        eval_dataset=eval_dict if do_eval else None,
        loss=loss, evaluator=evaluator)
    print(f"[train] starting: loss={args.loss} batch={args.batch_size} "
          f"lr={args.lr} epochs={args.epochs} max_steps={args.max_steps}",
          flush=True)
    train_result = trainer.train()

    # DDP (torchrun) runs: only rank 0 saves / evaluates
    if int(os.environ.get("RANK", "0")) != 0:
        return

    # ---- save + reload sanity check ----
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"[save] final model -> {final_dir}", flush=True)
    reloaded = SentenceTransformer(final_dir, device=args.device)
    emb = reloaded.encode(["Prove that $2+2=4$.", "Find all primes $p$."],
                          normalize_embeddings=True)
    print(f"[save] reload OK: encoded shape {emb.shape}", flush=True)

    run_config = {
        "argv": sys.argv,
        "args": vars(args),
        "data_stats": stats,
        "train_runtime_seconds": round(train_result.metrics.get(
            "train_runtime", time.time() - t0), 1),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "log_history_tail": trainer.state.log_history[-6:],
        "final_model": final_dir,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"[save] run_config.json written; total {time.time()-t0:.0f}s",
          flush=True)

    # ---- optional MathNet-Retrieve eval ----
    if args.eval_after:
        run_name = os.path.basename(os.path.normpath(args.output_dir))
        out_json = os.path.join(PROJECT_ROOT, "results",
                                f"eval_{args.eval_tier}_{run_name}.json")
        cmd = [sys.executable,
               os.path.join(PROJECT_ROOT, "scripts", "eval_retrieve.py"),
               "--tier", args.eval_tier, "--model", final_dir,
               "--batch-size", str(args.eval_batch_size),
               "--output", out_json]
        if args.query_prompt_name:
            cmd += ["--query-prompt-name", args.query_prompt_name]
        elif args.query_prompt_text:
            cmd += ["--query-prompt", args.query_prompt_text]
        if args.max_seq_length:
            cmd += ["--max-seq-length", str(args.max_seq_length)]
        if args.model_dtype:
            cmd += ["--model-dtype", args.model_dtype]
        if args.eval_max_queries:
            cmd += ["--max-queries", str(args.eval_max_queries)]
        if args.eval_max_corpus:
            cmd += ["--max-corpus", str(args.eval_max_corpus)]
        print(f"[eval] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
