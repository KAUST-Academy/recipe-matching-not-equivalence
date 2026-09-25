#!/usr/bin/env python3
"""Late-interaction arms (2026-09-17). Fine-tunes ColBERTv2
(colbert-ir/colbertv2.0, via PyLate) on the SAME training file, row cap,
contamination gates and 5% by-source dev split as the dense arms, by reusing
scripts/train_invarembed.py's build_datasets; loss = PyLate's in-batch
contrastive (MaxSim scores, one example per negative, exactly the dense
trainer's row-to-example rule). No checkpoint selection: the final model is
kept (a difference from the dense arms, stated in the paper).

  python scripts/colbert_train.py --train-file data/llm_pairs/pairs.jsonl \
      --max-rows 6145 --seed 42 --output-dir models/colbert-ctrl-llm-6145
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--max-rows", type=int, default=6145)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base", default="colbert-ir/colbertv2.0")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--query-length", type=int, default=256)
    ap.add_argument("--document-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    a = ap.parse_args()
    os.chdir(ROOT)
    import train_invarembed as T
    from pylate import losses, models
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    import torch
    targs = argparse.Namespace(train_file=a.train_file, max_rows=a.max_rows, max_negatives=0,
                               dev_frac=a.dev_frac, seed=a.seed, loss="cmnrl", dev_max_triplets=2000,
                               anchor_mapping=None)
    train_dict, eval_dict, dev_triplets, stats = T.build_datasets(targs)
    def rename(ds):
        cols = list(ds.column_names)
        m = {}
        for c in cols:
            if c in ("anchor", "query"): m[c] = "query"
            elif c in ("positive",): m[c] = "positive"
            elif c in ("negative", "negatives"): m[c] = "negative"
        ds = ds.rename_columns(m) if m else ds
        keep = [c for c in ("query", "positive", "negative") if c in ds.column_names]
        return ds.select_columns(keep)
    train = {k: rename(v) for k, v in train_dict.items()}
    print("[data]", {k: (len(v), v.column_names) for k, v in train.items()}, stats, flush=True)
    kw = {}
    try:
        model = models.ColBERT(model_name_or_path=a.base, query_length=a.query_length,
                               document_length=a.document_length)
    except TypeError:
        model = models.ColBERT(model_name_or_path=a.base)
        print("[model] query/document length args not accepted; defaults kept", flush=True)
    loss = {k: losses.Contrastive(model=model) for k in train}
    n_ex = sum(len(v) for v in train.values())
    targs_st = SentenceTransformerTrainingArguments(
        output_dir=a.output_dir, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size, learning_rate=a.lr, warmup_ratio=0.05,
        bf16=torch.cuda.is_available(), seed=a.seed, logging_steps=10, save_strategy="no",
        report_to=[], dataloader_drop_last=False, remove_unused_columns=False)
    trainer = SentenceTransformerTrainer(model=model, args=targs_st, train_dataset=train, loss=loss)
    t0 = time.time(); trainer.train()
    model.save_pretrained(os.path.join(a.output_dir, "final"))
    json.dump({"args": vars(a), "data_stats": stats, "n_examples": n_ex,
               "global_step": trainer.state.global_step, "train_seconds": round(time.time() - t0, 1),
               "base": a.base, "loss": "pylate.Contrastive (MaxSim, in-batch + explicit negatives)",
               "checkpoint_selection": "none (final model)"},
              open(os.path.join(a.output_dir, "run_config.json"), "w"), indent=2)
    print(f"[done] {a.output_dir}/final after {trainer.state.global_step} steps, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
