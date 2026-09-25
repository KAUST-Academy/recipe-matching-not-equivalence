#!/usr/bin/env python3
"""Certificates for three training-set counts the paper quotes that no other
shipped result file states directly. Writes results/count_certificates.json.

1. The uncapped verified arm (Appendix A, "Training-signal volume does not
   predict the score"): rows, source problems and contrastive examples of
   models/qwen3-0.6b-cas-cmnrl, read from that run's run_config.json. Its
   contrastive examples are the training triplets plus the training pairs
   (the dev split held out by source is not counted).
2. The SABER document channel (Appendix D, the summary-channel re-attack):
   rows, distinct source problems and distinct positive documents of
   data/saber_attack/pairs_doc_channel.jsonl, and the same for the summary
   channel. The document-channel file is over the bundle's 25 MB cap, so this
   certificate is what ships in its place.
3. The step-matched summary arm (models/saber-labelmatched-2368-s{42,43,44}):
   the 2,368 rows each seed trained on, drawn exactly as
   scripts/train_invarembed.py draws them (random.Random(seed).shuffle of the
   gated rows, then the first --max-rows), with their distinct sources and
   distinct positives.

    python scripts/count_certificates.py
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "count_certificates.json")


def channel(path):
    rows, sources, positives = 0, set(), set()
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            sources.add(r["source_id"])
            positives.add(r["positive_text"])
    return {"file": path, "rows": rows, "distinct_source_ids": len(sources), "distinct_positives": len(positives)}


def matched_draws(path, max_rows, seeds):
    """No row of the summary channel is dropped by the trainer's gates (run_config
    input_rows == rows_used before the cap), so its draw is a seeded shuffle of the file."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    out = {"file": path, "max_rows": max_rows, "per_seed": {}}
    for seed in seeds:
        rc = json.load(open(os.path.join(ROOT, f"models/saber-labelmatched-{max_rows}-s{seed}/run_config.json"),
                            encoding="utf-8"))["data_stats"]
        assert rc["input_rows"] == len(rows) and rc["rows_dropped_eval_anchor_overlap"] == 0 \
            and rc["rows_dropped_crosslingual_eval_overlap"] == 0 and rc["rows_dropped_source_id_not_in_corpus"] == 0
        kept = list(rows)
        random.Random(seed).shuffle(kept)
        kept = kept[:max_rows]
        assert len({r["source_id"] for r in kept}) == rc["n_source_ids"], seed
        out["per_seed"][str(seed)] = {"rows": len(kept), "distinct_source_ids": len({r["source_id"] for r in kept}),
                                      "distinct_positives": len({r["positive_text"] for r in kept})}
    return out


def main():
    rc_path = "models/qwen3-0.6b-cas-cmnrl/run_config.json"
    ds = json.load(open(os.path.join(ROOT, rc_path), encoding="utf-8"))["data_stats"]
    out = {
        "uncapped_verified_arm": {
            "run_config": rc_path,
            "rows_used": ds["rows_used"],
            "source_problems": ds["n_source_ids"],
            "train_triplets": ds["n_train_triplets"],
            "train_pairs": ds["n_train_pairs"],
            "contrastive_examples": ds["n_train_triplets"] + ds["n_train_pairs"],
            "definition": "contrastive examples = training triplets + training pairs, after the 5% by-source dev split",
        },
        "saber_document_channel": channel("data/saber_attack/pairs_doc_channel.jsonl"),
        "saber_summary_channel": channel("data/saber_attack/pairs_summary_channel.jsonl"),
        "saber_matched_summary_arm_draws": matched_draws("data/saber_attack/pairs_summary_channel.jsonl", 2368, (42, 43, 44)),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
