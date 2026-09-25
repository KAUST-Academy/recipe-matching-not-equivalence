#!/bin/bash
#SBATCH --job-name=invarembed-train-p2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# InvarEmbed PHASE 2: mixed training data, two variants, one GPU, sequential.
#
# Phase-1 diagnosis (models/qwen3-0.6b-cas-cmnrl): training saw only
# difficulty EXTREMES (CAS minimal-pair negatives + random in-batch) ->
# R@1 doubled on every tier (easy 8.32->15.92, medium 1.78->3.86,
# hard 0.00->0.16) but recall depth COLLAPSED (medium R@5 60.91->40.80).
#
# Phase-2 data (scripts/build_phase2_data.py -> data/phase2/mixed_pairs.jsonl,
# 30,323 rows): cas 8,747 (minimal negs capped at 2) + cas_midneg 5,000
# (1 extra same-topic mid-difficulty negative) + replay 12,000 (problem ->
# own official solution, in-batch diversity) + cross_transform 4,576
# (positive_i, positive_j). Both contamination lists excluded at build time
# AND re-gated by the trainer.
#
# V1 "mixed":       data change only               -> models/qwen3-0.6b-p2-mixed
# V2 "mixed+instr": same data + MATH-SPECIFIC anchor instruction (the
#                   trainer's --query-prompt-text path, mirrored at eval
#                   with eval_retrieve.py --query-prompt)
#                                                  -> models/qwen3-0.6b-p2-instr
# Both runs write <output-dir>/run_config.json (auditable). Each is followed
# by easy+medium+hard tier evals; a comparison table prints at job end.
#
# EXPECTED RUNTIME: ~41K exploded examples -> ~75 steps/epoch at batch 512
# -> ~150 steps/run; grad-cached batch 512 x seq 1024 ~ 40-60 s/step
# => ~2-2.5 h per training. 6 tier evals ~25-30 min each => ~3 h.
# Total ~7-8 h; the 12 h limit leaves slack.
#
# Submit with:  sbatch scripts/train_phase2.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results models

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="

MATH_PROMPT=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

# ---------------- V1 "mixed": data change only ------------------------------
python scripts/train_invarembed.py \
    --train-file data/phase2/mixed_pairs.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 512 --mini-batch-size 16 --scale 20 \
    --epochs 2 --lr 1e-5 --warmup-ratio 0.05 --seed 42 \
    --dev-frac 0.05 --eval-steps 50 \
    --max-seq-length 1024 --query-prompt-name query \
    --output-dir models/qwen3-0.6b-p2-mixed

echo "== V1 (mixed) training finished =="

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" \
        --model models/qwen3-0.6b-p2-mixed/final \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --output "results/eval_${TIER}_qwen3-0.6b-p2-mixed.json"
done

echo "== V1 (mixed) all-tier eval finished =="

# ---------------- V2 "mixed+instr": + math-specific instruction -------------
python scripts/train_invarembed.py \
    --train-file data/phase2/mixed_pairs.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 512 --mini-batch-size 16 --scale 20 \
    --epochs 2 --lr 1e-5 --warmup-ratio 0.05 --seed 42 \
    --dev-frac 0.05 --eval-steps 50 \
    --max-seq-length 1024 --query-prompt-text "${MATH_PROMPT}" \
    --output-dir models/qwen3-0.6b-p2-instr

echo "== V2 (mixed+instr) training finished =="

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" \
        --model models/qwen3-0.6b-p2-instr/final \
        --device cuda --model-dtype bfloat16 --query-prompt "${MATH_PROMPT}" \
        --batch-size 64 --max-seq-length 1024 \
        --output "results/eval_${TIER}_qwen3-0.6b-p2-instr.json"
done

echo "== V2 (mixed+instr) all-tier eval finished =="

# ---------------- comparison table ------------------------------------------
python - <<'EOF'
import json, os
runs = [("qwen3-0.6b-base",      "base 0.6B (untrained)"),
        ("qwen3-0.6b-cas-cmnrl", "phase-1 CAS-only"),
        ("qwen3-0.6b-p2-mixed",  "phase-2 V1 mixed"),
        ("qwen3-0.6b-p2-instr",  "phase-2 V2 mixed+instr")]
tiers = ["easy", "medium", "hard"]
hdr = f"{'model':24s} | " + " | ".join(f"{t + ' R@1/5/10':>22s}" for t in tiers)
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
for run, label in runs:
    cells = []
    for t in tiers:
        p = f"results/eval_{t}_{run}.json"
        if os.path.exists(p):
            o = json.load(open(p))["overall"]
            cells.append(f"{o['recall@1']:5.2f}/{o['recall@5']:5.2f}/"
                         f"{o['recall@10']:5.2f}")
        else:
            cells.append("(missing)")
    print(f"{label:24s} | " + " | ".join(f"{c:>22s}" for c in cells))
print("=" * len(hdr))
print("Reference bars: phase-1 R@1 easy/med/hard = 15.92/3.86/0.16 "
      "(match or beat) | base medium R@5 = 60.91 (the collapsed-depth "
      "metric phase-2 must recover; phase-1 fell to 40.80)")
EOF

echo "== phase-2 training + eval sweep finished =="
