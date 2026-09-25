#!/bin/bash
#SBATCH --job-name=mathnet-retrieve-eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# MathNet-Retrieve full evaluation on 1x A100 (InvarEmbed project).
# Written in the style of ~/run_1a100.slurm.
#
# Job 1: all-mpnet-base-v2 sanity check on the easy tier -- the UPDATED paper
#        (v2, per-difficulty Table 4) reports for it, per tier:
#          easy   6.75 R@1 / 82.47 R@5 / 91.07 R@10
#          medium 1.26 R@1 / 51.01 R@5 / 65.52 R@10
#          hard   0.00 R@1 /  1.24 R@5 /  5.73 R@10
#        so the easy-tier run should land near 6.75/82.47/91.07.
# Job 2: Qwen/Qwen3-Embedding-4B on all three tiers (the baseline to beat).
#        Reference (v2 Table 4): easy 14.76/83.53/92.13, medium 2.91/71.61/87.22,
#        hard 0.01/8.40/36.87. Hard tier is the headline: ALL models ~0% R@1.
#
# Submit with:  sbatch scripts/eval_retrieve.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"          # keep model weights on /ibex
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="

# ---- sanity check: reproduce the paper's all-mpnet-base-v2 easy-tier row ----
python scripts/eval_retrieve.py \
    --tier easy \
    --model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --batch-size 256 \
    --output results/eval_easy_all-mpnet-base-v2.json

# ---- headline baseline: Qwen3-Embedding-4B on all tiers ----
# Qwen3-Embedding uses an instruction prompt on the query side only
# (prompt name "query" ships with the model's sentence-transformers config).
for TIER in easy medium hard; do
    python scripts/eval_retrieve.py \
        --tier "${TIER}" \
        --model Qwen/Qwen3-Embedding-4B \
        --device cuda \
        --model-dtype bfloat16 \
        --query-prompt-name query \
        --batch-size 32 \
        --max-seq-length 1024 \
        --output "results/eval_${TIER}_qwen3-embedding-4b.json"
done

echo "== all evaluations finished =="
