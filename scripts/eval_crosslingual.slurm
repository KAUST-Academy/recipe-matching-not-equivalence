#!/bin/bash
#SBATCH --job-name=crosslingual-eval
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# Cross-lingual duplicate-retrieval baselines on 1x A100 (InvarEmbed project).
# Benchmark: data/crosslingual_eval (393 non-EN queries, 27,817-doc corpus,
# mined real official duplicates -- see data/crosslingual_eval/README.md).
#
# Job 1: all-mpnet-base-v2 (the paper's sanity baseline).
# Job 2: Qwen/Qwen3-Embedding-4B, same config that reproduced the direction
#        of v2 Table 4 (job 49566055): bf16, query prompt, max_seq_len 1024.
# CPU reference: all-MiniLM-L6-v2 already run on login node ->
#        results/crosslingual_smoke_test.json
#
# Submit with:  sbatch scripts/eval_crosslingual.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="

python scripts/eval_crosslingual.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --batch-size 256 \
    --output results/crosslingual_all-mpnet-base-v2.json

python scripts/eval_crosslingual.py \
    --model Qwen/Qwen3-Embedding-4B \
    --device cuda \
    --model-dtype bfloat16 \
    --query-prompt-name query \
    --batch-size 32 \
    --max-seq-length 1024 \
    --output results/crosslingual_qwen3-embedding-4b.json

echo "== all cross-lingual evaluations finished =="
