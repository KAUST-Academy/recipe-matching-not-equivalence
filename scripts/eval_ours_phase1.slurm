#!/bin/bash
#SBATCH --job-name=invarembed-eval-p1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# Completes the phase-1 picture:
#  (a) our trained model on medium + hard tiers (easy already done in-job)
#  (b) UNTRAINED base Qwen3-Embedding-0.6B on all three tiers — isolates the
#      training delta (the key ablation for the paper)
#  (c) our model on the cross-lingual regression set (must not degrade)

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

OURS=models/qwen3-0.6b-cas-cmnrl/final
for TIER in medium hard; do
    python scripts/eval_retrieve.py --tier "$TIER" --model "$OURS" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --output "results/eval_${TIER}_qwen3-0.6b-cas-cmnrl.json"
done
for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "$TIER" --model Qwen/Qwen3-Embedding-0.6B \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --output "results/eval_${TIER}_qwen3-0.6b-base.json"
done
python scripts/eval_crosslingual.py --model "$OURS" \
    --device cuda --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
    --output results/crosslingual_qwen3-0.6b-cas-cmnrl.json
echo "== phase-1 eval sweep finished =="
