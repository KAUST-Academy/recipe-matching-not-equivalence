#!/bin/bash
#SBATCH --job-name=tf4-baselines
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# Rerun of the two baselines whose custom remote code needs transformers 4.x
# (transformers 5 removed Config.rope_theta -> AttributeError in job 49569448):
#   Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical
#   reasonir/ReasonIR-8B
# Uses the 'baselines-tf4' conda env (transformers 4.x + ST 4.x + torch cu128).

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate baselines-tf4
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

MATH_INSTR="Given a Math problem, retrieve relevant examples that help answer the problem"

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" --device cuda \
        --model Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --pooling lasttoken --append-eos --trust-remote-code \
        --query-prompt $'query: '"${MATH_INSTR}"$'\n' \
        --doc-prompt 'document: ' \
        --output "results/eval_${TIER}_rader-gte-qwen2-7b.json"
done

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" --device cuda \
        --model reasonir/ReasonIR-8B \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --trust-remote-code \
        --query-prompt $'<|user|>\n'"${MATH_INSTR}"$'\n<|embed|>\n' \
        --doc-prompt $'<|embed|>\n' \
        --output "results/eval_${TIER}_reasonir-8b.json"
done
echo "== tf4 baselines rerun finished =="
