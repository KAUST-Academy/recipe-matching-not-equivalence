#!/bin/bash
#SBATCH --job-name=impliret-cal
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# Re-runs the two rows that failed in 49700521:
#   mpnet     : bfloat16 is not supported on that model's path -> run in fp32
#   ReasonIR  : baselines-tf4 has no `datasets` module -> read the local dump
# ReasonIR is the calibration row, so this job decides whether ImpliRet numbers
# may be compared to their Table 2 at all.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

conda activate mathnet
python scripts/eval_impliret.py --model sentence-transformers/all-mpnet-base-v2 \
    --device cuda --max-seq-length 384 --batch-size 64 --local \
    --output results/impliret_mpnet.json || echo "FAILED mpnet"
conda deactivate

conda activate baselines-tf4
REASONIR_PROMPT=$'<|user|>\nGiven a question, retrieve relevant passages that answer the question\n<|embed|>\n'
python scripts/eval_impliret.py --model reasonir/ReasonIR-8B --device cuda \
    --model-dtype bfloat16 --max-seq-length 2048 --batch-size 8 --local \
    --trust-remote-code --query-prompt "${REASONIR_PROMPT}" \
    --output results/impliret_reasonir-8b.json || echo "FAILED reasonir"
conda deactivate

conda activate mathnet
python scripts/impliret_calibration.py
