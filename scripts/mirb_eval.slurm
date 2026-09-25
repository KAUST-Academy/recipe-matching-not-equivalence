#!/bin/bash
#SBATCH --job-name=mirb-eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-16: the untrained base and the two seed-42 arms on
# MIRB's eleven natural-language tasks (the Lean/Isabelle/HOL/Mathlib formal-
# library premise tasks are outside the paper's domain). Query prompt = the
# model's "query" prompt, as for every other external benchmark in the paper.
set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"; mkdir -p results/mirb
declare -A MODEL
MODEL[qwen3-0.6b-base]="Qwen/Qwen3-Embedding-0.6B"
MODEL[ctrl-cas-6145]="models/ctrl-cas-6145/final"
MODEL[ctrl-llm-6145]="models/ctrl-llm-6145/final"
for task in numbertheoryps realanalysisps wikiformula mseformula stacksqa stacksps proofwikiqa proofwikips mseqa modup msedup; do
  for tag in qwen3-0.6b-base ctrl-cas-6145 ctrl-llm-6145; do
    out="results/mirb/${task}_${tag}.json"
    [ -s "$out" ] && { echo "=== skip ${task}/${tag}"; continue; }
    echo "=== [$(date +%H:%M:%S)] ${task} / ${tag} ==="
    python scripts/eval_mirb.py --task "${task}" --model "${MODEL[$tag]}" --device cuda \
      --model-dtype bfloat16 --batch-size 128 --max-seq-length 1024 --output "$out" \
      || echo "!!! FAILED ${task}/${tag}"
  done
done
echo "== end: $(date) =="
