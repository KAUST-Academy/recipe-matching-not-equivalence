#!/bin/bash
#SBATCH --job-name=bright-pq
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-16: re-score the seed-42 arms and the base on the
# two BRIGHT math splits with per-query metrics dumped, so a query-level paired
# bootstrap can put an interval on the arm difference (the original runs kept
# only aggregates). Same flags as scripts/eval_external_ood.slurm; the runs are
# deterministic, so the aggregate must reproduce results/bright_<split>_<tag>.json.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
declare -A MODEL
MODEL[qwen3-0.6b-base]="Qwen/Qwen3-Embedding-0.6B"
MODEL[ctrl-cas-6145]="models/ctrl-cas-6145/final"
MODEL[ctrl-llm-6145]="models/ctrl-llm-6145/final"
for split in theoremqa_theorems aops; do
  for tag in ctrl-llm-6145 ctrl-cas-6145 qwen3-0.6b-base; do
    echo "=== [$(date +%H:%M:%S)] ${split} / ${tag} ==="
    python scripts/eval_bright.py --split "${split}" --model "${MODEL[$tag]}" \
      --device cuda --model-dtype bfloat16 --batch-size 128 --max-seq-length 1024 \
      --query-prompt-name query \
      --output "results/bright_pq/${split}_${tag}.agg.json" \
      --per-query-out "results/bright_pq/${split}_${tag}.perquery.json"
  done
done
echo "== end: $(date) =="
