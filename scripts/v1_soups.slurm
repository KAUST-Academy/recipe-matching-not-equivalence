#!/bin/bash
#SBATCH --job-name=v1-soups
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-16: the forgetting-remedy comparison sets weight
# mixing (soups of the uncapped verified arm, no replay) against data mixing
# (V1/V2, verified pairs + 12,000 replay rows), so replay content and mixing
# mechanism are confounded. Control: soup the replay-trained V1
# (models/qwen3-0.6b-p2-mixed/final, seed 42) with the untrained base at the
# same alphas and score it on the three tiers and both duplicate probes with
# the flags of scripts/eval_soups.slurm / samelang_eval.slurm.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"; mkdir -p models/v1soups results/ranks
python scripts/soup_models.py --finetuned models/qwen3-0.6b-p2-mixed/final --alphas 0.3,0.5,0.7 \
    --output-root models/v1soups --skip-existing
for A in 0.3 0.5 0.7; do
  M="models/v1soups/soup-a${A}"
  for TIER in easy medium hard; do
    echo "=== [$(date +%H:%M:%S)] v1soup-a${A} / ${TIER} ==="
    python scripts/eval_retrieve.py --tier "${TIER}" --model "$M" --device cuda --model-dtype bfloat16 \
        --query-prompt-name query --batch-size 64 --max-seq-length 1024 \
        --output "results/eval_${TIER}_v1soup-a${A}.json"
  done
  echo "=== [$(date +%H:%M:%S)] v1soup-a${A} / cross-lingual ==="
  python scripts/eval_crosslingual.py --model "$M" --device cuda --model-dtype bfloat16 \
      --query-prompt-name query --max-seq-length 1024 --output "results/crosslingual_v1soup-a${A}.json"
  echo "=== [$(date +%H:%M:%S)] v1soup-a${A} / same-language ==="
  python scripts/eval_crosslingual.py --eval-dir data/samelang_eval --model "$M" --device cuda \
      --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
      --dump-ranks "results/ranks/samelang_v1soup-a${A}.ranks.jsonl" \
      --output "results/ranks/samelang_v1soup-a${A}.summary.json"
done
echo "== end: $(date) =="
