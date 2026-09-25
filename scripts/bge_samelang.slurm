#!/bin/bash
#SBATCH --job-name=bge-samelang
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-16: the BGE-large-en-v1.5 backbone is English-only,
# so its retention was read on the cross-lingual probe under a handicap. Score the
# untrained BGE base and the two seed-42 BGE arms on the English same-language
# probe (data/samelang_eval, primary slice = 125 reworded reprints; protocol of
# scripts/samelang_eval.slurm), and the base on the cross-lingual probe too, so
# the BGE rows can be read the way the Qwen rows are. Same query instruction and
# dtype as the arms' training and their cross-lingual runs.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results/ranks
Q='Represent this sentence for searching relevant passages: '
declare -A MODEL
MODEL[bge-base]="BAAI/bge-large-en-v1.5"
MODEL[ctrl-cas-bge-s42]="models/ctrl-cas-6145-bge-s42/final"
MODEL[ctrl-llm-bge-s42]="models/ctrl-llm-6145-bge-s42/final"
for tag in bge-base ctrl-cas-bge-s42 ctrl-llm-bge-s42; do
  echo "=== [$(date +%H:%M:%S)] samelang ${tag} ==="
  python scripts/eval_crosslingual.py --eval-dir data/samelang_eval \
      --model "${MODEL[$tag]}" --device cuda --model-dtype bfloat16 \
      --query-prompt "${Q}" \
      --dump-ranks "results/ranks/samelang_${tag}.ranks.jsonl" \
      --output "results/ranks/samelang_${tag}.summary.json"
done
echo "=== [$(date +%H:%M:%S)] crosslingual bge-base ==="
python scripts/eval_crosslingual.py --model "${MODEL[bge-base]}" --device cuda \
    --model-dtype bfloat16 --query-prompt "${Q}" \
    --output results/crosslingual_bge-base.json
echo "== end: $(date) =="
