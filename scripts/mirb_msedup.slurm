#!/bin/bash
#SBATCH --job-name=mirb-msedup
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err
# ============================================================================
# The unscored eleventh MIRB task (2026-09-17):
# msedup, Math StackExchange duplicate questions, 1,350,505 documents, 25,116
# queries, 991,836,631 excluded ids. One job per model so the three corpus
# encodes (about 2 h each at batch 128, 1,024 tokens, bf16, one A100) run in
# parallel; the document embeddings are cached under results/mirb_cache/ so
# a rerun skips the encode. Protocol unchanged from the other ten tasks
# (scripts/eval_mirb.py: title+text documents, per-query exclusion, nDCG@10
# over the top 1,000, the model's own query prompt). Gate: sbatch this with
# --dependency=afterok:<mirb_modup_check job>, which reproduces the committed
# modup result to 1e-4 and checks the new masked path against the old loop.
#
# PRE-REGISTERED READING (written before any msedup number existed; the ten
# scored tasks have the base ahead of both arms on nine, and modup, the other
# duplicate-question task, reads 72.67 / 52.37 / 48.22 base / verified / recipe):
#   R1  base > verified > recipe, like modup: ood.tex keeps its shape and reads
#       "eleven ... the base leads both arms on ten".
#   R2  base leads both arms but recipe > verified: the tab:mirb count of tasks
#       where the recipe arm trails the verified arm by an interval excluding
#       zero is re-derived from results/mirb_summary.json; ood.tex still "ten".
#   R3  an arm beats the base: say so in the tab:mirb caption and in reproducibility.json
#       and re-derive the ood.tex count; do not soften the main-text sentence.
#   Paired query bootstrap (5,000 resamples, seed 0) as for the other tasks.
#
# Submit (TAG selects the model):
#   for TAG in qwen3-0.6b-base ctrl-cas-6145 ctrl-llm-6145; do
#     sbatch --job-name=mirb-msedup-$TAG --export=ALL,TAG=$TAG scripts/mirb_msedup.slurm
#   done
# PROJECT and CONDA_SH default to the original clone; export them to run
# from another checkout (HF_HOME likewise, default ${PROJECT}/.hf_cache).
# ============================================================================
set -euo pipefail
PROJECT="${PROJECT:-/ibex/user/habiam0b/MathNet_Follow_Up}"
CONDA_SH="${CONDA_SH:-/ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate mathnet
export HF_HOME="${HF_HOME:-${PROJECT}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"; mkdir -p results/mirb results/mirb_cache

TAG="${TAG:?set TAG=qwen3-0.6b-base|ctrl-cas-6145|ctrl-llm-6145}"
case "${TAG}" in
  qwen3-0.6b-base) MODEL=Qwen/Qwen3-Embedding-0.6B ;;
  ctrl-cas-6145)   MODEL=models/ctrl-cas-6145/final ;;
  ctrl-llm-6145)   MODEL=models/ctrl-llm-6145/final ;;
  *) echo "unknown TAG=${TAG}"; exit 1 ;;
esac
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  tag: ${TAG} =="
python scripts/eval_mirb.py --task msedup --model "${MODEL}" --device cuda --model-dtype bfloat16 \
    --batch-size 128 --max-seq-length 1024 \
    --doc-cache "results/mirb_cache/msedup_${TAG}.npy" \
    --output "results/mirb/msedup_${TAG}.json"
echo "== msedup ${TAG} finished: $(date) =="
