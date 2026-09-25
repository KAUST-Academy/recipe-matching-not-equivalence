#!/bin/bash
#SBATCH --job-name=samesrc-train
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err

# ============================================================================
# Same-source regenerated control, step 2d (2026-09-17): train and evaluate the
# same-source regenerated recipe arm, eight seeds, controlled-experiment
# hyperparameters, corrected v2 gate. The training file is
# data/llm_pairs_samesrc/pairs.jsonl (scripts/build_samesrc_llm.py: the 4,101
# survivors plus the first 2,044 verified second rewrites of the SAME sources,
# 6,145 rows over exactly 4,101 sources). The verified arm is NOT retrained:
# the comparison is models/cleanfull-cas-6145-s{42..49} and their results/
# ranks/*_cleanfull-cas-s*.summary.json (same gate, same 6,145 cap, same
# source list). Reference: the cleanfull llm arm took 26,132 s for eight
# seeds of train + easy/medium/hard + xling (job 50923606).
#
# REGISTERED READOUT (scripts/samesrc_verdict.py, written before results):
#   PRIMARY  eight-seed mean easy-tier gap (samesrc-llm - cleanfull-cas).
#            G1 CONFIRMS  >= 35.0 and positive in every seed (predicted about
#                         44, as for the regenerated retrain's 44.45)
#            G2 PARTIAL   15.0 <= gap < 35.0
#            G3 REFUTES   gap < 15.0
#   THE POINT OF THE CONTROL  the real-duplicate (xling strict R@1) gap. If
#            the source-list difference explains the regenerated column's
#            17.40, the same-source gap sits near the filtered column's 11.04
#            (nearer to 11.04 than to 17.40); if it stays near 17.40, the
#            movement is the budget, not the sources. The verdict script
#            prints the distance to each and which is nearer.
#   Also     hard-tier gap (reference 9.21 / 9.26), the difference-in-
#            differences (easy gap minus xling gap; reference 27.05 = 44.45 -
#            17.40, filtered 32.07, headline 35.59).
#
# Submit:  sbatch --dependency=afterok:<samesrc-gen job> scripts/samesrc_train.slurm
# PROJECT, CONDA_SH and HF_HOME default to the original clone; export them
# to run from another checkout.
# ============================================================================

set -euo pipefail
PROJECT="${PROJECT:-/ibex/user/habiam0b/MathNet_Follow_Up}"
CONDA_SH="${CONDA_SH:-/ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate mathnet
export HF_HOME="${HF_HOME:-${PROJECT}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

FILE=data/llm_pairs_samesrc/pairs.jsonl
[[ -s "${FILE}" ]] || python scripts/build_samesrc_llm.py

for SEED in 42 43 44 45 46 47 48 49; do
  OUT="models/samesrc-llm-6145-s${SEED}"
  TAG="samesrc-llm-s${SEED}"
  if [[ -d "${OUT}/final" ]]; then
    echo "=== skip train ${TAG} (exists)"
  else
    echo "=== [${SECONDS}s] train ${TAG}"
    python scripts/train_invarembed.py \
        --train-file "${FILE}" --max-rows 6145 \
        --anchor-mapping anchor_to_corpus_mapping_v2.json \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${OUT}"
  fi
  for TIER in easy medium hard; do
    BD="results/ranks/${TIER}_${TAG}.ranks.jsonl"
    [[ -s "${BD}" ]] && { echo "=== skip ${TIER} ${TAG}"; continue; }
    echo "=== [${SECONDS}s] eval ${TIER} ${TAG}"
    python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
        --dump-ranks "${BD}" \
        --output "results/ranks/${TIER}_${TAG}.summary.json"
  done
  XD="results/ranks/xling_${TAG}.ranks.jsonl"
  if [[ -s "${XD}" ]]; then
    echo "=== skip xling ${TAG}"
  else
    echo "=== [${SECONDS}s] eval xling ${TAG}"
    python scripts/eval_crosslingual.py --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --dump-ranks "${XD}" \
        --output "results/ranks/xling_${TAG}.summary.json"
  fi
done
python scripts/samesrc_verdict.py
echo "== samesrc arm finished: $(date) =="
