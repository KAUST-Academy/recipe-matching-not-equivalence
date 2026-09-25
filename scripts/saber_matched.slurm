#!/bin/bash
#SBATCH --job-name=saber-matched
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-17: the SABER summary-channel arm (models/saber-label-4733*)
# ran 105 optimiser steps over 4,706 distinct positives where the document-channel
# arm ran 57 steps over 2,368 (reproducibility.json, entry saber_matched_budget_summary_arm). This control trains
# the summary channel at the document arm's budget on both counts: --max-rows 2368
# (one row per source; 2,360 to 2,365 distinct positives by seed, results/count_certificates.json) and --max-steps 57, everything
# else as in scripts/saber_label_seeds.slurm; three seeds; scored with the same
# eval_saber.py call (statement-full, no prompt, per-query dump) and on the
# cross-language probe.
set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
FILE=data/saber_attack/pairs_summary_channel.jsonl
ROWS=2368; STEPS=57
for S in 42 43 44; do
  OUT="models/saber-labelmatched-${ROWS}-s${S}"
  if [[ ! -d "${OUT}/final" ]]; then
    echo "== [$(date +%H:%M:%S)] train matched summary arm seed ${S} =="
    python scripts/train_invarembed.py \
      --train-file "${FILE}" --max-rows "${ROWS}" --max-steps "${STEPS}" \
      --model Qwen/Qwen3-Embedding-0.6B --model-dtype bfloat16 --bf16 \
      --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
      --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${S}" \
      --dev-frac 0.05 --eval-steps 100 --max-seq-length 1024 \
      --query-prompt-name query --output-dir "${OUT}" \
      || { echo "!!! FAILED train s${S}"; continue; }
  fi
  echo "== [$(date +%H:%M:%S)] eval SABER seed ${S} =="
  python scripts/eval_saber.py --model "${OUT}/final" --device cuda \
      --batch-size 64 --setting statement-full --emb-cache-dir .emb_cache \
      --model-dtype bfloat16 --max-seq-length 1024 \
      --per-query-out "results/saber_pq_labelmatched${ROWS}_s${S}.json" \
      --output "results/saber_labelmatched_${ROWS}_s${S}_noprompt.json" \
      || echo "!!! FAILED eval s${S}"
  echo "== [$(date +%H:%M:%S)] eval cross-lingual seed ${S} =="
  python scripts/eval_crosslingual.py --model "${OUT}/final" --device cuda --model-dtype bfloat16 \
      --query-prompt-name query --max-seq-length 1024 \
      --output "results/crosslingual_saber-labelmatched-${ROWS}-s${S}.json" \
      || echo "!!! FAILED xling s${S}"
done
echo "== end: $(date) =="
