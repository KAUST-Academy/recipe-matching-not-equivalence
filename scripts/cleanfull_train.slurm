#!/bin/bash
#SBATCH --job-name=cleanfull-train
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# E-R4 step 4: the fully-clean 6,145-row matched
# budget, eight seeds per arm, controlled-experiment hyperparameters, both
# arms trained UNDER the corrected v2 gate (--anchor-mapping v2; for the llm
# arm the file is already clean, the flag makes the gate explicit in
# run_config.json).
#
#   ARM=cas  data/cas_pairs/pairs.jsonl        v2 gate keeps 8,952 rows,
#            --max-rows caps to 6,145 (seeded shuffle-truncate)
#   ARM=llm  data/llm_pairs_cleanfull/pairs.jsonl  exactly 6,145 clean rows
#            (4,101 survivors + 2,044 fresh-source regenerations); the job
#            assembles the file first if missing (needs cleanfull-gen done).
#
# REGISTERED READOUT (scripts/cleanfull_verdict.py, written before results):
# eight-seed mean easy-tier gap (llm - cas). Published accounting predicts
# ~44 (45.33 headline - 1.24 leakage charge). Bands mirror the clean-gate
# retrain: G1 CONFIRMS >= 35.0 all seeds positive; G2 PARTIAL 15-35;
# G3 REFUTES < 15.
# Submit:  sbatch --export=ALL,ARM=cas scripts/cleanfull_train.slurm
#          sbatch --export=ALL,ARM=llm --dependency=afterok:<gen-job> ...
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

ARM="${ARM:?set ARM=cas|llm}"
case "${ARM}" in
  cas) FILE=data/cas_pairs/pairs.jsonl ;;
  llm) FILE=data/llm_pairs_cleanfull/pairs.jsonl
       [[ -s "${FILE}" ]] || python scripts/build_cleanfull_llm.py ;;
  *) echo "unknown ARM=${ARM}"; exit 1 ;;
esac

for SEED in 42 43 44 45 46 47 48 49; do
  OUT="models/cleanfull-${ARM}-6145-s${SEED}"
  TAG="cleanfull-${ARM}-s${SEED}"
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
echo "== E-R4 arm ${ARM} finished =="
