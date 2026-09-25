#!/bin/bash
#SBATCH --job-name=colbert-arms
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# Late-interaction arms (2026-09-17). Fine-tune ColBERTv2 on
# the recipe arm's file and on the verified arm's file (same rows, gates and
# dev split as the dense arms, via train_invarembed.build_datasets), then score
# the untrained ColBERTv2 and both arms on the three MathNet-Retrieve tiers and
# the two duplicate probes with scripts/colbert_eval.py (PLAID top-1000).
set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate pylate
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"; mkdir -p results/colbert results/ranks
IDX="${TMPDIR:-/tmp}/colbert_index_${SLURM_JOB_ID}"
echo "== self-test: base model on 300 same-language docs =="
python scripts/colbert_eval.py --data-dir data/samelang_eval --dup --model colbert-ir/colbertv2.0 --max-docs 300 --k 50 \
    --index-root "$IDX" --output results/colbert/selftest.json || { echo "!!! self-test failed"; exit 1; }
declare -A FILE=( [ctrl-llm]="data/llm_pairs/pairs.jsonl" [ctrl-cas]="data/cas_pairs/pairs.jsonl" )
for ARM in ctrl-llm ctrl-cas; do
  OUT="models/colbert-${ARM}-6145"
  if [[ ! -d "${OUT}/final" ]]; then
    echo "== [$(date +%H:%M:%S)] train colbert ${ARM} =="
    python scripts/colbert_train.py --train-file "${FILE[$ARM]}" --max-rows 6145 --seed 42 --output-dir "${OUT}" \
      || { echo "!!! FAILED train ${ARM}"; continue; }
  fi
done
declare -A MODEL=( [colbertv2-base]="colbert-ir/colbertv2.0" [ctrl-llm]="models/colbert-ctrl-llm-6145/final" [ctrl-cas]="models/colbert-ctrl-cas-6145/final" )
for TAG in colbertv2-base ctrl-llm ctrl-cas; do
  M="${MODEL[$TAG]}"; [[ "$TAG" == colbertv2-base || -d "$M" ]] || { echo "!!! missing $M"; continue; }
  for TIER in easy medium hard; do
    OUT="results/colbert/${TIER}_${TAG}.json"; [[ -s "$OUT" ]] && continue
    echo "== [$(date +%H:%M:%S)] eval ${TAG} / ${TIER} =="
    python scripts/colbert_eval.py --data-dir "data/retrieve/${TIER}" --model "$M" --index-root "$IDX" --output "$OUT" || echo "!!! FAILED eval ${TAG} ${TIER}"
  done
  echo "== [$(date +%H:%M:%S)] eval ${TAG} / cross-lingual =="
  python scripts/colbert_eval.py --data-dir data/crosslingual_eval --dup --model "$M" --index-root "$IDX" \
      --output "results/colbert/crosslingual_${TAG}.json" --dump-ranks "results/ranks/xling_colbert_${TAG}.ranks.jsonl" || echo "!!! FAILED xling ${TAG}"
  echo "== [$(date +%H:%M:%S)] eval ${TAG} / same-language =="
  python scripts/colbert_eval.py --data-dir data/samelang_eval --dup --model "$M" --index-root "$IDX" \
      --output "results/colbert/samelang_${TAG}.json" --dump-ranks "results/ranks/samelang_colbert_${TAG}.ranks.jsonl" || echo "!!! FAILED samelang ${TAG}"
done
rm -rf "$IDX"
echo "== end: $(date) =="
