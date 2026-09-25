#!/bin/bash
#SBATCH --job-name=fill-crosseval
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# Table fill (2026-09-08): the empty cells of paper Table tab:crosseval.
#  (a) the reference (fact-d4casnegs, seed 42: the seed the defense table uses)
#      scored on the five regenerated-gold evaluation variants of
#      scripts/defense_regen_eval.py (orig / exact / paraphrase / style / mixed),
#      exactly as scripts/defense_regen.slurm scored the other six models;
#  (b) the D3 arm (dose-style, seeds 42-44) on the generator-free near-miss
#      probe, both rankings, exactly as scripts/factorial_cells.slurm scores a cell
#      (skipped where the summary already exists).
# Evaluation only; no retraining. No directional prediction registered.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
echo "== node: $(hostname)  start: $(date) =="
REF=models/fact-d4casnegs-6145-s42/final
for NAME in orig exact paraphrase style mixed; do
  OUT="results/defense_eval/eval_${NAME}_reference.json"
  if [[ -f "${OUT}" ]]; then echo "[skip] ${OUT}"; continue; fi
  echo "=== [${SECONDS}s] defense ${NAME} reference"
  python scripts/eval_retrieve.py --data-dir "data/defense_eval/${NAME}" --model "${REF}" \
      --device cuda --model-dtype bfloat16 --query-prompt-name query \
      --batch-size 64 --max-seq-length 1024 --output "${OUT}"
done
for SEED in 42 43 44; do
  M="models/dose-style-6145$([[ ${SEED} == 42 ]] || echo -s${SEED})/final"
  TAG="dose-style-s${SEED}"
  for PROBE in casprobe:data/casprobe_eval casprobepairs:data/casprobe_pairs_eval; do
    PFX="${PROBE%%:*}"; PDIR="${PROBE#*:}"
    PD="results/ranks/${PFX}_${TAG}.ranks.jsonl"
    [[ -s "results/ranks/${PFX}_${TAG}.summary.json" ]] && { echo "[skip] ${PFX} ${TAG}"; continue; }
    echo "=== [${SECONDS}s] eval ${PFX} ${TAG}"
    python scripts/eval_crosslingual.py --eval-dir "${PDIR}" --model "${M}" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
        --dump-ranks "${PD}" --output "results/ranks/${PFX}_${TAG}.summary.json"
  done
done
echo "== done: $(date) =="
