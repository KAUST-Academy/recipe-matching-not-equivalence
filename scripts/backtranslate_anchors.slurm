#!/bin/bash
#SBATCH --job-name=bt-anchors
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R9 (2026-09-04): back-translated positives, the non-LLM
# deep-paraphrase control. scripts/backtranslate_anchors.py
# round-trips every reference-cell source problem through NLLB-200-distilled-
# 600M (English -> French -> English; other languages via English), formulas
# masked, sentence-level validation with a prose-only fallback; see its
# docstring. Pilot (13 stratified anchors, results/bt_pilot.json, 2026-09-04):
# every anchor round-tripped under every placeholder style; the bracket style
# kept the most character overlap (mean char-3 Jaccard 0.80) and is used here.
# Then: build the cell (scripts/build_review2_cells.py --which r9) and submit
# its training (scripts/factorial_cells.slurm CELL=btcasnegs; registered
# reading R9 in that header, written before this job ran).
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="
if [[ -s data/bt_pairs/pairs.jsonl ]]; then
    echo "== data/bt_pairs/pairs.jsonl exists; skipping translation =="
else
    python scripts/backtranslate_anchors.py --style BR --device cuda --beams 4 --batch 48
fi
echo "== [${SECONDS}s] build the E-R9 cell =="
python scripts/build_review2_cells.py --which r9
echo "== [${SECONDS}s] submit the training job =="
sbatch --export=ALL,CELL=btcasnegs scripts/factorial_cells.slurm
echo "== done: $(date) =="
