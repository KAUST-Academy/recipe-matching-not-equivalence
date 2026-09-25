#!/bin/bash
#SBATCH --job-name=bt2-anchors
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R13 (2026-09-05): MULTI-HOP back-translation.
# scripts/backtranslate_anchors.py --pivots fra_Latn,deu_Latn,rus_Cyrl applies
# the E-R9 machinery three times in sequence (source -> French -> source ->
# German -> source -> Russian -> source; a hop whose pivot equals the source
# language uses English), formulas masked at every hop, a row kept only if
# every hop validates. Depth is measured against the original source
# (E-R9 single hop: char-3 Jaccard 0.783). Then: build the cell
# (build_review2_cells.py --which r13 -> data/bt2_pairs_casnegs, the
# reference's verified negatives) and submit its training
# (factorial_cells.slurm CELL=bt2casnegs; registered reading R13 in its
# header, written before this job ran).
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
if [[ -s data/bt2_pairs/pairs.jsonl ]]; then
    echo "== data/bt2_pairs/pairs.jsonl exists; skipping translation =="
else
    python scripts/backtranslate_anchors.py --style BR --device cuda --beams 4 --batch 48 \
        --pivots fra_Latn,deu_Latn,rus_Cyrl \
        --out-dir data/bt2_pairs --stats results/bt2_pairs_build.json
fi
echo "== [${SECONDS}s] build the E-R13 cell =="
python scripts/build_review2_cells.py --which r13
echo "== [${SECONDS}s] submit the training job =="
sbatch --export=ALL,CELL=bt2casnegs scripts/factorial_cells.slurm
echo "== done: $(date) =="
