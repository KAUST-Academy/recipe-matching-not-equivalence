#!/bin/bash
#SBATCH --job-name=gen-split
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R11 (2026-09-05): the recipe arm's positives and
# near-misses written in SEPARATE calls (prompt variant `exact_split`: the
# Appendix-F prompt with the other step removed, derived from GEN_SYSTEM by
# string surgery so the wording is verbatim D1; judge pass unchanged; same
# generator, source ids and seed as D1). Then trainer conversion and the
# training job (scripts/factorial_cells.slurm CELL=d1split; registered
# readouts R11-a/b/c in its header, written before this job ran).
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
GENMODEL="${GENMODEL:-Qwen/Qwen3-32B-AWQ}"
SOURCE_IDS=data/phase2/cas_source_ids.txt
V=exact_split
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p data/pairs results
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="
conda activate vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -s "data/pairs/llm_pairs_${V}.jsonl" ]]; then
    echo "== generation output exists; skipping =="
else
    echo "== [${SECONDS}s] smoke (30 problems) =="
    python scripts/generate_llm_pairs.py \
        --model "${GENMODEL}" --backend vllm --seed 0 \
        --prompt-variant "${V}" --num-problems 30 \
        --positives-per-problem 1 --negatives-per-problem 3 \
        --output "data/pairs/dose_smoke_${V}.jsonl"
    python scripts/validate_pairs_schema.py "data/pairs/dose_smoke_${V}.jsonl"
    N_SMOKE=$(wc -l < "data/pairs/dose_smoke_${V}.jsonl")
    if (( N_SMOKE < 60 )); then
        echo "FATAL: smoke produced only ${N_SMOKE} rows (< 60): prompt/JSON issue"
        exit 1
    fi
    echo "== [${SECONDS}s] full generation over $(wc -l < ${SOURCE_IDS}) source ids (two calls each) =="
    python scripts/generate_llm_pairs.py \
        --model "${GENMODEL}" --backend vllm --seed 0 \
        --prompt-variant "${V}" \
        --source-ids-file "${SOURCE_IDS}" \
        --positives-per-problem 1 --negatives-per-problem 3 \
        --output "data/pairs/llm_pairs_${V}.jsonl" \
        --summary "results/llm_pairs_${V}_summary.json"
    python scripts/validate_pairs_schema.py "data/pairs/llm_pairs_${V}.jsonl"
fi
conda activate mathnet
echo "== [${SECONDS}s] convert to trainer format =="
python scripts/convert_llm_pairs.py --input "data/pairs/llm_pairs_${V}.jsonl" --output "data/llm_pairs_${V}/pairs.jsonl"
echo "== [${SECONDS}s] submit the training job =="
sbatch --export=ALL,CELL=d1split scripts/factorial_cells.slurm
echo "== done: $(date) =="
