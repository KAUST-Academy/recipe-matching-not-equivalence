#!/bin/bash
#SBATCH --job-name=gen-survey
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R10 (2026-09-05): a SECOND recipe-free
# restatement prompt (D5, prompt variant `survey`: a survey author restating
# another author's problem for a themed collection; own wording, notation and
# order; level and mathematics unchanged; positives only, no near-miss
# channel, exactly as D4). Same generator (Qwen3-32B-AWQ), judge pass, source
# ids and generation seed as the ladder. Then: trainer conversion (verified
# rows only), the two cells (D5 alone = data/llm_pairs_survey; D5 + the
# reference's verified negatives = build_review2_cells.py --which r10), and
# their training jobs (scripts/factorial_cells.slurm CELL=d5nonegs /
# d5casnegs; registered readouts R10-a/b/c in its header, written before this
# job ran).
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
GENMODEL="${GENMODEL:-Qwen/Qwen3-32B-AWQ}"
SOURCE_IDS=data/phase2/cas_source_ids.txt
V=survey
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
        --positives-per-problem 1 --negatives-per-problem 0 \
        --output "data/pairs/dose_smoke_${V}.jsonl"
    python scripts/validate_pairs_schema.py "data/pairs/dose_smoke_${V}.jsonl"
    N_SMOKE=$(wc -l < "data/pairs/dose_smoke_${V}.jsonl")
    if (( N_SMOKE < 20 )); then
        echo "FATAL: smoke produced only ${N_SMOKE} rows (< 20): prompt/JSON issue"
        exit 1
    fi
    echo "== [${SECONDS}s] full generation over $(wc -l < ${SOURCE_IDS}) source ids =="
    python scripts/generate_llm_pairs.py \
        --model "${GENMODEL}" --backend vllm --seed 0 \
        --prompt-variant "${V}" \
        --source-ids-file "${SOURCE_IDS}" \
        --positives-per-problem 1 --negatives-per-problem 0 \
        --output "data/pairs/llm_pairs_${V}.jsonl" \
        --summary "results/llm_pairs_${V}_summary.json"
    python scripts/validate_pairs_schema.py "data/pairs/llm_pairs_${V}.jsonl"
fi
conda activate mathnet
echo "== [${SECONDS}s] convert to trainer format =="
python scripts/convert_llm_pairs.py --input "data/pairs/llm_pairs_${V}.jsonl" --output "data/llm_pairs_${V}/pairs.jsonl"
echo "== [${SECONDS}s] build the E-R10 cell (D5 + reference negatives) =="
python scripts/build_review2_cells.py --which r10
echo "== [${SECONDS}s] submit the two training jobs =="
sbatch --export=ALL,CELL=d5nonegs scripts/factorial_cells.slurm
sbatch --export=ALL,CELL=d5casnegs scripts/factorial_cells.slurm
echo "== done: $(date) =="
