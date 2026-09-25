#!/bin/bash
#SBATCH --job-name=gen-unrelnegs
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R7 (2026-09-04): near-miss negatives written under a
# prompt UNRELATED to MathNet-Retrieve's Appendix F, so that the factorial can
# separate "near-misses written by the recipe's prompt" from "LLM-written
# near-misses of any prompt". Same
# generator (Qwen3-32B-AWQ), judge pass, source ids and generation seed as the
# ladder (scripts/dose_response.slurm); prompt variant unrelated_negs in
# scripts/generate_llm_pairs.py. The positives this run writes are discarded
# (the D4 positives on disk are reused); only the companions are new.
#
# Stages: 30-problem smoke -> full generation -> schema check -> trainer
# conversion (verified rows only, as for D1-D4) -> build the two cells
# (scripts/build_review2_cells.py --which r7) -> submit their training jobs
# (scripts/factorial_cells.slurm, registered readouts R7-a/R7-b in its header,
# written before this job ran).
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
GENMODEL="${GENMODEL:-Qwen/Qwen3-32B-AWQ}"
SOURCE_IDS=data/phase2/cas_source_ids.txt
V=unrelated_negs
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p data/pairs results
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="

conda activate vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on Ibex nodes (see generate_llm_pairs.slurm)
python - <<'PYCHECK'
import sys
import torch, vllm
cu = torch.version.cuda or "none"
print(f"[env] vllm {vllm.__version__} | torch {torch.__version__} | cuda build {cu}")
if not cu.startswith("12."):
    sys.exit(f"FATAL: torch CUDA build {cu} is not cu12x")
PYCHECK

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
    echo "== [${SECONDS}s] full generation over $(wc -l < ${SOURCE_IDS}) source ids =="
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
python scripts/convert_llm_pairs.py \
    --input "data/pairs/llm_pairs_${V}.jsonl" \
    --output "data/llm_pairs_${V}/pairs.jsonl"
echo "== [${SECONDS}s] build the E-R7 cells =="
python scripts/build_review2_cells.py --which r7
echo "== [${SECONDS}s] submit the two training jobs =="
sbatch --export=ALL,CELL=d4unrelnegs scripts/factorial_cells.slurm
sbatch --export=ALL,CELL=d4unrelnegsfull scripts/factorial_cells.slurm
echo "== done: $(date) =="
