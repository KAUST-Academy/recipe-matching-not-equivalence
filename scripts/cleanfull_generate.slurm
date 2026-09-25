#!/bin/bash
#SBATCH --job-name=cleanfull-gen
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# E-R4 step 2: regenerate recipe-arm pairs from
# 3,000 FRESH v2-clean sources (data/pairs/source_ids_cleanfull.txt, written
# by scripts/make_clean_sources.py) to restore a fully-clean 6,145-row
# matched budget. Identical pipeline to the original arm (Appendix-F prompt
# variant "exact", Qwen3-32B-AWQ generator + judge, 1 pos + 3 near-miss per
# problem, seed 0); the ONLY differences are the source list and the
# exclusion file, which is the CORRECTED v2 gate. Output paths are new --
# the original data/pairs/llm_pairs.jsonl is never touched.
# Env/model rationale: see scripts/generate_llm_pairs.slurm (copied verbatim).
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate vllm
python - <<'PYCHECK'
import sys
import torch, vllm
cu = torch.version.cuda or "none"
print(f"[env] vllm {vllm.__version__} | torch {torch.__version__} | cuda build {cu}")
if not cu.startswith("12."):
    sys.exit("FATAL: torch CUDA build is not cu12x")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False on the GPU node")
PYCHECK
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${PROJECT}"

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="

# ---- 30-problem smoke first (fails fast on model/JSON issues) --------------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 0 \
    --exclusion-file anchor_to_corpus_mapping_v2.json \
    --num-problems 30 \
    --output data/pairs/llm_pairs_cleanfull_smoke.jsonl
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_cleanfull_smoke.jsonl

# ---- full run --------------------------------------------------------------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 0 \
    --source-ids-file data/pairs/source_ids_cleanfull.txt \
    --exclusion-file anchor_to_corpus_mapping_v2.json \
    --positives-per-problem 1 --negatives-per-problem 3 \
    --output data/pairs/llm_pairs_cleanfull_raw.jsonl \
    --summary results/llm_pairs_cleanfull_summary.json
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_cleanfull_raw.jsonl
echo "== cleanfull generation finished =="
