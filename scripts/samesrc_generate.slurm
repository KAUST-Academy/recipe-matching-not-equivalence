#!/bin/bash
#SBATCH --job-name=samesrc-gen
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err

# ============================================================================
# Same-source regenerated control, step 2b (2026-09-17): the same-source regenerated
# control. The regenerated retrain (cleanfull) gave the recipe arm 2,044 rows
# from fresh sources the verified arm never sees. Here the missing third is
# regenerated from the SAME 4,101 clean sources instead (data/pairs/
# source_ids_survivors.txt, scripts/make_samesrc_sources.py): a second rewrite
# per source under the same prompt (variant "exact"), generator and judge
# (Qwen3-32B-AWQ), 1 positive + 3 near-miss negatives per problem, the
# corrected v2 gate; the ONLY change from the original arm's pipeline is the
# generation seed (--seed 1; decoding is sampled at temperature 0.7, so the
# rewrite differs). Reference: the 3,000-source cleanfull run took 3,620 s and
# verified a positive for 86.7% of sources, so expect about 3,550 verified rows
# here in about 1.5 h; 2,044 are used. New output paths; nothing existing is
# touched. Env/model rationale: scripts/generate_llm_pairs.slurm.
#
# Submit:  sbatch scripts/samesrc_generate.slurm
# PROJECT, CONDA_SH and HF_HOME default to the original clone; export them
# to run from another checkout.
# ============================================================================

set -euo pipefail
PROJECT="${PROJECT:-/ibex/user/habiam0b/MathNet_Follow_Up}"
CONDA_SH="${CONDA_SH:-/ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh}"
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
source "${CONDA_SH}"
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
export HF_HOME="${HF_HOME:-${PROJECT}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${PROJECT}"
mkdir -p data/pairs results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="

# ---- survivor list (asserts 4,101, no eval members, all gated CAS sources) --
python scripts/make_samesrc_sources.py

# ---- 30-problem smoke first (fails fast on model/JSON issues) --------------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 1 \
    --exclusion-file anchor_to_corpus_mapping_v2.json \
    --num-problems 30 \
    --output data/pairs/llm_pairs_samesrc_smoke.jsonl
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_samesrc_smoke.jsonl

# ---- full run: second rewrite of every survivor source ---------------------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 1 \
    --source-ids-file data/pairs/source_ids_survivors.txt \
    --exclusion-file anchor_to_corpus_mapping_v2.json \
    --positives-per-problem 1 --negatives-per-problem 3 \
    --output data/pairs/llm_pairs_samesrc_raw.jsonl \
    --summary results/llm_pairs_samesrc_summary.json
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_samesrc_raw.jsonl
echo "== samesrc generation finished: $(date) =="
