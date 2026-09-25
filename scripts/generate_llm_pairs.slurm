#!/bin/bash
#SBATCH --job-name=invarembed-llm-pairs
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# InvarEmbed TASK E: LLM-judged supervision arm (contrast condition) on 1x A100.
# Runs scripts/generate_llm_pairs.py: pass 1 rewrites each source problem into
# 1 claimed-equivalent + 3 claimed near-miss variants (MathNet Appendix F
# prompt), pass 2 LLM-judges every pair. No symbolic verification (that is the
# CAS arm's job) — this arm reproduces the RaDeR/ReasonIR/MELD supervision
# style at the SAME data budget as the CAS arm.
#
# MODEL CHOICE (verified against the HF API 2026-07-29, all ungated):
#   * Qwen/Qwen3-32B-AWQ  <-- CHOSEN. Official Qwen 4-bit AWQ of the strongest
#     dense open model that fits: 19.3 GB weights = 24% of 80 GB, so at
#     --gpu-memory-utilization 0.92 about 54 GB remain for KV cache (~68% of
#     the GPU; KV is 0.25 MB/token -> ~200K cached tokens -> ~50 concurrent
#     4K-context sequences). AWQ-Marlin kernels are supported on A100 (sm80).
#     Hybrid-thinking model, so the script's enable_thinking=False works.
#   * Qwen/Qwen3-32B bf16 (65.5 GB) and Qwen3-30B-A3B-Instruct-2507 bf16
#     (61 GB) leave <20% VRAM headroom at util 0.92 -> rejected.
#   * Qwen3-30B-A3B-Instruct-2507-FP8 (31 GB) is faster (3B active) but FP8
#     block-quant on Ampere relies on Marlin W8A16 fallbacks and its chat
#     template has no enable_thinking switch -> rejected as riskier.
#   * Qwen/Qwen3-14B(-AWQ) exists (9.99 GB AWQ) = the fallback if 32B-AWQ
#     misbehaves; strictly weaker judge, only use if forced.
#
# RUNTIME ESTIMATE (7,089 problems, honest):
#   pass 1: 7,089 gen calls, ~1K prompt + ~0.9-1.3K output tokens each;
#   pass 2: ~28.4K judge calls, ~0.6-1.1K prompt + ~0.1K output (thinking off)
#   => ~10-12M output tokens total. Qwen3-32B-AWQ on one A100-80 with
#   continuous batching sustains roughly 400-1000 aggregate output tok/s
#   => ~3-9 h generation + ~15 min model load. 24 h limit = >2.5x slack.
#
# ENV: the 'vllm' conda env was pre-built on the login node 2026-07-29
# (python 3.12.13, vllm 0.26.0+cu129, torch 2.11.0+cu129 — the cu129 stack is
# the CUDA-12 build line; the default PyPI vllm/torch wheels are CUDA-13 and
# CANNOT run on Ibex's driver 12080 = CUDA <= 12.8. cu129 user-space runs on
# any r525+ driver via CUDA minor-version compatibility). This job only
# activates + sanity-checks the env; it never installs anything.
# The full engine was smoke-tested live on an A100-SXM4-80GB under driver
# 570.86.15 (= CUDA 12.8): load + greedy + top-p generation OK with
# VLLM_USE_FLASHINFER_SAMPLER=0 (see below). Only the AWQ-Marlin path of the
# 32B model itself remains unverified until this job's 30-problem smoke.
#
# MATCHED-BUDGET CONTRACT: SOURCE_IDS_FILE must be the exact id list the CAS
# arm (generate_cas_pairs.py) used. Default: data/phase2/cas_source_ids.txt =
# the 7,397 unique source_ids of data/cas_pairs/pairs.jsonl minus the 308
# cross-lingual-eval members (contamination gate #2) = 7,089 ids, identical
# to data/pairs/source_ids.txt (kept for compatibility). The job REFUSES to
# run without it unless you export SOURCE_IDS_FILE=none (then it takes a
# seeded 5K sample and writes the id list for the CAS arm to consume).
#
# Submit with:
#   sbatch scripts/generate_llm_pairs.slurm
#   SOURCE_IDS_FILE=/path/to/ids.txt sbatch --export=ALL scripts/generate_llm_pairs.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
SOURCE_IDS_FILE="${SOURCE_IDS_FILE:-${PROJECT}/data/phase2/cas_source_ids.txt}"
NUM_PROBLEMS="${NUM_PROBLEMS:-5000}"   # only used when SOURCE_IDS_FILE=none

source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh

# ---- separate pre-built 'vllm' conda env (built on the login node
# ---- 2026-07-29) so vLLM's pinned
# ---- torch/CUDA stack can never break the mathnet env.
conda activate vllm
python - <<'PYCHECK'
import sys
import torch, vllm
cu = torch.version.cuda or "none"
print(f"[env] vllm {vllm.__version__} | torch {torch.__version__} | cuda build {cu}")
if not cu.startswith("12."):
    sys.exit(f"FATAL: torch CUDA build {cu} is not a cu12x build -- Ibex driver "
             "12080 supports CUDA <= 12.x user-space only. Rebuild the vllm env "
             "(vllm +cu129 wheel + torch 2.11.0+cu129, see requirements.txt section 3).")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False on the GPU node")
PYCHECK

export HF_HOME="${PROJECT}/.hf_cache"        # keep model weights on /ibex
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# FlashInfer's top-k/top-p sampling kernels JIT-compile with nvcc, which Ibex
# nodes don't expose (/usr/local/cuda absent). Attention still uses the
# precompiled FLASH_ATTN backend; sampling falls back to native torch ops.
# Verified live 2026-07-29 on an A100 (driver 570.86.15): engine + greedy +
# top-p sampling all work with this set; without it, engine init dies in
# flashinfer JIT ("Could not find nvcc").
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "${PROJECT}"
mkdir -p data/pairs results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== model: ${MODEL}  vllm: $(python -c 'import vllm; print(vllm.__version__)') =="

# ---- sanity: prompts + schema contract (no GPU needed, seconds) ------------
python scripts/generate_llm_pairs.py --dry-run --num-problems 3 > results/llm_pairs_dryrun_prompts.txt
python scripts/generate_llm_pairs.py --emit-mock data/pairs/llm_pairs_mock.jsonl --num-problems 3
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_mock.jsonl

# ---- matched-budget source selection ---------------------------------------
if [[ "${SOURCE_IDS_FILE}" == "none" ]]; then
    echo "== WARNING: no source-ids file — sampling ${NUM_PROBLEMS} problems (seed 0)"
    echo "==          and writing data/pairs/source_ids.txt for the CAS arm to reuse."
    SRC_ARGS=(--num-problems "${NUM_PROBLEMS}" --write-source-ids data/pairs/source_ids.txt)
elif [[ -f "${SOURCE_IDS_FILE}" ]]; then
    echo "== matched-budget mode: source ids from ${SOURCE_IDS_FILE}"
    SRC_ARGS=(--source-ids-file "${SOURCE_IDS_FILE}")
else
    echo "FATAL: SOURCE_IDS_FILE=${SOURCE_IDS_FILE} not found. Point it at the CAS arm's"
    echo "       id list, or export SOURCE_IDS_FILE=none to define the set from this arm."
    exit 1
fi

# ---- 30-problem smoke test first (fails fast on model/JSON issues) ---------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 0 \
    --num-problems 30 \
    --output data/pairs/llm_pairs_smoke.jsonl
python scripts/validate_pairs_schema.py data/pairs/llm_pairs_smoke.jsonl

# ---- full run ---------------------------------------------------------------
python scripts/generate_llm_pairs.py \
    --model "${MODEL}" --backend vllm --seed 0 \
    "${SRC_ARGS[@]}" \
    --positives-per-problem 1 --negatives-per-problem 3 \
    --output data/pairs/llm_pairs.jsonl \
    --summary results/llm_pairs_summary.json

python scripts/validate_pairs_schema.py data/pairs/llm_pairs.jsonl
echo "== LLM-judged arm finished =="
