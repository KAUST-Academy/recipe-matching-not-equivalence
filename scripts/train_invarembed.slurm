#!/bin/bash
#SBATCH --job-name=invarembed-train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# InvarEmbed contrastive training, phase 1.
#
# PHASE 1 (active below): Qwen3-Embedding-0.6B FULL fine-tune on 1x A100 with
# CachedMultipleNegativesRankingLoss (effective contrastive batch 256 via
# gradient caching; mini-batch 16 is only a memory knob). After training it
# self-evaluates on the MathNet-Retrieve easy tier (--eval-after) and writes
#   models/qwen3-0.6b-cas-cmnrl/{final,run_config.json}
#   results/eval_easy_qwen3-0.6b-cas-cmnrl.json
# Reference numbers to compare against (results/eval_*.json, v2 Table 4):
#   qwen3-embedding-4B easy 14.76/83.53/92.13; all models ~0% R@1 on hard.
#
# EXPECTED RUNTIME: ~20-30 s/optimizer step at batch 256 x seq 1024
# (768 sequences fwd+bwd per step under grad caching). 30K exploded triplets
# -> ~350 steps over 3 epochs = ~2.5 h; 100K -> ~8 h. Easy-tier eval adds
# ~15-20 min. The 24 h limit leaves ample slack.
#
# SUPERVISION-CONTROLLED EXPERIMENT: the LLM-judged arm is THIS EXACT command
# with only --train-file (and --output-dir name) changed to
# data/llm_pairs/pairs.jsonl, plus --max-rows N on BOTH arms where
# N = min(#rows cas_pairs, #rows llm_pairs). Nothing else may change.
#
# Submit with:  sbatch scripts/train_invarembed.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"          # keep model weights on /ibex
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results models

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="

# ---------------- PHASE 1: Qwen3-Embedding-0.6B full FT, 1x A100 ----------------
python scripts/train_invarembed.py \
    --train-file data/cas_pairs/pairs.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
    --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
    --dev-frac 0.05 --eval-steps 100 \
    --max-seq-length 1024 --query-prompt-name query \
    --output-dir models/qwen3-0.6b-cas-cmnrl \
    --eval-after --eval-tier easy --eval-batch-size 128

echo "== phase-1 training + easy-tier eval finished =="

# ---------------- PHASE 2 (commented): Qwen3-Embedding-4B + LoRA, 2-4 GPUs ------
# VERIFIED 2026-07-30 (CPU): every flag below exists in train_invarembed.py,
# bare `--lora` + `--lora-r 16` parses as a fresh r16 adapter, and the
# LoRA + --gradient-checkpointing + cmnrl combination trains, checkpoints,
# load-best-reloads and save/reloads (adapter-only final/ + base reference)
# at 0.6B/MiniLM scale; both eval harnesses load adapter-only dirs. Still
# NOT GPU-verified at 4B under torchrun/DDP. For the SINGLE-GPU 4B
# controlled-experiment replication use scripts/train_ctrl_4b.slurm (batch
# 256 grad cache fits 1x A100 80GB -- see its memory estimate).
# Edit the header first:   --gres=gpu:a100:4  --mem=400G  --cpus-per-task=32
# (2-GPU variant: --gres=gpu:a100:2 --mem=200G, halve the throughput estimate).
# DDP via torchrun; --gather-across-devices shares in-batch negatives across
# ranks so the effective contrastive batch is 4 x 128 = 512. LoRA (r=16,
# all-linear) keeps the 4B backbone trainable on 80 GB cards together with
# --gradient-checkpointing. Only rank 0 saves; run the eval as a separate
# command afterwards (do NOT pass --eval-after under torchrun).
#
# torchrun --nproc_per_node=4 scripts/train_invarembed.py \
#     --train-file data/cas_pairs/pairs.jsonl \
#     --model Qwen/Qwen3-Embedding-4B \
#     --model-dtype bfloat16 --bf16 --gradient-checkpointing \
#     --lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
#     --loss cmnrl --batch-size 128 --mini-batch-size 8 --scale 20 \
#     --gather-across-devices \
#     --epochs 3 --lr 1e-4 --warmup-ratio 0.05 --seed 42 \
#     --dev-frac 0.05 --eval-steps 100 \
#     --max-seq-length 1024 --query-prompt-name query \
#     --output-dir models/qwen3-4b-cas-cmnrl-lora
#
# for TIER in easy medium hard; do
#     python scripts/eval_retrieve.py \
#         --tier "${TIER}" \
#         --model models/qwen3-4b-cas-cmnrl-lora/final \
#         --device cuda --model-dtype bfloat16 \
#         --query-prompt-name query --batch-size 32 --max-seq-length 1024 \
#         --output "results/eval_${TIER}_qwen3-4b-cas-cmnrl-lora.json"
# done
