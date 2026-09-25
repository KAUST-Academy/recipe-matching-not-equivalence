#!/bin/bash
#SBATCH --job-name=missing-cells
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# Fills the three "---"/missing ablation cells of the draft tables listed below:
#  (a) soups on real duplicates — adjudicates "data-space vs weight-space mixing"
#      (soup-a0.3 is 70% base weights; base scores 84.73 — it may WIN retention)
#  (b) base + V2's instruction prompt, all tiers + xling — isolates prompt effect
#  (c) V2 WITHOUT its instruction, all tiers + xling — the symmetric cell

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
cd "${PROJECT}"
INSTR=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

# (a) soups on real duplicates
for A in 0.3 0.5 0.7; do
    python scripts/eval_crosslingual.py --model "models/soup-a${A}" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
        --output "results/crosslingual_soup-a${A}.json"
done

# (b) base + V2 instruction
for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "$TIER" --model Qwen/Qwen3-Embedding-0.6B \
        --device cuda --model-dtype bfloat16 --query-prompt "$INSTR" \
        --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_qwen3-0.6b-base-instr.json"
done
python scripts/eval_crosslingual.py --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --query-prompt "$INSTR" --max-seq-length 1024 \
    --output results/crosslingual_qwen3-0.6b-base-instr.json

# (c) V2 without instruction
for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "$TIER" --model models/qwen3-0.6b-p2-instr/final \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_qwen3-0.6b-p2-instr-noinstr.json"
done
python scripts/eval_crosslingual.py --model models/qwen3-0.6b-p2-instr/final \
    --device cuda --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
    --output results/crosslingual_qwen3-0.6b-p2-instr-noinstr.json

echo "=== MISSING CELLS SUMMARY ==="
python - <<'PY'
import json
print("soups real-duplicate strict R@1/5/10 (V2 ref: 74.30, base: 84.73):")
for a in ["0.3","0.5","0.7"]:
    s = json.load(open(f"results/crosslingual_soup-a{a}.json"))["strict_crosslingual_gold"]
    print(f"  soup-a{a}: {s['recall@1']:.2f}/{s['recall@5']:.2f}/{s['recall@10']:.2f}")
for tag, name in [("qwen3-0.6b-base-instr","base+instr"), ("qwen3-0.6b-p2-instr-noinstr","V2-noinstr")]:
    row = []
    for t in ["easy","medium","hard"]:
        o = json.load(open(f"results/eval_{t}_{tag}.json"))["overall"]
        row.append(f"{o['recall@1']:.2f}")
    s = json.load(open(f"results/crosslingual_{tag}.json"))["strict_crosslingual_gold"]
    print(f"{name}: tiers R@1 {'/'.join(row)} | xling {s['recall@1']:.2f}  (V2-with-instr ref: 21.70/2.71/0.01 | 74.30)")
PY
echo "== done =="
