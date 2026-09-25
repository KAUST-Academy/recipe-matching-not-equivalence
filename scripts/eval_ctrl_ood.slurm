#!/bin/bash
#SBATCH --job-name=ctrl-ood-eval
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# THE DISCRIMINATING TEST for the controlled-experiment anomaly:
# ctrl-llm scored 9.37 hard R@1 (vs <=0.16 for everything else) — suspected
# benchmark circularity (trained on the benchmark's own generation recipe).
# If its advantage vanishes on REAL equivalence (mined cross-language
# duplicates) while ctrl-cas holds, circularity is confirmed.

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

for M in ctrl-cas-6145 ctrl-llm-6145; do
    python scripts/eval_crosslingual.py --model "models/${M}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --output "results/crosslingual_${M}.json"
done
python scripts/eval_crosslingual.py --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --max-seq-length 1024 \
    --output "results/crosslingual_qwen3-0.6b-base.json"

echo "=== OOD (real-duplicate) summary ==="
python - <<'PY'
import json
for tag in ["qwen3-0.6b-base", "ctrl-cas-6145", "ctrl-llm-6145", "qwen3-0.6b-cas-cmnrl"]:
    try:
        s = json.load(open(f"results/crosslingual_{tag}.json"))["strict_crosslingual_gold"]
        print(f"  {tag:24s} R@1/5/10 = {s['recall@1']:.2f}/{s['recall@5']:.2f}/{s['recall@10']:.2f}")
    except FileNotFoundError:
        print(f"  {tag:24s} (missing)")
PY
echo "== OOD eval finished =="
