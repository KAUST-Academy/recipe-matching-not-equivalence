#!/bin/bash
#SBATCH --job-name=p2-ood-eval
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
cd "${PROJECT}"
for M in qwen3-0.6b-p2-mixed qwen3-0.6b-p2-instr; do
    ARGS=(--device cuda --model-dtype bfloat16 --max-seq-length 1024)
    if [ "$M" = "qwen3-0.6b-p2-instr" ]; then
        ARGS+=(--query-prompt $'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:')
    else
        ARGS+=(--query-prompt-name query)
    fi
    python scripts/eval_crosslingual.py --model "models/${M}/final" "${ARGS[@]}" \
        --output "results/crosslingual_${M}.json"
done
python - <<'PY'
import json
print("=== real-duplicate (forgetting) check ===")
for tag in ["qwen3-0.6b-base", "qwen3-0.6b-p2-mixed", "qwen3-0.6b-p2-instr", "ctrl-cas-6145", "qwen3-0.6b-cas-cmnrl"]:
    try:
        s = json.load(open(f"results/crosslingual_{tag}.json"))["strict_crosslingual_gold"]
        print(f"  {tag:24s} {s['recall@1']:.2f}/{s['recall@5']:.2f}/{s['recall@10']:.2f}")
    except FileNotFoundError: print(f"  {tag:24s} missing")
PY
echo "== p2 OOD eval finished =="
