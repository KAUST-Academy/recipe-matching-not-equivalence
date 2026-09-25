#!/bin/bash
#SBATCH --job-name=invarembed-controlled
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# THE CONTROLLED EXPERIMENT (paper headline): verified (CAS) vs LLM-judged
# supervision at MATCHED budget. Both arms: 6,145 grouped rows (= all LLM-arm
# rows; CAS capped to match), identical hyperparameters = phase-1 recipe.
# Only --train-file differs. Avg negatives/row: CAS ~0.65 (capped2 in file),
# LLM ~2.63 — recorded in run_config.json, reported honestly.

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

train_and_eval() {  # <arm> <train-file> <outdir>
  local arm="$1" file="$2" out="$3"
  echo "=== [${SECONDS}s] ARM=${arm} training ==="
  python scripts/train_invarembed.py \
      --train-file "${file}" --max-rows 6145 \
      --model Qwen/Qwen3-Embedding-0.6B \
      --model-dtype bfloat16 --bf16 \
      --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
      --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
      --dev-frac 0.05 --eval-steps 100 \
      --query-prompt-name query \
      --output-dir "${out}"
  for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "$TIER" --model "${out}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_ctrl-${arm}.json"
  done
}

train_and_eval cas data/cas_pairs/pairs.jsonl   models/ctrl-cas-6145
train_and_eval llm data/llm_pairs/pairs.jsonl   models/ctrl-llm-6145

echo "=== CONTROLLED EXPERIMENT SUMMARY ==="
python - <<'PY'
import json
for arm in ["cas", "llm"]:
    row = []
    for tier in ["easy", "medium", "hard"]:
        o = json.load(open(f"results/eval_{tier}_ctrl-{arm}.json"))["overall"]
        row.append(f"{o['recall@1']:.2f}/{o['recall@5']:.2f}/{o['recall@10']:.2f}")
    print(f"  {arm:4s}  easy {row[0]:>20s}  medium {row[1]:>20s}  hard {row[2]:>20s}")
print("  ref: base 8.32/78.61/87.78 | 1.78/60.91/75.49 | 0.00/4.08/17.60")
PY
echo "== controlled experiment finished =="
