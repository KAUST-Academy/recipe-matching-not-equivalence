#!/bin/bash
#SBATCH --job-name=saber-base-fill
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# Fills the two MISSING base-0.6B rows of the SABER table (job 49590171 steps
# OOMed at the model's raw 32768 default; local checkpoints carry 1024 baked in).
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
cd "${PROJECT}"
python scripts/eval_saber.py --model Qwen/Qwen3-Embedding-0.6B --device cuda \
    --model-dtype bfloat16 --max-seq-length 1024 --emb-cache-dir .emb_cache \
    --output results/saber_base0.6b_noprompt.json
python scripts/eval_saber.py --model Qwen/Qwen3-Embedding-0.6B --device cuda \
    --model-dtype bfloat16 --max-seq-length 1024 --query-prompt-name query \
    --emb-cache-dir .emb_cache \
    --output results/saber_base0.6b_queryprompt.json
python - <<'PY'
import json
for tag in ["noprompt","queryprompt"]:
    d = json.load(open(f"results/saber_base0.6b_{tag}.json"))
    nd = d.get("overall",{}).get("ndcg@10") or d.get("ndcg@10")
    rho = d.get("score_correlations_spearman_mean",{}).get("with_summary_jaccard_signal")
    print(f"base0.6b {tag}: nDCG@10={nd}  rho_jacc={rho}  [attacked: 0.5620 / 0.3082; paper anchor 0.575]")
PY
echo "== P-S1 verdict now computable =="
