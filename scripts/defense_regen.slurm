#!/bin/bash
#SBATCH --job-name=defense-regen
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# DEFENSE EXPERIMENT: regenerate 1,700 easy-tier golds at
# three prompt distances (exact / paraphrase / style, the D1-D3 dose prompts),
# rebuild five eval variants (orig / exact / paraphrase / style / mixed) on
# the judge-verified shared slice, then score the EXISTING checkpoints on
# each variant. No retraining anywhere.
#
# PRE-REGISTERED predictions P-D1..P-D3 + falsifier: see the docstring of
# scripts/defense_regen_eval.py (written before this job first ran).
#
# Stage 1  generation + judging  (vllm env, Qwen/Qwen3-32B-AWQ -- the same
#          generator, judge and decoding config as the training arms)
# Stage 2  build eval dirs       (mathnet env, CPU)
# Stage 3  30 evals: 6 models x 5 eval dirs (mathnet env, GPU)
#
# Estimated: ~3h generation + ~10h evals on one A100-80.

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${PROJECT}"

# ---- stage 1: generation (skipped if the manifest already exists) ---------
if [[ ! -f data/defense_eval/regen_manifest.json ]] || \
   ! python -c "import json,sys; m=json.load(open('data/defense_eval/regen_manifest.json')); sys.exit(0 if m['config']['backend']=='vllm' else 1)" 2>/dev/null; then
  conda activate vllm
  python scripts/defense_regen_eval.py generate \
      --backend vllm --model Qwen/Qwen3-32B-AWQ \
      --n-queries 1700 --seed 42
  conda deactivate
else
  echo "[defense] vllm manifest already present -- skipping generation"
fi

# ---- stage 2: build the five eval dirs ------------------------------------
conda activate mathnet
python scripts/defense_regen_eval.py build

# ---- stage 3: score existing checkpoints on every variant -----------------
mkdir -p results/defense_eval
declare -A MODELS=(
  [base]="Qwen/Qwen3-Embedding-0.6B"
  [ctrl-cas]="models/ctrl-cas-6145/final"
  [d1-ctrl-llm]="models/ctrl-llm-6145/final"
  [d2-paraphrase]="models/dose-paraphrase-6145/final"
  [d3-style]="models/dose-style-6145/final"
  [d4-unrelated]="models/dose-unrelated-6145/final"
)
for NAME in orig exact paraphrase style mixed; do
  for TAG in base ctrl-cas d1-ctrl-llm d2-paraphrase d3-style d4-unrelated; do
    OUT="results/defense_eval/eval_${NAME}_${TAG}.json"
    if [[ -f "${OUT}" ]]; then echo "[skip] ${OUT}"; continue; fi
    python scripts/eval_retrieve.py \
        --data-dir "data/defense_eval/${NAME}" \
        --model "${MODELS[${TAG}]}" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --output "${OUT}"
  done
done

python scripts/defense_verdict.py
echo "[defense] complete"
