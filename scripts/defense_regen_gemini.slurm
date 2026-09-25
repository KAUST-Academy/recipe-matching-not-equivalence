#!/bin/bash
#SBATCH --job-name=defense-gemini
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-17: the regeneration defence redone with the
# benchmark's own generator family. Generation ran through the Gemini API
# (gemini-3-flash-preview; scripts/defense_regen_eval.py generate --backend gemini,
# manifest data/defense_eval_gemini/regen_manifest.json, stub judge). Here:
# re-judge every candidate with the same local Qwen3-32B-AWQ judge the paper
# used, build the five eval sets, score the six checkpoints as in
# scripts/defense_regen.slurm, results under results/defense_eval_gemini/.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "${PROJECT}"
ROOTDIR=data/defense_eval_gemini
if ! python3 -c "import json,sys; m=json.load(open('${ROOTDIR}/regen_manifest.json')); sys.exit(0 if m['config'].get('judge_backend')=='vllm' else 1)" 2>/dev/null; then
  conda activate vllm
  python scripts/defense_regen_eval.py judge --backend vllm --model Qwen/Qwen3-32B-AWQ --out-root "${ROOTDIR}"
  conda deactivate
else
  echo "[defense-gemini] manifest already judged by vllm -- skipping"
fi
conda activate mathnet
python scripts/defense_regen_eval.py build --out-root "${ROOTDIR}"
mkdir -p results/defense_eval_gemini
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
    OUT="results/defense_eval_gemini/eval_${NAME}_${TAG}.json"
    if [[ -f "${OUT}" ]]; then echo "[skip] ${OUT}"; continue; fi
    echo "=== [$(date +%H:%M:%S)] ${NAME} / ${TAG} ==="
    python scripts/eval_retrieve.py --data-dir "${ROOTDIR}/${NAME}" --model "${MODELS[${TAG}]}" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 --output "${OUT}"
  done
done
python scripts/defense_gemini_readout.py || echo "[defense-gemini] readout failed"
echo "== end: $(date) =="
