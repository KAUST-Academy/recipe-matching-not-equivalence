#!/bin/bash
#SBATCH --job-name=casprobe
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R14 (2026-09-05): a near-miss-bearing,
# generator-free equivalence probe. scripts/build_cas_probe.py builds
# computer-algebra-verified positives and counterexampled minimal-edit
# negatives (the verified arm's own machinery, no LLM anywhere) for benchmark
# query problems that no training file of the campaign contains (see the
# script's docstring for the exact held-out definition), then two BEIR-style
# evaluation directories (full public corpus + probe documents; probe
# documents only). scripts/casprobe_eval_driver.py scores every campaign model
# under its recorded encoding convention. Readouts R14-a/b/c were registered
# before this job ran; the noise band they use is restated in
# scripts/review3_verdict.py.
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="
if [[ -s data/cas_probe/probe.jsonl && -s data/casprobe_eval/queries.jsonl ]]; then
    echo "== probe exists; skipping build =="
else
    python scripts/build_cas_probe.py --target "${PROBE_TARGET:-2000}" --workers 10 --timeout 60
fi
echo "== [${SECONDS}s] evaluate every campaign model =="
python scripts/casprobe_eval_driver.py run
echo "== done: $(date) =="
