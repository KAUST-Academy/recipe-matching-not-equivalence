#!/bin/bash
#SBATCH --job-name=bm25-all
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-16: BM25 lexical baseline on every evaluation set.
# Job 51976682 stopped at the same-language step (eval_bm25.py then had no fallback
# to the shared cross-lingual corpus); the steps from there on were run by hand with
# exactly the commands below, which reproduce the committed files.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
cd "${PROJECT}"; mkdir -p results/ranks
for tier in easy medium hard; do
  echo "=== [$(date +%H:%M:%S)] tier ${tier} ==="
  python scripts/eval_bm25.py --data-dir data/retrieve/${tier} --output results/bm25_${tier}.json
done
echo "=== [$(date +%H:%M:%S)] cross-lingual ==="
python scripts/eval_bm25.py --data-dir data/crosslingual_eval --dup --output results/bm25_crosslingual.json --dump-ranks results/ranks/xling_bm25.ranks.jsonl
echo "=== [$(date +%H:%M:%S)] same-language ==="
python scripts/eval_bm25.py --data-dir data/samelang_eval --dup --output results/bm25_samelang.json --dump-ranks results/ranks/samelang_bm25.ranks.jsonl
echo "=== [$(date +%H:%M:%S)] near-miss probe ==="
python scripts/eval_bm25.py --data-dir data/casprobe_eval --output results/bm25_casprobe.json
echo "=== [$(date +%H:%M:%S)] near-miss probe, near-misses only ==="
python scripts/eval_bm25.py --data-dir data/casprobe_pairs_eval --output results/bm25_casprobe_pairs.json
echo "=== [$(date +%H:%M:%S)] same-language slices ==="
python scripts/bm25_samelang_readout.py
echo "== end: $(date) =="
