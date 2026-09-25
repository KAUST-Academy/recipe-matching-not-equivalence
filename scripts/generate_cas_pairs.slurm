#!/bin/bash
#SBATCH --job-name=mathnet-cas-pairs
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# CAS-verified contrastive pair generation over ALL eligible MathNet problems
# (InvarEmbed project, Task A). CPU-only: no --gres on purpose.
#
# Eligible = >=1 census-relational math span (results/cas_expr_results.json)
# AND id not in anchor_to_corpus_mapping.json exclude_corpus_ids (8,698 ids
# matched to MathNet-Retrieve eval anchors are NEVER used for training data).
#
# Reference numbers from the 8-worker login-node run of 2026-07-29
# (results/cas_pairs_stats.json): 9,012 eligible problems -> 7,397 with >=1
# verified positive, 14,361 positive records, 9,408 verified negatives,
# 510 s wall at 8 workers. At 30 workers this job should finish in well
# under 1 h; the 4 h limit is generous headroom.
#
# Output: data/cas_pairs/pairs.jsonl (written atomically via .tmp + rename)
#         results/cas_pairs_stats.json
#
# Submit with:  sbatch scripts/generate_cas_pairs.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

cd "${PROJECT}"
mkdir -p results data/cas_pairs

echo "== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK} =="

# sanity gate: the normalizer must still beat the census baseline before we
# regenerate the training pairs (fast: ~30 s on 8 workers)
python scripts/normalize_latex.py --workers 8 --timeout 2.0

python scripts/generate_cas_pairs.py \
    --target 0 \
    --workers 30 \
    --timeout 30 \
    --out  data/cas_pairs/pairs.jsonl \
    --stats results/cas_pairs_stats.json

echo "== pair generation finished =="
