#!/bin/bash
#SBATCH --job-name=item16-cas-relaxed
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err
# ============================================================================
# Fresh-source CAS census, step 1 (2026-09-17): can the SymPy pipeline supply a like-for-like verified
# arm on the regenerated retrain's fresh sources? scripts/
# item16_census.py shows that 1,131 of the 7,585-problem fresh pool carry a
# census-relational span (289 of the 2,044 fresh sources used, 345 of the
# 2,602 verified), that the full CAS run attempted every one of them at the
# default budgets (15 s soft, 30 s hard, 8 spans) and verified nothing. This
# job reruns scripts/generate_cas_pairs.py over exactly those 1,131 at four
# times every budget, so the paper reports a measured yield instead of
# "cannot". CPU only. New output paths; data/cas_pairs/pairs.jsonl is never
# touched.
#
# PRE-REGISTERED READING (written before the rerun):
#   B1  rows recovered from the 289 used sources < 2,044 (the arithmetic
#       ceiling at the verified pool's 1.94 rows per source is about 560):
#       the like-for-like verified arm on the fresh sources stays
#       unbuildable; Appendix B states the measured
#       yield (sources recovered of 289 / 345 / 1,131, rows, rows / 2,044).
#   B2  rows from the 289 (or the 345) >= 2,044: stop and report; the
#       like-for-like verified arm on the fresh sources would then be buildable.
#   Readout: python scripts/item16_census.py -> results/item16_census.json
#   ("rerun" block: sources with a positive and rows inside each set).
#
# Submit:  sbatch scripts/item16_cas_relaxed.slurm
# The job is CPU-only and the scheduler account used here accepts GPU jobs only,
# so on 2026-09-17 the commands below were run on the login node instead, as the original 9,012-
# problem run was (generate_cas_pairs.slurm header); wall time is in
# results/cas_pairs_item16_stats.json meta.wall_s.
# PROJECT and CONDA_SH default to the original clone; export them to run
# from another checkout.
# ============================================================================
set -euo pipefail
PROJECT="${PROJECT:-/ibex/user/habiam0b/MathNet_Follow_Up}"
CONDA_SH="${CONDA_SH:-/ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate mathnet
export PROJECT
cd "${PROJECT}"
echo "== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK:-8} =="

python scripts/item16_census.py > /dev/null      # writes data/pairs/item16_relational_pool.txt
wc -l data/pairs/item16_relational_pool.txt
python scripts/generate_cas_pairs.py \
    --target 0 --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --source-ids-file data/pairs/item16_relational_pool.txt \
    --soft-budget 60 --timeout 120 --max-tried-spans 32 \
    --out  data/cas_pairs_item16/pairs.jsonl \
    --stats results/cas_pairs_item16_stats.json
python scripts/item16_census.py | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['rerun'], indent=1))"
echo "== fresh-source relaxed rerun finished: $(date) =="
