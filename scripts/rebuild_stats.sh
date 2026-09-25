#!/bin/bash
# ONE-COMMAND STATISTICS REBUILD for the paper's statistical claims. Run from a login shell AFTER the seed jobs land
# (49593312-16 = train_ctrl_param.slurm SEED 45..49); safe to run any time --
# it uses whatever seeds have complete results.
#
#   bash scripts/rebuild_stats.sh              # full rebuild; if per-query
#                                              # rank dumps are missing it
#                                              # submits scripts/dump_ranks.slurm
#                                              # and BLOCKS until it finishes
#                                              # (sbatch --wait, <=10 h)
#   NO_SUBMIT=1 bash scripts/rebuild_stats.sh  # never submits; falls back to
#                                              # seed-summary statistics
#
# Steps:
#   1. aggregate_ctrl_stats.py over every complete seed -> results/ctrl_seed_stats.json
#   2. ensure per-query rank dumps under results/ranks/ (dump_ranks.slurm)
#   3. bootstrap_stats.py -> results/final_stats.json + paper-ready table
#      (hierarchical cluster bootstrap, paired per-query bootstrap, TOST,
#      difference-in-differences -- see its docstring)
#
# After a successful full run: add a reproducibility.json entry citing the
# dump-ranks job id and results/final_stats.json before quoting any number
# in the paper (numbers are read from result files, never typed).
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
cd "${PROJECT}"

SEEDS=$(python scripts/bootstrap_stats.py --print-seeds)
if [[ -z "${SEEDS}" ]]; then
  echo "FATAL: no complete ctrl seed results found" >&2
  exit 1
fi
echo "== [1/3] ctrl seeds with complete summaries: ${SEEDS}"
python scripts/aggregate_ctrl_stats.py --seeds "${SEEDS}"

echo "== [2/3] per-query rank dumps"
rc=0
python scripts/bootstrap_stats.py --check-dumps || rc=$?
if (( rc != 0 )); then
  if [[ "${NO_SUBMIT:-0}" == "1" ]]; then
    echo "== dumps missing and NO_SUBMIT=1 -> continuing at seed-summary level"
  else
    echo "== submitting scripts/dump_ranks.slurm and waiting (this blocks; <=10 h)"
    sbatch --wait scripts/dump_ranks.slurm
    python scripts/bootstrap_stats.py --check-dumps || true
  fi
fi

echo "== [3/3] bootstrap statistics -> results/final_stats.json"
python scripts/bootstrap_stats.py --seeds "${SEEDS}" \
  --output results/final_stats.json
echo "== rebuild_stats finished =="
