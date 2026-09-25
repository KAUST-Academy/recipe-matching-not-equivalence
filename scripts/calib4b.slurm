#!/bin/bash
#SBATCH --job-name=calib4b
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R15 (2026-09-05): the Qwen3-Embedding-4B
# calibration gap (our best 11.96 easy R@1 against the 14.76 the benchmark
# publishes). The benchmark does not state its encoding convention, so this
# is a grid over the conventions an evaluator could plausibly have used:
# dtype {bfloat16, float16} x query instruction {none; the model's own
# "query" prompt; a math-retrieval instruction} x max sequence length
# {2048, 8192}, plus mean pooling at the best cell so far, all on the easy
# tier over the full corpus. Registered readout R15 (written before the grid ran):
# reported as the closest convention and its distance to 14.76; no cell is
# promoted into the paper's baseline tables unless it reproduces the published
# number within 0.5 R@1, and then Appendix A names it.
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results/calib4b
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="
MATHINSTR=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it\nQuery: '
run_cell() {  # tag dtype seq promptmode [extra...]
  local TAG=$1 DT=$2 SEQ=$3 PM=$4; shift 4
  local OUT="results/calib4b/easy_${TAG}.json"
  [[ -s "${OUT}" ]] && { echo "=== skip ${TAG}"; return; }
  local BS=32; (( SEQ > 4096 )) && BS=8
  local ARGS=(--tier easy --model Qwen/Qwen3-Embedding-4B --device cuda --model-dtype "${DT}" \
              --max-seq-length "${SEQ}" --batch-size "${BS}" --output "${OUT}")
  case "${PM}" in
    none) ;;
    query) ARGS+=(--query-prompt-name query) ;;
    math) ARGS+=(--query-prompt "${MATHINSTR}") ;;
  esac
  echo "=== [${SECONDS}s] ${TAG}: dtype=${DT} seq=${SEQ} prompt=${PM} $*"
  python scripts/eval_retrieve.py "${ARGS[@]}" "$@" || echo "=== FAILED ${TAG}"
}
for DT in bfloat16 float16; do
  for SEQ in 2048 8192; do
    for PM in none query math; do
      run_cell "${DT}-${SEQ}-${PM}" "${DT}" "${SEQ}" "${PM}"
    done
  done
done
run_cell "bfloat16-2048-query-meanpool" bfloat16 2048 query --pooling mean
run_cell "float32-2048-query" float32 2048 query
python - <<'PY'
import json,glob,os
rows=[]
for f in sorted(glob.glob("results/calib4b/easy_*.json")):
    d=json.load(open(f)); o=d.get("overall",{})
    rows.append((os.path.basename(f)[5:-5], o.get("recall@1"), o.get("recall@5"), o.get("recall@10")))
rows.sort(key=lambda r: -(r[1] or 0))
print("cell\tR@1\tR@5\tR@10  (published 4B row: 14.76 / -- / 36.87 hard R@10 not shown)")
for r in rows: print("\t".join(map(str,r)))
json.dump({"published_easy_r1": 14.76, "cells": [dict(zip(("cell","recall@1","recall@5","recall@10"),r)) for r in rows]},
          open("results/calib4b_summary.json","w"), indent=2)
PY
echo "== done: $(date) =="
