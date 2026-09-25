#!/bin/bash
#SBATCH --job-name=rejudge
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# E-R12 (2026-09-05): a SECOND, independent judge
# over the recipe arm's candidate pairs. The recipe arm's single judge is its
# own generator (Qwen3-32B-AWQ, Alibaba); the benchmark uses two independent
# judges. Here every one of the 25,656 candidate rows of
# data/pairs/llm_pairs.jsonl is re-judged by Mistral-Small-24B-Instruct-2501
# (Mistral AI, a different vendor and family, bf16, no quantisation) with the
# SAME judge prompt (generate_llm_pairs.py --rejudge-input). Output:
# data/pairs/llm_pairs_twojudge.jsonl with verification.verified = judge1 AND
# judge2, agreement statistics in results/llm_pairs_twojudge_summary.json.
# Then: trainer conversion (both-judge-verified rows only) and the training
# job (scripts/factorial_cells.slurm CELL=d1twojudge; registered readouts
# R12-a/b in its header, written before this job ran). Budget: the arm keeps
# --max-rows 6145 if that many rows survive both judges; if fewer survive, the
# cell trains on all survivors and the shortfall is disclosed (the matched
# budget then holds only against a verified-arm truncation of the same size,
# which factorial_cells.slurm reports as CELL=cas6145trunc if needed).
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
JUDGE2="${JUDGE2:-mistralai/Mistral-Small-24B-Instruct-2501}"
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="
conda activate vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -s data/pairs/llm_pairs_twojudge.jsonl ]]; then
    echo "== rejudge output exists; skipping =="
else
    echo "== [${SECONDS}s] smoke: first 60 rows =="
    head -60 data/pairs/llm_pairs.jsonl > data/pairs/rejudge_smoke_in.jsonl
    python scripts/generate_llm_pairs.py --backend vllm --seed 0 \
        --model "${JUDGE2}" --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --rejudge-input data/pairs/rejudge_smoke_in.jsonl \
        --output data/pairs/rejudge_smoke_out.jsonl
    python - <<'PY'
import json
s=json.load(open("data/pairs/rejudge_smoke_out.jsonl.summary.json"))
fail=s["n_second_judge_json_failures"]; n=s["n_rows"]
print("smoke: rows", n, "json failures", fail)
if fail > n*0.2: raise SystemExit("FATAL: second judge unparseable on >20% of smoke rows")
PY
    echo "== [${SECONDS}s] full rejudge of $(wc -l < data/pairs/llm_pairs.jsonl) rows =="
    python scripts/generate_llm_pairs.py --backend vllm --seed 0 \
        --model "${JUDGE2}" --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --rejudge-input data/pairs/llm_pairs.jsonl \
        --output data/pairs/llm_pairs_twojudge.jsonl \
        --summary results/llm_pairs_twojudge_summary.json
    python scripts/validate_pairs_schema.py data/pairs/llm_pairs_twojudge.jsonl
fi
conda activate mathnet
echo "== [${SECONDS}s] convert (rows verified by BOTH judges) =="
python scripts/convert_llm_pairs.py --input data/pairs/llm_pairs_twojudge.jsonl --output data/llm_pairs_twojudge/pairs.jsonl
N=$(wc -l < data/llm_pairs_twojudge/pairs.jsonl)
echo "== two-judge trainer rows: ${N} (recipe arm: 6145) =="
echo "== [${SECONDS}s] submit the training job =="
sbatch --export=ALL,CELL=d1twojudge scripts/factorial_cells.slurm
echo "== done: $(date) =="
