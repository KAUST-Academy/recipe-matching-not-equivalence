#!/bin/bash
#SBATCH --job-name=saber-lab-seeds
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# ============================================================================
# SEED REPLICATION FOR THE SABER LABEL-CHANNEL ATTACK
#
# WHY. The label-channel result (+0.0354 [0.0299, 0.0408] over base) now carries
# a rewritten taxonomy: the paper no longer says any benchmark resists. That
# claim currently rests on ONE training run per channel with an
# evaluation-sampling interval -- the exact inference this paper's own appendix
# calls "confidently wrong where the seeds disagree". The dose ladder taught us
# that lesson at cost: easy D2-D3 looked decisive at seed 42 and reversed sign
# across seeds.
#
# Trains BOTH channels at seeds 43 and 44 from the same gated pair files, so
# results/saber_disjointness.json still certifies exactly what these models see.
# Nothing is regenerated: only --seed and --output-dir differ.
#
# PRE-REGISTERED, fixed before submission and before any seed-43/44 output
# exists. Reference points measured: base 0.5747, label seed-42 0.6101,
# doc seed-42 0.5515.
#   S1 REPLICATES if all three label seeds exceed the base, AND the three-seed
#      label mean exceeds the three-seed doc mean. The taxonomy rewrite stands.
#   S2 DOES NOT REPLICATE if any label seed falls at or below the base, or if
#      the label/doc ordering flips in any seed. Then the SABER label result is
#      a single-run finding, the taxonomy rewrite must be withdrawn back to a
#      scoping correction of our original negative, and ood.tex, the abstract,
#      intro, conclusion and discussion all revert.
# S2 would undo work committed today. That is precisely why it is worth running.
# ============================================================================
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
ROWCAP=4733
FAILLOG="results/saber_label_seeds_failures_${SLURM_JOB_ID:-local}.log"; : > "${FAILLOG}"

for CH in label doc; do
  case "${CH}" in
    label) FILE=data/saber_attack/pairs_summary_channel.jsonl ;;
    doc)   FILE=data/saber_attack/pairs_doc_channel.jsonl ;;
  esac
  for S in 43 44; do
    OUT="models/saber-${CH}-${ROWCAP}-s${S}"
    if [[ ! -d "${OUT}/final" ]]; then
      echo "== [${SECONDS}s] ${CH} seed ${S} =="
      python scripts/train_invarembed.py \
        --train-file "${FILE}" --max-rows "${ROWCAP}" \
        --model Qwen/Qwen3-Embedding-0.6B --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${S}" \
        --dev-frac 0.05 --eval-steps 100 --max-seq-length 1024 \
        --query-prompt-name query --output-dir "${OUT}" \
        || { echo "FAILED train ${CH} s${S}" | tee -a "${FAILLOG}"; continue; }
    fi
    [[ -s "results/saber_${CH}_${ROWCAP}_s${S}_noprompt.json" ]] || \
    python scripts/eval_saber.py --model "${OUT}/final" --device cuda \
        --batch-size 64 --setting statement-full --emb-cache-dir .emb_cache \
        --model-dtype bfloat16 --max-seq-length 1024 \
        --per-query-out "results/saber_pq_${CH}${ROWCAP}_s${S}.json" \
        --output "results/saber_${CH}_${ROWCAP}_s${S}_noprompt.json" \
        || echo "FAILED eval ${CH} s${S}" | tee -a "${FAILLOG}"
  done
done

python scripts/saber_label_seed_verdict.py || echo "FAILED verdict" | tee -a "${FAILLOG}"
echo "== failures (empty=clean): =="; cat "${FAILLOG}"
echo "== finished: $(date) =="
