#!/bin/bash
#SBATCH --job-name=clean-gate
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# CLEAN-GATE RETRAIN (acceptance-review finding E1) — retrain BOTH controlled
# arms under the CORRECTED near-verbatim-twin contamination gate, at a matched
# budget, three seeds each.
#
# WHY. The obvious objection is that the ~33% item leakage both arms carry "could have been removed by a clean-gated
# matched retrain at 4,101 rows but was not." Disclosure does not fix an unrun
# experiment. This job runs it.
#
# The published campaign trained under anchor_to_corpus_mapping.json (v1), an
# EXACT-TEXT rule that caught 8,698 corpus ids / 8,761 of 15,000 anchors. It
# missed 6,160 near-verbatim twins differing only by a literal "Problem:" OCR
# header. anchor_to_corpus_mapping_v2.json rebuilds the gate and excludes
# 15,244 corpus ids. VERIFIED drop-in: same "exclude_corpus_ids" key.
#
# Applying v2 to the two pair files (measured, not assumed, 2026-08-01):
#     ctrl-LLM   6,145 rows -> 4,101 survive  (4,101 sources)
#     ctrl-CAS  14,361 rows -> 8,952 survive  (4,749 sources)
# so a MATCHED 4,101-row budget is constructible and is the largest one that
# is. No regeneration is needed: the pair files are FILTERED, not rebuilt.
#
# Supporting change (additive, default-preserving): train_invarembed.py gained
# --anchor-mapping. Omitting it loads v1, so every existing run reproduces
# byte-for-byte; the flag is recorded in run_config.json's argv either way.
#
# ---------------------------------------------------------------------------
# PRE-REGISTRATION. Written and committed BEFORE this job was submitted and
# before any v2-gated number existed. The verdict is computed IN CODE by
# scripts/clean_gate_verdict.py so the branch cannot be chosen after the fact.
#
# PRIMARY READOUT: the three-seed mean easy-tier arm gap (ctrl-LLM - ctrl-CAS,
# R@1) with both arms v2-gated at 4,101 rows.
#
# The paper's published clean-slice accounting attributes only 1.24 +/- 0.08
# of the 45.33-point gap to leakage, so it predicts a v2-gated gap near 44 --
# MINUS whatever the 33% budget cut costs, which is NOT zero and which this
# design deliberately does not control (both arms shrink together, so the
# comparison is matched to each other but NOT budget-matched to the published
# 45.33). The bands below are set with that slack built in.
#
#   G1 CONFIRMS  three-seed mean easy gap >= 35.0 AND positive in all 3 seeds.
#                Leakage is not a first-order driver. The 1.24-point
#                attribution stands qualitatively and this becomes a headline
#                robustness result: the gap survives a gate that removes every
#                near-verbatim twin from BOTH arms.
#
#   G2 PARTIAL   15.0 <= three-seed mean easy gap < 35.0, all seeds positive.
#                The effect is real but materially smaller than published.
#                The contamination section MUST revise its 1.24-point
#                attribution upward, and the abstract MUST carry the v2-gated
#                number alongside the headline.
#
#   G3 REFUTES   three-seed mean easy gap < 15.0, OR any seed non-positive.
#                Leakage is a first-order driver of the headline. The paper's
#                central claim must be rebuilt on the v2-gated numbers, and
#                the abstract/intro/gaming/ood sections all revert.
#
# SECONDARY (reported, not gated): hard-tier gap; and the easy-vs-real-data
# difference-in-differences recomputed under the clean gate, which is the
# paper's central statistic (published 35.59 [29.51, 41.47]).
#
# FALSIFIER FOR THE WHOLE PAPER: G3. We commit to reporting it either way.
# ---------------------------------------------------------------------------
#
# WHAT THIS TRAINS (6 trainings; identical to the controlled-experiment config
# of models/ctrl-llm-6145/run_config.json except --train-file, --output-dir,
# --seed, --max-rows 4101 and --anchor-mapping):
#   ctrl-llm-v2gate-4101-s42, -s43, -s44
#   ctrl-cas-v2gate-4101-s42, -s43, -s44
#
# COST: 6 x ~40 min training + 12 tier evals x ~350 s + 6 crosslingual evals
#       ~= 6 GPU-hours. Wall limit 16 h for headroom.
# ============================================================================

set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
# NB: source conda.sh directly, NOT ~/.bashrc -- under `set -u` the login
# rc chain trips on /etc/bashrc's unbound BASHRCSOURCED and kills the job
# before stage 1 (job 49722213 died exactly this way).
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
cd "${PROJECT}"

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

FAILLOG="results/clean_gate_failures.txt"
: > "${FAILLOG}"
MAPPING="anchor_to_corpus_mapping_v2.json"
ROWS=4101
# Seed list is overridable so the run can be extended to the headline's eight
# seeds without retraining what already exists (stage 1 skips finished models).
SEEDS="${SEEDS:-42 43 44}"

echo "== clean-gate retrain starting: $(date) =="
echo "== gate: ${MAPPING} | matched budget: ${ROWS} rows/arm | seeds: ${SEEDS} =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Stage 1 — train
# ---------------------------------------------------------------------------
for ARM in llm cas; do
    if [[ "${ARM}" == "llm" ]]; then TRAIN="data/llm_pairs/pairs.jsonl"
    else                             TRAIN="data/cas_pairs/pairs.jsonl"; fi

    for SEED in ${SEEDS}; do
        NAME="ctrl-${ARM}-v2gate-${ROWS}-s${SEED}"
        OUT="models/${NAME}"

        if [[ -f "${OUT}/final/config.json" || -f "${OUT}/final/adapter_config.json" ]]; then
            echo "== [${SECONDS}s] SKIP ${NAME} (already trained) =="
            continue
        fi

        echo "== [${SECONDS}s] TRAIN ${NAME} =="
        if python scripts/train_invarembed.py \
            --train-file "${TRAIN}" --max-rows "${ROWS}" \
            --anchor-mapping "${MAPPING}" \
            --model Qwen/Qwen3-Embedding-0.6B \
            --model-dtype bfloat16 --bf16 \
            --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
            --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
            --dev-frac 0.05 --eval-steps 100 \
            --query-prompt-name query \
            --output-dir "${OUT}"
        then
            :
        else
            echo "FAILED: training ${NAME}" | tee -a "${FAILLOG}"
            if [[ -e "${OUT}" ]]; then mv "${OUT}" "${OUT}.failed.$$" || true; fi
            continue
        fi
    done
done

# ---------------------------------------------------------------------------
# Stage 2 — evaluate (easy + hard tiers, then real duplicates)
# ---------------------------------------------------------------------------
for ARM in llm cas; do
    for SEED in ${SEEDS}; do
        NAME="ctrl-${ARM}-v2gate-${ROWS}-s${SEED}"
        OUT="models/${NAME}"
        [[ -f "${OUT}/final/config.json" ]] || { echo "SKIP eval ${NAME} (no model)" | tee -a "${FAILLOG}"; continue; }

        for TIER in easy hard; do
            EVJ="results/eval_${TIER}_${NAME}.json"
            if [[ -f "${EVJ}" ]]; then echo "== SKIP eval ${TIER} ${NAME} =="; continue; fi
            echo "== [${SECONDS}s] EVAL ${TIER} ${NAME} =="
            python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
                --device cuda --model-dtype bfloat16 --query-prompt-name query \
                --batch-size 64 --max-seq-length 1024 \
                --emb-cache-dir .emb_cache \
                --output "${EVJ}" \
                || echo "FAILED: eval ${TIER} ${NAME}" | tee -a "${FAILLOG}"
        done

        XJ="results/crosslingual_${NAME}.json"
        if [[ ! -f "${XJ}" ]]; then
            echo "== [${SECONDS}s] EVAL crosslingual ${NAME} =="
            python scripts/eval_crosslingual.py --model "${OUT}/final" \
                --device cuda --model-dtype bfloat16 --query-prompt-name query \
                --max-seq-length 1024 \
                --output "${XJ}" \
                || echo "FAILED: crosslingual ${NAME}" | tee -a "${FAILLOG}"
        fi
    done
done

# ---------------------------------------------------------------------------
# Stage 3 — pre-registered verdict, evaluated in code
# ---------------------------------------------------------------------------
echo "== [${SECONDS}s] VERDICT =="
python scripts/clean_gate_verdict.py || echo "FAILED: verdict" | tee -a "${FAILLOG}"

echo "== failures (empty = clean): =="
cat "${FAILLOG}"
echo "== finished: $(date) =="
