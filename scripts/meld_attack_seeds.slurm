#!/bin/bash
#SBATCH --job-name=meld-seeds
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# MELD ATTACK: TRAINING-SEED REPLICATION  (round-2 review finding R2-H7)
#
# WHY. Every MELD verdict in the paper -- P-M1 (+26.48), P-M3a (+17.78), P-M4
# (+19.44) and P-M2's refutation (real-duplicate 90.84, above the base) -- rests
# on ONE seed-42 training run per arm, with an interval that resamples MELD's
# 270 pairs and carries no training-seed variance. The paper's own
# appendix_mechanism.tex spends a paragraph establishing that exactly this kind
# of interval "is confidently wrong where the seeds disagree" (on easy D2-D3 it
# excluded zero in OPPOSITE directions at seeds 42 and 44), and then the draft
# applied it, without that caveat, to the result the abstract leads with.
#
# The largest effect (+26.48) is safe against any plausible seed sd. P-M2's
# refutation is NOT: 90.84 against a base of 84.73 is a 6.11-point margin, and
# the campaign's own real-duplicate run-to-run sd is 2.31 for a single run
# (results/ctrl_seed_stats.json, 8 ctrl-LLM seeds). P-M2 is the verdict that
# scopes the incentive-inversion claim, so it is the one that most needs seeds.
#
# WHAT. Retrain BOTH arms at seeds 43 and 44 -- 4 trainings -- from the pair
# files the seed-42 run already gated (data/meld_attack/pairs_{meld9,heldout}
# .jsonl). NOTHING is regenerated: no vllm, no new prompts, no new gates, so
# the disjointness certificates in results/meld_disjointness_*.json still
# describe exactly the data these models train on. Only --seed and
# --output-dir differ from the seed-42 command in
# scripts/attack_second_benchmark.slurm:318-328, which is itself byte-identical
# to the controlled experiment's hyperparameters.
#
# RUNTIME. The seed-42 meld9 training took 673 s (logged). 4 trainings ~45 min,
# + 4 MELD evals (~1 min each), + 4 real-duplicate evals (~1.5 min each),
# + 2 MathNet hard/easy evals ~10 min => well under the 4 h cap.
#
# PRE-REGISTERED READING, fixed BEFORE these runs (2026-07-31), so it cannot be
# chosen after the fact. sd_1 below = the single-run real-duplicate sd of 2.31.
#   R1  P-M1 REPLICATES if all three seeds of meld9 clear base 10.19 by >= +10.0.
#   R2  P-M2's REFUTATION HOLDS if the three-seed mean real-duplicate R@1 stays
#       ABOVE base 84.73. If the mean lands below 84.73, P-M2 is NOT refuted, the
#       seed-42 90.84 was a lucky run, and the incentive-inversion scoping in
#       ood.tex/limitations.tex must be withdrawn -- report it either way.
#   R3  P-M4's genre reading HOLDS if heldout's three-seed mean clears base by
#       >= +10.0. The meld9-minus-heldout domain increment (+7.04 at seed 42,
#       R@1 only) is REPORTED WITH ITS THREE-SEED SPREAD and is not claimed as
#       replicated unless all three seeds keep its sign.
# Anything else is reported as it lands, exactly as the dose ladder's two
# flagged orderings were (one replicated and grew, one did not replicate).
#
# Submit:  sbatch scripts/meld_attack_seeds.slurm
# Then:    python scripts/meld_seed_aggregate.py
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

ROWCAP=3150          # identical to attack_second_benchmark.slurm
FAILLOG="results/meld_seeds_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"

echo "== node: $(hostname) =="
echo "== start: $(date) =="

# --- pre-flight: the pair files must be the ones the seed-42 arms trained on
for GRP in meld9 heldout; do
    F="data/meld_attack/pairs_${GRP}.jsonl"
    [[ -s "${F}" ]] || { echo "FATAL: ${F} missing"; exit 1; }
    echo "== pre-flight: ${F} = $(wc -l < "${F}") rows =="
done
for GRP in meld9 heldout; do
    for S in 43 44; do
        D="models/meld-attack-${GRP}-s${S}"
        if [[ -e "${D}" && ! -e "${D}/final" ]]; then
            echo "FATAL: ${D} exists without final/ — inspect before resubmitting"
            exit 1
        fi
    done
done

soft () { echo "== [${SECONDS}s] $1 =="; shift; "$@" || echo "FAILED: $*" | tee -a "${FAILLOG}"; }

# --- STAGE 1: train + evaluate each (arm, seed)
for GRP in meld9 heldout; do
    for S in 43 44; do
        OUT="models/meld-attack-${GRP}-s${S}"
        TAG="attack_${GRP}_s${S}"

        if [[ -d "${OUT}/final" ]]; then
            echo "== [${SECONDS}s] ${TAG}: final exists — skipping training =="
        else
            echo "== [${SECONDS}s] ${TAG}: training (seed ${S}, ${ROWCAP} rows) =="
            if ! python scripts/train_invarembed.py \
                --train-file "data/meld_attack/pairs_${GRP}.jsonl" --max-rows "${ROWCAP}" \
                --model Qwen/Qwen3-Embedding-0.6B \
                --model-dtype bfloat16 --bf16 \
                --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
                --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${S}" \
                --dev-frac 0.05 --eval-steps 100 \
                --query-prompt-name query \
                --output-dir "${OUT}"
            then
                echo "FAILED: training ${TAG}" | tee -a "${FAILLOG}"
                [[ -e "${OUT}" ]] && mv "${OUT}" "${OUT}.FAILED-${SLURM_JOB_ID:-local}"
                continue
            fi
        fi

        # MELD target eval, matched prompt condition (threshold-bearing)
        if [[ ! -s "results/meld_atk_${TAG}.json" ]]; then
            soft "${TAG}: eval_meld" \
            python scripts/eval_meld.py --model "${OUT}/final" --device cuda \
                --model-dtype bfloat16 --max-seq-length 1024 --batch-size 64 \
                --query-prompt-name query \
                --output "results/meld_atk_${TAG}.json" \
                --per-query-out "results/meld_pq_${TAG}.json"
        fi

        # P-M2: real-duplicate ability
        if [[ ! -s "results/crosslingual_meld-attack-${GRP}-s${S}.json" ]]; then
            soft "${TAG}: eval_crosslingual (P-M2)" \
            python scripts/eval_crosslingual.py --model "${OUT}/final" \
                --device cuda --model-dtype bfloat16 --query-prompt-name query \
                --max-seq-length 1024 \
                --output "results/crosslingual_meld-attack-${GRP}-s${S}.json"
        fi
    done
done

# P-M3b: MathNet inheritance, meld9 only (matches the seed-42 design)
for S in 43 44; do
    OUT="models/meld-attack-meld9-s${S}"
    [[ -d "${OUT}/final" ]] || continue
    if [[ ! -s "results/eval_hard_meld-attack-meld9-s${S}.json" ]]; then
        soft "meld9 s${S}: eval_retrieve hard (P-M3b)" \
        python scripts/eval_retrieve.py --tier hard --model "${OUT}/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
            --output "results/eval_hard_meld-attack-meld9-s${S}.json"
    fi
done

# --- STAGE 2: paired cluster-bootstrap CIs against the shared base dump
for S in 43 44; do
    for GRP in meld9 heldout; do
        A="results/meld_pq_attack_${GRP}_s${S}.json"
        [[ -s "${A}" && -s "results/meld_pq_base.json" ]] || continue
        soft "compare attack_${GRP}_s${S} vs base" \
        python scripts/generate_meld_pairs.py --stage compare \
            --compare "${A}" "results/meld_pq_base.json" \
            --out "results/meld_compare_attack_${GRP}_s${S}_vs_base.json"
    done
    A="results/meld_pq_attack_meld9_s${S}.json"
    B="results/meld_pq_attack_heldout_s${S}.json"
    if [[ -s "${A}" && -s "${B}" ]]; then
        soft "compare meld9 vs heldout s${S}" \
        python scripts/generate_meld_pairs.py --stage compare \
            --compare "${A}" "${B}" \
            --out "results/meld_compare_attack_meld9_s${S}_vs_attack_heldout_s${S}.json"
    fi
done

# --- STAGE 3: across-seed aggregate + the pre-registered verdicts above
echo "=== [${SECONDS}s] MELD SEED SUMMARY ==="
python scripts/meld_seed_aggregate.py || echo "FAILED: meld_seed_aggregate.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== meld-seeds finished: $(date) =="
