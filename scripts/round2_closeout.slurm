#!/bin/bash
#SBATCH --job-name=r2-closeout
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# CLOSEOUT: four cheap, decisive experiments. Each closes a point the paper
# otherwise only *states*.
#
# A. R2-C2, the decisive alternative. An "above the best model in MELD's own
#    published table (28.9)" comparison would cross harnesses: the one model
#    measured in both, Qwen3-4B, reads 18.89 in our harness against their
#    published 13.7. The like-for-like way to make it is to run MELD's best
#    published model through OUR harness and quote the in-harness comparison. Weights are
#    already cached. Both MathLeap models are run, in the matched prompt
#    condition, so the comparison is like-for-like.
#    PRE-REGISTERED: if meld9's 37.16 exceeds MathLeap-Octen-8B measured here,
#    the paper may state that in-harness comparison. If it does not, the paper
#    says so plainly and the MELD result stands on its base-relative gain alone.
#
# B. R2-M5. The H2 budget control matched D3's row COUNT but not its SELECTION:
#    D3's rows are the survivors of the ladder's worst generation-failure rate,
#    and its dropped sources skew longer and proof-only. This trains D2 on the
#    5,332 sources D3 also has (scripts/make_d2_on_d3_sources.py), so the two
#    arms are matched on sources as well as count and the only difference left
#    is the prompt.
#    PRE-REGISTERED: if hard R@1 stays near D2's own 6.76 +-1.01 three-seed
#    mean, selection is not driving the D2->D3 step. If it falls toward D3's
#    4.67, part of what the paper attributes to prompt distance is selection,
#    and mechanism.tex must say so.
#
# C. R2-M7. The H2 control is n=1, so it bounds the row-count effect at
#    [-1.30, +2.06] rather than measuring it. Two more seeds tighten that.
#
# D. R2-M11. The two SABER control rows were encoded at the model default
#    32,768 tokens while the base and attacked rows were capped at 1,024. The
#    direction is conservative, but it is a cross-condition comparison and the
#    paper currently footnotes it. Re-running at 1,024 removes the footnote.
#
# RUNTIME. A ~15 min (two 8B models, 1,081 texts) + B ~40 min + C ~1.4 h
# + D ~15 min => ~2.5 h against an 8 h cap.
#
# Submit:  sbatch scripts/round2_closeout.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

FAILLOG="results/r2_closeout_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"
soft () { echo "== [${SECONDS}s] $1 =="; shift; "$@" || echo "FAILED: $*" | tee -a "${FAILLOG}"; }

echo "== node: $(hostname) =="
echo "== start: $(date) =="

# ---------------------------------------------------------------------------
# A — MELD's published models through OUR harness (R2-C2)
# ---------------------------------------------------------------------------
for M in Octen Qwen; do
    OUTF="results/meld_mathleap_${M,,}_8b.json"
    [[ -s "${OUTF}" ]] && { echo "== [${SECONDS}s] A: ${M} exists, skipping =="; continue; }
    soft "A: MathLeap-${M}-8B on our MELD harness" \
    python scripts/eval_meld.py --model "uw-math-ai/MathLeap-${M}-8B" \
        --device cuda --model-dtype bfloat16 --max-seq-length 1024 --batch-size 16 \
        --query-prompt-name query \
        --output "${OUTF}" --per-query-out "results/meld_pq_mathleap_${M,,}_8b.json"
done

# ---------------------------------------------------------------------------
# B — D2 trained on D3's own 5,332 sources (R2-M5)
# ---------------------------------------------------------------------------
PF=data/llm_pairs_paraphrase_d3sources/pairs.jsonl
[[ -s "${PF}" ]] || python scripts/make_d2_on_d3_sources.py
NROWS=$(wc -l < "${PF}")
echo "== B: ${PF} has ${NROWS} rows (expect 5332) =="
(( NROWS == 5332 )) || echo "WARNING: unexpected row count ${NROWS}" | tee -a "${FAILLOG}"

OUT=models/dose-paraphrase-d3sources
if [[ -d "${OUT}/final" ]]; then
    echo "== [${SECONDS}s] B: ${OUT}/final exists, skipping training =="
else
    echo "== [${SECONDS}s] B: training D2 on D3's sources (no --max-rows cap) =="
    if ! python scripts/train_invarembed.py \
        --train-file "${PF}" \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${OUT}"
    then
        echo "FAILED: training D2-on-D3-sources" | tee -a "${FAILLOG}"
        [[ -e "${OUT}" ]] && mv "${OUT}" "${OUT}.FAILED-${SLURM_JOB_ID:-local}"
    fi
fi
if [[ -d "${OUT}/final" ]]; then
    for TIER in hard easy; do
        [[ -s "results/eval_${TIER}_dose-paraphrase-d3sources.json" ]] && continue
        soft "B: eval ${TIER}" \
        python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
            --dump-ranks "results/ranks/${TIER}_dose-paraphrase-d3sources.ranks.jsonl" \
            --output "results/eval_${TIER}_dose-paraphrase-d3sources.json"
    done
    [[ -s results/crosslingual_dose-paraphrase-d3sources.json ]] || \
    soft "B: eval real-duplicate" \
    python scripts/eval_crosslingual.py --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --output results/crosslingual_dose-paraphrase-d3sources.json
fi

# ---------------------------------------------------------------------------
# C — the H2 row-count control at seeds 43 and 44 (R2-M7)
# ---------------------------------------------------------------------------
for S in 43 44; do
    OUT="models/dose-paraphrase-5591-s${S}"
    if [[ -d "${OUT}/final" ]]; then
        echo "== [${SECONDS}s] C: ${OUT}/final exists, skipping =="
    else
        echo "== [${SECONDS}s] C: D2@5591 seed ${S} =="
        if ! python scripts/train_invarembed.py \
            --train-file data/llm_pairs_paraphrase/pairs.jsonl --max-rows 5591 \
            --model Qwen/Qwen3-Embedding-0.6B \
            --model-dtype bfloat16 --bf16 \
            --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
            --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${S}" \
            --dev-frac 0.05 --eval-steps 100 \
            --query-prompt-name query \
            --output-dir "${OUT}"
        then
            echo "FAILED: D2@5591 seed ${S}" | tee -a "${FAILLOG}"
            [[ -e "${OUT}" ]] && mv "${OUT}" "${OUT}.FAILED-${SLURM_JOB_ID:-local}"
            continue
        fi
    fi
    for TIER in hard easy; do
        [[ -s "results/eval_${TIER}_dose-paraphrase-5591-s${S}.json" ]] && continue
        soft "C: eval ${TIER} s${S}" \
        python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
            --output "results/eval_${TIER}_dose-paraphrase-5591-s${S}.json"
    done
done

# ---------------------------------------------------------------------------
# D — the two SABER control rows at the campaign-wide 1,024 cap (R2-M11)
# ---------------------------------------------------------------------------
for ARM in llm cas; do
    OUTF="results/saber_ctrl_${ARM}_cap1024.json"
    [[ -s "${OUTF}" ]] && continue
    soft "D: SABER ctrl-${ARM} at 1024 tokens" \
    python scripts/eval_saber.py --model "models/ctrl-${ARM}-6145/final" \
        --device cuda --batch-size 64 --setting statement-full \
        --emb-cache-dir .emb_cache --model-dtype bfloat16 \
        --query-prompt-name query --max-seq-length 1024 \
        --output "${OUTF}"
done

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo "=== [${SECONDS}s] ROUND-2 CLOSEOUT SUMMARY ==="
python scripts/round2_closeout_summary.py \
    || echo "FAILED: round2_closeout_summary.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== r2-closeout finished: $(date) =="
