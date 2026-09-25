#!/bin/bash
#SBATCH --job-name=e2-negmatch
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# E2 (acceptance-review): SOURCE- AND NEGATIVE-MATCHED CAS ARM, three seeds.
#
# WHY. The strongest remaining objection to the controlled experiment is that ctrl-LLM enjoys more training signal than
# ctrl-CAS at the same row budget: 6,145 vs ~4,621 distinct sources, 2.63 vs
# 1.52 negatives/row, 15,431 vs 9,921 contrastive examples, 177 vs 114
# optimizer steps. The paper currently answers by proxy (phase-1 pure-CAS,
# which beats ctrl-LLM on rows/sources/examples and still loses by 46
# points). This job answers it directly: a CAS arm matched to ctrl-LLM on
# every quantity axis, differing ONLY in what the supervision is.
#
# WHAT IS MATCHED (construction certificate:
# results/e2_negmatched_construction.json, built by
# scripts/build_e2_negmatched_cas.py, build seed 20260802, committed with
# this header before submission):
#   rows       6,145, one per source (ctrl-LLM's exact 6,145 source ids)
#   negatives  16,169 attachments total = 2.6312/row, ctrl-LLM's number,
#              via WITHIN-SOURCE reuse of counterexampled negatives only
#   steps      ~ctrl-LLM's (~16k triplets -> ~177 steps at batch 256)
#   dev split  identical to ctrl-LLM's per seed (same source set + same
#              trainer split rule; verified by simulation)
#   config     byte-identical trainer args to the headline arms (v1 gate,
#              cmnrl, batch 256/16, lr 2e-5, 3 epochs, no train-time cap)
#
# RESIDUALS, DISCLOSED (not fixable without changing what CAS negatives ARE):
#   * per-row histogram differs: E2 {0:1797, 3:1223, 4:3125} vs ctrl-LLM
#     {0:79, 1:356, 2:1317, 3:4393} -- 1,797 sources have no counterexampled
#     negative at all, and cross-source filler would not be a same-anchor
#     near-miss, so totals and means are matched instead of row shapes;
#   * duplicated attachments are oversampling (NO_DUPLICATES sampler places
#     copies in different batches);
#   * E2 feeds 1,797 rows to the trainer's "pairs" dataset vs ctrl-LLM's 79.
#
# ---------------------------------------------------------------------------
# PRE-REGISTRATION. Written and committed BEFORE this job was submitted and
# before any E2 number existed. Verdict evaluated IN CODE by
# scripts/e2_verdict.py.
#
# PRIMARY READOUT: three-seed mean easy-tier R@1 of the negmatched arm.
# Reference points: ctrl-LLM 62.38 +/- 0.85 (8 seeds), ctrl-CAS 17.05 +/-
# 0.79 (8 seeds), phase-1 pure-CAS 15.92.
#
#   E1 CONFIRMS  mean easy R@1 <= 25.0 AND gap to ctrl-LLM positive in every
#                seed. Matching rows, sources, negative volume and steps does
#                not close the gap; the training-signal objection is answered
#                directly and setup.tex upgrades its proxy control.
#
#   E2 PARTIAL   25.0 < mean < 40.0. A material share of the published
#                45.33-point gap is training signal; setup.tex MUST concede
#                the share explicitly and the abstract's framing softens.
#
#   E3 REFUTES   mean >= 40.0. The signal asymmetry is a first-order driver;
#                the controlled-experiment interpretation must be rebuilt.
#
# SECONDARY (reported, not gated): hard-tier R@1, real-duplicate strict R@1,
# realized rows/negatives/steps (assert rows_used==6145, all gate-drop
# counters 0, n_dev_source_ids==307 -- identical to ctrl-LLM).
#
# We commit to reporting the outcome whichever branch it lands in.
# ---------------------------------------------------------------------------
# COST: 3 trainings x ~45 min + 6 tier evals + 3 crosslingual evals ~= 3 GPU-h.
# ============================================================================

set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
# conda.sh directly, NOT ~/.bashrc (set -u trips /etc/bashrc; see E1 header).
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
cd "${PROJECT}"

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

FAILLOG="results/e2_negmatched_failures.txt"
: > "${FAILLOG}"
TRAIN="data/cas_pairs/pairs_negmatched.jsonl"
SEEDS="${SEEDS:-42 43 44}"

echo "== e2-negmatch starting: $(date) =="
[[ -f "${TRAIN}" ]] || { echo "FATAL: ${TRAIN} missing -- run scripts/build_e2_negmatched_cas.py" | tee -a "${FAILLOG}"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Stage 1 -- train (identical args to the headline arms; v1 gate by default)
# ---------------------------------------------------------------------------
for SEED in ${SEEDS}; do
    NAME="ctrl-cas-negmatched-6145-s${SEED}"
    OUT="models/${NAME}"
    if [[ -f "${OUT}/final/config.json" ]]; then
        echo "== [${SECONDS}s] SKIP ${NAME} (already trained) =="
        continue
    fi
    echo "== [${SECONDS}s] TRAIN ${NAME} =="
    if python scripts/train_invarembed.py \
        --train-file "${TRAIN}" --max-rows 6145 \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${OUT}"
    then
        # Comparability asserts: nothing gated out, dev split as registered.
        python - "$OUT" <<'PY' || echo "FAILED: run_config asserts ${OUT}" | tee -a "${FAILLOG}"
import json, sys
d = json.load(open(sys.argv[1] + "/run_config.json"))["data_stats"]
assert d["rows_used"] == 6145, d["rows_used"]
assert d["n_source_ids"] == 6145, d["n_source_ids"]
assert d["n_dev_source_ids"] == 307, d["n_dev_source_ids"]
print("run_config asserts pass:", {k: d[k] for k in ("rows_used","n_source_ids","n_train_triplets","n_train_pairs","n_dev_source_ids")})
PY
    else
        echo "FAILED: training ${NAME}" | tee -a "${FAILLOG}"
        if [[ -e "${OUT}" ]]; then mv "${OUT}" "${OUT}.failed.$$" || true; fi
        continue
    fi
done

# ---------------------------------------------------------------------------
# Stage 2 -- evaluate (easy + hard tiers, then real duplicates)
# ---------------------------------------------------------------------------
for SEED in ${SEEDS}; do
    NAME="ctrl-cas-negmatched-6145-s${SEED}"
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

# ---------------------------------------------------------------------------
# Stage 3 -- pre-registered verdict, evaluated in code
# ---------------------------------------------------------------------------
echo "== [${SECONDS}s] VERDICT =="
python scripts/e2_verdict.py || echo "FAILED: verdict" | tee -a "${FAILLOG}"

echo "== failures (empty = clean): =="
cat "${FAILLOG}"
echo "== finished: $(date) =="
