#!/bin/bash
#SBATCH --job-name=saber-label
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# THE SABER LABEL-STYLE ATTACK
#
# WHY. The paper's SABER negative is the only member of the taxonomy's
# "resists" cell, and it is qualified: our attack targeted SABER's document
# style, not its label style. Auditing the run for this job found the
# qualification is stronger than the paper admits.
# data/saber_attack/pairs.jsonl is ordered channel-by-channel -- 11,461
# `saber_pair` rows then 4,733 `saber_summary` rows -- so the published
# attack's `--max-rows 6145` selected 6,145 saber_pair rows and ZERO summary
# rows. The label channel was generated and never reached the model.
#
# >>> CORRECTION APPENDED 2026-07-31 (round-3 data audit, finding D-H1). The
# >>> paragraph immediately above is WRONG and is left in place unaltered
# >>> because this is a pre-registration header: it records what we believed
# >>> BEFORE this job ran, and rewriting it would destroy the audit trail.
# >>> train_invarembed.py shuffles with random.Random(--seed) BEFORE applying
# >>> --max-rows, so the cap is a seeded random subsample, NOT a file-order
# >>> prefix. The published run actually trained on 4,359 doc + 1,786 summary
# >>> rows (29.1% label), which its recorded data_stats confirm on every field.
# >>> NOTHING BELOW IS AFFECTED: this job's design -- two pure channels at a
# >>> matched 4,733-row budget -- is exactly the right experiment either way,
# >>> and its results stand. Only the motivating diagnosis changes, from
# >>> "the label channel never reached the model" to "it reached the model
# >>> diluted to 29%, and dilution was enough to erase the gain".
#
# The distinction matters because SABER's relevance rule is a Jaccard test over
# LLM-written CORE-IDEA SUMMARIES, not over the documents. An attacker aligning
# to the document template is imitating the wrong surface; an attacker aligning
# a problem statement to its own core-idea summary is imitating the labeler.
#
# WHAT THIS TRAINS. Two arms at a MATCHED 4,733-row budget -- the size of the
# summary channel -- so that channel and budget are separated, which the
# original 6,145-row run could not do:
#   saber-label-4733  summary channel: anchor = problem statement,
#                     positive = its own core-idea summary (SABER's verbatim
#                     extraction prompt), negatives = summaries of J=0 problems
#   saber-doc-4733    document channel at the same budget, i.e. a like-for-like
#                     re-run of the published attack's surface
# Both use the controlled-experiment hyperparameters with the 1,024-token cap
# the campaign settled on. Nothing is regenerated: both files come from the
# already-gated pairs.jsonl, so results/saber_disjointness.json still certifies
# exactly what these models see.
#
# PRE-REGISTERED READINGS, fixed before submission and before either arm
# existed. Reference points, all measured: base nDCG@10 0.5747 (matches SABER's
# published 0.575), published attack 0.5618, ctrl-LLM 0.5069, ctrl-CAS 0.4498.
#   L1 THE LABEL SURFACE IS EXPOSED. saber-label-4733 >= base + 0.02 nDCG@10,
#      the same bar the original P-S1 used. If this fires, the paper's taxonomy
#      CHANGES: SABER does not resist recipe-matching, it resisted the wrong
#      attack, and the "organic corpora resist" cell becomes empty. Abstract,
#      intro, discussion, conclusion and limitations all have to be rewritten,
#      and the paper's headline claim narrows to "benchmarks are exposed on
#      whichever surface their construction procedure defines".
#   L2 THE NEGATIVE HOLDS AND STRENGTHENS. Both arms stay below base. The
#      paper may then say SABER resisted attacks on BOTH its document and its
#      label surface at a matched budget, which is a materially stronger
#      negative than the one currently reported.
#   L3 CHANNEL MATTERS BUT DOES NOT CLEAR THE BAR. label > doc by >= 0.02 while
#      still below base: report as a partial exposure and a corrected
#      description of what the original attack tested.
# Report whichever lands. L1 is against our interest and is the reason to run.
#
# RUNTIME. 2 trainings ~35 min each at 4,733 rows + 2 SABER evals ~12 min each
# + 2 real-duplicate evals => ~2 h against an 8 h cap.
#
# Submit:  sbatch scripts/saber_label_attack.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

ROWCAP=4733
FAILLOG="results/saber_label_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"
soft () { echo "== [${SECONDS}s] $1 =="; shift; "$@" || echo "FAILED: $*" | tee -a "${FAILLOG}"; }

echo "== node: $(hostname) =="
echo "== start: $(date) =="

for F in pairs_summary_channel pairs_doc_channel; do
    P="data/saber_attack/${F}.jsonl"
    [[ -s "${P}" ]] || python scripts/make_saber_channels.py
    echo "== ${P}: $(wc -l < "${P}") rows =="
done

train_arm () {   # <tag> <file>
    local tag="$1" file="$2" out="models/saber-${1}-${ROWCAP}"
    if [[ -d "${out}/final" ]]; then
        echo "== [${SECONDS}s] ${tag}: final exists, skipping =="; return
    fi
    echo "== [${SECONDS}s] ${tag}: training at ${ROWCAP} rows =="
    if ! python scripts/train_invarembed.py \
        --train-file "${file}" --max-rows "${ROWCAP}" \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
        --dev-frac 0.05 --eval-steps 100 --max-seq-length 1024 \
        --query-prompt-name query \
        --output-dir "${out}"
    then
        echo "FAILED: training ${tag}" | tee -a "${FAILLOG}"
        [[ -e "${out}" ]] && mv "${out}" "${out}.FAILED-${SLURM_JOB_ID:-local}"
    fi
}

train_arm label data/saber_attack/pairs_summary_channel.jsonl
train_arm doc   data/saber_attack/pairs_doc_channel.jsonl

for TAG in label doc; do
    OUT="models/saber-${TAG}-${ROWCAP}"
    [[ -d "${OUT}/final" ]] || continue
    # SABER, in the promptless condition where our base row reproduces 0.575
    [[ -s "results/saber_${TAG}_${ROWCAP}_noprompt.json" ]] || \
    soft "${TAG}: eval_saber (promptless, the calibrated condition)" \
    python scripts/eval_saber.py --model "${OUT}/final" --device cuda \
        --batch-size 64 --setting statement-full --emb-cache-dir .emb_cache \
        --model-dtype bfloat16 --max-seq-length 1024 \
        --output "results/saber_${TAG}_${ROWCAP}_noprompt.json"
    # and in the trained prompt condition, for symmetry with the published attack
    [[ -s "results/saber_${TAG}_${ROWCAP}.json" ]] || \
    soft "${TAG}: eval_saber (query prompt)" \
    python scripts/eval_saber.py --model "${OUT}/final" --device cuda \
        --batch-size 64 --setting statement-full --emb-cache-dir .emb_cache \
        --model-dtype bfloat16 --max-seq-length 1024 --query-prompt-name query \
        --output "results/saber_${TAG}_${ROWCAP}.json"
    # P-S2: does it cost real equivalence ability, as the published attack did?
    [[ -s "results/crosslingual_saber-${TAG}-${ROWCAP}.json" ]] || \
    soft "${TAG}: eval_crosslingual" \
    python scripts/eval_crosslingual.py --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --output "results/crosslingual_saber-${TAG}-${ROWCAP}.json"
done

echo "=== [${SECONDS}s] LABEL-ATTACK VERDICT ==="
python scripts/saber_label_verdict.py || echo "FAILED: saber_label_verdict.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== saber-label finished: $(date) =="
