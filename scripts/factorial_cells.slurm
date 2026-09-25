#!/bin/bash
#SBATCH --job-name=fact-cells
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# E-R3 -- the two missing cells of the
# positives x negatives factorial, 3 seeds each:
#
#   CELL=d4casnegs  D4 recipe-free positives + CAS counterexampled negatives
#                   (data/llm_pairs_unrelated_casnegs/pairs.jsonl, 6,768 rows,
#                   1.28 negs/row -- CAS's per-row distribution)
#   CELL=casnonegs  ctrl-CAS positives with negatives stripped
#                   (data/cas_pairs_nonegs/pairs.jsonl, 14,361 rows)
#
# Existing cells: ctrl-CAS (CAS-pos+CAS-negs, 8 seeds) and dose-unrelated
# (D4-pos+no-negs, 3 seeds). Hyperparameters are the controlled-experiment
# recipe verbatim; --max-rows 6145 shuffle-truncates with --seed, so each
# seed's retained rows are IDENTICAL to the corresponding ctrl-CAS / D4 run
# (same file order, same shuffle) -- the cells pair row-for-row.
#
# Registered readout (no directional prediction; this cell exists to
# deconfound positive style from the negative axis): easy R@1 of d4casnegs vs ctrl-CAS isolates positive
# style at identical negatives; vs dose-unrelated isolates the negative axis.
# Submit:  sbatch --export=ALL,CELL=d4casnegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=casnonegs scripts/factorial_cells.slurm
#
# ----------------------------------------------------------------------------
# E-R6 (2026-09-03; written BEFORE any result existed).
# Two further cells, built by scripts/build_llmnegs_cells.py, plus five more
# seeds of d4casnegs, plus the same-language duplicate eval (E-R5) for every
# seed of every cell:
#
#   CELL=d4llmnegs     D4 positives + the ctrl-LLM (D1, Appendix-F) row's own
#                      LLM-written near-miss negatives, COUNT-MATCHED per
#                      source to d4casnegs (first k of D1's negatives, k = the
#                      d4casnegs count; 5,978 rows, the 790 D4 sources with no
#                      D1 row dropped).
#   CELL=d4llmnegsfull The same rows with ALL of D1's negatives (<= 3/row, the
#                      recipe arm's own negative volume).
#   SEEDS="45 46 47 48 49" CELL=d4casnegs   -> eight seeds for the reference
#                      cell, so the reference-control difference-in-differences
#                      (scripts/reference_control_did.py) can be quoted at the
#                      headline's eight-seed standard.
#
# Registered readouts (scripts/factorial_verdict.py extends the 2x2 to a
# 2x3 over negatives {CAS counterexamples, LLM-written, none}; seeds 42-44
# unless stated):
#   R6-a  d4llmnegs vs d4casnegs (identical positives, identical per-row
#         counts up to D1's cap of 3): isolates the PROVENANCE of the
#         negatives. No directional prediction is registered. Reading map,
#         fixed now: within 2*sigma_1 = 1.72 hard-tier points of 10.97 ->
#         "provenance does not matter, the pair structure does"; more than
#         1.72 ABOVE -> the recipe's LLM-written negatives carry a specific
#         increment the paper must report as recipe-specific; more than 1.72
#         BELOW -> verified negatives are the better half and the hard tier
#         is reached without any LLM-written negative.
#   R6-b  d4llmnegsfull vs D1 (= ctrl-LLM; 9.90 three-seed / 9.42 eight-seed
#         hard R@1): isolates the positive PROMPT at the recipe arm's own
#         negatives. Claim under test: the paper's "does not require the
#         template" reading. It SURVIVES if d4llmnegsfull's three-seed hard
#         R@1 is within 2*sigma_1 (1.72) of D1's 9.90 or above it; it is
#         NARROWED if d4llmnegsfull sits more than 1.72 below 9.90 (the
#         template's positives then add something at the recipe's negatives,
#         and the abstract's wording must say so).
#   R6-c  Real duplicates (strict cross-lingual R@1) and the same-language
#         set for both new cells: reported, no direction registered.
#   sigma_1 = 0.86 hard-tier points is the run-to-run sd of a difference of
#   two trainings from the eight-seed ctrl-LLM arm (Appendix C convention).
# Submit:  sbatch --export=ALL,CELL=d4llmnegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d4llmnegsfull scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d4casnegs,SEEDS="45 46 47 48 49" scripts/factorial_cells.slurm
#
# ----------------------------------------------------------------------------
# E-R7 / E-R8 / E-R9 (2026-09-04; written BEFORE any of
# these cells was trained). Built by scripts/build_review2_cells.py; three
# seeds each (42-44); same hyperparameters, same evals (three tiers,
# cross-lingual, same-language).
#
#   CELL=d4unrelnegs     D4 positives + near-misses written under a prompt
#                        UNRELATED to Appendix F (generate_llm_pairs.py variant
#                        unrelated_negs: the D4 audience persona plus a "spot
#                        the difference" drill; judge pass identical),
#                        count-matched per source to the reference exactly as
#                        d4llmnegs is (first k, k = d4casnegs count, cap 3).
#   CELL=d4unrelnegsfull The same rows with all of that prompt's negatives.
#   CELL=casllmnegs      CAS positives (one seeded CAS row per D1 source) +
#                        the D1 row's own Appendix-F near-misses, all of them.
#   CELL=btcasnegs       Back-translated (NMT, non-LLM) positives of the same
#                        source problems + the reference's CAS negatives.
#
# Registered readouts (sigma_1 = 0.86 hard, 1.20 easy, 3.27 cross-lingual
# points, the Appendix C convention; bands are 2*sigma_1 = 1.72 / 2.40 / 6.54):
#   R7-a  d4unrelnegs vs d4llmnegs (identical positives, count-matched
#         LLM-written negatives; only the PROMPT that wrote them differs).
#         Reading map, fixed now: within 1.72 hard-tier points of d4llmnegs
#         (20.71 three-seed) -> the near-miss GENRE, not the recipe's prompt,
#         is the potent half, and the paper must retire "recipe-specific" for
#         the negatives in favour of "LLM-written near-miss"; more than 1.72
#         BELOW -> the recipe's prompt carries a specific increment, reported
#         as such with its size; more than 1.72 ABOVE -> reported, and the
#         "recipe-specific" wording is retired all the same.
#   R7-b  d4unrelnegsfull vs d4llmnegsfull (26.97), same map at full volume.
#   R8-a  casllmnegs vs ctrl-CAS (near-copy positives held fixed; SymPy
#         counterexamples -> LLM near-misses). Hard tier: more than 1.72 above
#         ctrl-CAS's 0.16 -> the recipe's near-misses lift even near-copy
#         positives off the floor (the negative half suffices for the tier);
#         within 1.72 -> the tier needs the deep-rewrite positive half too.
#   R8-b  casllmnegs vs ctrl-CAS on cross-lingual strict R@1 (tests whether
#         the CAS negatives drive the retention collapse). More than 6.54
#         above ctrl-CAS's 56.52 -> the counterexampled negatives are the
#         driver; within 6.54 -> they are not (the near-copy positives or the
#         narrow fine-tune are).
#   R9    btcasnegs vs the reference (46.91 easy, eight seeds). Within 2.40
#         easy-tier points -> LLM authorship of the positives does not matter
#         for the genre share (paraphrase depth does), and the paper's "genre"
#         reading must say so; more than 2.40 BELOW -> LLM authorship carries
#         a share of the reference's gain, reported as reference minus BT;
#         cross-lingual and same-language reported, no direction registered.
#         The BT control is disclosed as what it is: an NMT round trip with
#         formulas masked, shallower than a free restatement.
# Submit:  sbatch --export=ALL,CELL=casllmnegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d4unrelnegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d4unrelnegsfull scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=btcasnegs scripts/factorial_cells.slurm
#
# ----------------------------------------------------------------------------
# E-R10 / E-R11 / E-R12 / E-R13 (2026-09-05; written BEFORE any of these
# cells was trained). Three seeds
# each (42-44); same hyperparameters; same evals plus the E-R14 generator-free
# near-miss probe (data/casprobe_eval, data/casprobe_pairs_eval) when it exists.
#
#   CELL=d5nonegs    D5 positives alone (prompt variant survey: a survey
#                    author's restatement, a recipe-free task framing that is
#                    NOT D4's audience change; scripts/gen_survey.slurm), no
#                    negatives, D4's shape.
#   CELL=d5casnegs   D5 positives + the reference's verified negatives per
#                    source (build_review2_cells.py --which r10): the
#                    reference cell replicated with a second recipe-free
#                    prompt.
#   CELL=d1split     The recipe arm's prompt with positives and near-misses
#                    written in SEPARATE calls (variant exact_split;
#                    scripts/gen_split.slurm), all verified rows, --max-rows
#                    6145 shuffle-truncated as for every arm.
#   CELL=d1twojudge  The recipe arm's rows verified by BOTH its own judge and
#                    a second, independent judge (Mistral-Small-24B-Instruct-
#                    2501; scripts/rejudge.slurm), --max-rows 6145 (or all
#                    survivors if fewer, disclosed).
#   CELL=bt2casnegs  Multi-hop back-translated positives (French, German,
#                    Russian in sequence; scripts/bt2.slurm) + the reference's
#                    verified negatives.
#
# Registered readouts (sigma_1 = 0.86 hard, 1.20 easy, 3.27 cross-lingual
# points; bands 2*sigma_1 = 1.72 / 2.40 / 6.54; the reference = 46.91 easy,
# 10.30 hard, 78.02 cross-lingual over eight seeds; the recipe arm = 62.38 /
# 9.42 / 66.26 over eight seeds; D4 alone = 30.03 / 0.04 / 93.81):
#   R10-a  recipe arm minus d5casnegs on the easy tier, against the paper's
#          recipe-specific share 15.47 (recipe minus reference). Within 2.40
#          -> the recipe-specific share does not depend on which recipe-free
#          prompt wrote the reference's positives; more than 2.40 SMALLER ->
#          part of the 15.47 was D4's prompt, reported as a range over the two
#          references and the paper's share narrows to the smaller value;
#          more than 2.40 LARGER -> reported, the share is a range.
#   R10-b  d5casnegs vs the reference on cross-lingual strict R@1 (78.02) and
#          d5nonegs vs D4 alone (93.81): within 6.54 -> D4 is not unusually
#          retention-friendly; more than 6.54 BELOW -> D4's audience framing
#          was retention-friendly and the difference-in-differences against
#          the reference (27.24) is reported as a range over both references.
#   R10-c  d5casnegs vs the reference on hard R@1 (10.30): within 1.72 ->
#          "recipe-free rewrites with verified negatives reach the tier"
#          replicates; more than 1.72 below -> reported, D4-specific.
#   R11-a  d1split vs the recipe arm on hard R@1 (9.42): within 1.72 ->
#          writing positives and near-misses in one call is not what
#          depresses the recipe arm's hard tier; more than 1.72 ABOVE -> the
#          shared-phrasing account holds (the one-call pair is less
#          discriminative) and Appendix C says so with the size; more than
#          1.72 BELOW -> reported.
#   R11-b  d1split vs d4llmnegsfull (26.97): reported; if d1split comes
#          within 1.72 of it, the positives' prompt does not matter at the
#          recipe's own negatives once the calls are separated.
#   R11-c  easy tier, cross-lingual and same-language for d1split: reported,
#          no direction.
#   R12-a  Agreement between the two judges on positives and on near-misses
#          (rate and Cohen's kappa), reported.
#   R12-b  d1twojudge vs the recipe arm: within 2.40 easy / 1.72 hard / 6.54
#          cross-lingual -> the recipe arm's numbers do not depend on its
#          single self-judge; outside any band -> reported with the size, and
#          the paper's recipe arm is disclosed as single-judge in Section 3.
#   R13    Depth of the multi-hop control vs the source (char-3 Jaccard;
#          single hop 0.783, D4 0.421) reported. bt2casnegs vs the reference
#          (46.91 easy): within 2.40 -> deeper non-LLM paraphrase recovers
#          the reference's gain and the LLM-authorship attribution FALLS (the
#          abstract's "LLM authorship rather than paraphrase depth" is
#          retired); more than 2.40 below -> the attribution survives to the
#          deeper depth, reported as reference minus BT2 at that depth.
# Submit:  sbatch --export=ALL,CELL=d5nonegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d5casnegs scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d1split scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=d1twojudge scripts/factorial_cells.slurm
#          sbatch --export=ALL,CELL=bt2casnegs scripts/factorial_cells.slurm
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

CELL="${CELL:?set CELL=d4casnegs|casnonegs|d4llmnegs|d4llmnegsfull|d4unrelnegs|d4unrelnegsfull|casllmnegs|btcasnegs|d5nonegs|d5casnegs|d1split|d1twojudge|bt2casnegs|casunrelnegs|btnonegs|btunrelnegs|btllmnegs|d1nonegs|d1casnegs|d1unrelnegs}"
case "${CELL}" in
  d4casnegs) FILE=data/llm_pairs_unrelated_casnegs/pairs.jsonl ;;
  casnonegs) FILE=data/cas_pairs_nonegs/pairs.jsonl ;;
  d4llmnegs) FILE=data/llm_pairs_unrelated_llmnegs/pairs.jsonl ;;
  d4llmnegsfull) FILE=data/llm_pairs_unrelated_llmnegs_full/pairs.jsonl ;;
  d4unrelnegs) FILE=data/llm_pairs_unrelated_unrelnegs/pairs.jsonl ;;
  d4unrelnegsfull) FILE=data/llm_pairs_unrelated_unrelnegs_full/pairs.jsonl ;;
  casllmnegs) FILE=data/cas_pairs_llmnegs/pairs.jsonl ;;
  btcasnegs) FILE=data/bt_pairs_casnegs/pairs.jsonl ;;
  d5nonegs) FILE=data/llm_pairs_survey/pairs.jsonl ;;
  d5casnegs) FILE=data/llm_pairs_survey_casnegs/pairs.jsonl ;;
  d1split) FILE=data/llm_pairs_exact_split/pairs.jsonl ;;
  d1twojudge) FILE=data/llm_pairs_twojudge/pairs.jsonl ;;
  bt2casnegs) FILE=data/bt2_pairs_casnegs/pairs.jsonl ;;
  # E-T1..E-T7 (table fill, 2026-09-08; scripts/build_review4_cells.py): the seven
  # empty cells of the positives x negatives grid. No directional prediction is
  # registered; the grid is reported as it lands (three seeds each).
  casunrelnegs) FILE=data/cas_pairs_unrelnegs/pairs.jsonl ;;
  btnonegs) FILE=data/bt_pairs_nonegs/pairs.jsonl ;;
  btunrelnegs) FILE=data/bt_pairs_unrelnegs/pairs.jsonl ;;
  btllmnegs) FILE=data/bt_pairs_llmnegs/pairs.jsonl ;;
  d1nonegs) FILE=data/llm_pairs_nonegs/pairs.jsonl ;;
  d1casnegs) FILE=data/llm_pairs_casnegs/pairs.jsonl ;;
  d1unrelnegs) FILE=data/llm_pairs_unrelnegs/pairs.jsonl ;;
  *) echo "unknown CELL=${CELL}"; exit 1 ;;
esac
[[ -s "${FILE}" ]] || { echo "FATAL: ${FILE} missing or empty"; exit 1; }
SEEDS="${SEEDS:-42 43 44}"

for SEED in ${SEEDS}; do
  OUT="models/fact-${CELL}-6145-s${SEED}"
  TAG="fact-${CELL}-s${SEED}"
  if [[ -d "${OUT}/final" ]]; then
    echo "=== skip train ${TAG} (exists)"
  else
    echo "=== [${SECONDS}s] train ${TAG}"
    python scripts/train_invarembed.py \
        --train-file "${FILE}" --max-rows 6145 \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${OUT}"
  fi
  for TIER in easy medium hard; do
    BD="results/ranks/${TIER}_${TAG}.ranks.jsonl"
    [[ -s "${BD}" ]] && { echo "=== skip ${TIER} ${TAG}"; continue; }
    echo "=== [${SECONDS}s] eval ${TIER} ${TAG}"
    python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
        --dump-ranks "${BD}" \
        --output "results/ranks/${TIER}_${TAG}.summary.json"
  done
  XD="results/ranks/xling_${TAG}.ranks.jsonl"
  if [[ -s "${XD}" ]]; then
    echo "=== skip xling ${TAG}"
  else
    echo "=== [${SECONDS}s] eval xling ${TAG}"
    python scripts/eval_crosslingual.py --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --dump-ranks "${XD}" \
        --output "results/ranks/xling_${TAG}.summary.json"
  fi
  # E-R5 same-language organic duplicates (added 2026-09-03; the set is built
  # by scripts/build_samelang_eval.py and shares the cross-lingual corpus).
  SD="results/ranks/samelang_${TAG}.ranks.jsonl"
  if [[ -s "${SD}" ]]; then
    echo "=== skip samelang ${TAG}"
  else
    echo "=== [${SECONDS}s] eval samelang ${TAG}"
    python scripts/eval_crosslingual.py --eval-dir data/samelang_eval \
        --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --dump-ranks "${SD}" \
        --output "results/ranks/samelang_${TAG}.summary.json"
  fi
  # E-R14 generator-free near-miss probe (2026-09-05)
  for PROBE in casprobe:data/casprobe_eval casprobepairs:data/casprobe_pairs_eval; do
    PFX="${PROBE%%:*}"; PDIR="${PROBE#*:}"
    [[ -s "${PDIR}/queries.jsonl" ]] || { echo "=== no ${PDIR}; skip ${PFX} ${TAG}"; continue; }
    PD="results/ranks/${PFX}_${TAG}.ranks.jsonl"
    [[ -s "${PD}" ]] && { echo "=== skip ${PFX} ${TAG}"; continue; }
    echo "=== [${SECONDS}s] eval ${PFX} ${TAG}"
    python scripts/eval_crosslingual.py --eval-dir "${PDIR}" \
        --model "${OUT}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --dump-ranks "${PD}" \
        --output "results/ranks/${PFX}_${TAG}.summary.json"
  done
done
echo "== cell ${CELL} seeds [${SEEDS}] finished =="
