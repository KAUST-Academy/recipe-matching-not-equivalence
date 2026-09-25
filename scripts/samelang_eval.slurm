#!/bin/bash
#SBATCH --job-name=samelang-eval
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# E-R5 (2026-09-03; written BEFORE any result existed):
# SAME-LANGUAGE organic duplicates.
#
# Motivation: every real duplicate in the first probe crosses languages, so an
# arm's loss there mixes lost retrieval with lost language transfer. Scoring
# reprints within one language removes the transfer from the measurement; the
# English queries alone give an English-only reading of the same set.
#
# Set: data/samelang_eval (scripts/build_samelang_eval.py) -- every
# known-language member of the 242 monolingual clusters of the mined
# duplicate graph queries the full 27,817-problem corpus for its same-
# language reprint(s), self-masked. PRIMARY slice = the clusters mined by
# formula, near-text or answer (not the exact-text rule) whose members occur
# in NO training pair file of the campaign (61 clusters, 125 queries, 115
# English). Leaked and exact-text clusters are reported separately.
#
# Models: every campaign model that has a cross-lingual result, under the
# encoding convention recorded in that result (scripts/samelang_eval_driver.py
# manifest -> results/samelang_models.tsv).
#
# REGISTERED READOUT (P-R5), verdict computed by scripts/samelang_verdict.py:
#   Quantity: DiD_same = (easy-tier arm gap, ctrl-LLM - ctrl-CAS, eight seeds)
#             - (same-language R@1 arm gap on the PRIMARY slice, eight seeds),
#             with a nested bootstrap over duplicate clusters and training seeds.
#   Claim under test: "most of the easy-tier advantage does not reach real
#             duplicates" holds on SAME-LANGUAGE organic duplicates too.
#   PASS      DiD_same >= 22.67 (half of the 45.33 easy gap: "most") AND its
#             95% interval excludes zero.
#   PARTIAL   0 < DiD_same < 22.67, or the interval includes zero: a real but
#             minority share of the gap fails to transfer; Section 6 must say
#             "a share", not "most", for same-language data.
#   REFUTED   DiD_same <= 0: the recipe arm's easy-tier lead reaches same-
#             language duplicates in full; the easy-tier circularity claim
#             would then rest on the cross-lingual probe alone and Section 6
#             would have to say so in its first paragraph.
#   No direction is registered for: either controlled arm vs the base, D4 vs
#   the base, or the reference cell (D4 positives + verified negatives); all
#   are reported as they land, on the primary, English, leaked and exact slices.
#   Falsifier for the surface-bias worry: if the recipe arm's same-language
#   loss vs the base is LARGER on the primary slice than its cross-lingual
#   loss (-18.48), the cross-lingual probe was not the thing understating it.
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

python scripts/samelang_eval_driver.py manifest
python scripts/samelang_eval_driver.py run
python scripts/samelang_verdict.py || echo "verdict script: some arms not yet complete"
echo "== E-R5 finished =="
