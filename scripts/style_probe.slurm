#!/bin/bash
#SBATCH --job-name=style-probe
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# RECIPE-SENSITIVITY AUDIT, GPU leg (scripts/style_probe.py --stage correlate
# with all three models). The CPU legs already ran on the login node
# (2026-07-30) and their artifacts are in results/:
#   style_probe_features.json / style_probe_model.joblib   (fit;   AUC .969)
#   style_probe_transfer.json / style_scores_retrieve.npz  (score; eq-vs-orig
#                                                           transfer AUC .714)
#   style_probe_correlations.json  (correlate, ctrl-llm + ctrl-cas only:
#       per-query rho +0.1145 vs -0.0168, delta +0.1313, p~5e-307)
#   analyze_hits_ctrl-llm_vs_ctrl-cas_hard.json (hit accounting)
#
# What is MISSING and what this job adds: the UNTRAINED base
# Qwen3-Embedding-0.6B column of the correlation table (its full-corpus
# doc/query embeddings were never cached). Steps:
#   1. re-run fit+score only if the CPU artifacts are absent (idempotent);
#   2. eval_retrieve on the hard tier for base 0.6B with --emb-cache-dir,
#      which (a) fills .emb_cache with the missing doc/query .npz and
#      (b) dumps per-query ranks for any future analyze_hits run against base
#      (the eval JSON goes to a -cachefill name so the recorded
#      eval_hard_qwen3-0.6b-base.json number is never clobbered);
#   3. re-run --stage correlate with base + ctrl-llm + ctrl-cas (overwrites
#      results/style_probe_correlations.json with the 3-model version).
# Expected outcome (pre-registered): base sits between the arms, well below
# ctrl-llm — the recipe-style tracking is TRAINED IN by the LLM arm, not
# inherited from the backbone. Falsifier: base rho ~= ctrl-llm rho would mean
# the style score measures something the backbone already tracks, and the
# correlation could not be attributed to recipe supervision.
# Budget: ~1 h encode + minutes of CPU correlate (3 h wall for safety).
#
# OUTCOME (job 49594071) + 2026-07-30 H1 RE-RUN: the pre-registered
# prediction held and then strengthened. Base 0.6B does NOT track recipe
# style (it mildly anti-tracks), so the tracking is trained in by the LLM
# arm. The base doc/query .npz are now cached, so steps 1-2 are unnecessary
# on a re-run; the whole audit is CPU-only on the login node (see the
# 3-command sequence in scripts/style_probe.py's docstring). Review finding
# H1 then showed the detector was reading a "Problem:" header + image-
# placeholder artifact, so all three CPU legs were re-run artifact-stripped
# (--strip-artifacts, now the default):
#   holdout AUC .969 -> .880; eq-vs-orig transfer .714 -> .804 (STRONGER);
#   per-query rho base -0.061 / ctrl-cas -0.019 / ctrl-llm +0.171,
#   llm-minus-cas +0.1895 on 94.2% of queries.
# Canonical numbers: results/style_probe_stripped.json (both columns);
# the v1 artifact-inflated files are kept as results/style_probe_*_v1_
# artifact.*. Step 1's guard below skips fit/score whenever
# style_scores_retrieve.npz exists -- delete it first if you want this job
# to regenerate the (stripped) detector from scratch.

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

# 1. CPU artifacts (skip if the login-node run already produced them)
if [[ ! -f results/style_scores_retrieve.npz ]]; then
    python scripts/style_probe.py --stage fit,score
fi

# 2. base-0.6B embeddings into the shared cache (+ rank dump, house eval flags)
python scripts/eval_retrieve.py --tier hard --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
    --dump-ranks results/ranks_qwen3-0.6b-base_hard.jsonl \
    --output results/eval_hard_qwen3-0.6b-base-cachefill.json

# sanity: the -cachefill run must reproduce the recorded base hard row
python - <<'EOF'
import json
new = json.load(open("results/eval_hard_qwen3-0.6b-base-cachefill.json"))["overall"]
old = json.load(open("results/eval_hard_qwen3-0.6b-base.json"))["overall"]
print(f"[sanity] base hard recorded={old} cachefill={new}")
assert abs(new["recall@10"] - old["recall@10"]) < 0.5, "base hard row drifted"
EOF

# 3. three-model correlation (CPU; overwrites the 2-model JSON)
python scripts/style_probe.py --stage correlate \
    --model-emb 'base-0.6b=.emb_cache/docs_Qwen__Qwen3-Embedding-0.6B_*.npz,.emb_cache/queries_Qwen__Qwen3-Embedding-0.6B_*.npz' \
    --model-emb 'ctrl-llm-6145=.emb_cache/docs_models__ctrl-llm-6145__final_*.npz,.emb_cache/queries_models__ctrl-llm-6145__final_*.npz' \
    --model-emb 'ctrl-cas-6145=.emb_cache/docs_models__ctrl-cas-6145__final_*.npz,.emb_cache/queries_models__ctrl-cas-6145__final_*.npz'

echo "== style-probe GPU leg finished =="
