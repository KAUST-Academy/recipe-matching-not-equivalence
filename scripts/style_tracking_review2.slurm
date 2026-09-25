#!/bin/bash
#SBATCH --job-name=style-track-r2
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# 2026-09-04: do the reference cell and the two D4+LLM-negative cells track
# the style fingerprint? Re-runs the correlate
# stage of scripts/style_probe.py (CPU; every embedding is already cached
# under .emb_cache by the tier evals) with the seed-42 models of those cells
# beside the three published rows, then parks the output as
# results/style_tracking_review2.json and restores the published
# results/style_probe_correlations.json byte-for-byte.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
cd "${PROJECT}"
cp -n results/style_probe_correlations.json results/style_probe_correlations_prereview2_backup.json
python scripts/style_probe.py --stage correlate \
  --model-emb 'base-0.6b=.emb_cache/docs_Qwen__Qwen3-Embedding-0.6B_*.npz,.emb_cache/queries_Qwen__Qwen3-Embedding-0.6B_*.npz' \
  --model-emb 'ctrl-llm-6145=.emb_cache/docs_models__ctrl-llm-6145__final_*.npz,.emb_cache/queries_models__ctrl-llm-6145__final_*.npz' \
  --model-emb 'ctrl-cas-6145=.emb_cache/docs_models__ctrl-cas-6145__final_*.npz,.emb_cache/queries_models__ctrl-cas-6145__final_*.npz' \
  --model-emb 'reference-s42=.emb_cache/docs_models__fact-d4casnegs-6145-s42__final_*.npz,.emb_cache/queries_models__fact-d4casnegs-6145-s42__final_*.npz' \
  --model-emb 'd4llmnegs-s42=.emb_cache/docs_models__fact-d4llmnegs-6145-s42__final_*.npz,.emb_cache/queries_models__fact-d4llmnegs-6145-s42__final_*.npz' \
  --model-emb 'd4llmnegsfull-s42=.emb_cache/docs_models__fact-d4llmnegsfull-6145-s42__final_*.npz,.emb_cache/queries_models__fact-d4llmnegsfull-6145-s42__final_*.npz' \
  --model-emb 'D4-unrelated-s42=.emb_cache/docs_models__dose-unrelated-6145__final_*.npz,.emb_cache/queries_models__dose-unrelated-6145__final_*.npz'
mv results/style_probe_correlations.json results/style_tracking_review2.json
cp results/style_probe_correlations_prereview2_backup.json results/style_probe_correlations.json
python - <<'PY'
import json
d=json.load(open("results/style_tracking_review2.json"))
for m,v in d["per_model"].items():
    print(m, {k: v[k] for k in v if isinstance(v[k],(int,float))})
PY
echo "== done =="
