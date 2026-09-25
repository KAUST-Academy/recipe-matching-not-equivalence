#!/bin/bash
#SBATCH --job-name=saber-sig
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err
# Re-run the four SABER rows with --per-query-out so the label-channel gain
# (+0.0354 over base) can carry a paired interval. Embeddings are cached, so
# this is cheap. Point estimates without an interval are exactly what round-2
# findings caught twice; the claim does not go in the paper without one.
set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

run () {  # <tag> <model> [flags...]
    local tag="$1" model="$2"; shift 2
    echo "== [${SECONDS}s] per-query ${tag} =="
    python scripts/eval_saber.py --model "${model}" --device cuda \
        --batch-size 64 --setting statement-full --emb-cache-dir .emb_cache \
        --model-dtype bfloat16 --max-seq-length 1024 \
        --per-query-out "results/saber_pq_${tag}.json" \
        --output "results/saber_pqrun_${tag}.json" "$@" || echo "FAILED ${tag}"
}
run base           Qwen/Qwen3-Embedding-0.6B
run pubattack      models/saber-attack-6145/final
run label4733      models/saber-label-4733/final
run doc4733        models/saber-doc-4733/final

python scripts/saber_label_bootstrap.py
echo "== finished: $(date) =="
