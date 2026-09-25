#!/bin/bash
#SBATCH --job-name=p2-param
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# TASK D1: SEED-parametrized variant of train_phase2.slurm (seed fill-ins for
# the V1/V2 error bars). Trains BOTH phase-2 variants at a given seed:
#
#   V1 "mixed":       4-channel mixed data          -> models/qwen3-0.6b-p2-mixed-s${SEED}
#   V2 "mixed+instr": same + math instruction       -> models/qwen3-0.6b-p2-instr-s${SEED}
#
# then runs the three MathNet-Retrieve tiers AND the real-duplicate
# (cross-lingual) eval per variant. Trainer commands are IDENTICAL to
# train_phase2.slurm (the seed-42 reference, job 49580953) except --seed and
# --output-dir; xling prompt conventions mirror eval_p2_ood{,_v2}.slurm
# (V1: registered 'query' prompt; V2: the raw math instruction).
# Tier evals additionally pass --emb-cache-dir .emb_cache (corpus/queries are
# md5-identical across tiers, so medium/hard reuse easy's embeddings; cache
# key includes the model path -- no cross-run sharing, numbers unchanged).
#
# Env vars (pass via sbatch --export):
#   SEED   training/shuffling/dev-split seed   [default 42; the unsuffixed
#          models/qwen3-0.6b-p2-{mixed,instr} ARE seed 42 -- submit 43/44]
#
# Submissions:
#   sbatch --export=ALL,SEED=43 scripts/train_phase2_param.slurm
#   sbatch --export=ALL,SEED=44 scripts/train_phase2_param.slurm
#
# Outputs: models/qwen3-0.6b-p2-{mixed,instr}-s${SEED}/
#          results/eval_{easy,medium,hard}_qwen3-0.6b-p2-{mixed,instr}-s${SEED}.json
#          results/crosslingual_qwen3-0.6b-p2-{mixed,instr}-s${SEED}.json
#
# EXPECTED RUNTIME: seed-42 reference took 4h45m for 2 trainings + 6 tier
# evals WITHOUT the embedding cache; with the cache ~4h + 2 xling evals
# (~2 min each at 0.6B). 8 h cap leaves slack.
#
# CPU-verified 2026-07-30: trainer path unchanged vs train_phase2.slurm
# (diff = --seed/--output-dir only); tiny CPU smoke of the exact V2 arg
# combination (--seed 43 + --query-prompt-text) trains/saves/reloads and
# records seed=43 in run_config.json; full-script dry-run with a stubbed
# python resolves every command as expected.
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results models

SEED="${SEED:-42}"
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) =="
echo "== p2-param SEED=${SEED} =="

MATH_PROMPT=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

for v in mixed instr; do
  if [[ -e "models/qwen3-0.6b-p2-${v}-s${SEED}" ]]; then
    echo "FATAL: models/qwen3-0.6b-p2-${v}-s${SEED} already exists -- refusing to clobber" >&2
    exit 1
  fi
done

# ---------------- V1 "mixed": data change only ------------------------------
# identical to train_phase2.slurm except --seed / --output-dir
python scripts/train_invarembed.py \
    --train-file data/phase2/mixed_pairs.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 512 --mini-batch-size 16 --scale 20 \
    --epochs 2 --lr 1e-5 --warmup-ratio 0.05 --seed "${SEED}" \
    --dev-frac 0.05 --eval-steps 50 \
    --max-seq-length 1024 --query-prompt-name query \
    --output-dir "models/qwen3-0.6b-p2-mixed-s${SEED}"

echo "== [${SECONDS}s] V1 (mixed, s${SEED}) training finished =="

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" \
        --model "models/qwen3-0.6b-p2-mixed-s${SEED}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 \
        --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_qwen3-0.6b-p2-mixed-s${SEED}.json"
done
python scripts/eval_crosslingual.py \
    --model "models/qwen3-0.6b-p2-mixed-s${SEED}/final" \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --max-seq-length 1024 \
    --output "results/crosslingual_qwen3-0.6b-p2-mixed-s${SEED}.json"

echo "== [${SECONDS}s] V1 (mixed, s${SEED}) evals finished =="

# ---------------- V2 "mixed+instr": + math-specific instruction -------------
python scripts/train_invarembed.py \
    --train-file data/phase2/mixed_pairs.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 512 --mini-batch-size 16 --scale 20 \
    --epochs 2 --lr 1e-5 --warmup-ratio 0.05 --seed "${SEED}" \
    --dev-frac 0.05 --eval-steps 50 \
    --max-seq-length 1024 --query-prompt-text "${MATH_PROMPT}" \
    --output-dir "models/qwen3-0.6b-p2-instr-s${SEED}"

echo "== [${SECONDS}s] V2 (mixed+instr, s${SEED}) training finished =="

for TIER in easy medium hard; do
    python scripts/eval_retrieve.py --tier "${TIER}" \
        --model "models/qwen3-0.6b-p2-instr-s${SEED}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt "${MATH_PROMPT}" \
        --batch-size 64 --max-seq-length 1024 \
        --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_qwen3-0.6b-p2-instr-s${SEED}.json"
done
python scripts/eval_crosslingual.py \
    --model "models/qwen3-0.6b-p2-instr-s${SEED}/final" \
    --device cuda --model-dtype bfloat16 --query-prompt "${MATH_PROMPT}" \
    --max-seq-length 1024 \
    --output "results/crosslingual_qwen3-0.6b-p2-instr-s${SEED}.json"

echo "== [${SECONDS}s] V2 (mixed+instr, s${SEED}) evals finished =="

# ---------------- comparison table ------------------------------------------
SEED="${SEED}" python - <<'EOF'
import json, os
seed = os.environ["SEED"]
runs = [("qwen3-0.6b-p2-mixed",           "V1 mixed  (s42 ref)"),
        (f"qwen3-0.6b-p2-mixed-s{seed}",  f"V1 mixed  (s{seed})"),
        ("qwen3-0.6b-p2-instr",           "V2 +instr (s42 ref)"),
        (f"qwen3-0.6b-p2-instr-s{seed}",  f"V2 +instr (s{seed})")]
tiers = ["easy", "medium", "hard"]
hdr = f"{'model':22s} | " + " | ".join(f"{t + ' R@1/5/10':>20s}" for t in tiers) \
      + f" | {'xling strict':>20s}"
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
for run, label in runs:
    cells = []
    for t in tiers:
        p = f"results/eval_{t}_{run}.json"
        if os.path.exists(p):
            o = json.load(open(p))["overall"]
            cells.append(f"{o['recall@1']:5.2f}/{o['recall@5']:5.2f}/{o['recall@10']:5.2f}")
        else:
            cells.append("(missing)")
    px = f"results/crosslingual_{run}.json"
    if os.path.exists(px):
        x = json.load(open(px))["strict_crosslingual_gold"]
        cells.append(f"{x['recall@1']:5.2f}/{x['recall@5']:5.2f}/{x['recall@10']:5.2f}")
    else:
        cells.append("(missing)")
    print(f"{label:22s} | " + " | ".join(f"{c:>20s}" for c in cells))
print("=" * len(hdr))
print("s42 reference bars: V1 easy/med/hard R@1 = 20.83/3.67/0.03, xling 59.80 | "
      "V2 = 21.70/2.71/0.01, xling 74.30 | base xling 84.73")
EOF

echo "== p2-param (s${SEED}) finished =="
