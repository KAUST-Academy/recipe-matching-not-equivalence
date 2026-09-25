#!/bin/bash
#SBATCH --job-name=invarembed-eval-soups
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# WiSE-FT weight-soup evaluation (phase-2, zero-training enhancement).
#
# Soups: models/soup-a{0.3,0.5,0.7} = alpha*finetuned + (1-alpha)*base
#   (built by scripts/soup_models.py; base = Qwen/Qwen3-Embedding-0.6B,
#    finetuned = models/qwen3-0.6b-cas-cmnrl/final; rebuilt below if missing).
# Evals: every soup on ALL THREE tiers (easy/medium/hard) — 9 runs. The 0.6B
# is fast and the tiers share byte-identical corpus/queries, so with
# --emb-cache-dir each soup needs one encode pass; tiers 2/3 are matmul-only.
#
# SUCCESS CRITERION (phase-1 refs: reproducibility.json id
# "invarembed_phase1_full_sweep", job 49575668):
#   phase-1 FT   easy R@1 15.92 | medium R@1 3.86 / R@5 40.80 | hard R@1 0.16
#   untrained    easy R@1  8.32 | medium R@1 1.78 / R@5 60.91 | hard R@1 0.00
# A good soup KEEPS >=80% of the phase-1 R@1 gain:
#   easy   R@1 >= 8.32 + 0.8*(15.92-8.32) = 14.40
#   medium R@1 >= 1.78 + 0.8*( 3.86-1.78) =  3.44
# AND RECOVERS >=70% of the base's medium recall depth:
#   medium R@5 >= 0.7*60.91 = 42.64   (phase-1 FT sits at 40.80 — collapsed)
# The summary table below checks all three thresholds per alpha.
#
# Runtime: ~30-45 min nominal (3 encode passes + 6 cached retrievals); 3 h wall.
# Submit with:  sbatch scripts/eval_soups.slurm
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results
EMB_CACHE="${PROJECT}/.emb_cache"

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="

# Idempotent: (re)build any missing soup on CPU (seconds; exact same recipe).
python scripts/soup_models.py --alphas 0.3,0.5,0.7 --skip-existing

for A in 0.3 0.5 0.7; do
    for TIER in easy medium hard; do
        echo "=== [$(date +%H:%M:%S) | ${SECONDS}s] soup-a${A} / ${TIER} ==="
        python scripts/eval_retrieve.py --tier "${TIER}" --model "models/soup-a${A}" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 \
            --emb-cache-dir "${EMB_CACHE}" \
            --output "results/eval_${TIER}_soup-a${A}.json"
    done
done

echo "== summary =="
python - <<'EOF'
import json

REF = {  # phase-1 sweep, job 49575668
    "base-0.6B (untrained)":   {"easy": (8.32, 78.61, 87.78), "medium": (1.78, 60.91, 75.49), "hard": (0.00, 4.08, 17.60)},
    "FT phase-1 (alpha=1.0)":  {"easy": (15.92, 78.18, 87.21), "medium": (3.86, 40.80, 55.92), "hard": (0.16, 6.59, 19.47)},
}
rows = dict(REF)
for a in ("0.3", "0.5", "0.7"):
    rows[f"soup-a{a}"] = {}
    for tier in ("easy", "medium", "hard"):
        o = json.load(open(f"results/eval_{tier}_soup-a{a}.json"))["overall"]
        rows[f"soup-a{a}"][tier] = (o["recall@1"], o["recall@5"], o["recall@10"])

hdr = f"{'model':<24}" + "".join(f"{t + ' R@1/5/10':>24}" for t in ("easy", "medium", "hard"))
print(hdr); print("-" * len(hdr))
for name, tiers in rows.items():
    cells = "".join(f"{v[0]:>7.2f}/{v[1]:.2f}/{v[2]:.2f}" for v in (tiers["easy"], tiers["medium"], tiers["hard"]))
    print(f"{name:<24}{cells}")

print("\ncriterion check (keep >=80% R@1 gain AND recover >=70% base medium R@5):")
for a in ("0.3", "0.5", "0.7"):
    t = rows[f"soup-a{a}"]
    checks = [("easy R@1 >= 14.40", t["easy"][0] >= 14.40),
              ("medium R@1 >= 3.44", t["medium"][0] >= 3.44),
              ("medium R@5 >= 42.64", t["medium"][1] >= 42.64)]
    verdict = "PASS" if all(ok for _, ok in checks) else "fail"
    detail = "  ".join(f"{name}: {'yes' if ok else 'NO'}" for name, ok in checks)
    print(f"  soup-a{a}: {verdict}   {detail}")
EOF

echo "== soup eval sweep finished: $(date) =="
