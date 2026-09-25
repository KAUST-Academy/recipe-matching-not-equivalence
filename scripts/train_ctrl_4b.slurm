#!/bin/bash
#SBATCH --job-name=ctrl-4b
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# TASK D2: 4B controlled replication -- BOTH supervision arms (CAS-verified vs
# LLM-judged) at the matched 6,145-row budget on Qwen3-Embedding-4B + LoRA
# (r16, all-linear), 1x A100 80GB, bf16 + gradient checkpointing. Then per
# arm: easy + hard tier evals + real-duplicate (cross-lingual) eval.
#
#   models/ctrl-{cas,llm}-6145-4b/
#   results/eval_{easy,hard}_ctrl-{cas,llm}-4b.json
#   results/crosslingual_ctrl-{cas,llm}-4b.json
#   (tag "4b" -> readable later by: python scripts/aggregate_ctrl_stats.py
#    --backbone-tags e5-s42,4b ; missing medium tier is reported + skipped)
#
# FAITHFULNESS vs the 0.6B controlled experiment (train_controlled_exp.slurm,
# job 49585887): same data files, same --max-rows 6145, same effective
# contrastive batch 256 (cmnrl grad cache), same epochs/warmup/seed/dev-frac,
# same 'query' prompt, same no---max-seq-length-at-train convention.
# DOCUMENTED DEVIATIONS (identical across arms, so the arm-vs-arm comparison
# stands): (1) LoRA r16/alpha32/dropout0.05 instead of full FT -- 4B full FT
# does not fit 1x80GB with AdamW; (2) --lr 1e-4 instead of 2e-5 -- LoRA
# convention, per the phase-2 stanza in train_invarembed.slurm;
# (3) --gradient-checkpointing on.
#
# MEMORY ESTIMATE (honest, for batch 256 grad-cache at 4B, mini-batch 16):
#   weights bf16 ~7.6 GB + LoRA r16 all-linear ~33M params (AdamW fp32 states
#   ~0.5 GB) + grad-cache embedding pass (no-grad, 16-seq chunks, transient)
#   + grad re-forward with checkpointing: ~36 layers x 16 seqs x ~1.3K tokens
#   x 2560 dim x 2 B ~ 4 GB boundaries + a few GB per-layer recompute
#   transient (SDPA). Peak ~15-25 GB << 80 GB, even though training passes no
#   --max-seq-length (faithful to the 0.6B reference): the corpus maxes out
#   at 3,585 chars ~ 1.2K tokens, so batch 256 FITS and mini-batch stays 16
#   (identical to the reference). If an OOM nonetheless occurs, resubmit with
#   MINI_BATCH=8 (cmnrl mini-batch is a pure memory knob -- the objective and
#   the effective batch 256 are unchanged; document the value used).
#
# RUNTIME ESTIMATE (honest): 0.6B full-FT reference steps/times: cas ~114
# steps @21 s, llm ~180 steps @15 s. At 4B: ~6.7x FLOPs, LoRA removes the
# weight-gradient half of backward (~-25%), checkpointing re-adds a forward
# (~+25%) -> ~60-140 s/step => training ~5.5-11 h for both arms; evals ~45
# min/arm (easy-tier embed ~30-35 min at batch 32; hard reuses the embedding
# cache -- corpus/queries are md5-identical across tiers; xling ~6 min).
# TOTAL ~7-12 h: the 12 h cap may not survive the pessimistic tail. The job
# is therefore RESUMABLE: finished arms/evals are skipped on resubmission
# (guards below), and ARMS=llm (or cas) restricts a run to one arm:
#     sbatch scripts/train_ctrl_4b.slurm                      # both arms
#     sbatch --export=ALL,ARMS=llm scripts/train_ctrl_4b.slurm  # one arm
#
# NOTE: models/ctrl-*-6145-4b/final stores ONLY the LoRA adapter + a
# reference to Qwen/Qwen3-Embedding-4B (base_model_name_or_path); eval jobs
# resolve the base from HF_HOME (already cached in .hf_cache). CPU-verified
# 2026-07-30: LoRA+gradient-checkpointing+cmnrl trains, checkpoints,
# load-best-reloads, saves and reloads at 0.6B/MiniLM; both eval harnesses
# load an adapter-only final/ dir.
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results models

ARMS="${ARMS:-cas llm}"
MINI_BATCH="${MINI_BATCH:-16}"
echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== ctrl-4b ARMS='${ARMS}' MINI_BATCH=${MINI_BATCH} =="

train_file() {  # <arm> -> train file (arms validated upfront below)
  case "$1" in
    cas) echo data/cas_pairs/pairs.jsonl ;;
    llm) echo data/llm_pairs/pairs.jsonl ;;
  esac
}

# validate ARMS upfront: `exit` inside $(train_file ...) could not abort the job
for arm in ${ARMS}; do
  case "${arm}" in
    cas|llm) ;;
    *) echo "FATAL: unknown arm '${arm}' in ARMS='${ARMS}' (use cas and/or llm)" >&2; exit 1 ;;
  esac
done

for arm in ${ARMS}; do
  out="models/ctrl-${arm}-6145-4b"

  # ---- train (skip if this arm already finished; refuse half-written dirs) --
  if [[ -e "${out}/final/modules.json" ]]; then
    echo "== [${SECONDS}s] ARM=${arm}: ${out}/final exists -- skipping training =="
  elif [[ -e "${out}" ]]; then
    echo "FATAL: ${out} exists but has no final/ (dead run?) -- inspect/remove it first" >&2
    exit 1
  else
    echo "== [${SECONDS}s] ARM=${arm} training (4B + LoRA r16) =="
    python scripts/train_invarembed.py \
        --train-file "$(train_file "${arm}")" --max-rows 6145 \
        --model Qwen/Qwen3-Embedding-4B \
        --model-dtype bfloat16 --bf16 --gradient-checkpointing \
        --lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
        --loss cmnrl --batch-size 256 --mini-batch-size "${MINI_BATCH}" --scale 20 \
        --epochs 3 --lr 1e-4 --warmup-ratio 0.05 --seed 42 \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${out}"
  fi

  # ---- evals (each skipped if its output already exists) --------------------
  for TIER in easy hard; do
    res="results/eval_${TIER}_ctrl-${arm}-4b.json"
    if [[ -e "${res}" ]]; then
      echo "== [${SECONDS}s] ARM=${arm} ${TIER}: ${res} exists -- skipping =="
      continue
    fi
    echo "== [${SECONDS}s] ARM=${arm} eval tier=${TIER} =="
    python scripts/eval_retrieve.py --tier "${TIER}" --model "${out}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 32 --max-seq-length 1024 \
        --emb-cache-dir .emb_cache \
        --output "${res}"
  done

  res="results/crosslingual_ctrl-${arm}-4b.json"
  if [[ -e "${res}" ]]; then
    echo "== [${SECONDS}s] ARM=${arm} xling: ${res} exists -- skipping =="
  else
    echo "== [${SECONDS}s] ARM=${arm} real-duplicate (cross-lingual) eval =="
    python scripts/eval_crosslingual.py --model "${out}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --output "${res}"
  fi
done

echo "=== CONTROLLED EXPERIMENT @4B SUMMARY ==="
python - <<'PY'
import json, os
def cell(p, key):
    if not os.path.exists(p):
        return "(missing)"
    o = json.load(open(p))[key]
    return f"{o['recall@1']:.2f}/{o['recall@5']:.2f}/{o['recall@10']:.2f}"
for arm in ["cas", "llm"]:
    e = cell(f"results/eval_easy_ctrl-{arm}-4b.json", "overall")
    h = cell(f"results/eval_hard_ctrl-{arm}-4b.json", "overall")
    x = cell(f"results/crosslingual_ctrl-{arm}-4b.json", "strict_crosslingual_gold")
    print(f"  {arm:4s}  easy {e:>20s}  hard {h:>20s}  xling-strict {x:>20s}")
print("  ref 0.6B ctrl seed42 (easy|hard|xling R@1): cas 17.31 | 0.13 | 61.32"
      "  --  llm 61.88 | 9.37 | 64.38")
print("  ref UNTRAINED Qwen3-4B ('query' prompt, seq 1024): easy 11.95/82.03/90.77"
      "  hard 0.00/5.05/25.70  xling 91.60/96.95/97.46")
PY
echo "== ctrl-4b finished; aggregate with: python scripts/aggregate_ctrl_stats.py --backbone-tags e5-s42,4b =="
