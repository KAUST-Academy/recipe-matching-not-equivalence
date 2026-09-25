#!/bin/bash
#SBATCH --job-name=dump-ranks
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# PER-QUERY RANK DUMPS for the bootstrap statistics machinery (TASK C).
# Re-runs, with --dump-ranks, (i) the real-duplicate (cross-lingual) eval and
# (ii) all three MathNet-Retrieve tiers for EVERY Qwen3-0.6B controlled-
# experiment checkpoint present under models/ (ctrl-{cas,llm}-6145[-sNN];
# the e5 backbone replication is excluded -- different prompt conventions,
# single seed, not part of the seed statistics), plus the single-run dumps
# needed for the V2-vs-RaDeR-gte paired CI and the V2/base real-duplicate
# pair CI. Consumed by scripts/bootstrap_stats.py; driver:
# scripts/rebuild_stats.sh (submits this job only when dumps are missing).
#
# Design notes:
#  * IDEMPOTENT + RESUMABLE: every step is skipped when its .ranks.jsonl
#    already exists and is non-empty; safe to resubmit after a timeout.
#  * Summary JSONs of these re-runs go to results/ranks/*.summary.json --
#    the CANONICAL results/eval_* / results/crosslingual_* files (the
#    numbers in README/reproducibility.json) are never overwritten.
#  * Benchmark tiers reuse .emb_cache (written by the original eval jobs and
#    by train_ctrl_param.slurm), so per-query ranks are bit-identical to the
#    canonical runs. The cross-lingual harness has no embedding cache; its
#    re-encode can differ by GPU nondeterminism at the ~0.1-pt level --
#    bootstrap_stats.py cross-checks dump-derived R@1 against the canonical
#    summaries and flags any gap > 0.5 pt.
#  * RaDeR-gte needs the baselines-tf4 env (transformers-5 removed
#    rope_theta); that stanza mirrors eval_tf4_baselines.slurm exactly.
#  * Per-step skip-on-failure (like eval_baselines.slurm): one broken
#    checkpoint must not kill the sweep; failures are listed at the end and
#    the job exits nonzero if any occurred.

set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results/ranks

FAILED=()
step() {  # <desc> <cmd ...>
  local desc="$1"; shift
  echo "=== [${SECONDS}s] ${desc}"
  if ! "$@"; then
    echo "[WARN] step FAILED: ${desc}" >&2
    FAILED+=("${desc}")
  fi
}

# ---- 1. controlled-experiment checkpoints (both arms, all Qwen3 seeds) ----
CKPTS=()   # "dir|tag" pairs
for arm in cas llm; do
  [[ -d "models/ctrl-${arm}-6145/final" ]] && \
    CKPTS+=("models/ctrl-${arm}-6145|ctrl-${arm}-s42")   # legacy seed-42 dir
  for d in models/ctrl-${arm}-6145-s*; do
    [[ -d "${d}/final" ]] || continue
    [[ "${d}" == *e5* ]] && continue                     # e5 backbone: skip
    CKPTS+=("${d}|ctrl-${arm}-${d##*-}")                 # ...-s43 -> tag s43
  done
done
echo "=== checkpoints: ${CKPTS[*]}"

for pair in "${CKPTS[@]}"; do
  DIR="${pair%%|*}"; TAG="${pair##*|}"
  XD="results/ranks/xling_${TAG}.ranks.jsonl"
  if [[ -s "${XD}" ]]; then
    echo "=== skip xling ${TAG} (exists)"
  else
    step "xling ${TAG}" \
      python scripts/eval_crosslingual.py --model "${DIR}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --dump-ranks "${XD}" \
        --output "results/ranks/xling_${TAG}.summary.json"
  fi
  for TIER in easy medium hard; do
    BD="results/ranks/${TIER}_${TAG}.ranks.jsonl"
    if [[ -s "${BD}" ]]; then
      echo "=== skip ${TIER} ${TAG} (exists)"
    else
      step "${TIER} ${TAG}" \
        python scripts/eval_retrieve.py --tier "${TIER}" --model "${DIR}/final" \
          --device cuda --model-dtype bfloat16 --query-prompt-name query \
          --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
          --dump-ranks "${BD}" \
          --output "results/ranks/${TIER}_${TAG}.summary.json"
    fi
  done
done

# ---- 2. single-run dumps: V2 (easy + xling) and base 0.6B (xling) ---------
V2_PROMPT=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

D="results/ranks/easy_qwen3-0.6b-p2-instr.ranks.jsonl"
[[ -s "${D}" ]] || step "easy V2 (p2-instr)" \
  python scripts/eval_retrieve.py --tier easy \
    --model models/qwen3-0.6b-p2-instr/final \
    --device cuda --model-dtype bfloat16 --query-prompt "${V2_PROMPT}" \
    --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
    --dump-ranks "${D}" \
    --output results/ranks/easy_qwen3-0.6b-p2-instr.summary.json

D="results/ranks/xling_qwen3-0.6b-p2-instr.ranks.jsonl"
[[ -s "${D}" ]] || step "xling V2 (p2-instr)" \
  python scripts/eval_crosslingual.py --model models/qwen3-0.6b-p2-instr/final \
    --device cuda --model-dtype bfloat16 --query-prompt "${V2_PROMPT}" \
    --max-seq-length 1024 \
    --dump-ranks "${D}" \
    --output results/ranks/xling_qwen3-0.6b-p2-instr.summary.json

D="results/ranks/xling_qwen3-0.6b-base.ranks.jsonl"
[[ -s "${D}" ]] || step "xling base 0.6B" \
  python scripts/eval_crosslingual.py --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --max-seq-length 1024 \
    --dump-ranks "${D}" \
    --output results/ranks/xling_qwen3-0.6b-base.summary.json

# ---- 3. RaDeR-gte easy (baselines-tf4 env; mirrors eval_tf4_baselines) ----
D="results/ranks/easy_rader-gte-qwen2-7b.ranks.jsonl"
if [[ -s "${D}" ]]; then
  echo "=== skip easy rader-gte (exists)"
else
  conda activate baselines-tf4
  MATH_INSTR="Given a Math problem, retrieve relevant examples that help answer the problem"
  step "easy rader-gte-qwen2-7b" \
    python scripts/eval_retrieve.py --tier easy --device cuda \
      --model Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical \
      --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
      --pooling lasttoken --append-eos --trust-remote-code \
      --query-prompt $'query: '"${MATH_INSTR}"$'\n' \
      --doc-prompt 'document: ' \
      --dump-ranks "${D}" \
      --output results/ranks/easy_rader-gte-qwen2-7b.summary.json
  conda activate mathnet
fi

# ---- 4. summary ------------------------------------------------------------
python scripts/bootstrap_stats.py --check-dumps || true
if ((${#FAILED[@]})); then
  echo "== dump-ranks finished WITH ${#FAILED[@]} FAILURES:"
  printf '   - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "== dump-ranks finished cleanly; next: bash scripts/rebuild_stats.sh =="
