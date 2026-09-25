#!/bin/bash
#SBATCH --job-name=ctrl-param
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# PARAMETERIZED CONTROLLED EXPERIMENT (statistical hardening of the paper's
# headline claim). Generalizes train_controlled_exp.slurm (which hardcodes
# seed 42 + Qwen3-Embedding-0.6B): trains BOTH supervision arms (CAS-verified
# vs LLM-judged) at the matched 6,145-row budget with identical
# hyperparameters, then runs all three MathNet-Retrieve tiers + the
# real-duplicate (cross-lingual) eval per arm.
#
# Env vars (pass via sbatch --export=ALL,VAR=value,...):
#   SEED      training/shuffling seed                  [default 42]
#   BACKBONE  HF model id or local path                [default Qwen/Qwen3-Embedding-0.6B]
#   TAG       suffix for models/ + results/ artifacts  [default: s${SEED} for
#             qwen3, e5-s${SEED} for multilingual-e5]
#   Per-backbone prompt/pooling conventions are auto-derived from BACKBONE
#   (qwen3 family: registered 'query' prompt, lasttoken pooling from the repo
#   ST config, eval seq len 1024; multilingual-e5 family: raw "query: " /
#   "passage: " prefixes, mean pooling from the repo ST config, eval seq len
#   512). Override explicitly for any other backbone:
#   QUERY_PROMPT_NAME  registered prompt name for anchors/queries (qwen3 path)
#   QUERY_PROMPT_TEXT  raw anchor/query prefix (e5 path; excl. w/ _NAME)
#   DOC_PROMPT_TEXT    raw positive/negative/document prefix (e5 path)
#   EVAL_MAX_SEQ       eval truncation length
#
# Two extra seeds and a second backbone:
#   sbatch --export=ALL,SEED=43 scripts/train_ctrl_param.slurm
#   sbatch --export=ALL,SEED=44 scripts/train_ctrl_param.slurm
#   sbatch --export=ALL,BACKBONE=intfloat/multilingual-e5-large scripts/train_ctrl_param.slurm
#
# Outputs:  models/ctrl-{cas,llm}-6145-${TAG}/
#           results/eval_{easy,medium,hard}_ctrl-{cas,llm}-${TAG}.json
#           results/crosslingual_ctrl-{cas,llm}-${TAG}.json
# Aggregate seeds afterwards with: python scripts/aggregate_ctrl_stats.py
#
# NOTE (faithfulness): the seed-42 reference run (train_controlled_exp.slurm)
# passed no --max-seq-length at TRAINING time (model-default truncation);
# this script keeps that behavior for every backbone. Eval seq lens mirror
# the published eval configs (1024 qwen3 / 512 = e5 model max).

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

SEED="${SEED:-42}"
BACKBONE="${BACKBONE:-Qwen/Qwen3-Embedding-0.6B}"

# ---- per-backbone conventions (override via env) --------------------------
case "${BACKBONE}" in
  *multilingual-e5*|*e5-large*|*e5-base*|*e5-small*)
    FAMILY=e5
    TAG="${TAG:-e5-s${SEED}}"
    QUERY_PROMPT_NAME="${QUERY_PROMPT_NAME:-}"
    QUERY_PROMPT_TEXT="${QUERY_PROMPT_TEXT:-query: }"
    DOC_PROMPT_TEXT="${DOC_PROMPT_TEXT:-passage: }"
    EVAL_MAX_SEQ="${EVAL_MAX_SEQ:-512}"     # XLM-R hard positional limit
    ;;
  *Qwen3-Embedding*|*qwen3*)
    FAMILY=qwen3
    TAG="${TAG:-s${SEED}}"
    QUERY_PROMPT_NAME="${QUERY_PROMPT_NAME:-query}"
    QUERY_PROMPT_TEXT="${QUERY_PROMPT_TEXT:-}"
    DOC_PROMPT_TEXT="${DOC_PROMPT_TEXT:-}"
    EVAL_MAX_SEQ="${EVAL_MAX_SEQ:-1024}"
    ;;
  *)
    FAMILY=custom
    TAG="${TAG:?custom BACKBONE needs an explicit TAG}"
    QUERY_PROMPT_NAME="${QUERY_PROMPT_NAME:-}"
    QUERY_PROMPT_TEXT="${QUERY_PROMPT_TEXT:-}"
    DOC_PROMPT_TEXT="${DOC_PROMPT_TEXT:-}"
    EVAL_MAX_SEQ="${EVAL_MAX_SEQ:-1024}"
    ;;
esac
if [[ -n "${QUERY_PROMPT_NAME}" && -n "${QUERY_PROMPT_TEXT}" ]]; then
  echo "FATAL: QUERY_PROMPT_NAME and QUERY_PROMPT_TEXT are mutually exclusive" >&2
  exit 1
fi

# trainer prompt args
TRAIN_PROMPT_ARGS=()
[[ -n "${QUERY_PROMPT_NAME}" ]] && TRAIN_PROMPT_ARGS+=(--query-prompt-name "${QUERY_PROMPT_NAME}")
[[ -n "${QUERY_PROMPT_TEXT}" ]] && TRAIN_PROMPT_ARGS+=(--query-prompt-text "${QUERY_PROMPT_TEXT}")
[[ -n "${DOC_PROMPT_TEXT}"   ]] && TRAIN_PROMPT_ARGS+=(--doc-prompt-text "${DOC_PROMPT_TEXT}")
# eval prompt args (eval_retrieve.py and eval_crosslingual.py share flag names)
EVAL_PROMPT_ARGS=()
[[ -n "${QUERY_PROMPT_NAME}" ]] && EVAL_PROMPT_ARGS+=(--query-prompt-name "${QUERY_PROMPT_NAME}")
[[ -n "${QUERY_PROMPT_TEXT}" ]] && EVAL_PROMPT_ARGS+=(--query-prompt "${QUERY_PROMPT_TEXT}")
[[ -n "${DOC_PROMPT_TEXT}"   ]] && EVAL_PROMPT_ARGS+=(--doc-prompt "${DOC_PROMPT_TEXT}")

echo "=== ctrl-param: SEED=${SEED} BACKBONE=${BACKBONE} FAMILY=${FAMILY} TAG=${TAG}"
echo "=== train prompts: ${TRAIN_PROMPT_ARGS[*]:-<none>} | eval prompts: ${EVAL_PROMPT_ARGS[*]:-<none>} | eval seq ${EVAL_MAX_SEQ}"

for arm in cas llm; do
  if [[ -e "models/ctrl-${arm}-6145-${TAG}" ]]; then
    echo "FATAL: models/ctrl-${arm}-6145-${TAG} already exists -- refusing to clobber (pick another TAG)" >&2
    exit 1
  fi
done

train_and_eval() {  # <arm> <train-file>
  local arm="$1" file="$2"
  local out="models/ctrl-${arm}-6145-${TAG}"
  echo "=== [${SECONDS}s] ARM=${arm} SEED=${SEED} TAG=${TAG} training ==="
  # identical to the seed-42 reference command except --seed/--model/prompts
  python scripts/train_invarembed.py \
      --train-file "${file}" --max-rows 6145 \
      --model "${BACKBONE}" \
      --model-dtype bfloat16 --bf16 \
      --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
      --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
      --dev-frac 0.05 --eval-steps 100 \
      "${TRAIN_PROMPT_ARGS[@]}" \
      --output-dir "${out}"
  for TIER in easy medium hard; do
    echo "=== [${SECONDS}s] ARM=${arm} eval tier=${TIER} ==="
    python scripts/eval_retrieve.py --tier "$TIER" --model "${out}/final" \
        --device cuda --model-dtype bfloat16 \
        "${EVAL_PROMPT_ARGS[@]}" \
        --batch-size 64 --max-seq-length "${EVAL_MAX_SEQ}" \
        --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_ctrl-${arm}-${TAG}.json"
  done
  echo "=== [${SECONDS}s] ARM=${arm} real-duplicate (cross-lingual) eval ==="
  python scripts/eval_crosslingual.py --model "${out}/final" \
      --device cuda --model-dtype bfloat16 \
      "${EVAL_PROMPT_ARGS[@]}" \
      --max-seq-length "${EVAL_MAX_SEQ}" \
      --output "results/crosslingual_ctrl-${arm}-${TAG}.json"
}

train_and_eval cas data/cas_pairs/pairs.jsonl
train_and_eval llm data/llm_pairs/pairs.jsonl

echo "=== CONTROLLED EXPERIMENT (${TAG}) SUMMARY ==="
TAG="${TAG}" python - <<'PY'
import json, os
tag = os.environ["TAG"]
for arm in ["cas", "llm"]:
    row = []
    for tier in ["easy", "medium", "hard"]:
        o = json.load(open(f"results/eval_{tier}_ctrl-{arm}-{tag}.json"))["overall"]
        row.append(f"{o['recall@1']:.2f}/{o['recall@5']:.2f}/{o['recall@10']:.2f}")
    x = json.load(open(f"results/crosslingual_ctrl-{arm}-{tag}.json"))["strict_crosslingual_gold"]
    xs = f"{x['recall@1']:.2f}/{x['recall@5']:.2f}/{x['recall@10']:.2f}"
    print(f"  {arm:4s}  easy {row[0]:>20s}  medium {row[1]:>20s}  hard {row[2]:>20s}  xling-strict {xs:>20s}")
print("  ref seed42/qwen3 R@1 (easy|med|hard|xling): cas 17.31 | 2.47 | 0.13 | 61.32  --  llm 61.88 | 9.06 | 9.37 | 64.38")
print("  ref base qwen3-0.6B: 8.32 | 1.78 | 0.00 | 84.73")
PY
echo "== ctrl-param (${TAG}) finished; aggregate with: python scripts/aggregate_ctrl_stats.py =="
