#!/bin/bash
#SBATCH --job-name=external-ood
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# EXTERNAL OOD EVALUATION: MELD + BRIGHT theorem splits (Task B, 2026-07-29).
#
# Purpose: test the campaign's models on math-equivalence / math-retrieval
# benchmarks we did NOT build (the independence our mined real-duplicate set
# lacks): MELD (uw-math-ai/MELD-dataset, 270 human-reviewed equivalence pairs
# + 541 hard distractors, MathLeap paper arXiv:2606.23959) and BRIGHT's two
# math splits (xlangai/BRIGHT theoremqa_theorems 76q/23,839d and aops
# 111q/188,002d, official protocol, headline NDCG@10).
#
# PRE-REGISTERED PREDICTION (written before this job first ran):
#   ctrl-llm-6145 (trained on pairs from MathNet-Retrieve's own Appendix-F
#   recipe; benchmark hard R@1 9.37 vs <=0.16 for all real models) shows NO
#   systematic advantage over ctrl-cas-6145 on MELD or BRIGHT-theorem --
#   i.e. the two arms land within noise of each other, far from the base
#   model's neighborhood, replicating the real-duplicate verdict (61.3 vs
#   64.4 R@1, statistically indistinguishable) on evals we did not
#   construct. If ctrl-llm instead showed a large external advantage, the
#   recipe-matching/circularity interpretation would be in trouble.
#   Secondary expectations: (a) V2 (p2-instr) retains the most external
#   performance among trained models (mirroring its real-duplicate
#   retention); (b) narrow synthetic fine-tuning (phase-1 pure-CAS) erodes
#   external scores below base (catastrophic forgetting).
#
# WHAT RUNS (skip-on-failure per step, summary table at the end):
#  S1  MELD for all 8 models            (~1-3 min each; tiny dataset)
#  S2  BRIGHT theoremqa_theorems, all 8 (~1-5 min each at 360 texts/s @0.6B)
#  S3  BRIGHT aops, all 8               (~9 min per 0.6B, ~30 min for 4B)
#  S4  prompt-sensitivity references: base-0.6B + 4B rerun with BRIGHT's
#      OFFICIAL e5-style task instruction (configs/e5/<task>.json) instead
#      of our house 'query' prompt      (cheap; theoremqa both, aops base)
#  S5  summary table + pre-registered-prediction check
#
# PROMPT CONVENTIONS (same as every campaign eval; recorded in each JSON):
#  * all Qwen3-0.6B derivatives + base + 4B: --query-prompt-name query
#    (the prompt they were trained/evaluated with all campaign) EXCEPT
#    p2-instr which uses its own math instruction (README models table);
#  * all-mpnet-base-v2: no prompt (BRIGHT's official sbert config is empty);
#  * prompts go on the query side only (BRIGHT + house convention).
#  Deviation from official BRIGHT leaderboard: --max-seq-length 1024 (house
#  standard; official used up to 8192 for 7B instruct models). mpnet
#  (native 384) is unaffected -- its rows are the external comparability
#  anchor against the official 'sbert' leaderboard entry.
#
# Nominal runtime ~2.5 h (measured throughput: 0.6B 360 texts/s, 4B 112
# texts/s on the 117K MathNet corpus); 6 h wall for downloads + margin.
# First run downloads BRIGHT (~a few hundred MB) into HF_HOME.
#
# Submit with:  sbatch scripts/eval_external_ood.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results
export FAILLOG="results/external_ood_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="

# The V2 instruction (README models table / models/qwen3-0.6b-p2-instr).
V2_INSTR=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'
# BRIGHT's official e5-style task instructions (configs/e5/<task>.json).
BRIGHT_TT_INSTR=$'Instruct: Given a Math problem, retrieve relevant theorems that help answer the problem\nQuery: '
BRIGHT_AOPS_INSTR=$'Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery: '

run_step() {                # run_step <label> <script> <args...>
    local label="$1" script="$2"; shift 2
    echo "=== [$(date +%H:%M:%S) | ${SECONDS}s] ${label} ==="
    if python "scripts/${script}" "$@"; then
        echo "=== done: ${label} ==="
    else
        echo "${label}" >> "${FAILLOG}"
        echo "!!! FAILED (continuing): ${label}"
    fi
}

# tag -> model path + prompt args, applied uniformly across S1-S3
declare -A MODEL PROMPT BATCH DTYPE
MODEL[qwen3-0.6b-base]="Qwen/Qwen3-Embedding-0.6B";      PROMPT[qwen3-0.6b-base]="name:query"
MODEL[ctrl-cas-6145]="models/ctrl-cas-6145/final";        PROMPT[ctrl-cas-6145]="name:query"
MODEL[ctrl-llm-6145]="models/ctrl-llm-6145/final";        PROMPT[ctrl-llm-6145]="name:query"
MODEL[qwen3-0.6b-p2-mixed]="models/qwen3-0.6b-p2-mixed/final"; PROMPT[qwen3-0.6b-p2-mixed]="name:query"
MODEL[qwen3-0.6b-p2-instr]="models/qwen3-0.6b-p2-instr/final"; PROMPT[qwen3-0.6b-p2-instr]="raw:${V2_INSTR}"
MODEL[qwen3-0.6b-cas-cmnrl]="models/qwen3-0.6b-cas-cmnrl/final"; PROMPT[qwen3-0.6b-cas-cmnrl]="name:query"
MODEL[qwen3-4b]="Qwen/Qwen3-Embedding-4B";                PROMPT[qwen3-4b]="name:query"
MODEL[mpnet]="sentence-transformers/all-mpnet-base-v2";   PROMPT[mpnet]="none"
for t in qwen3-0.6b-base ctrl-cas-6145 ctrl-llm-6145 qwen3-0.6b-p2-mixed \
         qwen3-0.6b-p2-instr qwen3-0.6b-cas-cmnrl; do BATCH[$t]=128; DTYPE[$t]=bfloat16; done
BATCH[qwen3-4b]=64;  DTYPE[qwen3-4b]=bfloat16
BATCH[mpnet]=256;    DTYPE[mpnet]=float32

# mpnet keeps its native max_seq_length (384); Qwen models use house 1024
seqlen_args() { [[ "$1" == "mpnet" ]] && echo "" || echo "--max-seq-length 1024"; }
# NUL-delimited (the V2 raw prompt contains a newline; read back with
# `mapfile -t -d ''`)
prompt_args() {
    case "${PROMPT[$1]}" in
        name:*) printf -- '--query-prompt-name\0%s\0' "${PROMPT[$1]#name:}" ;;
        raw:*)  printf -- '--query-prompt\0%s\0'      "${PROMPT[$1]#raw:}" ;;
        none)   ;;
    esac
}

ALL_TAGS=(qwen3-0.6b-base ctrl-cas-6145 ctrl-llm-6145 qwen3-0.6b-p2-mixed \
          qwen3-0.6b-p2-instr qwen3-0.6b-cas-cmnrl mpnet qwen3-4b)

# ---------------- S1: MELD, all models -------------------------------------
for tag in "${ALL_TAGS[@]}"; do
    mapfile -t -d '' PARGS < <(prompt_args "$tag")
    run_step "meld/${tag}" eval_meld.py \
        --model "${MODEL[$tag]}" --device cuda --model-dtype "${DTYPE[$tag]}" \
        --batch-size "${BATCH[$tag]}" $(seqlen_args "$tag") "${PARGS[@]}" \
        --output "results/meld_${tag}.json"
done

# ---------------- S2: BRIGHT theoremqa_theorems, all models ----------------
for tag in "${ALL_TAGS[@]}"; do
    mapfile -t -d '' PARGS < <(prompt_args "$tag")
    run_step "bright-tt/${tag}" eval_bright.py --split theoremqa_theorems \
        --model "${MODEL[$tag]}" --device cuda --model-dtype "${DTYPE[$tag]}" \
        --batch-size "${BATCH[$tag]}" $(seqlen_args "$tag") "${PARGS[@]}" \
        --output "results/bright_theoremqa_theorems_${tag}.json"
done

# ---------------- S3: BRIGHT aops, all models (4B last: most expensive) ----
for tag in "${ALL_TAGS[@]}"; do
    mapfile -t -d '' PARGS < <(prompt_args "$tag")
    run_step "bright-aops/${tag}" eval_bright.py --split aops \
        --model "${MODEL[$tag]}" --device cuda --model-dtype "${DTYPE[$tag]}" \
        --batch-size "${BATCH[$tag]}" $(seqlen_args "$tag") "${PARGS[@]}" \
        --output "results/bright_aops_${tag}.json"
done

# ---------------- S4: BRIGHT-official-instruction references ---------------
run_step "bright-tt/qwen3-0.6b-base-brightinstr" eval_bright.py \
    --split theoremqa_theorems --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --batch-size 128 --max-seq-length 1024 \
    --query-prompt "${BRIGHT_TT_INSTR}" \
    --output results/bright_theoremqa_theorems_qwen3-0.6b-base-brightinstr.json
run_step "bright-aops/qwen3-0.6b-base-brightinstr" eval_bright.py \
    --split aops --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda --model-dtype bfloat16 --batch-size 128 --max-seq-length 1024 \
    --query-prompt "${BRIGHT_AOPS_INSTR}" \
    --output results/bright_aops_qwen3-0.6b-base-brightinstr.json
run_step "bright-tt/qwen3-4b-brightinstr" eval_bright.py \
    --split theoremqa_theorems --model Qwen/Qwen3-Embedding-4B \
    --device cuda --model-dtype bfloat16 --batch-size 64 --max-seq-length 1024 \
    --query-prompt "${BRIGHT_TT_INSTR}" \
    --output results/bright_theoremqa_theorems_qwen3-4b-brightinstr.json

# ---------------- S5: summary table + prediction check ---------------------
python - <<'PY'
import json, os

TAGS = ["qwen3-0.6b-base", "ctrl-cas-6145", "ctrl-llm-6145",
        "qwen3-0.6b-p2-mixed", "qwen3-0.6b-p2-instr", "qwen3-0.6b-cas-cmnrl",
        "qwen3-4b", "mpnet"]

def get(path, *keys, default=None):
    try:
        d = json.load(open(path))
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default

print()
print("=== EXTERNAL OOD SUMMARY (evals we did NOT build) ===")
hdr = (f"{'model':24s} {'MELD R@1':>9s} {'MELD R@5':>9s} {'MELD AUChard':>12s} "
       f"{'tt NDCG@10':>11s} {'aops NDCG@10':>13s}")
print(hdr); print("-" * len(hdr))
vals = {}
for tag in TAGS:
    m_r1 = get(f"results/meld_{tag}.json", "retrieval_pairs_only", "recall@1")
    m_r5 = get(f"results/meld_{tag}.json", "retrieval_pairs_only", "recall@5")
    m_auc = get(f"results/meld_{tag}.json", "separation_auc",
                "auc_same_framing_distractors")
    tt = get(f"results/bright_theoremqa_theorems_{tag}.json", "metrics", "NDCG@10")
    ao = get(f"results/bright_aops_{tag}.json", "metrics", "NDCG@10")
    vals[tag] = dict(m_r1=m_r1, tt=tt, ao=ao)
    fmt = lambda v, w: f"{v:>{w}.2f}" if isinstance(v, (int, float)) else f"{'--':>{w}s}"
    print(f"{tag:24s} {fmt(m_r1,9)} {fmt(m_r5,9)} {fmt(m_auc,12)} "
          f"{fmt(100*tt if tt is not None else None,11)} "
          f"{fmt(100*ao if ao is not None else None,13)}")
for ref in ["qwen3-0.6b-base-brightinstr", "qwen3-4b-brightinstr"]:
    tt = get(f"results/bright_theoremqa_theorems_{ref}.json", "metrics", "NDCG@10")
    ao = get(f"results/bright_aops_{ref}.json", "metrics", "NDCG@10")
    fmt = lambda v, w: f"{100*v:>{w}.2f}" if isinstance(v, (int, float)) else f"{'--':>{w}s}"
    print(f"{ref:24s} {'':9s} {'':9s} {'':12s} {fmt(tt,11)} {fmt(ao,13)}")

print()
print("=== PRE-REGISTERED PREDICTION CHECK ===")
print("Claim: ctrl-llm (benchmark hard R@1 9.37, 60x every real model) shows")
print("NO systematic advantage over ctrl-cas on external evals.")
llm, cas = vals.get("ctrl-llm-6145", {}), vals.get("ctrl-cas-6145", {})
for name, key, scale in [("MELD pairs-only R@1", "m_r1", 1),
                         ("BRIGHT theoremqa_theorems NDCG@10", "tt", 100),
                         ("BRIGHT aops NDCG@10", "ao", 100)]:
    a, b = llm.get(key), cas.get(key)
    if a is None or b is None:
        print(f"  {name}: MISSING RESULT -- check {os.environ.get('FAILLOG','faillog')}")
    else:
        print(f"  {name}: ctrl-llm {scale*a:.2f} vs ctrl-cas {scale*b:.2f} "
              f"(delta {scale*(a-b):+.2f})")
print("Interpret deltas against the model spread in the table above; the")
print("prediction is REPLICATED if ctrl-llm's deltas are small relative to")
print("that spread (as on real duplicates: 64.4 vs 61.3 R@1), and REFUTED if")
print("ctrl-llm leads by a margin resembling its 9.37-vs-0.13 benchmark gap.")
PY

echo "== failures (empty = none): =="
cat "${FAILLOG}"
echo "== end: $(date) =="
