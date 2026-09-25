#!/bin/bash
#SBATCH --job-name=impliret-base
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# IMPLIRET: HARNESS CALIBRATION + BASELINES
#
# The cross-domain scope test needs two things before any attack is meaningful,
# and this job supplies both.
#
# 1. CALIBRATION. Every benchmark in this campaign carries a credential or an
#    explicit admission that it has none. MathNet: our all-mpnet easy row
#    matches their Table 4 to the second decimal. SABER: our base row 0.5747
#    against their published 0.575. MELD: NO calibration, which is why
#    no cross-table claim is made for it. ImpliRet can be
#    calibrated: their Table 2 reports ReasonIR, and we have ReasonIR-8B
#    locally. Their figures, nDCG@10 averaged over the two discourse styles:
#        ReasonIR   W.Know 18.88   Arithmetic 10.78   Temporal 11.25   Avg 13.64
#        Dragon+    17.46          14.61              12.66            14.91  <- their best
#    If our ReasonIR rows land near theirs, ImpliRet numbers are comparable to
#    their table. If they do not, we say so in the paper and compare only
#    across our own models -- the rule the MELD episode taught us.
#    ReasonIR needs the baselines-tf4 env (transformers 4.51.3; 5.x removed
#    Config.rope_theta) and its GritLM-style query prompt, reused verbatim from
#    the convention this campaign already validated for it.
#
# 2. BASELINES. Where do our existing models sit on a benchmark in a domain
#    they have never seen? ctrl-LLM is trained on MathNet's recipe; if it gains
#    on ImpliRet that is same-genre transfer, and if it does not, the later
#    attack arm has a clean floor to beat. This costs minutes and decides how
#    the attack is interpreted.
#
# NO ATTACK ARM HERE. Nothing is trained. This job only measures, so that the
# attack's pre-registration can be written against real reference points rather
# than guesses -- the same order the MELD attack followed.
#
# RUNTIME. 9,000 documents + 9,000 queries per model. 0.6B models ~2 min each;
# ReasonIR-8B at 2,048 tokens ~25 min. Well under the cap.
#
# Submit:  sbatch scripts/impliret_baselines.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

FAILLOG="results/impliret_baselines_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"
soft () { echo "== [${SECONDS}s] $1 =="; shift; "$@" || echo "FAILED: $*" | tee -a "${FAILLOG}"; }

echo "== node: $(hostname) =="
echo "== start: $(date) =="

# ---------------------------------------------------------------------------
# Our own models (mathnet env)
# ---------------------------------------------------------------------------
conda activate mathnet

run_ours () {   # <tag> <model> [extra flags...]
    local tag="$1" model="$2"; shift 2
    [[ -s "results/impliret_${tag}.json" ]] && { echo "== ${tag} exists, skipping =="; return; }
    soft "impliret ${tag}" \
    python scripts/eval_impliret.py --model "${model}" --device cuda \
        --model-dtype bfloat16 --max-seq-length 1024 --batch-size 64 \
        --output "results/impliret_${tag}.json" \
        --per-query-out "results/impliret_pq_${tag}.json" "$@"
}

# matched prompt condition, as everywhere else in the campaign
run_ours base       Qwen/Qwen3-Embedding-0.6B  --query-prompt-name query
run_ours ctrl_llm   models/ctrl-llm-6145/final --query-prompt-name query
run_ours ctrl_cas   models/ctrl-cas-6145/final --query-prompt-name query
# promptless base, the prompt-sensitivity reference
run_ours base_noprompt Qwen/Qwen3-Embedding-0.6B
# a general-purpose sanity row on the same protocol
run_ours mpnet      sentence-transformers/all-mpnet-base-v2

conda deactivate

# ---------------------------------------------------------------------------
# ReasonIR-8B: the calibration row (baselines-tf4 env)
# ---------------------------------------------------------------------------
conda activate baselines-tf4
REASONIR_PROMPT=$'<|user|>\nGiven a question, retrieve relevant passages that answer the question\n<|embed|>\n'
if [[ ! -s results/impliret_reasonir-8b.json ]]; then
    soft "impliret ReasonIR-8B (calibration vs their Table 2)" \
    python scripts/eval_impliret.py --model reasonir/ReasonIR-8B --device cuda \
        --model-dtype bfloat16 --max-seq-length 2048 --batch-size 8 \
        --trust-remote-code --query-prompt "${REASONIR_PROMPT}" \
        --output results/impliret_reasonir-8b.json \
        --per-query-out results/impliret_pq_reasonir-8b.json
fi
conda deactivate

# ---------------------------------------------------------------------------
# Calibration verdict + baseline table
# ---------------------------------------------------------------------------
conda activate mathnet
echo "=== [${SECONDS}s] IMPLIRET CALIBRATION + BASELINES ==="
python scripts/impliret_calibration.py || echo "FAILED: impliret_calibration.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== impliret-baselines finished: $(date) =="
