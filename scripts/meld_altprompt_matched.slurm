#!/bin/bash
#SBATCH --job-name=meld-altmatched
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# THE MELD PROMPT CONTROL  (round-2 review finding R2-H6)
#
# WHY. The paper's evidence that MELD's *recipe* is what buys +26.97 R@1 is the
# P-M3a contrast against ctrl-LLM (+17.78). That control differs from the attack
# arm in FOUR ways at once -- task (competition-problem paraphrase vs
# cross-subfield dialect restatement), source data (MathNet anchors vs a domain
# pair + prose connection), budget (6,145 vs 3,150 rows) and negative design
# (minimal-edit corruptions vs same-framing statements from the generated pool).
# Beating it therefore shows that task-matched data beats task-mismatched data.
# It does NOT separate "MELD's recipe" from "cross-dialect equivalence training
# in general", and the paper's own P-M4 already shows a topic-disjoint arm
# recovering 73% of the effect.
#
# This is the missing rung, built the way the MathNet ladder is built: the SAME
# nine MELD domain pairs, the SAME output schema, generator, judge, gates, row
# budget and negatives-per-row -- and a deliberately different authorial
# framing (MELD_GEN_SYSTEM_ALT: a graduate reading-course translation exercise
# rather than a research mathematician building an evaluation set, and WITHOUT
# the lexical-distinctness instruction that is the clause of MELD's procedure
# most specific to what its benchmark measures). Only the prompt varies.
#
# PRE-REGISTERED READINGS, fixed before this job was submitted and before any
# meld9alt output existed. Reference points, all already measured:
#   base 10.19 | meld9 37.16 +-1.19 (three seeds) | heldout 30.80 +-1.12
#   ctrl-LLM 18.89 | ctrl-CAS 9.44
# Let A = meld9alt's pairs-only R@1 at seed 42.
#   C1  RECIPE-SPECIFIC. A <= 30.80 (i.e. the alt prompt does no better than
#       the topic-disjoint arm) AND meld9 - A >= 3.6 (the MDE floor from
#       results/meld_attack_power.json). Reading: the wording of MELD's
#       procedure carries a measurable part of the effect, and "MELD's own
#       recipe buys more than generic cross-dialect training" survives.
#   C2  GENRE-ONLY. |meld9 - A| < 3.6 with a paired CI containing zero.
#       Reading: the effect is cross-dialect equivalence training as such. The
#       paper must then DROP any recipe-specific reading of the MELD result,
#       keep it as a genre/scope result, and say the reconstruction's wording
#       did not matter -- which also removes the last support for treating the
#       MELD attack as evidence about *recipes* rather than about *formats*.
#   C3  Anything between is reported as a gradient, exactly as the MathNet
#       ladder's partial outcomes were.
# Whichever lands is reported. C2 is a real possibility and would weaken a
# claim we currently make; that is the point of running it.
#
# COST. Generation 24 rounds x 9 domains (~45 min on one A100 at the measured
# throughput of job 49639491), screen + gates ~5 min, training ~11 min at 3,150
# rows, evals ~4 min. Under 1.5 h; the 8 h cap is slack for a queue restart.
#
# Submit:  sbatch scripts/meld_prompt_control.slurm
# ============================================================================

set -euo pipefail
# 2026-09-17: the 84/16 prompt control regenerated with meld9's exact
# generation procedure (greedy first round, --target-pairs stop rule, same round cap), so
# the only difference from meld9 is the prompt. Group meld9altm = MELD9 domain pairs + ALT prompt.

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

GRP=meld9altm
ROWCAP=3150            # identical to attack_second_benchmark.slurm
ROUNDS=24              # meld9's round cap (matched generation)
FAILLOG="results/meld_altprompt_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"

echo "== node: $(hostname) =="
echo "== start: $(date) =="

soft () { echo "== [${SECONDS}s] $1 =="; shift; "$@" || echo "FAILED: $*" | tee -a "${FAILLOG}"; }

# ---------------------------------------------------------------------------
# STAGE A — generation under the ALT prompt (vllm env)
# ---------------------------------------------------------------------------
conda activate vllm
# The two env vars the meld9/heldout generation ran under. Omitting the second
# is what killed jobs 49677956 and 49678119 after ~75 s: flashinfer's top-p
# sampler JIT-compiles and there is no nvcc on these nodes. Setting them here
# also keeps this arm on the SAME sampling path as the arm it controls for.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -s "data/meld_attack/raw_${GRP}.jsonl" ]]; then
    echo "== [${SECONDS}s] STAGE A skipped: raw_${GRP}.jsonl exists =="
else
    echo "== [${SECONDS}s] STAGE A generate ${GRP} (${ROUNDS} rounds, ALT prompt) =="
    python scripts/generate_meld_pairs.py --stage generate \
        --backend vllm --model Qwen/Qwen3-32B-AWQ \
        --group "${GRP}" --rounds "${ROUNDS}" --target-pairs 4000 --first-round-greedy --device cuda
fi
conda deactivate

# ---------------------------------------------------------------------------
# STAGE B — MELD's own screen + the eval-disjointness gates, identical settings
# ---------------------------------------------------------------------------
conda activate mathnet
if [[ -s "data/meld_attack/pairs_${GRP}.jsonl" ]]; then
    echo "== [${SECONDS}s] STAGE B skipped: pairs_${GRP}.jsonl exists =="
else
    soft "STAGE B gates + rows: ${GRP}" \
    python scripts/generate_meld_pairs.py --stage pairs --group "${GRP}" --device cuda
fi

N_ROWS=$(wc -l < "data/meld_attack/pairs_${GRP}.jsonl" 2>/dev/null || echo 0)
echo "== ${GRP}: ${N_ROWS} trainer rows generated (cap ${ROWCAP}) =="
if (( N_ROWS < ROWCAP )); then
    echo "WARNING: ${GRP} yielded ${N_ROWS} < ${ROWCAP} rows, so the budget is NOT" \
         "matched to meld9's 3,150 and the comparison is confounded by volume." \
         "Report it, exactly as D3's 9% deficit was reported." | tee -a "${FAILLOG}"
fi

# ---------------------------------------------------------------------------
# STAGE C — training. Byte-identical to the meld9 arm except --train-file.
# ---------------------------------------------------------------------------
OUT="models/meld-attack-${GRP}"
if [[ -d "${OUT}/final" ]]; then
    echo "== [${SECONDS}s] STAGE C skipped: ${OUT}/final exists =="
else
    echo "== [${SECONDS}s] STAGE C train ${GRP} at ${ROWCAP} rows =="
    if ! python scripts/train_invarembed.py \
        --train-file "data/meld_attack/pairs_${GRP}.jsonl" --max-rows "${ROWCAP}" \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${OUT}"
    then
        echo "FAILED: training ${GRP}" | tee -a "${FAILLOG}"
        [[ -e "${OUT}" ]] && mv "${OUT}" "${OUT}.FAILED-${SLURM_JOB_ID:-local}"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# STAGE D — evaluation, matched prompt condition, plus P-M2 and P-M3b
# ---------------------------------------------------------------------------
soft "${GRP}: eval_meld" \
python scripts/eval_meld.py --model "${OUT}/final" --device cuda \
    --model-dtype bfloat16 --max-seq-length 1024 --batch-size 64 \
    --query-prompt-name query \
    --output "results/meld_atk_attack_${GRP}.json" \
    --per-query-out "results/meld_pq_attack_${GRP}.json"

soft "${GRP}: eval_crosslingual (P-M2)" \
python scripts/eval_crosslingual.py --model "${OUT}/final" \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --max-seq-length 1024 \
    --output "results/crosslingual_meld-attack-${GRP}.json"

soft "${GRP}: eval_retrieve hard (P-M3b)" \
python scripts/eval_retrieve.py --tier hard --model "${OUT}/final" \
    --device cuda --model-dtype bfloat16 --query-prompt-name query \
    --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
    --output "results/eval_hard_meld-attack-${GRP}.json"

for REF in base attack_meld9 attack_heldout attack_meld9alt; do
    if [[ -s "results/meld_pq_attack_${GRP}.json" && -s "results/meld_pq_${REF}.json" ]]; then
        soft "compare ${GRP} vs ${REF}" \
        python scripts/generate_meld_pairs.py --stage compare \
            --compare "results/meld_pq_attack_${GRP}.json" "results/meld_pq_${REF}.json" \
            --out "results/meld_compare_attack_${GRP}_vs_${REF}.json"
    fi
done

# ---------------------------------------------------------------------------
# STAGE E — the pre-registered verdict, evaluated by the script, not by hand
# ---------------------------------------------------------------------------
echo "=== [${SECONDS}s] PROMPT-CONTROL VERDICT ==="
true \
    || echo "FAILED: meld_prompt_control_verdict.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== meld-altprompt finished: $(date) =="
