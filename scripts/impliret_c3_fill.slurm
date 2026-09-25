#!/bin/bash
#SBATCH --job-name=impliret-c3
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# C3 FILL (acceptance-review C3, round-6 audit follow-up).
#
# tab:external has models as rows (base, ctrl-CAS, ctrl-LLM, V1, V2,
# Qwen3-4B) and its caption promises "Every cell was measured; none is
# omitted." Adding the ImpliRet column therefore requires the three cells
# that were never run: V1 (qwen3-0.6b-p2-mixed), V2 (qwen3-0.6b-p2-instr,
# evaluated with its own instruction prompt, house convention per
# eval_external_ood.slurm) and Qwen3-Embedding-4B.
#
# Also repairs two manifest gaps found by the round-6 ImpliRet audit: the
# mpnet and ReasonIR rows have no per-query dumps (the 49700762 re-runs
# omitted --per-query-out). They are re-run here WITH dumps into *_pqrun.json
# files -- the verified originals (impliret_mpnet.json 12.77,
# impliret_reasonir-8b.json 19.45) are NOT overwritten; agreement to 2 dp is
# checked and reported.
#
# Conventions copied from the verified runs: bf16, cap 1,024, batch 64,
# query-side prompt only (impliret_baselines.slurm); mpnet fp32 at its native
# 384 cap and ReasonIR bf16/2,048/GritLM prompt in baselines-tf4 with --local
# (impliret_calibration.slurm). New rows read the local dump (--local,
# deterministic from the committed data/impliret_eval/ dump).
# ============================================================================

set -uo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"; export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

FAILLOG="results/impliret_c3_failures.txt"
: > "${FAILLOG}"
echo "== impliret-c3 starting: $(date) =="

conda activate mathnet
V2_INSTR=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

run_house () {  # tag, model, extra prompt args...
    local tag="$1" model="$2"; shift 2
    local out="results/impliret_${tag}.json"
    if [[ -f "${out}" ]]; then echo "== SKIP ${tag} (exists) =="; return; fi
    echo "== [${SECONDS}s] EVAL ${tag} =="
    python scripts/eval_impliret.py --model "${model}" --device cuda \
        --model-dtype bfloat16 --max-seq-length 1024 --batch-size 64 --local \
        --output "${out}" \
        --per-query-out "results/impliret_pq_${tag}.json" "$@" \
        || echo "FAILED ${tag}" | tee -a "${FAILLOG}"
}

run_house v1        models/qwen3-0.6b-p2-mixed/final  --query-prompt-name query
run_house v2        models/qwen3-0.6b-p2-instr/final  --query-prompt "${V2_INSTR}"
run_house qwen3-4b  Qwen/Qwen3-Embedding-4B           --query-prompt-name query

# --- per-query re-runs for the two dump-less verified rows (new filenames) --
if [[ ! -f results/impliret_pq_mpnet.json ]]; then
    echo "== [${SECONDS}s] EVAL mpnet (pq re-run) =="
    python scripts/eval_impliret.py --model sentence-transformers/all-mpnet-base-v2 \
        --device cuda --max-seq-length 384 --batch-size 64 --local \
        --output results/impliret_mpnet_pqrun.json \
        --per-query-out results/impliret_pq_mpnet.json \
        || echo "FAILED mpnet-pq" | tee -a "${FAILLOG}"
fi
conda deactivate

conda activate baselines-tf4
REASONIR_PROMPT=$'<|user|>\nGiven a question, retrieve relevant passages that answer the question\n<|embed|>\n'
if [[ ! -f results/impliret_pq_reasonir-8b.json ]]; then
    echo "== [${SECONDS}s] EVAL reasonir (pq re-run) =="
    python scripts/eval_impliret.py --model reasonir/ReasonIR-8B --device cuda \
        --model-dtype bfloat16 --max-seq-length 2048 --batch-size 8 --local \
        --trust-remote-code --query-prompt "${REASONIR_PROMPT}" \
        --output results/impliret_reasonir-8b_pqrun.json \
        --per-query-out results/impliret_pq_reasonir-8b.json \
        || echo "FAILED reasonir-pq" | tee -a "${FAILLOG}"
fi
conda deactivate

conda activate mathnet
python - <<'PY'
import json, os
print("== summary ==")
for tag in ("v1", "v2", "qwen3-4b"):
    p = f"results/impliret_{tag}.json"
    if os.path.exists(p):
        print(tag, json.load(open(p)).get("macro_avg"))
for orig, rerun in (("impliret_mpnet.json", "impliret_mpnet_pqrun.json"),
                    ("impliret_reasonir-8b.json", "impliret_reasonir-8b_pqrun.json")):
    po, pr = f"results/{orig}", f"results/{rerun}"
    if os.path.exists(po) and os.path.exists(pr):
        a = json.load(open(po)).get("macro_avg"); b = json.load(open(pr)).get("macro_avg")
        print(f"{orig}: original {a} vs pq-rerun {b} -> agree@2dp={round(a,2)==round(b,2)}")
PY

echo "== failures (empty = clean): =="
cat "${FAILLOG}"
echo "== finished: $(date) =="
