#!/bin/bash
#SBATCH --job-name=saber-attack
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# The SABER-Math recipe-matching attack (chained, 1x A100, <24 h).
#
# Does the MathNet circularity finding generalize to a GENRE problem with
# LLM-constructed retrieval benchmarks? We manufacture training pairs with
# SABER-Math's OWN construction recipe (their verbatim core-idea prompt +
# their verbatim summary-Jaccard relevance logic, tau=0.211) via
# Qwen3-32B-AWQ — a DIFFERENT generator than their GPT-OSS-120B, exactly as
# the MathNet attack used Qwen3-32B against GPT-5-built eval data — on 4,917
# source problems verified DISJOINT from their eval texts
# (results/saber_disjointness.json: 5-gram containment gates vs all 1,000
# queries AND all 71,117 candidate docs + numeric-fingerprint gate; built
# 2026-07-29 on the login node from data/pairs/source_ids.txt).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS (written 2026-07-29, BEFORE this job ever ran;
# the attacked model has never been trained or evaluated at registration
# time):
#  LABEL SCHEME (renamed 2026-08-23, content unchanged): this registration's
#  predictions were originally written as P1..Pn. They are now namespaced
#  P-S* because three registrations (SABER, MELD, dose ladder) each used
#  P1/P2. The renaming is 1:1 and changes no threshold, clause or verdict;
#  git history preserves the original labels.
#  P-S1 on-genre gaming: saber-attack-6145 gains >= +0.02 overall nDCG@10
#     (statement-full) over base Qwen3-Embedding-0.6B evaluated identically,
#     and its mean per-query Spearman with the summary-Jaccard construction
#     signal at least doubles vs base.
#  P-S2 no real ability: its real-duplicate strict R@1 (eval_crosslingual)
#     stays BELOW base (84.73) and BELOW V2 (74.30) — i.e. in ctrl-arm
#     territory — so the SABER jump is not explained by any gain in real
#     equivalence ability (sibling MELD/duplicate evals give the same test).
#  P-S3 recipe specificity: ctrl-llm-6145 (the MathNet-recipe attack model)
#     shows NO comparable SABER gain: |nDCG(ctrl-llm) - nDCG(ctrl-cas)| is
#     smaller than the attacked model's delta over base. Each benchmark is
#     gamed by ITS OWN recipe, not by generic synthetic training.
#  FALSIFIER: if the attacked model lands within noise of base on SABER,
#     the genre-generalization claim FAILS and only the MathNet-specific
#     circularity claim stands. Report either way.
#
# HONEST FIDELITY LIMITS (documented in generate_saber_pairs.py):
#  * light analog — replicates the summary-Jaccard branch of their recipe
#    (2/3 of candidate mass; their Fig. 4 shows summary-selected candidates
#    rank >= topic-selected under the final ranking) but NOT the MathWorld
#    ontology BMA signal nor the Swiss tournament;
#  * tau=0.211 was tuned by them on a 283K corpus; our mining pool is 4,917
#    problems, so pair density differs (reported in saber_pairs_summary);
#  * cross-language duplicate caveat for the DOCUMENT gate (see
#    saber_disjointness.json); the load-bearing QUERY gate has a
#    language-independent numeric-fingerprint backstop.
#
# CHAIN: [vllm env] summaries smoke+full -> [mathnet env] pairs -> train
# 0.6B (EXACT controlled-experiment hyperparameters, --max-rows 6145) ->
# eval_saber on attacked + base + V2 + ctrl-llm + ctrl-cas + 3 stock
# baselines (published anchors: base 0.575, e5-large 0.488, bge-m3 0.511)
# -> eval_crosslingual on attacked. Skip-on-failure per eval step; summary
# table at the end.
#
# Submit: sbatch scripts/saber_attack.slurm
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
GENMODEL="${GENMODEL:-Qwen/Qwen3-32B-AWQ}"
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p data/saber_attack results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="

if [[ ! -s data/saber_attack/source_ids.txt ]]; then
    echo "FATAL: data/saber_attack/source_ids.txt missing — run"
    echo "       python scripts/generate_saber_pairs.py --stage filter  (login node) first"
    exit 1
fi

# ---------------------------------------------------------------------------
# STAGE A — SABER-style core-idea summaries via Qwen3-32B-AWQ (vllm env)
# ---------------------------------------------------------------------------
if [[ -s data/saber_attack/summaries.jsonl ]]; then
    echo "== [${SECONDS}s] STAGE A skipped: summaries.jsonl already exists =="
else
    conda activate vllm
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on Ibex nodes (see generate_llm_pairs.slurm)

    echo "== [${SECONDS}s] STAGE A smoke: 30 problems =="
    python scripts/generate_saber_pairs.py --stage summaries \
        --backend vllm --model "${GENMODEL}" --max-problems 30 \
        --summaries-file data/saber_attack/summaries_smoke.jsonl
    N_SMOKE=$(wc -l < data/saber_attack/summaries_smoke.jsonl)
    if (( N_SMOKE < 20 )); then
        echo "FATAL: smoke produced only ${N_SMOKE}/30 summaries — prompt/JSON issue"
        exit 1
    fi

    echo "== [${SECONDS}s] STAGE A full: all filtered sources =="
    python scripts/generate_saber_pairs.py --stage summaries \
        --backend vllm --model "${GENMODEL}" \
        --summaries-file data/saber_attack/summaries.jsonl
    conda deactivate
fi

# ---------------------------------------------------------------------------
# STAGE B — recipe mining (their Jaccard logic) + training (mathnet env)
# ---------------------------------------------------------------------------
conda activate mathnet

echo "== [${SECONDS}s] STAGE B pairs: their tokenizer + tau=0.211 =="
python scripts/generate_saber_pairs.py --stage pairs \
    --summaries-file data/saber_attack/summaries.jsonl \
    --pairs-file data/saber_attack/pairs.jsonl

echo "== [${SECONDS}s] STAGE B train: EXACT controlled-experiment recipe =="
# Hyperparameters byte-identical to train_controlled_exp.slurm (only
# --train-file and --output-dir differ) => saber-attack-6145 is directly
# comparable to ctrl-cas-6145 / ctrl-llm-6145 at the same 6,145-row budget.
python scripts/train_invarembed.py \
    --train-file data/saber_attack/pairs.jsonl --max-rows 6145 \
    --model Qwen/Qwen3-Embedding-0.6B \
    --model-dtype bfloat16 --bf16 \
    --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
    --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
    --dev-frac 0.05 --eval-steps 100 \
    --query-prompt-name query \
    --max-seq-length 1024 \
    --output-dir models/saber-attack-6145
# NOTE (job 49589998 post-mortem): without --max-seq-length the model default
# is 32768; SABER docs are long (problem+solution) -> OOM at batch 256.
# CAS/LLM-arm texts are short, which is why the controlled arms trained fine
# uncapped. 1024 matches phase-1 training and all eval configs.

# ---------------------------------------------------------------------------
# STAGE C — SABER-Math evals (attacked first, then anchors + baselines)
# ---------------------------------------------------------------------------
run_saber () {  # <output-name> <model> [extra flags...]
    local name="$1" model="$2"; shift 2
    echo "== [${SECONDS}s] eval_saber ${name} =="
    python scripts/eval_saber.py --model "${model}" --device cuda \
        --batch-size 64 --setting statement-full \
        --emb-cache-dir .emb_cache \
        --output "results/saber_${name}.json" "$@" \
        || echo "== STEP FAILED (continuing): saber_${name} =="
}

# our models mirror their training-time query instruction; doc side bare
run_saber attack       models/saber-attack-6145/final --model-dtype bfloat16 --query-prompt-name query
run_saber attack_noprompt models/saber-attack-6145/final --model-dtype bfloat16

echo "== [${SECONDS}s] eval_crosslingual saber-attack (P-S2: real equivalence) =="
python scripts/eval_crosslingual.py --model models/saber-attack-6145/final \
    --device cuda --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
    --output results/crosslingual_saber-attack.json \
    || echo "== STEP FAILED (continuing): crosslingual_saber-attack =="

# anchors: their protocol = NO prompts (models.txt); published 0.575
run_saber base0.6b_noprompt   Qwen/Qwen3-Embedding-0.6B --model-dtype bfloat16
run_saber base0.6b_queryprompt Qwen/Qwen3-Embedding-0.6B --model-dtype bfloat16 --query-prompt-name query

# our campaign models
run_saber p2instr  models/qwen3-0.6b-p2-instr/final --model-dtype bfloat16 \
    --query-prompt $'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'
run_saber ctrl_llm models/ctrl-llm-6145/final --model-dtype bfloat16 --query-prompt-name query
run_saber ctrl_cas models/ctrl-cas-6145/final --model-dtype bfloat16 --query-prompt-name query

# stock baselines (published anchors: e5-large 0.488 chunked, bge-m3 0.511)
run_saber e5large intfloat/multilingual-e5-large --chunk-long-docs
run_saber bgem3   BAAI/bge-m3
run_saber mpnet   sentence-transformers/all-mpnet-base-v2 --chunk-long-docs

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo "=== SABER-ATTACK SUMMARY (overall statement-full nDCG@10) ==="
python - <<'PY'
import json, os
rows = [
    ("saber-attack-6145 (+query prompt)", "attack"),
    ("saber-attack-6145 (no prompt)",     "attack_noprompt"),
    ("base 0.6B no prompt [paper 0.575]", "base0.6b_noprompt"),
    ("base 0.6B query prompt",            "base0.6b_queryprompt"),
    ("V2 p2-instr (honest champion)",     "p2instr"),
    ("ctrl-llm-6145 (MathNet-recipe)",    "ctrl_llm"),
    ("ctrl-cas-6145 (verified arm)",      "ctrl_cas"),
    ("multilingual-e5-large [paper 0.488]", "e5large"),
    ("BGE-m3 [paper 0.511]",              "bgem3"),
    ("all-mpnet-base-v2",                 "mpnet"),
]
print(f"  {'model':42s} {'nDCG@10':>8s} {'rho_jacc':>9s} {'rho_bma':>8s} {'rho_rel':>8s}")
for label, name in rows:
    p = f"results/saber_{name}.json"
    if not os.path.exists(p):
        print(f"  {label:42s}  MISSING")
        continue
    r = json.load(open(p))["settings"]["statement-full"]
    c = r.get("score_correlations_spearman_mean", {})
    if not isinstance(c, dict):
        c = {}
    print(f"  {label:42s} {r['ndcg@10']:8.4f} "
          f"{c.get('with_summary_jaccard_signal', float('nan')):9.4f} "
          f"{c.get('with_topic_bma_signal', float('nan')):8.4f} "
          f"{c.get('with_final_relevance', float('nan')):8.4f}")
xp = "results/crosslingual_saber-attack.json"
if os.path.exists(xp):
    x = json.load(open(xp))
    ov = x.get("strict_crosslingual_gold", {})
    print(f"\n  attacked real-duplicate strict R@1/5/10 (P-S2): "
          f"{ov.get('recall@1')}/{ov.get('recall@5')}/{ov.get('recall@10')}")
    print("  refs: base 84.73 R@1 | V2 74.30 | ctrl-llm 64.38 | ctrl-cas 61.32")
print("\n  pre-registered: P-S1 attack >= base+0.02 & rho_jacc doubles; "
      "P-S2 attacked xling R@1 < V2; P-S3 |ctrl-llm - ctrl-cas| < attack delta")
PY
echo "== saber-attack job finished =="
