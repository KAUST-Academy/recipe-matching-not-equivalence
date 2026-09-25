#!/bin/bash
#SBATCH --job-name=mathnet-baselines
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# Trained-baseline evaluation on MathNet-Retrieve (InvarEmbed project).
# House style follows scripts/eval_retrieve.slurm.
#
# WHAT THIS JOB RUNS (all via scripts/eval_retrieve.py, skip-on-failure):
#  S1  Qwen/Qwen3-Embedding-4B rerun, corrected config: NO query instruction,
#      max_seq_length 2048 (v2 paper reports 14.76/83.53/92.13 easy; our
#      2026-07-29 run with the "query" instruction + seqlen 1024 got
#      11.95/82.03/90.77). The paper documents no instruction ("similarities
#      between problem statements", Sec 4.1/4.2; Appendix F has only
#      generation/grading prompts), so no-instruction is the most faithful
#      candidate. Truncation is a near-null factor: only 1 of 117,088 corpus
#      docs exceeds 1024 tokens (p99.9 = ~323 tokens).
#  S2  Qwen3-4B diagnostic: WITH the registered "query" instruction at
#      seqlen 2048, easy tier only. Together with S1 and the 2026-07-29 run
#      this isolates instruction-vs-truncation as the cause of the gap.
#  S3  uw-math-ai/MathLeap-Qwen-8B    (MELD paper; base Qwen3-Embedding-8B)
#  S4  uw-math-ai/MathLeap-Octen-8B   (MELD paper; base Octen-Embedding-8B)
#  S5  Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical (best BRIGHT RaDeR)
#  S6  reasonir/ReasonIR-8B           (cc-by-nc-4.0, research use)
#  S7  Qwen/Qwen3-Embedding-8B        (zero-shot base control for MathLeap:
#      isolates what MELD fine-tuning added)          [time-guarded]
#  S8  Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes
#      (RaDeR trained on all query types, largest data) [time-guarded]
#  S9  Cross-lingual eval (data/crosslingual_eval, built by sibling agent;
#      guarded by a directory-exists check) for the best model by easy R@1.
#  S10 Results summary table echoed at the end.
#
# PER-MODEL ENCODING CONVENTIONS (faithful to each model's released usage):
#  * MathLeap-*: native sentence-transformers repos; registered prompts
#    "query" ("Instruct: Given a web search query, ...\nQuery:") and
#    "document" (per config_sentence_transformers.json). NOTE: their shipped
#    sentence_bert_config.json truncates at 128 tokens (their training
#    length); we evaluate at 2048 so the baseline is not crippled on
#    longer problems -- deviation is recorded in the result JSONs.
#  * RaDeR-*: plain-transformers checkpoints, NO sentence-transformers
#    config. Released eval code (github.com/Debrup-61/RaDeR retrievers.py,
#    retrieval_trainedmodel) encodes "query: {instr}{text}<eos>" /
#    "document: {text}<eos>" with last-hidden-state-of-EOS pooling + L2
#    norm; model card serves via vLLM with pooling_type LAST. Our
#    --pooling lasttoken --append-eos path reproduces their encoding to
#    float precision (verified 1e-7 max abs diff on Qwen2.5-0.5B).
#    The gte variant needs trust_remote_code (bidirectional gte attention,
#    is_causal=false via auto_map to Alibaba-NLP/gte-Qwen2-7B-instruct).
#    Their BRIGHT eval prepends the benchmark's task instruction to
#    queries; we use ReasonIR's released math instruction (below) so both
#    trained baselines see the same task description.
#  * ReasonIR-8B: native ST integration (custom bidirectional Llama,
#    trust_remote_code required; bf16 per model card). GritLM-style
#    prompts from their released BRIGHT configs (aops.json):
#      query: "<|user|>\n{instruction}\n<|embed|>\n"   doc: "<|embed|>\n"
#  * Dtype: bf16 everywhere (RaDeR ships fp32 weights; bf16 halves load
#    time/VRAM and is standard for retrieval eval -- recorded in JSONs).
#  * VRAM: 4B bf16 ~8 GB, 7-8B bf16 ~16 GB weights; batch 32 x 2048 tokens
#    of activations is far below the 80 GB of an A100 -- no sharding needed.
#
# RUNTIME BUDGET (~7.5 h nominal on 1x A100, 10 h wall):
#  The three tiers share byte-identical corpus.jsonl AND queries.jsonl
#  (md5-verified), so --emb-cache-dir makes tier 2/3 of each model nearly
#  free (retrieval matmul only). Per 7-8B model: ~55-60 min for all tiers;
#  Qwen3-4B: ~30 min. Model downloads: ~120 GB total to $HF_HOME.
#
# Submit with:  sbatch scripts/eval_baselines.slurm
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet

export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT}"
mkdir -p results
EMB_CACHE="${PROJECT}/.emb_cache"          # ~12 GB of cached embeddings
FAILLOG="results/eval_baselines_failures_${SLURM_JOB_ID:-local}.log"
: > "${FAILLOG}"

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="

# The one shared task instruction for the two instruction-trained baselines
# (wording from ReasonIR's released BRIGHT math config, evaluation/bright/
# configs/reasonir/aops.json).
MATH_INSTR="Given a Math problem, retrieve relevant examples that help answer the problem"

run_eval() {                       # run_eval <tag> <tier> <extra args...>
    local tag="$1" tier="$2"; shift 2
    local out="results/eval_${tier}_${tag}.json"
    echo "=== [$(date +%H:%M:%S) | ${SECONDS}s] ${tag} / ${tier} ==="
    if python scripts/eval_retrieve.py --tier "${tier}" --device cuda \
            --emb-cache-dir "${EMB_CACHE}" --output "${out}" "$@"; then
        echo "=== done: ${tag} / ${tier} ==="
    else
        echo "${tag}/${tier}" >> "${FAILLOG}"
        echo "!!! FAILED (continuing): ${tag} / ${tier}"
    fi
}

# ---------------- S1: Qwen3-4B corrected config (no instruction) -----------
for TIER in easy medium hard; do
    run_eval qwen3-4b-noinstr "${TIER}" \
        --model Qwen/Qwen3-Embedding-4B \
        --model-dtype bfloat16 --batch-size 64 --max-seq-length 2048
done

# ---------------- S2: Qwen3-4B diagnostic (instruction, 2048), easy only ---
run_eval qwen3-4b-instr2048 easy \
    --model Qwen/Qwen3-Embedding-4B \
    --model-dtype bfloat16 --batch-size 64 --max-seq-length 2048 \
    --query-prompt-name query

# ---------------- S3/S4: MathLeap trained baselines (MELD paper) -----------
for TIER in easy medium hard; do
    run_eval mathleap-qwen-8b "${TIER}" \
        --model uw-math-ai/MathLeap-Qwen-8B \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --query-prompt-name query --doc-prompt-name document
done
for TIER in easy medium hard; do
    run_eval mathleap-octen-8b "${TIER}" \
        --model uw-math-ai/MathLeap-Octen-8B \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --query-prompt-name query --doc-prompt-name document
done

# ---------------- S5: RaDeR gte-Qwen2-7B (best BRIGHT variant) -------------
for TIER in easy medium hard; do
    run_eval rader-gte-qwen2-7b "${TIER}" \
        --model Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --pooling lasttoken --append-eos --trust-remote-code \
        --query-prompt $'query: '"${MATH_INSTR}"$'\n' \
        --doc-prompt 'document: '
done

# ---------------- S6: ReasonIR-8B ------------------------------------------
for TIER in easy medium hard; do
    run_eval reasonir-8b "${TIER}" \
        --model reasonir/ReasonIR-8B \
        --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
        --trust-remote-code \
        --query-prompt $'<|user|>\n'"${MATH_INSTR}"$'\n<|embed|>\n' \
        --doc-prompt $'<|embed|>\n'
done

# ---------------- S7: Qwen3-Embedding-8B base control (time-guarded) -------
if (( SECONDS < 25200 )); then      # < 7 h elapsed
    for TIER in easy medium hard; do
        run_eval qwen3-8b-base "${TIER}" \
            --model Qwen/Qwen3-Embedding-8B \
            --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
            --query-prompt-name query
    done
else
    echo "== skipping S7 qwen3-8b-base (elapsed ${SECONDS}s > 7h) =="
fi

# ---------------- S8: RaDeR Qwen2.5-7B all-query-types (time-guarded) ------
if (( SECONDS < 28800 )); then      # < 8 h elapsed
    for TIER in easy medium hard; do
        run_eval rader-qwen25-7b "${TIER}" \
            --model Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes \
            --model-dtype bfloat16 --batch-size 32 --max-seq-length 2048 \
            --pooling lasttoken --append-eos \
            --query-prompt $'query: '"${MATH_INSTR}"$'\n' \
            --doc-prompt 'document: '
    done
else
    echo "== skipping S8 rader-qwen25-7b (elapsed ${SECONDS}s > 8h) =="
fi

# ---------------- S9: cross-lingual eval for the best baseline -------------
XLING_DIR="${PROJECT}/data/crosslingual_eval"
if [ -d "${XLING_DIR}" ] && [ -f "${XLING_DIR}/corpus.jsonl" ] \
        && [ -f "${XLING_DIR}/qrels/test.tsv" ] && (( SECONDS < 32400 )); then
    BEST_TAG=$(python - <<'PY' || true
import json, os
tags = ["qwen3-4b-noinstr", "qwen3-4b-instr2048", "mathleap-qwen-8b",
        "mathleap-octen-8b", "rader-gte-qwen2-7b", "reasonir-8b",
        "qwen3-8b-base", "rader-qwen25-7b"]
best, best_r1 = "", -1.0
for t in tags:
    p = f"results/eval_easy_{t}.json"
    if os.path.exists(p):
        r1 = json.load(open(p))["overall"]["recall@1"]
        if r1 > best_r1:
            best, best_r1 = t, r1
print(best)
PY
)
    echo "== best baseline by easy R@1: ${BEST_TAG} =="
    XARGS=()
    case "${BEST_TAG}" in
        qwen3-4b-noinstr)   XARGS=(--model Qwen/Qwen3-Embedding-4B --batch-size 64) ;;
        qwen3-4b-instr2048) XARGS=(--model Qwen/Qwen3-Embedding-4B --batch-size 64 --query-prompt-name query) ;;
        qwen3-8b-base)      XARGS=(--model Qwen/Qwen3-Embedding-8B --batch-size 32 --query-prompt-name query) ;;
        mathleap-qwen-8b)   XARGS=(--model uw-math-ai/MathLeap-Qwen-8B --batch-size 32 --query-prompt-name query --doc-prompt-name document) ;;
        mathleap-octen-8b)  XARGS=(--model uw-math-ai/MathLeap-Octen-8B --batch-size 32 --query-prompt-name query --doc-prompt-name document) ;;
        rader-gte-qwen2-7b) XARGS=(--model Raderspace/RaDeR_gte_Qwen2-7B_MATH_LLMq_CoT_lexical --batch-size 32 --pooling lasttoken --append-eos --trust-remote-code --query-prompt $'query: '"${MATH_INSTR}"$'\n' --doc-prompt 'document: ') ;;
        rader-qwen25-7b)    XARGS=(--model Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes --batch-size 32 --pooling lasttoken --append-eos --query-prompt $'query: '"${MATH_INSTR}"$'\n' --doc-prompt 'document: ') ;;
        reasonir-8b)        XARGS=(--model reasonir/ReasonIR-8B --batch-size 32 --trust-remote-code --query-prompt $'<|user|>\n'"${MATH_INSTR}"$'\n<|embed|>\n' --doc-prompt $'<|embed|>\n') ;;
        *) echo "no successful baseline found; skipping cross-lingual eval" ;;
    esac
    if [ "${#XARGS[@]}" -gt 0 ]; then
        # NOTE (2026-07-30): this stanza is what produced
        # results/eval_crosslingual_rader-qwen25-7b.json.INVALID -- eval_retrieve.py
        # had no self-masking, so every query retrieved its own corpus document at
        # rank 1 (R@1 0.25 / R@5 95.93). eval_retrieve.py now self-masks by default,
        # so a rerun is valid; but it reports ANY-gold recall, while the paper's
        # real-duplicate column is the STRICT cross-language metric from
        # scripts/eval_crosslingual.py. Prefer that harness for any number that goes
        # into a table (it also gives the per-language / confidence / anchor-overlap
        # breakdowns); it currently lacks --pooling/--append-eos, which RaDeR needs.
        echo "=== [$(date +%H:%M:%S)] cross-lingual eval: ${BEST_TAG} ==="
        python scripts/eval_retrieve.py --data-dir "${XLING_DIR}" --device cuda \
            --model-dtype bfloat16 --max-seq-length 2048 \
            --emb-cache-dir "${EMB_CACHE}" \
            --output "results/eval_crosslingual_${BEST_TAG}.json" \
            "${XARGS[@]}" \
            || { echo "crosslingual/${BEST_TAG}" >> "${FAILLOG}"; echo "!!! FAILED: cross-lingual"; }
    fi
else
    echo "== cross-lingual eval skipped (data/crosslingual_eval missing/incomplete or out of time) =="
fi

# ---------------- S10: results summary table -------------------------------
echo ""
echo "==================== RESULTS SUMMARY ===================="
python - <<'PY'
import json, os

rows = [
    ("qwen3-4b-instr1024 (jul29 run)", "qwen3-embedding-4b"),
    ("qwen3-4b-noinstr (corrected)",   "qwen3-4b-noinstr"),
    ("qwen3-4b-instr2048 (diag)",      "qwen3-4b-instr2048"),
    ("qwen3-8b-base",                  "qwen3-8b-base"),
    ("mathleap-qwen-8b",               "mathleap-qwen-8b"),
    ("mathleap-octen-8b",              "mathleap-octen-8b"),
    ("rader-gte-qwen2-7b",             "rader-gte-qwen2-7b"),
    ("rader-qwen25-7b",                "rader-qwen25-7b"),
    ("reasonir-8b",                    "reasonir-8b"),
]
paper = {"qwen3-embedding-4B (paper v2)": {"easy": (14.76, 83.53, 92.13),
                                           "medium": (2.91, 71.61, 87.22),
                                           "hard": (0.01, 8.40, 36.87)},
         "gemini-embedding-001 (paper v2)": {"easy": (11.36, 90.68, 96.93),
                                             "medium": (2.46, 77.07, 91.63),
                                             "hard": (0.01, 9.31, 53.67)}}
hdr = f"{'model':<34}" + "".join(f"{t+' R@1':>9}{'R@5':>7}{'R@10':>7}" for t in ("easy", "med", "hard"))
print(hdr); print("-" * len(hdr))
for name, vals in paper.items():
    line = f"{name:<34}"
    for tier in ("easy", "medium", "hard"):
        r1, r5, r10 = vals[tier]
        line += f"{r1:>9.2f}{r5:>7.2f}{r10:>7.2f}"
    print(line)
print("-" * len(hdr))
for label, tag in rows:
    line, seen = f"{label:<34}", False
    for tier in ("easy", "medium", "hard"):
        p = f"results/eval_{tier}_{tag}.json"
        if os.path.exists(p):
            o = json.load(open(p))["overall"]
            line += f"{o['recall@1']:>9.2f}{o['recall@5']:>7.2f}{o['recall@10']:>7.2f}"
            seen = True
        else:
            line += f"{'--':>9}{'--':>7}{'--':>7}"
    if seen:
        print(line)
xl = [f for f in sorted(os.listdir("results")) if f.startswith("eval_crosslingual_")]
for f in xl:
    o = json.load(open(os.path.join("results", f)))
    print(f"\ncross-lingual ({o['model']}): R@1 {o['overall']['recall@1']} "
          f"R@5 {o['overall']['recall@5']} R@10 {o['overall']['recall@10']} "
          f"({o['n_queries_evaluated']} queries, {o['n_corpus']} docs)")
PY
echo "========================================================="
if [ -s "${FAILLOG}" ]; then
    echo "!! FAILED steps (skipped gracefully):"; cat "${FAILLOG}"
else
    echo "all steps completed without failures"
fi
echo "== end: $(date)  (elapsed ${SECONDS}s) =="
