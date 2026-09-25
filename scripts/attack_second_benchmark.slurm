#!/bin/bash
#SBATCH --job-name=meld-attack
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# The SECOND synthetic-corpus recipe-matching attack (chained,
# 1x A100, <20 h). Target: MELD, from the MathLeap paper arXiv:2606.23959;
# dataset released as uw-math-ai/MELD-dataset, local copy data/external/meld/.
#
# WHY THIS TARGET. The vulnerability taxonomy currently rests on n=1 per cell
# (review finding M8). Cell A -- benchmarks whose retrieval CORPUS is itself
# LLM output -- contains only MathNet-Retrieve. MELD qualifies as the second
# member on the authors' own account (§3, p.4): they iterated over nine
# complementary-domain pairs, described the connection between the fields, and
# prompted Claude Opus 4.7 for 30 equivalent-but-lexically-distinct statement
# pairs, then manually revised them and screened all statements with GPT-5.5
# for validity, equivalence and dissimilarity. Query side, gold-document side
# and the 541 distractors are ALL LLM-authored; human effort is review of LLM
# text, not a source corpus. That is exactly the structure that broke under
# recipe-matching on MathNet-Retrieve, and exactly what SABER (organic
# problems + LLM-assigned labels, cell B) does NOT have -- where the same
#  LABEL SCHEME (renamed 2026-08-23, content unchanged): this registration's
#  predictions were originally written as P1..Pn. They are now namespaced
#  P-M* because three registrations (SABER, MELD, dose ladder) each used
#  P1/P2. The renaming is 1:1 and changes no threshold, clause or verdict;
#  git history preserves the original labels.
# attack FAILED (results/saber_attack_results, P-S1 delta -0.0127).
# Alternatives checked and rejected: MIRB mixes organic sources
# (MathStackExchange duplicates, Lean premise retrieval) = cell B/C; BRIGHT's
# math splits are organic StackExchange/AoPS = cell C; SABER is already the
# cell-B member. Full justification + quotes: generate_meld_pairs.py docstring.
#
# WHAT IS MANUFACTURED. Training pairs built with MELD's own construction
# recipe through a DIFFERENT generator (Qwen3-32B-AWQ vs their Claude Opus
# 4.7) -- the same design as the MathNet attack (Qwen3-32B against GPT-5-built
# eval data) and the SABER attack. Two matched-budget arms:
#   meld9    MELD's own nine domain pairs (maximum recipe fidelity)  = PRIMARY
#   heldout  27 authored domain pairs, no MELD subfield name reused  = SPECIFICITY
#
# DISJOINTNESS. MELD's recipe has no source corpus (its only inputs are a
# domain pair and a prose connection), so disjointness is enforced ITEM-level
# against all 1,081 released eval texts (540 pair statements + 541
# distractors) by four gates: normalized exact match; word-5-gram containment
# >= 0.35 (the SABER gate); char-8-gram Jaccard >= 0.60; embedding cosine
# >= 0.90 under all-mpnet-base-v2, an encoder that is never a base model and
# never fine-tuned in this campaign. Every gate can only DELETE the training
# pairs closest to eval items, so a P-M1 success is a lower bound on exposure.
# Counts -> results/meld_disjointness_{meld9,heldout}.json; the domain-name
# gate for the heldout arm -> results/meld_domain_disjointness.json (already
# run on the login node: 0 exact collisions, 2 substring collisions reported).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS (written 2026-07-30, BEFORE this job ever ran.
# No attacked model exists at registration time: nothing under
# models/meld-attack-* has been trained, and no results/meld_*attack* file
# exists. The only MELD numbers in hand are the campaign's published ones,
# quoted below as fixed reference points, plus results/meld_attack_power.json,
# which was computed from OFF-THE-SHELF encoders only.)
#
# PRIMARY METRIC: MELD pairs_only Recall@1 -- the MathLeap paper's own Table 2
# protocol (540 statements, each queries the other 539, gold = its partner),
# with EVERY model evaluated in the identical prompt condition
# (--query-prompt-name query). Reference rows, same harness, same condition:
#   base Qwen3-Embedding-0.6B 10.19 | ctrl-cas-6145 9.44 | ctrl-llm-6145 18.89
#   Qwen3-Embedding-4B 18.89 | best published model in their Table 2: 28.9
#
# POWER, STATED UP FRONT (results/meld_attack_power.json). MELD gives 540
# queries in 270 correlated clusters. Paired cluster-bootstrap SE between two
# off-the-shelf encoders is 1.27 R@1 points (MDE 3.55 at 80% power); the SE
# grows with how many queries the two models disagree on, and at MELD's
# realistic R@1 levels discordance caps near 200-300 queries, i.e. an MDE of
# 7.8-9.6 points. Every threshold below clears that ceiling. Effects smaller
# than ~3.6 points are undetectable on this benchmark at ANY budget, so a null
# is "no evidence", never "evidence of resistance" -- the same caveat that
# applies to the SABER negative.
#
#  P-M1 ON-TARGET GAMING (the taxonomy's positive prediction -- the attack is
#     predicted to SUCCEED here, unlike on SABER's organic corpus):
#     meld-attack-meld9 exceeds base by >= +10.0 points pairs_only R@1
#     (10.19 -> >= 20.19) in the matched prompt condition, with the paired
#     95% cluster-bootstrap CI (generate_meld_pairs.py --stage compare)
#     excluding zero.
#     Secondary, higher-power, same run: pct_pos_above_all_same_framing_
#     distractors >= base + 8.0 (54.07 -> >= 62.07) and
#     auc_same_framing_distractors >= base + 0.02 (0.8872 -> >= 0.9072).
#  P-M2 NO REAL ABILITY: the attacked model's real-duplicate strict R@1
#     (eval_crosslingual) stays BELOW base (84.73) and BELOW V2 (74.30) --
#     identical test to SABER P-S2, which held at 34.86.
#  P-M3 RECIPE SPECIFICITY, a double dissociation, both halves required:
#     (a) meld-attack-meld9 exceeds ctrl-llm-6145 (18.89 -- the generic
#         LLM-rewrite-genre control, trained on MathNet's recipe) by >= +8.0
#         points R@1, i.e. >= 26.89: MELD's OWN recipe buys more than "some
#         LLM-rewrite training". (+8.0, not +5.0, because +5 would sit under
#         the MDE at the discordance this comparison will have.)
#     (b) meld-attack-meld9 does NOT inherit MathNet-Retrieve's exposure:
#         hard-tier R@1 < 4.69 (half of ctrl-llm's 9.37).
#  P-M4 GENRE vs TEMPLATE (the meld9-vs-heldout contrast; both outcomes are
#     informative and both readings are fixed HERE so neither can be told
#     post hoc):
#     - if meld-attack-heldout ALSO clears base by >= +10.0 R@1, the exposure
#       is GENRE-level (LLM-written cross-dialect equivalence pairs), matching
#       the D1-D3 plateau of the MathNet dose panel;
#     - if heldout lands within +-3.6 of base while meld9 clears +10.0, the
#       exposure is TOPIC-level and the taxonomy claim must be narrowed to
#       "same domains, same recipe".
#     - anything between is reported as a gradient, not resolved either way.
#
#  FALSIFIER (P-M1): if |meld-attack-meld9 - base| < 3.6 points R@1 with a
#     paired 95% CI containing zero, cell A's prediction FAILS at n=2: MELD is
#     a synthetic-corpus benchmark that RESISTED recipe-matching, the taxonomy
#     reverts to a single-benchmark observation about MathNet-Retrieve, and
#     the abstract/conclusion must say so. A gain in the 3.6-10.0 band is a
#     PARTIAL success and must be reported as a gradient, not as confirmation.
#     Report whichever lands, exactly as the SABER negative was reported.
#
# HONEST FIDELITY LIMITS (must accompany any number this job produces):
#  * MELD's generation prompt is NOT published; MELD_GEN_SYSTEM is a faithful
#    reconstruction of the procedure paragraph into the released record schema
#    -- weaker fidelity than the SABER attack, which used their repo prompt
#    byte-for-byte, and stronger than nothing, since the output schema and the
#    per-call count (30) are pinned by the release and the paper;
#  * their connection descriptions are unpublished; ours are in
#    generate_meld_pairs.py (both groups);
#  * their manual review-and-revise pass is not reproducible; we keep only
#    their automated GPT-5.5 screen, run through Qwen3-32B-AWQ, gating on
#    validity + equivalence and recording the dissimilarity question;
#  * MELD ran one call per domain pair; we run --rounds calls per domain pair
#    with the already-covered topics fed back, which is a SCALE deviation;
#  * training negatives are same-framing statements from our own generated
#    pool (MELD never describes how its distractors were built);
#  * MELD is small (270 pairs / 541 distractors) -- see POWER above.
#
# CHAIN: [mathnet] domains gate -> [vllm] smoke + generate meld9 + heldout ->
# [mathnet] gates + trainer rows -> matched row cap R = min(meld9, heldout,
# 6145) -> train BOTH arms at the EXACT controlled-experiment hyperparameters
# (byte-identical to train_controlled_exp.slurm; only --train-file,
# --max-rows and --output-dir differ) -> eval on the TARGET (MELD, all models,
# matched prompt condition, per-query dumps) + on our REAL-DUPLICATE eval +
# MathNet-Retrieve hard/easy for P-M3(b) -> paired bootstrap CIs -> summary with
# the pre-registered verdicts. Skip-on-failure per eval step.
#
# Submit: sbatch scripts/attack_second_benchmark.slurm
# ============================================================================

set -euo pipefail
PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
GENMODEL="${GENMODEL:-Qwen/Qwen3-32B-AWQ}"
ROWCAP_MAX="${ROWCAP_MAX:-6145}"        # the controlled experiment's budget
MELD9_ROUNDS="${MELD9_ROUNDS:-24}"      # 9 domain pairs  -> <= 6,480 raw pairs
HELDOUT_ROUNDS="${HELDOUT_ROUNDS:-8}"   # 27 domain pairs -> <= 6,480 raw pairs
TARGET_PAIRS="${TARGET_PAIRS:-4000}"    # early stop per group (6,145 rows needs ~3,073)

source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p data/meld_attack results

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date) =="

# ---------------------------------------------------------------------------
# STAGE 0 — domain-pair table + domain-name disjointness (cheap, must pass)
# ---------------------------------------------------------------------------
conda activate mathnet
echo "== [${SECONDS}s] STAGE 0 domain gate =="
python scripts/generate_meld_pairs.py --stage domains
if [[ ! -s results/meld_attack_power.json ]]; then
    echo "== [${SECONDS}s] STAGE 0 power analysis (no attacked model involved) =="
    python scripts/generate_meld_pairs.py --stage power --device cuda
fi
conda deactivate

# ---------------------------------------------------------------------------
# STAGE A — MELD's recipe through Qwen3-32B-AWQ (vllm env), both groups
# ---------------------------------------------------------------------------
conda activate vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on Ibex nodes (see generate_llm_pairs.slurm)

# ---- smoke gate -----------------------------------------------------------
# The old gate compared `wc -l raw_smoke.jsonl` against 30 of a nominal 60 and
# killed job 49630276 at 28 pairs.  That threshold was not measuring anything:
# with 2 generation calls it cannot tell "both calls parsed, each a little
# short" from "one call parsed at full yield and the other returned garbage",
# and a single unparseable call is not by itself abnormal for this generator
# (measured per-call generation JSON failure rates on the SAME model and the
# SAME parser: 232/7089 = 3.3%, 345/7089 = 4.9%, 675/7089 = 9.5%,
# 1266/7089 = 17.9% across the four generate_llm_pairs runs in results/).  At
# n=2 calls a ~10% per-call failure rate trips a "both must be perfect" gate
# about one run in five, which is what happened.
#
# The gate now (a) takes a big enough sample to estimate the rate at all,
# (b) reads the generator's own per-call accounting instead of a row count,
# and (c) names the condition that tripped.  Conditions, all required:
#     >= 12 generation calls         (6 domain pairs x 2 rounds; also the
#                                     smallest sample where the 90% rule
#                                     tolerates one failure instead of
#                                     demanding perfection, and it exercises
#                                     the round>=2 continuation prompt)
#     >= 90% of calls parse as JSON  (11/12 passes, 10/12 is FATAL: one
#                                     unparseable call is ordinary noise at
#                                     the 3-18% rates measured above, two out
#                                     of twelve is not)
#     >= 20 pairs from the WORST successful call (of the 30 requested) --
#                                     catches truncation / prompt drift, which
#                                     a whole-run pair total would average away
SMOKE_RAW=data/meld_attack/raw_smoke.jsonl
SMOKE_SUM="${SMOKE_RAW}.summary.json"

smoke_health () {   # 0 = healthy; prints the failing condition(s) otherwise
    python - "${SMOKE_SUM}" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(f"FATAL: smoke gate: no summary at {p} (generator did not finish)")
    sys.exit(1)
s = json.load(open(p))
calls = int(s.get("n_generation_calls") or 0)
bad = []
if calls < 12:
    # 0 also lands here: a summary written before the per-call accounting
    # existed (i.e. by the pre-fix run that produced 28 pairs and died).
    bad.append(f"SAMPLE TOO SMALL — summary reports {calls} generation calls, "
               f"need >= 12; a JSON success rate cannot be estimated from "
               f"fewer" + (" (0 means the summary predates the per-call "
                           "accounting, i.e. it is a stale pre-fix file)"
                           if calls == 0 else ""))
    print("  smoke health: unusable summary "
          f"({s.get('n_pairs_generated')} pairs, {calls} calls recorded)")
else:
    fails = int(s.get("n_generation_json_failures") or 0)
    ok, rate = calls - fails, (calls - fails) / calls
    minpc = int(s.get("min_pairs_per_successful_call") or 0)
    print(f"  smoke health: {ok}/{calls} generation calls parsed "
          f"({100 * rate:.0f}%), worst successful call {minpc} pairs, "
          f"{s.get('n_pairs_generated')} pairs total, "
          f"{s.get('n_judge_json_failures')} judge JSON failures")
    if rate < 0.90:
        bad.append(f"JSON SUCCESS RATE — only {ok}/{calls} calls parsed "
                   f"({100 * rate:.0f}% < 90%); read the raw completions "
                   f"dumped to data/meld_attack/genfail_*.txt")
    if minpc < 20:
        bad.append(f"PAIRS PER CALL — worst successful call returned {minpc} "
                   f"of the {s.get('pairs_per_call')} pairs requested "
                   f"(floor 20); suspect truncation at --gen-max-tokens or "
                   f"prompt drift")
for b in bad:
    print("FATAL: smoke gate: " + b)
sys.exit(1 if bad else 0)
PY
}

# Resume must never accept a smoke file left behind by a FAILED run — the old
# `[[ -s raw_smoke.jsonl ]]` skip would have waved through job 49630276's own
# 28-pair wreckage on the very next submit.  Existence is not the test; the
# health check is, and a smoke that does not pass it is deleted and redone.
if [[ -s "${SMOKE_RAW}" && -s "${SMOKE_SUM}" ]] && smoke_health; then
    echo "== [${SECONDS}s] STAGE A smoke skipped: existing smoke passes the gate =="
else
    rm -f "${SMOKE_RAW}" "${SMOKE_SUM}" data/meld_attack/genfail_*.txt
    echo "== [${SECONDS}s] STAGE A smoke: 6 domain pairs x 2 rounds = 12 generation calls =="
    python scripts/generate_meld_pairs.py --stage generate \
        --backend vllm --model "${GENMODEL}" --group meld9 \
        --rounds 2 --max-domain-pairs 6 --first-round-greedy \
        --out "${SMOKE_RAW}"
    smoke_health || exit 1
    echo "== smoke OK =="
fi

for GRP in meld9 heldout; do
    if [[ -s "data/meld_attack/raw_${GRP}.jsonl" ]]; then
        echo "== [${SECONDS}s] STAGE A ${GRP} skipped: raw_${GRP}.jsonl exists =="
        continue
    fi
    RNDS=$([[ "${GRP}" == meld9 ]] && echo "${MELD9_ROUNDS}" || echo "${HELDOUT_ROUNDS}")
    echo "== [${SECONDS}s] STAGE A generate ${GRP} (${RNDS} rounds) =="
    python scripts/generate_meld_pairs.py --stage generate \
        --backend vllm --model "${GENMODEL}" --group "${GRP}" \
        --rounds "${RNDS}" --target-pairs "${TARGET_PAIRS}" --first-round-greedy
done
conda deactivate

# ---------------------------------------------------------------------------
# STAGE B — disjointness gates + trainer rows (mathnet env)
# ---------------------------------------------------------------------------
conda activate mathnet
for GRP in meld9 heldout; do
    echo "== [${SECONDS}s] STAGE B gates + rows: ${GRP} =="
    python scripts/generate_meld_pairs.py --stage pairs --group "${GRP}" --device cuda
done

ROWS_MELD9=$(wc -l < data/meld_attack/pairs_meld9.jsonl)
ROWS_HELDOUT=$(wc -l < data/meld_attack/pairs_heldout.jsonl)
ROWCAP=${ROWCAP_MAX}
(( ROWS_MELD9   < ROWCAP )) && ROWCAP=${ROWS_MELD9}
(( ROWS_HELDOUT < ROWCAP )) && ROWCAP=${ROWS_HELDOUT}
echo "== matched row budget: meld9 ${ROWS_MELD9}, heldout ${ROWS_HELDOUT}, cap ${ROWCAP} (controlled experiment used ${ROWCAP_MAX}) =="
if (( ROWCAP < 1000 )); then
    echo "FATAL: only ${ROWCAP} usable rows — the attack would be budget-starved, not tested"
    exit 1
fi

# ---------------------------------------------------------------------------
# STAGE C — training. Hyperparameters byte-identical to
# train_controlled_exp.slurm (no --max-seq-length override: MELD statements
# average 138 chars, so unlike the SABER attack nothing needs capping and the
# "exact controlled-experiment hyperparameters" claim is literal here).
# ---------------------------------------------------------------------------
train_arm () {  # <arm> <train-file> <outdir>
    local arm="$1" file="$2" out="$3"
    if [[ -d "${out}/final" ]]; then
        echo "== [${SECONDS}s] STAGE C ${arm} skipped: ${out}/final exists =="
        return
    fi
    echo "== [${SECONDS}s] STAGE C train ${arm} at ${ROWCAP} rows =="
    python scripts/train_invarembed.py \
        --train-file "${file}" --max-rows "${ROWCAP}" \
        --model Qwen/Qwen3-Embedding-0.6B \
        --model-dtype bfloat16 --bf16 \
        --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
        --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 \
        --dev-frac 0.05 --eval-steps 100 \
        --query-prompt-name query \
        --output-dir "${out}"
}
train_arm meld9   data/meld_attack/pairs_meld9.jsonl   models/meld-attack-meld9
train_arm heldout data/meld_attack/pairs_heldout.jsonl models/meld-attack-heldout

# ---------------------------------------------------------------------------
# STAGE D — evaluation on the TARGET. Every threshold-bearing row uses the
# SAME prompt condition (--query-prompt-name query); the extra no-prompt rows
# are the prompt-sensitivity reference that review finding M2 asks for, and
# V2's own-instruction row is context only, never part of a threshold.
# ---------------------------------------------------------------------------
V2_INSTR=$'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'

run_meld () {  # <name> <model> [extra flags...]
    local name="$1" model="$2"; shift 2
    echo "== [${SECONDS}s] eval_meld ${name} =="
    python scripts/eval_meld.py --model "${model}" --device cuda \
        --model-dtype bfloat16 --max-seq-length 1024 --batch-size 64 \
        --output "results/meld_atk_${name}.json" \
        --per-query-out "results/meld_pq_${name}.json" "$@" \
        || echo "== STEP FAILED (continuing): meld_atk_${name} =="
}

# matched condition (threshold-bearing)
run_meld attack_meld9   models/meld-attack-meld9/final   --query-prompt-name query
run_meld attack_heldout models/meld-attack-heldout/final --query-prompt-name query
run_meld base           Qwen/Qwen3-Embedding-0.6B        --query-prompt-name query
run_meld ctrl_llm       models/ctrl-llm-6145/final       --query-prompt-name query
run_meld ctrl_cas       models/ctrl-cas-6145/final       --query-prompt-name query
run_meld v2_queryprompt models/qwen3-0.6b-p2-instr/final --query-prompt-name query
# context / prompt-sensitivity rows (never threshold-bearing)
run_meld attack_meld9_noprompt models/meld-attack-meld9/final
run_meld base_noprompt         Qwen/Qwen3-Embedding-0.6B
run_meld v2_owninstr           models/qwen3-0.6b-p2-instr/final --query-prompt "${V2_INSTR}"

# paired cluster-bootstrap CIs on the pre-registered contrasts
for CMP in "attack_meld9 base" "attack_meld9 ctrl_llm" "attack_heldout base" \
           "attack_meld9 attack_heldout" "ctrl_llm base"; do
    set -- ${CMP}
    if [[ -s "results/meld_pq_$1.json" && -s "results/meld_pq_$2.json" ]]; then
        echo "== [${SECONDS}s] compare $1 vs $2 =="
        python scripts/generate_meld_pairs.py --stage compare \
            --compare "results/meld_pq_$1.json" "results/meld_pq_$2.json" \
            --out "results/meld_compare_$1_vs_$2.json" > /dev/null \
            || echo "== STEP FAILED (continuing): compare $1 vs $2 =="
    fi
done

# ---------------------------------------------------------------------------
# STAGE E — P-M2 (real-duplicate ability) and P-M3(b) (recipe specificity)
# ---------------------------------------------------------------------------
for ARM in meld9 heldout; do
    echo "== [${SECONDS}s] eval_crosslingual meld-attack-${ARM} (P-M2) =="
    python scripts/eval_crosslingual.py --model "models/meld-attack-${ARM}/final" \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --max-seq-length 1024 \
        --output "results/crosslingual_meld-attack-${ARM}.json" \
        || echo "== STEP FAILED (continuing): crosslingual_meld-attack-${ARM} =="
done

for TIER in hard easy; do
    echo "== [${SECONDS}s] eval_retrieve ${TIER} meld-attack-meld9 (P-M3b) =="
    python scripts/eval_retrieve.py --tier "${TIER}" \
        --model models/meld-attack-meld9/final \
        --device cuda --model-dtype bfloat16 --query-prompt-name query \
        --batch-size 64 --max-seq-length 1024 --emb-cache-dir .emb_cache \
        --output "results/eval_${TIER}_meld-attack-meld9.json" \
        || echo "== STEP FAILED (continuing): eval_${TIER}_meld-attack-meld9 =="
done

# ---------------------------------------------------------------------------
# SUMMARY + pre-registered verdicts
# ---------------------------------------------------------------------------
echo "=== MELD-ATTACK SUMMARY ==="
ROWCAP="${ROWCAP}" python - <<'PY'
import json, os

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

rows = [
    ("meld-attack-meld9   (PRIMARY)",   "attack_meld9"),
    ("meld-attack-heldout (SPECIFIC)",  "attack_heldout"),
    ("base 0.6B [ref 10.19]",           "base"),
    ("ctrl-llm-6145 [ref 18.89]",       "ctrl_llm"),
    ("ctrl-cas-6145 [ref 9.44]",        "ctrl_cas"),
    ("V2 p2-instr, query prompt",       "v2_queryprompt"),
    ("-- context: attacked, no prompt", "attack_meld9_noprompt"),
    ("-- context: base, no prompt [ref 10.19 w/ prompt]", "base_noprompt"),
    ("-- context: V2, own instruction [ref 10.19]", "v2_owninstr"),
]
print(f"  {'model':52s} {'R@1':>6s} {'R@5':>6s} {'MRR':>7s} {'aucSF':>7s} {'above%':>7s}")
vals = {}
for label, name in rows:
    r = load(f"results/meld_atk_{name}.json")
    if r is None:
        print(f"  {label:52s}  MISSING")
        continue
    p, s = r["retrieval_pairs_only"], r["separation_auc"]
    vals[name] = (p["recall@1"], s["auc_same_framing_distractors"],
                  s["pct_pos_above_all_same_framing_distractors"])
    print(f"  {label:52s} {p['recall@1']:6.2f} {p['recall@5']:6.2f} "
          f"{p['mrr']:7.4f} {s['auc_same_framing_distractors']:7.4f} "
          f"{s['pct_pos_above_all_same_framing_distractors']:7.2f}")

# harness calibration: these rows were produced by eval_external_ood.slurm with
# byte-identical flags (bfloat16, max_seq_length 1024, --query-prompt-name
# query), so a drift here means the harness moved, not the attack.
print("\n  harness calibration vs results/meld_{qwen3-0.6b-base,ctrl-llm-6145,"
      "ctrl-cas-6145}.json:")
for name, ref in (("base", 10.19), ("ctrl_llm", 18.89), ("ctrl_cas", 9.44)):
    got = vals.get(name, (None,))[0]
    if got is None:
        print(f"    {name}: MISSING")
    else:
        print(f"    {name}: {got:.2f} vs published {ref:.2f}  "
              f"{'OK' if abs(got - ref) < 0.01 else 'DRIFT — investigate before reading any verdict'}")

print("\n  paired cluster-bootstrap (270-pair resampling unit):")
for a, b in [("attack_meld9", "base"), ("attack_meld9", "ctrl_llm"),
             ("attack_heldout", "base"), ("attack_meld9", "attack_heldout"),
             ("ctrl_llm", "base")]:
    c = load(f"results/meld_compare_{a}_vs_{b}.json")
    if not c:
        print(f"    {a} - {b}: MISSING")
        continue
    m = c["metrics"]["recall@1_pairs_only"]
    flag = "" if c.get("prompt_conditions_match") else "  [PROMPT MISMATCH]"
    print(f"    {a} - {b}: {m['diff']:+.2f} R@1  CI95 [{m['ci95'][0]:+.2f},"
          f"{m['ci95'][1]:+.2f}]  {m['n_sig']} sd  d={m['n_discordant_queries']}{flag}")

pw = load("results/meld_attack_power.json") or {}
print(f"\n  MDE floor (off-the-shelf reference): "
      f"{pw.get('mde_recall@1_80pct_power_alpha05')} R@1 at 80% power; "
      f"read mde_recall@1_by_discordance at the d printed above")
print(f"  matched training budget: {os.environ.get('ROWCAP')} rows/arm "
      f"(controlled experiment: 6145)")

xl = {}
for arm in ("meld9", "heldout"):
    x = load(f"results/crosslingual_meld-attack-{arm}.json")
    if x:
        g = x.get("strict_crosslingual_gold", {})
        xl[arm] = g.get("recall@1")
        print(f"  real-duplicate strict R@1/5/10 (P-M2, {arm}): "
              f"{g.get('recall@1')}/{g.get('recall@5')}/{g.get('recall@10')}")
print("  refs: base 84.73 | V2 74.30 | ctrl-llm 64.38 | ctrl-cas 61.32 | saber-attack 34.86")

hard = load("results/eval_hard_meld-attack-meld9.json")
h1 = hard["overall"]["recall@1"] if hard else None
if hard:
    o = hard["overall"]
    print(f"  MathNet-Retrieve hard R@1/5/10 (P-M3b): "
          f"{o['recall@1']:.2f}/{o['recall@5']:.2f}/{o['recall@10']:.2f}"
          f"   refs: base 0.00 | ctrl-llm 9.37 | ctrl-cas 0.13")

# ---- pre-registered verdicts (thresholds fixed in this file's header) ------
print("\n=== PRE-REGISTERED VERDICTS ===")
BASE_R1, CTRL_LLM_R1, BASE_AUC, BASE_ABOVE = 10.19, 18.89, 0.8872, 54.07
a = vals.get("attack_meld9")
h = vals.get("attack_heldout")
if a:
    d_base = a[0] - BASE_R1
    c = load("results/meld_compare_attack_meld9_vs_base.json")
    ci = c["metrics"]["recall@1_pairs_only"]["ci95"] if c else None
    ci_ok = bool(ci) and (ci[0] > 0 or ci[1] < 0)
    print(f"  P-M1  attack - base = {d_base:+.2f} R@1 (need >= +10.00, CI excludes 0): "
          f"{'PASS' if d_base >= 10.0 and ci_ok else 'FAIL'}"
          f"{'' if ci else '  [CI missing]'}")
    print(f"      secondary: above% {a[2] - BASE_ABOVE:+.2f} (need >= +8.00) | "
          f"aucSF {a[1] - BASE_AUC:+.4f} (need >= +0.0200)")
    if abs(d_base) < 3.6 and not ci_ok:
        print("      FALSIFIER TRIGGERED: within the MDE and CI contains 0 -> "
              "cell A's prediction FAILS at n=2; report MELD as a "
              "synthetic-corpus benchmark that RESISTED, and narrow the "
              "taxonomy to a single-benchmark observation.")
    elif 3.6 <= d_base < 10.0:
        print("      PARTIAL: in the 3.6-10.0 gradient band -> report as a "
              "gradient, NOT as confirmation.")
    d_ctrl = a[0] - CTRL_LLM_R1
    print(f"  P-M3a attack - ctrl-llm = {d_ctrl:+.2f} R@1 (need >= +8.00): "
          f"{'PASS' if d_ctrl >= 8.0 else 'FAIL'}")
if h:
    d_h = h[0] - BASE_R1
    verdict = ("GENRE-level" if d_h >= 10.0 else
               "TOPIC-level" if abs(d_h) < 3.6 else "gradient (unresolved)")
    print(f"  P-M4  heldout - base = {d_h:+.2f} R@1 -> {verdict}")
r1 = xl.get("meld9")
if r1 is not None:
    print(f"  P-M2  real-duplicate R@1 {r1} < base 84.73 and < V2 74.30: "
          f"{'PASS' if r1 < 74.30 else 'FAIL'}")
if h1 is not None:
    print(f"  P-M3b MathNet-Retrieve hard R@1 {h1:.2f} < 4.69: "
          f"{'PASS' if h1 < 4.69 else 'FAIL'}")
print("\n  Report every verdict, pass or fail — the SABER negative is in the "
      "paper for exactly this reason.")
PY
echo "== meld-attack job finished: $(date) =="
