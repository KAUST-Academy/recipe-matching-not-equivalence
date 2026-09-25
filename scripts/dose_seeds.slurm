#!/bin/bash
#SBATCH --job-name=dose-seeds
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --output=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.out
#SBATCH --error=/ibex/user/habiam0b/MathNet_Follow_Up/results/slurm_%x_%j.err

# ============================================================================
# DOSE-RESPONSE HARDENING (review findings H2 + M4) — seed error bars for the
# D2-D4 rungs + the row-count confound control for D2 vs D3.
#
# WHY (review findings, 2026-07-30):
#  M4  the two-regime dose-response centerpiece (tab:dose) rests on single
#      seed-42 runs. Confirmed by scripts/dose_bootstrap.py on the existing
#      seed-42 rank dumps: the *eval-sampling* CIs are tight and all six
#      adjacent-rung differences exclude zero, BUT measured against the arm's
#      own training-seed noise (results/ctrl_seed_stats.json, 8 ctrl-llm
#      seeds: sd 0.85 easy / 0.61 hard R@1, so sd of a run-to-run DIFFERENCE
#      = 1.20 easy / 0.86 hard) two of them are under 2 sigma:
#          hard D1-D2  +1.47  [ 1.07,  1.89]  = 1.71 sigma_seed
#          easy D2-D3  -2.19  [-2.89, -1.48]  = 1.82 sigma_seed
#      (the other four are 2.8-31 sigma_seed and are safe on one seed).
#      D1 already has 8 seeds (models/ctrl-llm-6145*, dumps in results/ranks);
#      this job adds seeds 43 and 44 for D2/D3/D4 -> 3 seeds per rung, which
#      is what turns those two differences into replicated claims.
#  H2  D3 trained on 5,591 rows, NOT the 6,145 the paper claimed. The style
#      prompt yielded only 5,591 judge-verified positives, all of which were
#      used. QUADRUPLE-SOURCED (re-verified 2026-07-30):
#        1. results/llm_pairs_style_summary.json
#           by_label.positive.verified = 5591 (n_generation_json_failures
#           1266 — the worst of the ladder; D2 345, D4 232)
#        2. the dose-response job's scheduler log (kept offline), line 792: "[convert] wrote 5591
#           trainer rows (5591 source problems, 14440 attached negatives)"
#           and :795 "[data] 5591 rows read" with NO "matched-budget cap"
#           line, vs :424 (D2 6553 -> 6145) and :1164 (D4 6768 -> 6145)
#        3. wc -l data/llm_pairs_style/pairs.jsonl = 5591
#           (paraphrase 6553, unrelated 6768, D1 llm_pairs 6145)
#        4. models/dose-style-6145/run_config.json data_stats:
#           input_rows 5591, rows_used 5591, n_source_ids 5591
#           (vs dose-paraphrase-6145: input_rows 6553, rows_used 6145;
#            dose-unrelated-6145: input_rows 6768, rows_used 6145)
#      So D3's 7.89 -> 4.92 hard R@1 drop conflates prompt distance with a 9%
#      row deficit. CONFOUND KILLER: retrain D2 at seed 42 with
#      --max-rows 5591. train_invarembed.py:211-214 shuffles with --seed
#      BEFORE truncating, so kept[:5591] is a strict PREFIX of the same
#      shuffled order the original D2 model saw — a nested subsample, not a
#      new random draw. If its hard R@1 stays near 7.9 rather than falling
#      toward 4.9, the D2-D3 gap is the prompt, not the row count; if it
#      falls, H2's confound is real and the paper must say so.
#      (Note the seed-43/44 D3 runs inherit the same 5,591 rows — they measure
#      seed noise AT D3's actual budget. Only the D2@5591 control separates
#      budget from prompt.)
#
# WHAT THIS JOB TRAINS (7 trainings, all at the EXACT controlled-experiment
# config of models/ctrl-llm-6145/run_config.json; only --train-file /
# --output-dir / --seed / --max-rows vary):
#   dose-paraphrase-6145-s43, -s44     6,145 of 6,553 rows (seed-specific cap)
#   dose-style-6145-s43, -s44          ALL 5,591 rows (the cap is a no-op; the
#                                      "-6145" suffix names the BUDGET arm,
#                                      matching the existing dose-style-6145)
#   dose-unrelated-6145-s43, -s44      6,145 of 6,768 rows; 0 attached
#                                      negatives BY DESIGN (see
#                                      dose_response.slurm honesty notes)
#   dose-paraphrase-5591  (seed 42)    the H2 row-count control above
# D1 seeds 43/44 already exist (models/ctrl-llm-6145-s43/-s44) with rank
# dumps (results/ranks/{easy,hard,xling}_ctrl-llm-s4{3,4}.ranks.jsonl) —
# nothing to train for D1.
#
# EVALS per new model: easy + hard tiers and the real-duplicate
# (cross-lingual) eval, each with --dump-ranks so scripts/dose_bootstrap.py
# can compute paired per-query CIs for the adjacent-rung differences and
# scripts/bootstrap_stats.py can cluster-bootstrap the real-duplicate line
# (pre-registered prediction P-L2). Stage 2 also backfills the seed-42 dose
# real-duplicate rank dumps, which the original dose_response job did not
# write (its easy/hard seed-42 dumps already exist).
#
# RUNTIME (measured anchors, models/*/run_config.json train_runtime_seconds
# and results/eval_*_dose-*.json runtime_seconds from job 49594070):
#   trainings: D2 2,828s + D3 2,423s + D4 855s = 6,106s per seed
#              x2 seeds                                       ~3.4 h
#              + D2@5591 control ~2,600s (5,591/6,145 x 2,828) ~0.7 h
#   evals: easy 363s + hard 30s (shares the per-model emb cache) + xling 80s
#          = ~8 min/model x 7 models                          ~0.9 h
#   + 3 seed-42 xling dump backfills (~80s each) + model loads ~0.2 h
#   TOTAL ~5.2-6 h  =>  the 20 h cap has >3x slack.
# RESUME-SAFE: finished finals, existing eval JSONs and existing rank dumps
# are skipped. A training that FAILS has its partial output dir moved aside to
# <dir>.FAILED-<jobid> and the sweep continues, so a requeue never deadlocks
# on a half-written model dir (a pre-flight check still refuses to run over a
# pre-existing partial dir it did not create).
#
# Submit:  sbatch scripts/dose_seeds.slurm
#   only the H2 control (highest value per GPU-hour, ~1 h total):
#          PLAN_FILTER=5591 sbatch --time=03:00:00 scripts/dose_seeds.slurm
#   only the seed replicas:
#          PLAN_FILTER='s4[34]' sbatch scripts/dose_seeds.slurm
# Then:    python scripts/dose_bootstrap.py   (also run by this job's stage 3)
# ============================================================================

set -euo pipefail

PROJECT=/ibex/user/habiam0b/MathNet_Follow_Up
source /ibex/user/habiam0b/miniconda3/etc/profile.d/conda.sh
conda activate mathnet
export HF_HOME="${PROJECT}/.hf_cache"
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"
mkdir -p results/ranks
JOBTAG="${SLURM_JOB_ID:-local}"
export FAILLOG="results/dose_seeds_failures_${JOBTAG}.log"
: > "${FAILLOG}"
PLAN_FILTER="${PLAN_FILTER:-.}"   # regex over the plan line; default = all

echo "== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) =="
echo "== start: $(date)  PLAN_FILTER='${PLAN_FILTER}' =="

run_soft() {  # evals are fail-soft so one bad step cannot kill the sweep
    echo "== [${SECONDS}s] $1 =="
    shift
    if ! "$@"; then
        echo "FAILED: $*" | tee -a "${FAILLOG}"
    fi
}

# TRAINING PLAN: "variant|seed|max_rows|model_dir|eval_tag" per line.
# eval_tag drives canonical eval names (results/eval_{tier}_<tag>.json,
# results/crosslingual_<tag>.json) and dump names
# (results/ranks/{easy,hard,xling}_<tag>.ranks.jsonl). The "-s42" suffix is
# stripped from canonical eval names only for the control, matching the
# seed-42 naming convention of the rest of the campaign.
ALL_PLAN=(
    "paraphrase|43|6145|models/dose-paraphrase-6145-s43|dose-paraphrase-s43"
    "paraphrase|44|6145|models/dose-paraphrase-6145-s44|dose-paraphrase-s44"
    "style|43|6145|models/dose-style-6145-s43|dose-style-s43"
    "style|44|6145|models/dose-style-6145-s44|dose-style-s44"
    "unrelated|43|6145|models/dose-unrelated-6145-s43|dose-unrelated-s43"
    "unrelated|44|6145|models/dose-unrelated-6145-s44|dose-unrelated-s44"
    "paraphrase|42|5591|models/dose-paraphrase-5591|dose-paraphrase-5591-s42"
)
PLAN=()
for ITEM in "${ALL_PLAN[@]}"; do
    [[ "${ITEM}" =~ ${PLAN_FILTER} ]] && PLAN+=("${ITEM}")
done
if (( ${#PLAN[@]} == 0 )); then
    echo "FATAL: PLAN_FILTER='${PLAN_FILTER}' matched no plan item"
    exit 1
fi
echo "== plan (${#PLAN[@]} trainings): =="
printf '   %s\n' "${PLAN[@]}"

# ---------------------------------------------------------------------------
# STAGE 0 — pre-flight: refuse to run over partly-trained model dirs, and gate
# on the training pair files being the exact ones the paper's counts describe
# (a silently regenerated pair file would invalidate every H2 statement).
# ---------------------------------------------------------------------------
declare -A EXPECT_ROWS=([paraphrase]=6553 [style]=5591 [unrelated]=6768)
for V in "${!EXPECT_ROWS[@]}"; do
    F="data/llm_pairs_${V}/pairs.jsonl"
    if [[ ! -s "${F}" ]]; then
        echo "FATAL: ${F} missing — run scripts/dose_response.slurm first"
        exit 1
    fi
    N=$(wc -l < "${F}")
    if (( N != EXPECT_ROWS[$V] )); then
        echo "FATAL: ${F} has ${N} rows, expected ${EXPECT_ROWS[$V]} (the count"
        echo "       the H2 fix and the paper's corrected table are built on)."
        echo "       The pair file changed since job 49594070 — re-verify before"
        echo "       training, and update the header's quadruple-sourced counts."
        exit 1
    fi
    echo "== pre-flight OK: ${F} = ${N} rows =="
done

for ITEM in "${PLAN[@]}"; do
    IFS='|' read -r V SEED MAXROWS OUT TAG <<< "${ITEM}"
    if [[ -e "${OUT}" && ! -e "${OUT}/final" ]]; then
        echo "FATAL: ${OUT} exists without final/ — partial run from an earlier"
        echo "       job; inspect and remove (or rename) it before resubmitting"
        exit 1
    fi
done

# seed-42 reference models the stage-2 backfill needs
for V in paraphrase style unrelated; do
    [[ -e "models/dose-${V}-6145/final" ]] || \
        echo "WARNING: models/dose-${V}-6145/final missing — stage-2 seed-42 dump backfill for D-${V} will be skipped" | tee -a "${FAILLOG}"
done

# ---------------------------------------------------------------------------
# STAGE 1 — train + eval (+rank dumps) per plan item
# ---------------------------------------------------------------------------
for ITEM in "${PLAN[@]}"; do
    IFS='|' read -r V SEED MAXROWS OUT TAG <<< "${ITEM}"
    NAME="${TAG%-s42}"          # canonical eval-file stem
    [[ "${TAG}" == *-s4[34] ]] && NAME="${TAG}"

    if [[ -e "${OUT}/final" ]]; then
        echo "== [${SECONDS}s] ${TAG}: ${OUT}/final exists — skipping training =="
    else
        echo "== [${SECONDS}s] ${TAG}: training (variant=${V} seed=${SEED} max-rows=${MAXROWS}) =="
        # identical to models/ctrl-llm-6145/run_config.json argv except
        # --train-file / --output-dir / --seed / --max-rows
        if python scripts/train_invarembed.py \
            --train-file "data/llm_pairs_${V}/pairs.jsonl" --max-rows "${MAXROWS}" \
            --model Qwen/Qwen3-Embedding-0.6B \
            --model-dtype bfloat16 --bf16 \
            --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 \
            --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed "${SEED}" \
            --dev-frac 0.05 --eval-steps 100 \
            --query-prompt-name query \
            --output-dir "${OUT}"
        then
            :
        else
            echo "FAILED: training ${TAG} (variant=${V} seed=${SEED})" | tee -a "${FAILLOG}"
            # quarantine the partial dir so a requeue is not blocked by stage 0
            if [[ -e "${OUT}" ]]; then
                mv "${OUT}" "${OUT}.FAILED-${JOBTAG}"
                echo "   partial dir moved to ${OUT}.FAILED-${JOBTAG}" | tee -a "${FAILLOG}"
            fi
            continue
        fi
    fi

    for TIER in easy hard; do
        EVJ="results/eval_${TIER}_${NAME}.json"
        DMP="results/ranks/${TIER}_${TAG}.ranks.jsonl"
        if [[ -s "${EVJ}" && -s "${DMP}" ]]; then
            echo "== [${SECONDS}s] ${TAG}: ${TIER} eval + dump exist — skipping =="
            continue
        fi
        run_soft "${TAG}: eval tier=${TIER}" \
        python scripts/eval_retrieve.py --tier "${TIER}" --model "${OUT}/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 \
            --emb-cache-dir .emb_cache \
            --dump-ranks "${DMP}" \
            --output "${EVJ}"
    done

    XJ="results/crosslingual_${NAME}.json"
    XD="results/ranks/xling_${TAG}.ranks.jsonl"
    if [[ -s "${XJ}" && -s "${XD}" ]]; then
        echo "== [${SECONDS}s] ${TAG}: real-duplicate eval + dump exist — skipping =="
    else
        run_soft "${TAG}: real-duplicate (cross-lingual) eval" \
        python scripts/eval_crosslingual.py --model "${OUT}/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --max-seq-length 1024 \
            --dump-ranks "${XD}" \
            --output "${XJ}"
    fi
done

# ---------------------------------------------------------------------------
# STAGE 2 — backfill seed-42 dose rank dumps. The easy/hard seed-42 dumps
# already exist (results/ranks/{easy,hard}_dose-*-s42.ranks.jsonl, written by
# the login-node CPU pass); the xling ones do not. Summaries go to
# results/ranks/*.summary.json so the canonical results/*.json are never
# overwritten by a re-run.
# ---------------------------------------------------------------------------
for V in paraphrase style unrelated; do
    [[ -e "models/dose-${V}-6145/final" ]] || continue
    for TIER in easy hard; do
        D="results/ranks/${TIER}_dose-${V}-s42.ranks.jsonl"
        [[ -s "${D}" ]] && continue
        run_soft "backfill s42 dump: ${TIER} dose-${V}" \
        python scripts/eval_retrieve.py --tier "${TIER}" \
            --model "models/dose-${V}-6145/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --batch-size 64 --max-seq-length 1024 \
            --emb-cache-dir .emb_cache \
            --dump-ranks "${D}" \
            --output "results/ranks/${TIER}_dose-${V}-s42.summary.json"
    done
    XD="results/ranks/xling_dose-${V}-s42.ranks.jsonl"
    if [[ ! -s "${XD}" ]]; then
        run_soft "backfill s42 dump: xling dose-${V}" \
        python scripts/eval_crosslingual.py \
            --model "models/dose-${V}-6145/final" \
            --device cuda --model-dtype bfloat16 --query-prompt-name query \
            --max-seq-length 1024 \
            --dump-ranks "${XD}" \
            --output "results/ranks/xling_dose-${V}-s42.summary.json"
    fi
done

# ---------------------------------------------------------------------------
# STAGE 3 — across-seed summary, what-actually-trained audit, H2 verdict
# ---------------------------------------------------------------------------
echo "=== [${SECONDS}s] DOSE-SEEDS SUMMARY ==="
python - <<'PY'
import json, os, statistics

def _get(path, *keys):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return None

r1 = lambda p: _get(p, "overall", "recall@1")
x1 = lambda p: _get(p, "strict_crosslingual_gold", "recall@1")

# rung label -> {seed: (easy, hard, xling) canonical files}
FILES = {
    "D1 exact": {42: ("results/eval_easy_ctrl-llm.json",
                      "results/eval_hard_ctrl-llm.json",
                      "results/crosslingual_ctrl-llm-6145.json")},
}
for s in (43, 44):
    FILES["D1 exact"][s] = (f"results/eval_easy_ctrl-llm-s{s}.json",
                            f"results/eval_hard_ctrl-llm-s{s}.json",
                            f"results/crosslingual_ctrl-llm-s{s}.json")
for rung, v in (("D2 paraphrase", "paraphrase"), ("D3 style", "style"),
                ("D4 unrelated", "unrelated")):
    FILES[rung] = {42: (f"results/eval_easy_dose-{v}.json",
                        f"results/eval_hard_dose-{v}.json",
                        f"results/crosslingual_dose-{v}.json")}
    for s in (43, 44):
        FILES[rung][s] = (f"results/eval_easy_dose-{v}-s{s}.json",
                          f"results/eval_hard_dose-{v}-s{s}.json",
                          f"results/crosslingual_dose-{v}-s{s}.json")
FILES["D2@5591 ctrl (s42)"] = {42: ("results/eval_easy_dose-paraphrase-5591.json",
                                    "results/eval_hard_dose-paraphrase-5591.json",
                                    "results/crosslingual_dose-paraphrase-5591.json")}

def fmt(vals):
    if not vals:
        return "  --  "
    m = statistics.mean(vals)
    sd = f" +-{statistics.stdev(vals):4.2f}" if len(vals) > 1 else "       "
    return f"{m:6.2f}{sd} (n={len(vals)})"

print(f"{'rung':22s} {'easy R@1 mean+-sd':>22s} {'hard R@1 mean+-sd':>22s} "
      f"{'realdup R@1 mean+-sd':>22s}")
point = {}
for rung, per_seed in FILES.items():
    cols = ([], [], [])
    for seed, (fe, fh, fx) in sorted(per_seed.items()):
        vals = (r1(fe), r1(fh), x1(fx))
        point[(rung, seed)] = vals
        for col, val in zip(cols, vals):
            if val is not None:
                col.append(val)
    print(f"{rung:22s} {fmt(cols[0]):>22s} {fmt(cols[1]):>22s} {fmt(cols[2]):>22s}")

print("\n--- what actually trained (run_config.json data_stats) ---")
for m in ("ctrl-llm-6145", "dose-paraphrase-6145", "dose-style-6145",
          "dose-unrelated-6145", "dose-paraphrase-6145-s43",
          "dose-paraphrase-6145-s44", "dose-style-6145-s43",
          "dose-style-6145-s44", "dose-unrelated-6145-s43",
          "dose-unrelated-6145-s44", "dose-paraphrase-5591"):
    ds = _get(f"models/{m}/run_config.json", "data_stats")
    if ds is None:
        continue
    print(f"  {m:26s} input_rows={ds['input_rows']:5d} rows_used={ds['rows_used']:5d} "
          f"train_triplets={ds['n_train_triplets']:6d} train_pairs={ds['n_train_pairs']:5d}")

print("\n--- H2 ROW-COUNT CONFOUND VERDICT (hard R@1, seed 42) ---")
d2 = r1("results/eval_hard_dose-paraphrase.json")
d3 = r1("results/eval_hard_dose-style.json")
ctl = r1("results/eval_hard_dose-paraphrase-5591.json")
print(f"  D2 @6145 rows (paraphrase prompt) : {d2}")
print(f"  D2 @5591 rows (paraphrase prompt) : {ctl}   <- control")
print(f"  D3 @5591 rows (style prompt)      : {d3}")
if None in (d2, d3, ctl):
    print("  VERDICT: pending — the control eval has not run yet")
else:
    row_eff, prompt_eff = ctl - d2, d3 - ctl
    print(f"  row-count effect  (D2@5591 - D2@6145) = {row_eff:+.2f}")
    print(f"  prompt effect     (D3@5591 - D2@5591) = {prompt_eff:+.2f}")
    tot = d3 - d2
    share = "n/a" if abs(tot) < 1e-9 else f"{100 * row_eff / tot:.0f}%"
    print(f"  total D2->D3 drop = {tot:+.2f}; row count explains {share} of it")
    if abs(row_eff) < 0.86:      # < 1 sd of a run-to-run difference (hard R@1)
        print("  VERDICT: row count is INSIDE seed noise (sd_diff 0.86) — the")
        print("           D2->D3 drop is the PROMPT. H2 is a reporting fix only.")
    else:
        print("  VERDICT: the 9% row deficit moves hard R@1 by more than seed")
        print("           noise — H2's confound is REAL and the paper must")
        print("           attribute part of the D2->D3 drop to data volume.")

print("\n--- easy tier, same control ---")
for lbl, p in (("D2 @6145", "results/eval_easy_dose-paraphrase.json"),
               ("D2 @5591", "results/eval_easy_dose-paraphrase-5591.json"),
               ("D3 @5591", "results/eval_easy_dose-style.json")):
    print(f"  {lbl}: {r1(p)}")
PY

echo "== [${SECONDS}s] paired per-query bootstrap CIs =="
python scripts/dose_bootstrap.py --seeds 42,43,44 \
    || echo "FAILED: dose_bootstrap.py" | tee -a "${FAILLOG}"

echo "== failures logged (empty = clean): =="
cat "${FAILLOG}"
echo "== dose-seeds finished: $(date) =="
echo "== next: update the reproducibility.json dose_seeds_hardening entry's key_numbers from"
echo "==       results/eval_{easy,hard}_dose-*-s4{3,4}.json, eval_*_dose-paraphrase-5591.json,"
echo "==       results/crosslingual_dose-*.json and results/dose_bootstrap.json =="
