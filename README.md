# Recipe-Matching, Not Equivalence

Code, data and result files for the paper *Recipe-Matching, Not Equivalence* (Ali Habibullah, Mohammad Alshiekh, Yazan Alshoibi, Salman Khan and Naeemullah Khan, 2026). Every number in the paper is regenerable from this repository; `reproducibility.json` records, for each result, the command, the scheduler job id, the inputs and the outputs, and the pre-registered predictions sit in the headers of the job scripts that produced them.

**The finding in one paragraph.** MathNet-Retrieve's golds are LLM paraphrases of its queries, written by a published prompt and filtered by LLM judges. We call that procedure the benchmark's *recipe*, and training on pairs built the way the benchmark's were *recipe-matching*. In a matched-budget experiment (6,145 training rows per arm, identical hyperparameters, eight seeds) a 0.6B embedder trained on pairs built with the benchmark's own prompt through a different vendor's generator and judge (the *recipe arm*) leads an arm whose pairs a computer algebra system (CAS) verifies (the *verified arm*) by 45.33 easy-tier recall-at-1 (R@1) points. A factorial decomposes the gap: a recipe-free reference (restatements from an unrelated prompt with the verified arm's own negatives) recovers about two thirds of it, so the easy tier rewards the genre of LLM rewrites; the remaining 15.47 points are recipe-specific and do not reach mined cross-language reprints that no generator wrote. The hard tier rewards the recipe's pair structure, a deep LLM rewrite against a minimal-edit near-miss, and needs both halves. Every negative that unlocks the hard tier costs cross-language retention, so on this benchmark one edit to a training file moves score and retention in opposite directions. We call the hard tier buyable and cannot yet call it measured. Two further benchmarks whose recipes we attacked, MELD and SABER-Math, each move on the surface their recipes write, MELD under a reconstruction of its unpublished prompt and SABER only slightly. We release the two mined duplicate evaluations as generator-free targets, the analysis artifacts behind every claim, and the training files of every MathNet-Retrieve arm and factorial cell. The generated MELD and SABER attack pair files and V2's mixed training set are included too; only SABER's document-channel split is left out, and `make_saber_channels.py` rebuilds it from the full pair file.

---

## Headline results (eight seeds unless noted; every number is read from the file named)

### The matched-budget controlled experiment

R@1, mean ± sd over training seeds 42–49 on Qwen3-Embedding-0.6B. The base is a single deterministic run. Cross-language and same-language duplicates are the two mined organic evaluations released under `data/`.

| Model | Easy | Medium | Hard | Cross-language duplicates (strict R@1) | Same-language reprints (primary slice) |
|---|---|---|---|---|---|
| Untrained base | 8.32 | 1.78 | 0.00 | 84.73 | 62.40 |
| Verified arm (`ctrl-CAS`) | 17.05 ± 0.79 | 3.14 ± 0.46 | 0.16 ± 0.06 | 56.52 ± 4.05 | 51.30 ± 2.44 |
| Recipe arm (`ctrl-LLM`) | 62.38 ± 0.85 | 8.84 ± 0.36 | 9.42 ± 0.61 | 66.26 ± 2.31 | 46.30 ± 1.89 |
| Reference (restatements from an unrelated prompt, ladder rung D4 below, + the verified arm's negatives) | 46.91 ± 1.25 | 4.22 ± 0.25 | 10.30 ± 1.15 | 78.02 ± 0.91 | 39.20 ± 1.28 |

Sources: arm means and sds `results/ctrl_seed_stats.json`; reference `results/factorial_2x2.json` and `results/reference_control_did.json`; base `results/eval_{easy,medium,hard}_qwen3-0.6b-base.json`, `results/crosslingual_qwen3-0.6b-base.json`; same-language `results/samelang_verdict.json`.

Contrasts (nested bootstrap over queries or duplicate clusters and seeds, 95% intervals):

| Contrast | Easy R@1 | Cross-language R@1 | Difference-in-differences (easy gap minus duplicate gap) | Source |
|---|---|---|---|---|
| Recipe arm minus verified arm | 45.33 [44.25, 46.41] | 9.74 [3.96, 15.71], positive in every seed | 35.59 [29.51, 41.47] | `results/final_stats.json` |
| Recipe arm minus reference | 15.47 [14.11, 16.73] | −11.77 [−17.33, −6.09] | 27.24 [21.40, 32.93] | `results/reference_control_did.json` |
| Reference minus verified arm | 29.85 [28.56, 31.15] | 21.50 [15.31, 27.67] | 8.35 [1.93, 14.78] | `results/reference_control_did.json` |

On the hard tier the reference is level with the recipe arm (recipe minus reference −0.88 [−1.85, 0.10], `results/reference_control_did.json`, key `tiers`) and above it at R@5. On same-language reprints the recipe arm holds no measurable lead over the verified arm (−5.00 [−13.42, 3.35]); the registered difference-in-differences there is 50.33 [41.85, 58.75] against a pass bar of 22.67 (`results/samelang_verdict.json`, key `P-R5`, the same-language readout registered in `scripts/samelang_eval.slurm`). No trained model beats the base beyond noise on that slice. The medium tier carries no recipe-specific inflation against the verified arm (against the reference the recipe arm leads by 4.62, `results/reference_control_did.json`, key `tiers`).

### The rewriter-prompt ladder (three seeds each; D1 is the recipe arm)

| Rung | Prompt | Easy R@1 | Hard R@1 | Cross-language R@1 |
|---|---|---|---|---|
| D1 | the benchmark's published rewriter prompt | 61.86 | 9.90 | 66.41 |
| D2 | close paraphrase of that prompt | 67.54 | 6.76 | 68.02 |
| D3 | a deliberately different rewriting style | 67.88 | 4.67 | 65.39 |
| D4 | an unrelated restatement prompt, no negatives | 30.03 | 0.04 | 93.81 |

Source: `results/dose_seed_stats.json`; the four prompts are printed verbatim in `paper/dose_prompts.tex` (generated by `scripts/dump_dose_prompts.py`). The registered monotone hard-tier decay passes in every seed; the registered easy-tier gradient fails (D2 and D3 are indistinguishable and land above D1), so the easy tier rewards the genre, not the template.

### The positives × negatives factorial (hard tier)

| Positives | Negatives | Easy R@1 | Hard R@1 | Cross-language R@1 |
|---|---|---|---|---|
| CAS (verified) | CAS counterexamples (= verified arm) | 17.05 ± 0.79 | 0.16 ± 0.06 | 56.52 ± 4.05 |
| CAS (verified) | none | 7.42 ± 0.24 | 0.00 | 52.08 ± 1.06 |
| D4 (recipe-free) | none (= D4) | 30.03 ± 1.45 | 0.04 ± 0.02 | 93.81 ± 0.39 |
| D4 (recipe-free) | CAS counterexamples (= reference, 8 seeds) | 46.91 ± 1.25 | 10.30 ± 1.15 | 78.02 ± 0.91 |
| D4 (recipe-free) | the recipe arm's own LLM-written near-misses, count-matched | 44.59 ± 1.33 | 20.71 ± 0.82 | 83.89 ± 3.82 |
| D4 (recipe-free) | the same, at the recipe arm's full volume | 37.79 ± 1.37 | 26.97 ± 0.97 | 78.88 ± 2.33 |
| D4 (recipe-free) | LLM-written near-misses from an unrelated prompt, count-matched | 47.69 ± 2.12 | 16.96 ± 2.08 | 85.50 ± 1.67 |
| D4 (recipe-free) | the same, full volume | 43.93 ± 0.51 | 20.93 ± 0.44 | 84.91 ± 1.49 |
| CAS (verified) | the recipe arm's near-misses, full volume | 11.28 ± 0.48 | 0.11 ± 0.03 | 12.47 ± 0.68 |
| Back-translated (NLLB-200) | CAS counterexamples | 18.81 ± 0.80 | 1.09 ± 0.04 | 84.57 ± 1.20 |

Source: `results/factorial_2x2.json` (means and sds; the D4-alone row is the three-seed ladder rung read from `results/dose_seed_stats.json`, `per_rung.D4`), `results/reference_control_did.json` key `matched_negatives` (contrasts). Readings: the positive must be a deep LLM rewrite (near-copy or back-translated positives stay near the floor even with negatives). LLM-written near-misses from an unrelated prompt carry most of the negatives' effect, and the recipe's own add 3.75 [1.83, 5.97] hard-tier points at matched count and 6.03 [5.01, 7.18] at full volume. Near-misses from an unrelated prompt leave cross-language retention at the base. Back-translated positives with the reference's negatives recover only 1.76 [0.62, 2.82] of the reference's 29.85 easy-tier points over the verified arm, so the recovered share is LLM authorship rather than paraphrase depth, as far as back-translation reaches.

### The style probe

A pair-matched n-gram detector (trained on our rewrites against their organic sources, split by source problem) reaches held-out AUC 0.880, transfers to the benchmark's own golds at 0.804 having seen no token from that vendor, sits near chance on the benchmark's near-miss distractors (0.56), and detects the recipe-free rung D4 at 0.676. Models trained under the recipe's rewrite prompts track that fingerprint (mean per-query Spearman ρ between similarity and style score: recipe arm +0.171, D2 +0.164, D3 +0.105). The verified arm (−0.019), the base (−0.061) and D4 (−0.000) do not. The cells that buy the most hard-tier score carry little of it (reference +0.025; D4 with the recipe's near-misses +0.046 and +0.050). Five further public models (Qwen3-Embedding-4B and -8B, RaDeR-Qwen2.5-7B, MathLeap-Octen-8B and MathLeap-Qwen-8B) sit in the base's band (−0.07 to +0.004). Sources: `results/style_probe_stripped.json`, `results/style_tracking_review2.json`, `results/style_probe_public_models.json`. On the hard tier the recipe arm alone ranks the gold first on 1,401 of 15,000 queries, the verified arm alone on 15; those wins skew toward query-gold wording overlap (AUC 0.639), not toward high-style golds (0.511) (`results/analyze_hits_ctrl-llm_vs_ctrl-cas_hard.json`).

### Contamination

The exact-text gate the campaign trained under missed near-verbatim twins that differ only by an OCR header, so about a third of both arms' rows anchor on a benchmark query's own statement (seed 42: recipe arm 33.26%, verified arm 34.91%). The leak is symmetric and marginally heavier in the arm that loses. The easy-tier gap is larger on the 8,761 queries the gate did catch (46.25 ± 0.86 against 45.33). Netting each arm's leaked-slice excess against its untrained-twin baseline charges 1.24 ± 0.08 points to leakage, or 0.59 ± 0.06 under the gate-caught counterfactual (recomputed from the same file's per-seed slices). Both arms retrained under the corrected gate, eight seeds each, keep the gap: 43.11 ± 0.90 at the 4,101 rows the gate leaves, 44.45 ± 0.87 at the full 6,145 with the recipe arm's excluded third regenerated from fresh sources; neither retrain carries any anchor overlap against the corrected list. Sources: `results/clean_slice_analysis.json`, `results/clean_gate_stats.json`, `results/cleanfull_verdict.json`; gates `anchor_to_corpus_mapping.json` (v1, 8,698 corpus ids) and `anchor_to_corpus_mapping_v2.json` (corrected, 15,244 ids).

### Backbones and scale (seed 42)

The easy-tier gap replicates on multilingual-e5-large (+19.12, shrinking to +2.29 on cross-language duplicates) and BGE-large-en-v1.5 (+15.91). At 4B (rank-16 LoRA) the recipe arm scores 77.77 easy R@1 against the verified arm's 39.51, with cross-language retention 70.99 against 83.72 under a 91.60 untrained-4B base: the inversion holds at that scale, and 39.51 from verified pairs alone shows the easy tier is learnable. Sources: `results/ctrl_seed_stats.json` (`backbone_replications`, e5 only), `results/eval_easy_ctrl-{cas,llm}-{e5-s42,bge-s42,4b}.json` (the BGE gap is read directly from the eval files), `results/crosslingual_ctrl-{cas,llm}-4b.json`, and the untrained 4B base `results/crosslingual_qwen3-embedding-4b.json`.

### Two external benchmarks

- **SABER-Math.** The registered document-surface attack fails (nDCG@10 drops; the construction-signal correlation rises 1.14× against a required 2×). Attacking each surface alone at a matched 4,733-row budget, three seeds each, the summary channel gains +0.0343 ± 0.0018 over the base (0.5747) in every seed and the document channel loses (−0.0163 ± 0.0065); the correlation still rises only 1.14×. A small, confounded gain on an artifact SABER never scores, at a cross-language retention cost (document channel 10.69, mixed run 34.86, summary channel 58.78, base 84.73). Sources: `results/saber_label_seed_stats.json`, `results/saber_label_4733_noprompt.json`, `results/saber_base0.6b_noprompt.json`, `results/crosslingual_saber-*.json`.
- **MELD.** A reconstruction of its unpublished evaluation-set prompt, run through Qwen3-32B-AWQ on 3,150 matched rows and gated item by item against the evaluation texts, lifts a 0.6B model from 10.19 to 37.16 ± 1.19 pairs-only R@1 over three seeds. At seed 42 it ties MELD's best published model in the same harness (+2.22 [−2.96, 7.22] against MathLeap-Octen-8B) and inherits nothing on MathNet's hard tier (0.11). Its cross-language retrieval rises above the base (90.67 ± 0.53) and same-language retention stays at the base: most of the gain is the task. A held-out-domain arm keeps +20.61 of the gain and a rewritten prompt keeps 84% of it (80% when the control is regenerated under the attack's own generation procedure, 31.48 pairs-only R@1). Sources: `results/meld_seed_stats.json`, `results/meld_compare_*.json`, `results/meld_prompt_control.json`, `results/meld_atk_attack_meld9altm.json`, `results/meld_disjointness_*.json`.

### Remedies, and a measured partial defense

- Mixing verified pairs with problem-to-solution replay under a task instruction (V2) reaches 22.19 ± 0.79 easy R@1 at 74.22 ± 2.42 cross-language retention (three seeds); WiSE-FT soups of the uncapped verified checkpoint buy retention and breadth (α = 0.3: 80.15 retention at 13.55 easy R@1); neither keeps the uncapped verified arm's rank-1 hard-tier discrimination (0.16). Base plus V2's instruction alone scores 90.59 on cross-language duplicates. Sources: `results/eval_*_qwen3-0.6b-p2-{mixed,instr}*.json`, `results/crosslingual_*.json`, `results/eval_*_soup-a*.json`.
- Regenerating 1,700 easy-tier golds with the benchmark's own recipe and scoring on the 1,525-query slice every variant shares is no defense: the recipe arm keeps 35.73 of its 45.64-point edge; a different-style prompt cuts it to 28.33 and drawing each gold from one of three prompts to 31.48; on the style-regenerated evaluation D3 overtakes the recipe arm by 11.28 points. Source: `results/defense_eval/`, verdicts by `scripts/defense_verdict.py`.

### Public models on MathNet-Retrieve (single runs; R@1 / R@5 / R@10; full 117,088-document corpus, 15,000 queries per tier)

| Model | Size | Easy | Medium | Hard |
|---|---|---|---|---|
| V2 (ours, mixed data + instruction) | 0.6B | 21.70 / 94.48 / 97.81 | 2.71 / 67.25 / 82.06 | 0.01 / 7.02 / 32.03 |
| V1 (ours, mixed data) | 0.6B | 20.83 / 91.96 / 96.71 | 3.67 / 63.57 / 78.80 | 0.03 / 7.67 / 30.01 |
| RaDeR-gte-Qwen2-7B | 7B | 18.94 / 95.57 / 99.05 | 1.12 / 63.88 / 83.24 | 0.01 / 8.63 / 34.63 |
| RaDeR-Qwen2.5-7B | 7B | 17.41 / 96.91 / 99.51 | 0.92 / 60.05 / 80.14 | 0.01 / 9.53 / 37.86 |
| Soup α = 0.7 (ours, WiSE-FT) | 0.6B | 16.41 / 90.44 / 96.04 | 2.33 / 52.23 / 69.83 | 0.01 / 7.54 / 28.14 |
| Uncapped verified arm (ours, `phase-1 pure-CAS` in the paper's tables) | 0.6B | 15.92 / 78.18 / 87.21 | 3.86 / 40.80 / 55.92 | 0.16 / 6.59 / 19.47 |
| MathLeap-Octen-8B | 8B | 15.87 / 84.17 / 94.40 | 3.51 / 75.90 / 91.97 | 0.01 / 17.53 / 62.57 |
| MathLeap-Qwen-8B | 8B | 13.79 / 81.32 / 93.16 | 3.37 / 75.56 / 91.15 | 0.01 / 17.01 / 60.86 |
| Qwen3-Embedding-4B (unprompted, 2,048 tokens) | 4B | 11.15 / 85.65 / 93.34 | 1.93 / 70.19 / 85.15 | 0.01 / 6.04 / 28.58 |
| ReasonIR-8B | 8B | 10.61 / 78.25 / 86.55 | 2.41 / 52.01 / 66.82 | 0.00 / 2.02 / 8.50 |
| Qwen3-Embedding-0.6B (base) | 0.6B | 8.32 / 78.61 / 87.78 | 1.78 / 60.91 / 75.49 | 0.00 / 4.08 / 17.60 |

Excluded from the ranking as benchmark-internal: the recipe arm (61.88 / 9.06 / 9.37 R@1 at seed 42), which trains on the benchmark's own recipe, and the 4B verified arm (39.51 easy R@1), whose symbolically verified positives inhabit the paraphrase genre the easy tier tests. Every cell is read from `results/eval_{easy,medium,hard}_<model>.json`.

---

## Repository map

Paths are relative to the repository root. The SLURM scripts, `scripts/rebuild_stats.sh` and three data-preparation scripts (`download_corpus.py`, `build_anchor_mapping.py`, `mine_duplicates.py`) carry the original project path or conda path as literals, and the construction certificates under `results/` and `data/` record the same absolute paths; edit those before running on another machine. The SLURM scripts request one A100 and set no scheduler account; add `#SBATCH --account=<yours>` if your cluster needs one.

### Top level

| File | What it is |
|---|---|
| `README.md` | This file |
| `reproducibility.json` | The ledger: one entry per produced result with its command, job id, inputs, outputs and key numbers; also the external data sources and environment notes |
| `requirements.txt` | The four Python environments (training and evaluation; third-party baselines; pair generation; the late-interaction arms), with pinned versions |
| `anchor_to_corpus_mapping.json` | Contamination gate v1: exact-text matches between the 15,000 MathNet-Retrieve anchors and the corpus (8,761 anchors, 8,698 corpus ids excluded from training). The gate the headline arms trained under |
| `anchor_to_corpus_mapping_v2.json` | Corrected gate: also strips OCR headers and image placeholders and unions in corpus-internal duplicates (14,921 anchors, 15,244 ids). Used by the corrected-gate retrains and the contamination audit |
| `.gitignore` | Names every regenerable intermediate the repository does not carry |
| `LICENSE`, `LICENSE-DATA`, `LICENSE-PAPER` | Licences of the code, the data and results, and the paper source (see Licence below) |

### `scripts/`

Each script carries a usage docstring; the experiment jobs listed under Registered predictions below hold their pre-registered predictions in their headers (the evaluation, generation and utility jobs do not).

**Data preparation and contamination gates**

| Script | What it does |
|---|---|
| `download_corpus.py` | Public MathNet corpus (27,817 problems, no images) → `data/mathnet_corpus.parquet` |
| `build_anchor_mapping.py`, `build_anchor_mapping_v2.py` | The v1 and corrected contamination gates (root JSON files above) |
| `cas_census.py`, `normalize_latex.py` | SymPy parse census over every LaTeX span and the normalizer that lifts unique-expression coverage from 86.61% to 92.29% (`results/cas_census.json`, `results/normalizer_recovery.json`) |
| `mine_duplicates.py`, `build_crosslingual_eval.py` | Mine 754 real cross-language reprint pairs and package them as the cross-language duplicate evaluation (`data/crosslingual_eval/`) |
| `build_samelang_eval.py`, `samelang_eval_driver.py`, `samelang_eval.slurm`, `samelang_verdict.py` | The same-language reprint evaluation from the monolingual clusters of the same duplicate graph, its scoring of every campaign model, and the registered readout |
| `make_source_ids.py` | The shared matched-budget source list (7,089 problems; tracked copy `data/phase2/cas_source_ids.txt`) |

**Pair generation (the arms, the ladder, the factorial cells)**

| Script | What it does |
|---|---|
| `generate_cas_pairs.py` + `.slurm` | Verified arm: SymPy-verified rename, reformulation and mirror positives; minimal-edit negatives kept only with a numeric counterexample → `data/cas_pairs/pairs.jsonl` |
| `generate_llm_pairs.py` + `.slurm`, `validate_pairs_schema.py`, `convert_llm_pairs.py` | Recipe arm: the benchmark's published rewriter prompt (and its ladder variants D2, D3, D4 and the unrelated-prompt negatives) through Qwen3-32B-AWQ with an LLM judge; schema check; conversion to the trainer format → `data/llm_pairs*/pairs.jsonl` |
| `dose_response.slurm`, `dose_seeds.slurm` | Generate and train the ladder rungs D2–D4; seed replication and the row-count control |
| `build_factorial_cells.py`, `build_llmnegs_cells.py`, `gen_unrelated_negs.slurm`, `build_review2_cells.py`, `build_review4_cells.py`, `backtranslate_anchors.py` + `.slurm` | Build the factorial cells: the reference and the negative-free CAS cell; D4 with the recipe arm's near-misses; near-misses from an unrelated prompt; CAS positives with the recipe's near-misses; back-translated positives through NLLB-200 |
| `gen_survey.slurm`, `gen_split.slurm`, `rejudge.slurm`, `bt2.slurm` | Generate the files of five more factorial cells, build them and submit their `factorial_cells.slurm` training: a second recipe-free restatement prompt D5 (a survey author restating another author's problem; `data/llm_pairs_survey/`, and with the reference's negatives `data/llm_pairs_survey_casnegs/` via `build_review2_cells.py --which r10`); the recipe prompt with the positive and the near-misses written in separate calls (`data/llm_pairs_exact_split/`); the recipe arm's 25,656 candidate rows re-judged by Mistral-Small-24B-Instruct-2501 with the same judge prompt, keeping the rows both judges accept (`data/llm_pairs_twojudge/`, agreement in `results/llm_pairs_twojudge_summary.json`); multi-hop back-translation through French, German and Russian (`data/bt2_pairs/`, `results/bt2_pairs_build.json`, then `data/bt2_pairs_casnegs/` via `build_review2_cells.py --which r13`) |
| `build_e2_negmatched_cas.py`, `e2_negmatched.slurm`, `e2_verdict.py` | The signal-matched verified arm (rows, sources, negative attachments and dev split matched to the recipe arm) and its registered verdict |
| `make_clean_sources.py`, `cleanfull_generate.slurm`, `build_cleanfull_llm.py`, `cleanfull_train.slurm`, `cleanfull_verdict.py` | The corrected-gate retrain at the full budget: fresh clean sources, regeneration, assembly, eight-seed training, verdict |
| `make_samesrc_sources.py`, `samesrc_generate.slurm`, `build_samesrc_llm.py`, `samesrc_train.slurm`, `samesrc_verdict.py` | The same-source control of the full-budget corrected-gate retrain: the 4,101 surviving sources (`data/pairs/source_ids_survivors.txt`), a second rewrite of each under the same prompt, generator and judge at generation seed 1 (`results/llm_pairs_samesrc_summary.json`), the 6,145-row file `data/llm_pairs_samesrc/` (`results/llm_pairs_samesrc_build.json`), eight seeds scored against the corrected-gate verified arm at 6,145 rows, which is not retrained, and the registered verdict (`results/samesrc_verdict.json`) |
| `item16_census.py`, `item16_cas_relaxed.slurm` | Counts which problems of the 7,585-problem fresh-source pool carry a relational span (1,131, of which 289 are among the 2,044 fresh sources the retrain used) and checks that the verified arm's generator attempted them all (`results/item16_census.json`, list `data/pairs/item16_relational_pool.txt`), then reruns that generator over those 1,131 problems at four times every time and span budget (CPU only; `data/cas_pairs_item16/pairs.jsonl`, `results/cas_pairs_item16_stats.json`) |
| `make_d2_on_d3_sources.py` | D2 restricted to D3's sources (selection control) |
| `build_phase2_data.py` | The four-channel mixed training set behind V1 and V2 |

**Training**

| Script | What it does |
|---|---|
| `train_invarembed.py` | The single contrastive trainer (sentence-transformers, cached MNRL, grouped rows, always-on contamination gates, seeded `--max-rows`, LoRA option, `run_config.json` audit record) |
| `train_controlled_exp.slurm`, `train_ctrl_param.slurm` | The controlled experiment at seed 42, and its parameterized form (SEED, BACKBONE, TAG) used for seeds 43–49 and the multilingual-e5 replication; the BGE run used the same job with `BACKBONE=BAAI/bge-large-en-v1.5 TAG=bge-s42 QUERY_PROMPT_TEXT='Represent this sentence for searching relevant passages: '` (the custom-backbone branch attaches no prompt unless one is exported; this run has no manifest entry, its `run_config.json` records the arguments) |
| `train_ctrl_4b.slurm` | Both arms on Qwen3-Embedding-4B with rank-16 LoRA |
| `clean_gate_retrain.slurm`, `clean_gate_verdict.py` | Both arms under the corrected gate at 4,101 rows (`SEEDS=`; the job defaults to the three pre-registered seeds, eight were run) and the registered verdict |
| `factorial_cells.slurm` | Trains any factorial cell (`CELL=` and `SEEDS=`) with the controlled recipe, then evaluates it on the three tiers and both duplicate sets; header holds the reading maps for every cell |
| `fill_crosseval.slurm` | Scores the reference (seed 42) on the five regenerated-gold variants and the D3 arm on the near-miss probe, for the paper's cross-evaluation table; evaluation only |
| `train_invarembed.slurm`, `train_phase2.slurm`, `train_phase2_param.slurm`, `soup_models.py`, `eval_soups.slurm`, `eval_missing_cells.slurm` | The uncapped verified arm, V1 and V2 (and their seeds), the WiSE-FT soups, and the instruction ablations |
| `v1_soups.slurm`, `v1_soups_readout.py` | WiSE-FT soups of V1 with the untrained base at α = 0.3, 0.5 and 0.7, scored on the three tiers and both duplicate sets (`results/eval_*_v1soup-a*.json`, `results/crosslingual_v1soup-a*.json`), and their readout beside the verified-arm soups (`results/v1_soups.json`) |
| `colbert_train.py`, `colbert_eval.py`, `colbert_arms.slurm`, `colbert_readout.py` | Late-interaction arms: ColBERTv2 fine-tuned through PyLate on the recipe and verified files with the dense arms' row cap, gates and dev split (seed 42), scored from a PLAID top-1,000 with the dense harnesses' semantics on the three tiers and both duplicate sets (`results/colbert/`), and the readout (`results/colbert_summary.json`) |
| `round2_closeout.slurm`, `round2_closeout_summary.py` | Four registered closeout blocks (MELD in-harness comparison, D2 on D3's sources, extra row-count control seeds, SABER context-length check) |

**Evaluation harnesses**

| Script | What it does |
|---|---|
| `eval_retrieve.py` + `.slurm` | MathNet-Retrieve harness: any tier or any BEIR-style directory, R@1/5/10 overall and per domain, near-miss separation, embedding cache, per-query rank dumps, self-masking. Reproduces the benchmark's published all-mpnet-base-v2 easy row to the second decimal |
| `eval_crosslingual.py` + `.slurm` | Real-duplicate harness for both mined sets (mandatory self-masking, strict cross-language metric, per-language and per-confidence slices, rank dumps) |
| `eval_bm25.py`, `bm25_all.slurm`, `bm25_samelang_readout.py` | A BM25 lexical baseline (bm25s, k1 1.5, b 0.75) scored with the dense harnesses' semantics on the three tiers, both duplicate sets and the near-miss probe with and without the corpus (`results/bm25_*.json`; rank dumps for the duplicate sets), and the same-language slices with a cluster-bootstrap gap to the untrained base (`results/bm25_samelang_slices.json`) |
| `bge_samelang.slurm`, `bge_samelang_readout.py` | The BGE-large-en-v1.5 base and its two seed-42 arms on the same-language set, and the base on the cross-language set (`results/crosslingual_bge-base.json`); the readout slices them with cluster-bootstrap intervals (`results/bge_samelang.json`) |
| `calib4b.slurm` | The Qwen3-Embedding-4B calibration grid on the easy tier over dtype, query instruction and input length, plus mean pooling and float32 (fourteen settings; `results/calib4b/`, `results/calib4b_summary.json`) |
| `eval_baselines.slurm`, `eval_tf4_baselines.slurm`, `eval_ours_phase1.slurm`, `eval_ctrl_ood.slurm`, `eval_p2_ood.slurm`, `eval_p2_ood_v2.slurm` | The public-model sweep, the environment rerun for models needing transformers 4.x, and the campaign's own evaluation jobs |
| `eval_bright.py`, `eval_meld.py`, `eval_saber.py`, `eval_impliret.py`, `eval_external_ood.slurm`, `impliret_baselines.slurm`, `impliret_calibration.slurm`, `impliret_c3_fill.slurm`, `impliret_calibration.py` | BRIGHT (official protocol), MELD (pairs-only retrieval and separation AUC), SABER-Math (nDCG@10 verified against the official code), ImpliRet (six subsets), and their calibration checks |
| `bright_perquery.slurm`, `bright_bootstrap.py` | The seed-42 arms and the base rerun on the two BRIGHT math splits with per-query nDCG@10 (`results/bright_pq/`), and the query-level paired bootstrap, which also checks that the reruns reproduce the committed aggregates to 1e-4 (`results/bright_bootstrap.json`) |
| `eval_mirb.py`, `mirb_eval.slurm`, `mirb_msedup.slurm`, `mirb_modup_check.py`, `mirb_readout.py` | MIRB's eleven natural-language tasks (its formal-library premise tasks are out of scope) for the base and the two seed-42 arms (`results/mirb/<task>_<tag>.json` and `.perquery.json`; data downloaded to `data/external/mirb/`); the largest task, msedup, runs one job per model with a document-embedding cache, gated by a check that the rewritten scorer reproduces the committed modup result to 1e-4 and matches the old ranking loop (`results/mirb_modup_check.json`); the readout adds per-task paired bootstraps (`results/mirb_summary.json`) |
| `defense_regen_eval.py`, `defense_regen.slurm`, `defense_verdict.py` | The regeneration defense: 1,700 sampled easy-tier golds rebuilt at three prompt distances, scored on the 1,525 whose regenerations the judge verified under all three prompts; thirty evaluations, registered verdicts |
| `defense_regen_gemini.slurm`, `defense_gemini_readout.py` | The regeneration defense rerun with golds rewritten by gemini-3-flash-preview (generated beforehand by `defense_regen_eval.py generate --backend gemini`; manifest `data/defense_eval_gemini/regen_manifest.json`): judged by the same Qwen3-32B-AWQ judge, built into five sets and scored on the six checkpoints (`results/defense_eval_gemini/`); the readout sets it beside the Qwen run (`results/defense_gemini_summary.json`) |
| `recompute_separation.py`, `test_eval_retrieve_selfmask.py` | A CPU recomputation of the near-miss separation credential, and the regression test for self-masking and separation alignment |
| `count_certificates.py` | Recomputes three training-set counts the paper quotes from the run config and pair files that hold them (`results/count_certificates.json`) |
| `build_cas_probe.py`, `casprobe.slurm`, `casprobe_eval_driver.py`, `casprobe_solewins.py` | The generator-free near-miss probe: computer-algebra-verified positives and counterexampled minimal-edit negatives, built with the verified arm's generator on benchmark queries that no pair file of the arms or factorial cells contains (one also appears as a replay row in V1 and V2's mixed set; `data/cas_probe/`, `data/casprobe_eval/`, `data/casprobe_pairs_eval/`, `results/cas_probe_build.json`); every campaign model scored on it under its recorded encoding (`results/ranks/casprobe_<tag>.*`, `results/ranks/casprobepairs_<tag>.*`); and the recipe arm's hard-tier sole wins on it (`results/casprobe_solewins.json`) |

**Attacks on the external benchmarks**

| Script | What it does |
|---|---|
| `generate_saber_pairs.py`, `saber_attack.slurm`, `make_saber_channels.py`, `saber_label_attack.slurm`, `saber_label_verdict.py`, `saber_label_bootstrap.py`, `saber_label_seeds.slurm`, `saber_label_seed_verdict.py`, `saber_base_fill.slurm`, `saber_label_significance.slurm` | The SABER-Math attack from its public pipeline (disjointness gates, its own core-idea prompt and Jaccard rule), the pure document and summary channels at a matched budget, seed replication and verdicts |
| `saber_matched.slurm`, `saber_matched_readout.py` | The SABER summary channel trained at the document channel's budget (2,368 rows, 57 steps), three seeds (`results/saber_labelmatched_2368_s*_noprompt.json`, `results/crosslingual_saber-labelmatched-2368-s*.json`), and its readout beside both channels (`results/saber_matched.json`) |
| `generate_meld_pairs.py`, `attack_second_benchmark.slurm`, `meld_attack_seeds.slurm`, `meld_seed_aggregate.py`, `meld_prompt_control.slurm`, `meld_prompt_control_verdict.py` | The MELD attack from a reconstruction of its procedure (four item-level disjointness gates against the evaluation texts), a held-out-domain arm, seed replication, and the rewritten-prompt control |
| `meld_altprompt_matched.slurm` | The rewritten-prompt control regenerated under the attack's own generation procedure (greedy first round, pair-target stop), so that only the prompt differs: `data/meld_attack/{raw,pairs}_meld9altm.jsonl`, `results/meld_atk_attack_meld9altm.json`, `results/meld_compare_attack_meld9altm_vs_*.json`, and its cross-language and hard-tier scores |
| `extract_impliret_prompts.py` | ImpliRet's published generation prompts, extracted verbatim (`data/impliret_attack/`); no attack was trained from them |

**Statistics and registered verdicts**

| Script | What it does |
|---|---|
| `rebuild_stats.sh`, `aggregate_ctrl_stats.py`, `dump_ranks.slurm`, `bootstrap_stats.py` | Seed aggregation, per-query rank dumps, and the nested bootstrap behind `results/final_stats.json` |
| `reference_control_did.py`, `factorial_verdict.py`, `did_scale_robustness.py` | The recipe-free reference decomposition, the factorial table, and the difference-in-differences on three scales |
| `clean_slice_analysis.py` | The contamination audit by query slice and the leakage attribution |
| `review3_verdict.py` | Registered readouts of the cells added by `gen_survey.slurm`, `gen_split.slurm`, `rejudge.slurm` and `bt2.slurm`, of the generator-free near-miss probe and of the 4B calibration grid (`results/review3_verdict.json`) |
| `dose_seed_aggregate.py`, `dose_bootstrap.py` | Ladder statistics across seeds and per-query bootstraps |
| `style_probe.py` + `.slurm`, `style_within_class.py`, `style_tracking_review2.slurm` | The recipe-style detector (fit, transfer, tracking), the within-document-class control, and the tracking of the factorial cells |
| `analyze_hits.py`, `paraphrase_depth.py`, `xling_slice_analysis.py` | Hard-tier hit accounting, paraphrase depth of every training file, and cross-language retention by mining slice |
| `build_hard_audit.py`, `audit_ingest.py`, `test_audit_ingest.py`, `test_audit_sheet.py` | The human-audit kit for the hard-tier wins (`data/hard_audit/`) and its tests |

**Paper and release**

| Script | What it does |
|---|---|
| `dump_dose_prompts.py` | Writes `paper/dose_prompts.tex` from the prompts in `generate_llm_pairs.py`, so the paper cannot drift from the prompts that ran |
| `make_smoke_pairs.py` | Dummy pairs for trainer smoke tests (never train on them) |

### `data/` (tracked)

| Directory | Contents |
|---|---|
| `cas_pairs/` | Verified arm: 14,361 positive records with 9,408 counterexampled negatives over 7,397 problems (`pairs.jsonl`); the signal-matched arm (`pairs_negmatched.jsonl`) |
| `llm_pairs/` | Recipe arm (ladder rung D1): 6,145 rows, 16,169 near-miss negatives, judge-verified |
| `llm_pairs_paraphrase/`, `llm_pairs_style/`, `llm_pairs_unrelated/` | Ladder rungs D2 (6,553 rows), D3 (5,591) and D4 (6,768, no negatives) |
| `llm_pairs_survey/`, `llm_pairs_survey_casnegs/` | A second recipe-free restatement prompt, D5 (a survey author restating another author's problem; positives only, as D4; 6,293 judge-verified rows), alone and with the reference's negatives (6,163 rows; the 605 reference rows whose source has no D5 positive are dropped; `results/review2_cells_build.json`) |
| `llm_pairs_exact_split/` | The recipe prompt with the positive and the near-misses written in separate calls (6,516 rows) |
| `llm_pairs_twojudge/` | The recipe arm's candidates re-judged by a second judge from another vendor (Mistral-Small-24B-Instruct-2501) with the same judge prompt: the 5,398 rows both judges accept |
| `llm_pairs_paraphrase_d3sources/` | D2 restricted to D3's 5,332 sources |
| `llm_pairs_unrelated_casnegs/`, `cas_pairs_nonegs/` | The reference (D4 with the verified arm's negatives) and the negative-free CAS cell |
| `llm_pairs_unrelated_llmnegs/`, `llm_pairs_unrelated_llmnegs_full/` | D4 with the recipe arm's own near-misses, count-matched and full (5,978 rows each) |
| `llm_pairs_unrelated_negs/`, `llm_pairs_unrelated_unrelnegs/`, `llm_pairs_unrelated_unrelnegs_full/` | Near-misses written under an unrelated prompt (6,665 rows), and D4 with them, count-matched and full (6,393 rows each) |
| `cas_pairs_llmnegs/` | CAS positives with the recipe arm's near-misses (6,145 rows) |
| `bt_pairs/`, `bt_pairs_casnegs/` | Back-translated positives (NLLB-200, 6,576 rows) alone and with the reference's negatives |
| `bt2_pairs/`, `bt2_pairs_casnegs/` | Multi-hop back-translated positives (NLLB-200 round trips through French, German and Russian in turn, English standing in for a pivot equal to the source language; 5,723 rows from 6,768 sources, `results/bt2_pairs_build.json`) alone and with the reference's negatives |
| `cas_pairs_unrelnegs/`, `bt_pairs_nonegs/`, `bt_pairs_unrelnegs/`, `bt_pairs_llmnegs/`, `llm_pairs_nonegs/`, `llm_pairs_casnegs/`, `llm_pairs_unrelnegs/` | The seven table-fill cells of the positives x negatives grid (E-T1..E-T7, 2026-09-08; `scripts/build_review4_cells.py`, certificate `results/review4_cells_build.json`): CAS positives with unrelated-prompt near-misses; back-translated positives alone, with unrelated-prompt near-misses and with D1 near-misses; D1 positives alone, with the reference's CAS negatives and with unrelated-prompt near-misses |
| `llm_pairs_cleanfull/` | The recipe arm under the corrected gate at the full budget: 4,101 surviving rows plus 2,044 regenerated from fresh sources (`new_rows.jsonl`: all 2,602 verified regenerated rows, of which the first 2,044 are used) |
| `llm_pairs_samesrc/` | The same-source control: the 4,101 surviving rows plus 2,044 second rewrites of the same sources (generation seed 1), 6,145 rows over 4,101 sources (`new_rows.jsonl`: the 3,783 converted second rewrites, of which the first 2,044 are used; `results/llm_pairs_samesrc_build.json`) |
| `cas_pairs_item16/` | The verified arm's generator rerun at four times every budget on the 1,131 problems of the 7,585-problem fresh-source pool that carry a relational span (289 of them among the 2,044 fresh sources the retrain used): 9 rows from 6 sources, none of them a source the retrain used (`results/cas_pairs_item16_stats.json`, `results/item16_census.json`) |
| `crosslingual_eval/` | The cross-language duplicate evaluation: 393 non-English queries, 457 qrels, the full 27,817-problem corpus, cluster provenance, the 779 ids training must exclude; own README; CC-BY-4.0 |
| `samelang_eval/` | The same-language reprint evaluation: 498 queries from the 242 monolingual clusters, with the exact-text and clean-of-training flags that define the primary slice (125 queries); own README; CC-BY-4.0 |
| `hard_audit/` | The human-audit kit: annotator guide, 100 stratified items, self-contained annotation sheet |
| `cas_probe/`, `casprobe_eval/`, `casprobe_pairs_eval/` | The generator-free near-miss probe: 1,107 benchmark queries that no pair file of the arms or factorial cells and no duplicate cluster touches (one appears as a replay row in V1 and V2's mixed set), each with a computer-algebra-verified positive and on average 1.885 counterexampled negatives (`probe.jsonl`; `results/cas_probe_build.json`), as BEIR-style sets inside the 27,817-problem public corpus (31,011 documents) and on the 3,194 probe documents alone |
| `defense_eval/` | The regeneration manifest (1,700 sampled queries; the 1,525-query shared slice has three verified regenerated golds each) and build summary; the rebuilt evaluation directories regenerate from it |
| `defense_eval_gemini/` | The regeneration manifest of the rerun whose golds gemini-3-flash-preview rewrote (the same 1,700 queries, judged by Qwen3-32B-AWQ; 1,127 queries have all three regenerations verified) and its build summary; the evaluation directories regenerate from it |
| `phase2/` | The shared source list and the build statistics of the mixed training set |
| `saber_attack/`, `saber/upstream_ref/` | SABER attack: the 4,917 source ids (`source_ids.txt`), the 4,733 generated summaries (`summaries.jsonl`) and the summary-channel pairs (`pairs_summary_channel.jsonl`, 4,733 rows), the full attack pair file (`pairs.jsonl`, 16,194 rows), and the smoke-test summaries (`summaries_smoke.jsonl`); the document-channel pairs are rebuilt by `make_saber_channels.py` (see below); byte-identical copies of the official SABER-Math code (CC BY-SA 4.0) |
| `meld_attack/`, `external/meld/` | MELD attack: the nine domain pairs (`domain_pairs.json`); for each arm the raw generations (`raw_<group>.jsonl`, with its `.summary.json`) and the gated trainer file (`pairs_<group>.jsonl`), groups `meld9` (the attack, 3,150 rows), `heldout` (the held-out-domain arm), `meld9alt` (the rewritten-prompt control) and `meld9altm` (that control regenerated under the attack's own generation procedure),  per-call generation failure dumps (`genfail_*.txt`); the two MELD benchmark files as downloaded from its Hugging Face release (Apache-2.0 per that release; no licence file is copied here) |
| `impliret_eval/`, `impliret_attack/` | The ImpliRet subsets used for evaluation, and its published prompts |

Not in this repository (see `.gitignore`): `data/mathnet_corpus.parquet` (`download_corpus.py`) and `data/retrieve/` (fetched from the benchmark's Hugging Face release by `eval_retrieve.py` on first use), `data/external/mirb/` (MIRB's Hugging Face datasets; see `eval_mirb.py`), `data/defense_eval/*/` and `data/defense_eval_gemini/*/` (rebuilt from the manifests by `defense_regen_eval.py build --out-root <dir>`), and `data/saber_attack/pairs_doc_channel.jsonl` (rebuilt byte for byte from `pairs.jsonl` by `make_saber_channels.py`). The generated training files that cost GPU or API time to rebuild are included: the MELD attack raw and pair files (`data/meld_attack/raw_*.jsonl`, `pairs_*.jsonl`), the SABER attack pairs, summaries and summary-channel pairs (`data/saber_attack/`), V2's mixed training set (`data/phase2/mixed_pairs.jsonl`, rebuilt by `build_phase2_data.py`), and the raw generator output of every LLM-written training set (`data/pairs/llm_pairs*.jsonl`, with the source-id lists `data/pairs/source_ids*.txt`), which the style detector and the conversion scripts read. The trained style detector and its per-document scores are included as well (`results/style_probe_model.joblib`, `results/style_scores_retrieve.npz`).

### `results/` (tracked)

One machine-written JSON per evaluation or analysis; `results/README.md` explains the `.INVALID` convention (one retained but never-cited file) and the one credential corrected after a harness fix.

| Family | Contents |
|---|---|
| `eval_<tier>_<model>.json`, `crosslingual_<model>.json` | Every MathNet-Retrieve tier evaluation and every cross-language duplicate evaluation (the paper's tables are read from these) |
| `ctrl_seed_stats.json`, `final_stats.json`, `did_scale_robustness.json` | The controlled experiment over eight seeds, its nested-bootstrap intervals and difference-in-differences, and the scale check |
| `reference_control_did.json`, `factorial_2x2.json` | The recipe-free reference decomposition and the factorial table, with the construction certificates `llmnegs_cells_build.json`, `review2_cells_build.json`, `bt_pairs_build.json`, `bt_pilot.json` |
| `dose_seed_stats.json`, `dose_bootstrap.json`, `d2_on_d3_sources_manifest.json` | The ladder over seeds, its bootstraps and the selection control |
| `samelang_verdict.json`, `samelang_models.tsv` | The same-language readout and the manifest of every model's encoding convention |
| `clean_slice_analysis.json`, `clean_gate_stats.json`, `cleanfull_verdict.json` | The contamination audit and both corrected-gate retrains |
| `style_probe_*.json`, `style_tracking_review2.json`, `style_within_class.json`, `style_probe_public_models.json` | The style probe (the `_stripped` file is the canonical, artifact-stripped fit; `style_probe_correlations.json`, `_features.json` and `_transfer.json` are its current stage outputs; the `_v1_artifact` and `_backup` files are retained as superseded), its tracking statistics and the public-model null |
| `analyze_hits_*.json`, `analyze_hits_perquery_hard.jsonl`, `ranks_*_hard.jsonl` | Hard-tier hit accounting and the per-query ranks behind the audit kit |
| `saber_*.json`, `meld_*.json`, `bright_*.json`, `impliret_*.json` | The external harnesses, the attacks, their seed replications, disjointness certificates and calibration checks |
| `defense_eval/` | The thirty regeneration-defense evaluations, plus the reference's five (`eval_<set>_reference.json`, from `fill_crosseval.slurm`) |
| `defense_eval_gemini/`, `defense_gemini_summary.json`, `gemini_usage_firstpass.json`, `gemini_usage.json` | The thirty evaluations of the Gemini-regenerated rerun, its summary beside the Qwen run, and the Gemini API call and token counts of the first pass and of the retry of unparseable outputs |
| `bm25_*.json` | The BM25 baseline on the tiers, both duplicate sets and the probe in both forms, and `bm25_samelang_slices.json`, the same-language slices and their gap to the untrained base (`bm25_samelang_readout.py`) |
| `count_certificates.json` | Certificates for three training-set counts the paper quotes: the uncapped verified arm's contrastive examples (training triplets plus pairs) and the SABER document channel's distinct sources and positives, whose pair file is rebuilt rather than stored (`count_certificates.py`) |
| `colbert/`, `colbert_summary.json` | The ColBERTv2 base and both late-interaction arms on each evaluation set, and their readout |
| `mirb/`, `mirb_summary.json`, `mirb_modup_check.json` | MIRB per task and model (`<task>_<tag>.json` with `.perquery.json`, eleven tasks by three models), the summary with paired bootstraps, and the scorer check |
| `bright_pq/` | The per-query BRIGHT reruns (`.agg.json`, `.perquery.json`) behind `bright_bootstrap.json` |
| `calib4b/`, `calib4b_summary.json` | The fourteen settings of the 4B calibration grid, ranked |
| `cas_probe_build.json`, `casprobe_solewins.json` | The near-miss probe's construction certificate and its sole-win breakdown (per-model probe scores are in `results/ranks/casprobe_*.summary.json` and `casprobepairs_*.summary.json`) |
| `review3_verdict.json`, `samesrc_verdict.json`, `bge_samelang.json`, `v1_soups.json`, `saber_matched.json` | Readouts of the added factorial cells, the probe and the grid; the same-source control; the BGE same-language slices; the V1 soups; the step-matched SABER summary arm |
| `bt2_pairs_build.json`, `llm_pairs_cleanfull_build.json`, `llm_pairs_samesrc_build.json`, `cas_pairs_item16_stats.json`, `item16_census.json` | Construction certificates of the multi-hop back-translations, both regenerated recipe files and the relaxed CAS rerun, and the relational-span census of the fresh sources |
| `e2_negmatched_*.json`, `paraphrase_depth.json`, `xling_slice_analysis.json`, `round2_closeout.json`, `separation_recompute_hard_qwen3-4b.json` | The signal-matched control, paraphrase depth, retention by slice, the closeout blocks, and the recomputed separation credential |
| `cas_census.json`, `cas_expr_results.json`, `cas_pairs_stats.json`, `normalizer_recovery.json`, `duplicate_candidates.jsonl`, `duplicate_mining_summary.json`, `llm_pairs*_summary.json` | Construction certificates for the corpus census, the verified pairs, the mined duplicates and every generation run (`llm_pairs_summary.json` is the recipe arm's) |
| `*_failures_<jobid>.log`, `*_failures.txt`, `dose_dryrun_*.txt`, `llm_pairs_dryrun_prompts.txt`, `*smoke*.json` | Per-job failure lists, dry-run prompt dumps and CPU smoke-test runs, retained for auditability and never cited |

Per-query rank dumps (`results/ranks/*.ranks.jsonl`) and per-run summaries (`results/ranks/*.summary.json`) of the per-seed, factorial, corrected-gate, same-language and near-miss-probe runs live under `results/ranks/`; they are regenerated by `scripts/dump_ranks.slurm` and by the evaluation legs of the jobs that wrote them. The summaries are in this repository, and they are all that `factorial_verdict.py` and `cleanfull_verdict.py` read. `samesrc_verdict.py` reads the same-source control's summaries (`*_samesrc-llm-s*.summary.json`), which are not included: rerun `sbatch scripts/samesrc_train.slurm` first; without them the script stops and leaves `results/samesrc_verdict.json` unchanged. The rank dumps are not included (1.8 GB); rebuild them before running the scripts that read them: `bootstrap_stats.py` (at per-query level), `reference_control_did.py`, `samelang_verdict.py`, `dose_bootstrap.py`, `clean_slice_analysis.py` (the leakage attribution skips seeds without dumps), `xling_slice_analysis.py`, `review3_verdict.py` (the same-language primary slice and the near-miss probe), `casprobe_solewins.py`, `bge_samelang_readout.py`, `bm25_samelang_readout.py`, `colbert_readout.py` (the same-language slices) and `v1_soups_readout.py` (the same-language column). Scheduler logs are kept offline; `reproducibility.json` records each run's job id.

### `paper/`

`main.tex`, `sections/*.tex`, `tab_design.tex`, `tab_crosseval.tex`, `fig_inversion.tex`, `fig_inversion_full.tex`, `dose_prompts.tex`, `references.bib`, `main.bbl` (kept because arXiv does not run BibTeX) and the arXiv style `PRIMEarxiv.sty`. Build with `cd paper && latexmk -pdf main.tex` or `cd paper && tectonic -X compile main.tex`.

### Model checkpoints

Each training run writes `final/` and a `run_config.json` audit record (command line, arguments, training-data statistics, runtime and final loss). The repository keeps every run's `models/<run>/run_config.json`; the weights of all but the three released models are left out (retrain them with the commands in `reproducibility.json`). The three models the paper releases, all seed 42:

- **D4, the recipe-free rung:** [KAUSTAcademy/RecipeMatching_D4_Qwen3-Embedding-0.6B](https://huggingface.co/KAUSTAcademy/RecipeMatching_D4_Qwen3-Embedding-0.6B) on Hugging Face; encode queries with the model's `query` prompt.
- **The 4B verified arm:** in this repository, `models/ctrl-cas-6145-4b/final/`, a rank-16 LoRA adapter over Qwen3-Embedding-4B; `SentenceTransformer("models/ctrl-cas-6145-4b/final")` loads it when `peft` is installed; encode queries with the `query` prompt.
- **V2, the mixed-supervision model:** [KAUSTAcademy/RecipeMatching_V2_Qwen3-Embedding-0.6B](https://huggingface.co/KAUSTAcademy/RecipeMatching_V2_Qwen3-Embedding-0.6B) on Hugging Face; encode queries with its instruction (below), not the stored `query` prompt.

Each is regenerated by the command listed:

| Model | Training file | Command |
|---|---|---|
| The recipe-free rung D4 (`models/dose-unrelated-6145`) | `data/llm_pairs_unrelated/pairs.jsonl` | the training block of `scripts/dose_response.slurm` (the full job regenerates the ladder's pairs first): `python scripts/train_invarembed.py --train-file data/llm_pairs_unrelated/pairs.jsonl --max-rows 6145 --model Qwen/Qwen3-Embedding-0.6B --model-dtype bfloat16 --bf16 --loss cmnrl --batch-size 256 --mini-batch-size 16 --scale 20 --epochs 3 --lr 2e-5 --warmup-ratio 0.05 --seed 42 --dev-frac 0.05 --eval-steps 100 --query-prompt-name query --output-dir models/dose-unrelated-6145` |
| The 4B verified arm (`models/ctrl-cas-6145-4b`, LoRA adapter over Qwen3-Embedding-4B) | `data/cas_pairs/pairs.jsonl` | `sbatch scripts/train_ctrl_4b.slurm` |
| V2, the mixed-supervision model (`models/qwen3-0.6b-p2-instr`) | `data/phase2/mixed_pairs.jsonl` (included; `scripts/build_phase2_data.py` regenerates it) | `sbatch scripts/train_phase2.slurm` (trains V1 then V2) |

Evaluate V2 with its instruction: `--query-prompt $'Instruct: Given a math problem, retrieve problems that are mathematically equivalent to it.\nQuery:'`.

---

## Environments

`requirements.txt` documents four stacks that are not co-installable: (1) training and evaluation (Python 3.11, torch 2.11.0 with CUDA 12.8, transformers 5.14.1, sentence-transformers 5.6.1, sympy 1.14.0, scikit-learn, pytrec-eval, bm25s 0.3.11 for the BM25 baseline); (2) the third-party 7–8B baselines, which need transformers 4.51.3; (3) pair generation, which serves Qwen3-32B-AWQ through vLLM 0.26.0; (4) the late-interaction ColBERTv2 arms, through PyLate 1.6.0 (sentence-transformers 5.3.0, transformers 5.3.0). The Gemini-regenerated defense golds were generated through the Gemini API with the standard library (`generate_llm_pairs.py` reads `GEMINI_API_KEY`). Every GPU job ran on a single A100 80GB.

---

## Reproducing the results

All commands run from the repository root with the training and evaluation environment active. Every `.slurm` job can be read as a shell script: the commands are sequential and the pre-registered predictions are in the header.

### 1. Data preparation

```bash
python scripts/download_corpus.py            # data/mathnet_corpus.parquet
python scripts/build_anchor_mapping.py       # anchor_to_corpus_mapping.json (v1 gate)
python scripts/build_anchor_mapping_v2.py    # anchor_to_corpus_mapping_v2.json (corrected gate)
python scripts/cas_census.py                 # results/cas_{census,expr_results}.json
python scripts/normalize_latex.py --workers 8
python scripts/mine_duplicates.py            # results/duplicate_candidates.jsonl
python scripts/build_crosslingual_eval.py    # data/crosslingual_eval/
python scripts/build_samelang_eval.py        # data/samelang_eval/
```

### 2. Training pairs

```bash
sbatch scripts/generate_cas_pairs.slurm      # verified arm -> data/cas_pairs/pairs.jsonl (CPU only)
python scripts/make_source_ids.py            # writes data/pairs/source_ids.txt; the tracked copy data/phase2/cas_source_ids.txt (7,089 ids) is what the next job reads
sbatch scripts/generate_llm_pairs.slurm      # recipe arm (generation environment; SOURCE_IDS_FILE defaults to data/phase2/cas_source_ids.txt) -> data/pairs/llm_pairs.jsonl
python scripts/convert_llm_pairs.py          # -> data/llm_pairs/pairs.jsonl
```

### 3. The controlled experiment and its statistics

```bash
sbatch scripts/train_controlled_exp.slurm    # seed 42, both arms, three tiers
sbatch scripts/eval_ctrl_ood.slurm           # seed 42, cross-language duplicates
sbatch --export=ALL,SEED=43 scripts/train_ctrl_param.slurm   # repeat for 44..49
sbatch scripts/train_ctrl_4b.slurm           # 4B replication
bash scripts/rebuild_stats.sh                # results/ctrl_seed_stats.json, results/final_stats.json (submits dump_ranks.slurm and waits if rank dumps are missing; NO_SUBMIT=1 skips that)
python scripts/did_scale_robustness.py
```

### 4. The ladder, the reference and the factorial

```bash
sbatch scripts/dose_response.slurm           # D2, D3, D4 generated and trained
sbatch scripts/dose_seeds.slurm              # seeds 43, 44 and the row-count control
python scripts/dose_seed_aggregate.py && python scripts/dose_bootstrap.py --seeds 42,43,44
python scripts/build_factorial_cells.py      # the reference and the negative-free CAS cell
sbatch --export=ALL,CELL=d4casnegs scripts/factorial_cells.slurm            # seeds 42-44
sbatch --export=ALL,CELL=d4casnegs,SEEDS="45 46 47 48 49" scripts/factorial_cells.slurm
python scripts/build_llmnegs_cells.py        # then CELL=d4llmnegs and CELL=d4llmnegsfull
sbatch scripts/gen_unrelated_negs.slurm      # unrelated-prompt near-misses; submits CELL=d4unrelnegs{,full}
sbatch scripts/backtranslate_anchors.slurm   # back-translated positives
python scripts/build_review2_cells.py --which r8   # then CELL=casllmnegs
python scripts/build_review4_cells.py --which all   # then CELL=casunrelnegs|btnonegs|btunrelnegs|btllmnegs|d1nonegs|d1casnegs|d1unrelnegs (E-T1..E-T7)
sbatch scripts/fill_crosseval.slurm                # cross-evaluation table: reference on regenerated golds, D3 on the probe
python scripts/build_review2_cells.py --which r9   # then CELL=btcasnegs
python scripts/factorial_verdict.py          # results/factorial_2x2.json
python scripts/reference_control_did.py --boot 5000   # results/reference_control_did.json
```

### 5. Duplicate evaluations, contamination and the probes

```bash
sbatch scripts/samelang_eval.slurm           # every campaign model on the same-language set; results/samelang_verdict.json
python scripts/clean_slice_analysis.py       # results/clean_slice_analysis.json
SEEDS="42 43 44 45 46 47 48 49" sbatch scripts/clean_gate_retrain.slurm && python scripts/clean_gate_verdict.py   # the job defaults to seeds 42-44
python scripts/make_clean_sources.py && sbatch scripts/cleanfull_generate.slurm
python scripts/build_cleanfull_llm.py && sbatch scripts/cleanfull_train.slurm && python scripts/cleanfull_verdict.py
python scripts/build_e2_negmatched_cas.py && sbatch scripts/e2_negmatched.slurm && python scripts/e2_verdict.py
python scripts/style_probe.py --stage fit,score --out-summary results/style_probe_stripped.json && sbatch scripts/style_probe.slurm
    # fit also reads the raw ladder files data/pairs/llm_pairs{,_paraphrase,_style,_unrelated}.jsonl (section 4);
    # the correlate stage needs one --model-emb per model; the job passes them from .emb_cache
python scripts/style_within_class.py && sbatch scripts/style_tracking_review2.slurm
python scripts/analyze_hits.py && python scripts/paraphrase_depth.py && python scripts/xling_slice_analysis.py
```

### 6. Public models, external benchmarks and the defense

```bash
sbatch scripts/eval_baselines.slurm          # the public-model sweep
sbatch scripts/eval_external_ood.slurm       # MELD and BRIGHT for the campaign models
sbatch scripts/impliret_baselines.slurm && python scripts/impliret_calibration.py
sbatch scripts/saber_attack.slurm            # SABER attack from its public pipeline
python scripts/make_saber_channels.py && sbatch scripts/saber_label_attack.slurm && sbatch scripts/saber_label_seeds.slurm
python scripts/saber_label_seed_verdict.py
sbatch scripts/attack_second_benchmark.slurm # MELD attack
sbatch scripts/meld_attack_seeds.slurm && python scripts/meld_seed_aggregate.py
sbatch scripts/meld_prompt_control.slurm && python scripts/meld_prompt_control_verdict.py
sbatch scripts/defense_regen.slurm && python scripts/defense_verdict.py
```

### 7. Any checkpoint on any evaluation

```bash
python scripts/eval_retrieve.py --tier hard --model models/<run>/final --device cuda \
    --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
    --output results/eval_hard_<run>.json
python scripts/eval_crosslingual.py --model models/<run>/final --device cuda \
    --model-dtype bfloat16 --query-prompt-name query --max-seq-length 1024 \
    --output results/crosslingual_<run>.json
python scripts/eval_crosslingual.py --eval-dir data/samelang_eval --model models/<run>/final ...
```

Self-masking is mandatory on both duplicate sets: every query also exists in the corpus under its own id, and `eval_crosslingual.py` masks it automatically.

---

## Registered predictions

Every registered reading is stated, with its threshold and falsifier, in the header of the job that produced the data, and scored by a verdict script that reads only result files: the ladder (`dose_response.slurm`, `dose_seeds.slurm`), the factorial cells (`factorial_cells.slurm`), the signal-matched arm (`e2_negmatched.slurm`, `e2_verdict.py`), the corrected-gate retrains (`clean_gate_retrain.slurm`, `clean_gate_verdict.py`, `cleanfull_train.slurm`, `cleanfull_verdict.py`), the same-language readout (`samelang_eval.slurm`, `samelang_verdict.py`), the SABER attack (`saber_attack.slurm`, `saber_label_attack.slurm`, `saber_label_seeds.slurm`), the MELD attack (`attack_second_benchmark.slurm`, `meld_attack_seeds.slurm`, `meld_prompt_control.slurm`), the external sweep (`eval_external_ood.slurm`) and the regeneration defense (`defense_regen_eval.py`). Later controls carry their readings the same way: the five cells that `gen_survey.slurm`, `gen_split.slurm`, `rejudge.slurm` and `bt2.slurm` add, in the header of `factorial_cells.slurm`; the same-source control in `samesrc_train.slurm` (scored by `samesrc_verdict.py`); the relaxed-budget CAS rerun in `item16_cas_relaxed.slurm` (read out by `item16_census.py`); the 4B calibration grid in `calib4b.slurm`; and the eleventh MIRB task in `mirb_msedup.slurm`. `review3_verdict.py` scores the added cells, the grid and the generator-free near-miss probe, whose readings (`casprobe.slurm`) were registered in a log that is not included. Registration is self-attested: each prediction was written into the header of the script before that job ran, so its wording can be checked against the code that ran, but no external timestamp is claimed. Readings that landed the other way are reported as such in the paper (the easy-tier ladder gradient, the ladder's real-duplicate band, the MELD retention prediction, the SABER document attack and the no-arm-difference readings on SABER, MELD and BRIGHT, and the template-to-paraphrase decay of the defense).

---

## Harness calibration

Our MathNet-Retrieve harness reproduces the benchmark's published all-mpnet-base-v2 easy-tier row (6.75 / 82.47 / 91.07, `results/eval_easy_all-mpnet-base-v2.json`) to the second decimal and its near-miss separation at full scale (11 of 14,993 Qwen3-Embedding-4B hard-tier queries rank the gold above every near-miss, `results/separation_recompute_hard_qwen3-4b.json`). Our SABER nDCG is bit-identical to the official metric code over 500 random cases (`reproducibility.json`, entry `saber_harness_validation`). Three checks fail and are reported as such: our best untrained Qwen3-Embedding-4B easy-tier row with the model's query prompt at 2,048 tokens is 11.96 R@1 (`results/eval_easy_qwen3-4b-instr2048.json`; the baseline table above lists the unprompted 11.15) against the published 14.76, and a fourteen-setting grid over dtype, query instruction and input length gets no closer than 13.19 (`scripts/calib4b.slurm`, `results/calib4b_summary.json`), so the gap stays unresolved; MELD has no official implementation and the two published rows we can compare read high in our harness; ImpliRet's one check fails (our ReasonIR-8B row reads 19.45, `results/impliret_reasonir-8b.json` key `macro_avg`, against the published 13.64 recorded in `results/impliret_calibration.json`). MELD and ImpliRet numbers are therefore comparable across our own models only. Details: the paper's Appendix A.

## Caveats carried into the paper

Duplicate-mining precision is about 85–90% on a hand-read sample, and the miner favours formula overlap (per-language counts and the language-identification caveats are in `data/crosslingual_eval/README.md`); neither duplicate set carries constructed near-misses, so neither can detect the rank-1 discrimination the hard tier targets; the same-language primary slice is small (from its interval widths in `results/samelang_verdict.json`, the smallest arm difference it would detect at 80% power is about 12 to 14 points); MathLeap models were evaluated at 2,048 tokens instead of their shipped 128-token truncation; the 4B replication moves three variables at once (LoRA, learning rate, gradient checkpointing), symmetrically across arms; counterexampled negatives treat variables as free reals or complexes; 220 of the 393 cross-language queries overlap MathNet-Retrieve anchors (`data/crosslingual_eval/build_summary.json`) and are sliced, not dropped. No human has yet read the hard-tier queries the recipe arm alone wins; the audit kit in `data/hard_audit/` is released for that.

## Licence

- **Code** (`scripts/`): MIT, see `LICENSE`.
- **Data and results** we produced (`data/`, `results/`, `models/*/run_config.json`, `reproducibility.json`, the anchor mappings): CC BY 4.0, see `LICENSE-DATA`. The pair files and evaluation sets are derived from the MathNet corpus, itself CC BY 4.0.
- **Paper source** (`paper/`): CC BY 4.0, see `LICENSE-PAPER`. `paper/PRIMEarxiv.sty` is adapted from George Kour's arxiv-style and keeps its MIT notice.
- **Third-party material** keeps its own licence: the MathNet corpus and both derived duplicate evaluations are CC BY 4.0; the copied SABER-Math reference code (`data/saber/upstream_ref/`) is CC BY-SA 4.0; the MELD benchmark files (`data/external/meld/`) are Apache-2.0 per their Hugging Face release; the ImpliRet subsets (`data/impliret_eval/`) are MIT. Cite the respective papers when using them.

## Citation

```bibtex
@misc{habibullah2026recipematching,
  title  = {Recipe-Matching, Not Equivalence},
  author = {Habibullah, Ali and Alshiekh, Mohammad and Alshoibi, Yazan and Khan, Salman and Khan, Naeemullah},
  year   = {2026}
}
```
