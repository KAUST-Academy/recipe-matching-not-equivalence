# `results/` — provenance notes

Every file here is a machine-written artifact of one command recorded in
`../reproducibility.json` (match by the `outputs` field). Two conventions:

* `*.json` — a result that may be cited.
* `*.json.INVALID` — a result that was produced by a harness misconfiguration
  and is **retained only for auditability**. Never cite these; never let a
  table-filling pass read them. Each one is explained below.

## Retained-but-invalid artifacts

### `eval_crosslingual_rader-qwen25-7b.json.INVALID`

* **What it claims:** RaDeR-Qwen25-7B on the mined real-duplicate cross-lingual
  eval set (`data/crosslingual_eval`, 393 non-English queries vs the full
  27,817-doc public MathNet corpus): R@1 `0.25` / R@5 `95.93` / R@10 `96.44`.
* **Why it is invalid:** it was produced with
  `scripts/eval_retrieve.py --data-dir data/crosslingual_eval` on 2026-07-29,
  when that harness had **no self-masking**. In that eval set every query
  problem also exists in the corpus under the query's own `_id`, so for each
  query its own document was retrieved at rank 1 and the true gold (an
  officially reprinted duplicate of the same problem in another language) was
  displaced to rank 2. The numbers are the arithmetic signature of that bug —
  R@1 near zero with R@5 near 100 — not a measurement of the model.
* **Fixed on 2026-07-30:** `eval_retrieve.py` now self-masks by default
  (`--no-self-mask` opts out and is recorded in the output JSON), and
  `scripts/test_eval_retrieve_selfmask.py` is a CPU regression test that
  reproduces both the bug and the fix on a synthetic fixture.
  `scripts/eval_crosslingual.py` has always self-masked and remains the
  preferred harness for this eval set: it additionally reports the strict
  cross-language metric and the per-language / per-confidence /
  anchor-overlap breakdowns that the paper uses.
* **Consequences:** none for the paper. No number from this file was ever
  reported in `README.md`, `paper/`, or `reproducibility.json`; RaDeR-Qwen25-7B
  simply has **no** real-duplicate number in the campaign. To obtain a valid
  one, run `scripts/eval_crosslingual.py` with RaDeR's encoding conventions
  (`--query-prompt 'query: Given a Math problem, retrieve relevant examples
  that help answer the problem\n' --doc-prompt 'document: '`, last-token
  pooling + appended EOS — see `results/eval_easy_rader-qwen25-7b.json` for the
  exact flags used on the benchmark tiers; note `eval_crosslingual.py` currently
  exposes no `--pooling/--append-eos`, so RaDeR needs the flags ported first).

## Corrections applied to existing files

`figure6_separation` blocks written **before 2026-07-30** paired each query's
positive similarity with a *later* query's near-miss similarities whenever some
query had a gold document but no near-miss documents (7 such queries on the full
15,000-query tiers). The bug moved only the separation block — `overall` and
`per_domain` recall are unaffected — and it is fixed in `eval_retrieve.py`
(records are now accumulated per query, and the block reports
`n_queries_gold_but_no_near_miss`). The one number quoted in the paper from an
affected block is the harness credential for Qwen3-Embedding-4B on the hard
tier: `pct_pos_above_all_hard_negs` **0.09 → 0.07** (13 → 11 of 14,993 queries),
recomputed from the cached embeddings in `.emb_cache` (see the
`eval_retrieve_selfmask_alignment_fix` entry in `reproducibility.json`). The
stored `results/eval_hard_*.json` files still carry the pre-fix separation
blocks; they are regenerated on the next eval run of the affected model.
