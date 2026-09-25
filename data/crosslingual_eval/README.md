# Cross-Lingual Duplicate-Retrieval Eval Set

Held-out evaluation benchmark of **real, officially-reprinted cross-language
duplicate problems** mined from the public MathNet corpus (ShadenA/MathNet,
27,817 problems). A query is a competition problem in a non-English language;
the gold documents are the *same problem* as it appears in another national
booklet (usually the English one) — e.g. an IMO/BMO/JBMO/Iberoamerican problem
printed in multiple countries' official olympiad booklets. No synthetic
paraphrasing is involved anywhere: every positive is a real-world reprint,
which makes this an evaluation channel that no LLM-generation pipeline can
contaminate.

Built 2026-07-29 by `scripts/build_crosslingual_eval.py` from
`results/duplicate_candidates.jsonl` (754 verified pairs mined by
`scripts/mine_duplicates.py`; unbiased-sample precision ~85–90%, see
`results/duplicate_mining_summary.json`).

## Files (BEIR-style)

| file | contents |
|---|---|
| `corpus.jsonl` | ALL 27,817 MathNet corpus problems (`_id`, `title`(empty), `text` = problem_markdown) — retrieval runs against the full realistic corpus |
| `queries.jsonl` | 393 queries (`_id` = corpus id of the query problem, `text`, `metadata`) |
| `qrels/test.tsv` | 457 rows `query-id \t corpus-id \t 1`; a query has >1 gold when its duplicate cluster has >2 members |
| `clusters.json` | provenance: all 661 transitive clusters with member languages and the underlying mined pairs (method + confidence per pair) |
| `leakage_exclude_ids.json` | the 779 corpus ids used as queries/golds (+ conservative list of all 1,377 pair members). **Any InvarEmbed training data must exclude these ids** |
| `build_summary.json` | construction statistics |

Query `metadata` fields: `lang`, `cluster_id`, `cluster_size`, `doc_lang`,
`gold_ids`, `gold_langs`, `n_crosslingual_golds`, `supporting_pairs`
(method + confidence of the mined pairs directly involving this query),
`methods`, `min_confidence` (worst confidence among the cluster's pairs),
`country`, `competition`, `query_in_retrieve_anchors`,
`n_golds_in_retrieve_anchors`, `overlaps_retrieve_anchor`.

## Construction

1. **Pairs → clusters.** The 754 verified pairs (all mining methods kept:
   `V2_rare_math` 519, `exact_text` 160, `V1_text` 45, `V4_answer_math` 29,
   `V5_answer_tag` 1) are transitively merged with union-find → 661 clusters
   over 1,377 problems (613 of size 2, 42 of 3, 5 of 4, 1 of 5).
2. **Language.** Member language = the mining pipeline's stopword-based
   lang-ID (conflict-free per problem; approximate for close language pairs,
   e.g. es/pt, hr/sr/sl).
3. **Query selection.** Per cluster, the document language `L_doc` is
   English if any member is English (346 multilingual clusters), otherwise
   the member language that is most frequent corpus-wide (24 clusters) — so
   the query side is always the *rarer* language. Queries = every member with
   a known language ≠ `L_doc`. Monolingual clusters (291, incl. en–en
   exact-reprints) yield no queries; members with lang-ID `unknown` are never
   queries but remain golds.
4. **Golds.** All other members of the query's cluster, any language. By
   construction every query has ≥1 gold in a different language, so the
   strict cross-lingual metric covers 100% of queries.

## Statistics

- **393 queries**, 457 qrels, 60 queries with >1 gold; corpus = 27,817 docs.
- Queries per language: nl 71, zh 63, sl 54, ro 52, de 51, fr 49, es 21,
  hu 16, mn 6, pt 3, mk 3, it 2, af 1, ru 1.
- Document language: en for 368 queries; fr 11, es 6, zh 2, de 2, pt 2,
  it 1, sl 1 for the non-English/non-English clusters.
- Confidence: 368 queries rest only on `high`-confidence pairs; 25 involve a
  `medium` (distinctive-answer-corroborated) pair — sliced separately by the
  harness.

## How to evaluate

```bash
python scripts/eval_crosslingual.py --model <hf-model> [--device cuda ...]
```

Metrics (same convention as `scripts/eval_retrieve.py`, which reproduced
MathNet v2 Table 4 exactly): Recall@1/5/10 = hit-rate of the best-ranked
gold; `strict_crosslingual_gold` restricts golds to those in a language
different from the query's; breakdowns per query language, per confidence,
and by `overlaps_retrieve_anchor`.

**Self-masking is mandatory.** Every query problem also exists verbatim in
`corpus.jsonl` under the same `_id` (kept there because in multi-query
clusters one query is another query's gold document). Any harness must
exclude the corpus document whose `_id` equals the query `_id` from that
query's ranking — `scripts/eval_crosslingual.py` does this automatically.
Off-the-shelf BEIR tooling that skips this step will deflate Recall@k by
ranking the query's own text first.

## Eval integrity

- **MathNet-Retrieve anchor overlap is reported, not dropped.** 220/393
  queries have the query or a gold among the 8,698 corpus problems that
  exactly match MathNet-Retrieve anchors (`anchor_to_corpus_mapping.json`).
  That exclusion list exists to keep *training* data disjoint from
  MathNet-Retrieve; since this set is itself an eval, the overlap causes no
  leakage between the two evals — but results can be sliced by the
  `overlaps_retrieve_anchor` flag (the harness does).
- **Training-side leakage is on you:** any model *trained* on MathNet data
  must exclude the ids in `leakage_exclude_ids.json` (779 eval members;
  conservatively all 1,377 mined-pair members) *in addition to* the 8,698
  MathNet-Retrieve exclusions, or its numbers on this benchmark are
  contaminated.

## Caveats

- **Mining precision ~85–90%** (unbiased hand-labeled sample): expect a few
  percent of qrels to be near-duplicates rather than true duplicates
  (e.g. same problem with an extra sub-question). `min_confidence` slicing
  gives a cleaner-subset view.
- **Recall is a lower bound / coverage is biased.** The public corpus is
  post-deduplication (the MathNet authors removed retrieval+LLM-detected
  duplicates), and the miner requires shared rare LaTeX 4-grams — formula-free
  problems (pure-text combinatorics) and duplicates removed by the authors'
  dedup are under-represented. Language coverage follows booklet
  availability, not language importance.
- **Lang-ID is approximate** (stopword-based); `es` vs `pt` and Slavic
  languages can be confused, and 71 members are `unknown` (golds only).
- Small per-language n (1–71): report per-language numbers with counts, and
  treat af/ru/it/mk/pt/mn rows as anecdotal.

## License

Derived from the public MathNet corpus (ShadenA/MathNet, arXiv:2604.18584),
released under **CC-BY-4.0**. This derived eval set is likewise
**CC-BY-4.0**; cite the MathNet paper when using it.
