# Same-language organic-duplicate evaluation set (built 2026-09-03)

Companion to `data/crosslingual_eval/` (the cross-lingual real-duplicate
probe). Built by `scripts/build_samelang_eval.py` from the 242
MONOLINGUAL duplicate clusters of `data/crosslingual_eval/clusters.json`,
which the cross-lingual set skipped by design. Every known-language member
of such a cluster is a query and the other members are its golds:
498 queries, 552 qrels, against the full 27817-problem
corpus (the cross-lingual set's `data/crosslingual_eval/corpus.jsonl`; this
directory ships no copy, and `scripts/eval_crosslingual.py` falls back to it).
Self-masking is mandatory, exactly as for the cross-lingual set.

Two per-query flags (in `metadata`) define the slices:

* `exact_text_cluster` -- 143 clusters were mined by the
  exact-text rule (near-byte-identical reprints): a ceiling check, not a
  paraphrase test. The 99 non-exact clusters (formula-,
  near-text- or answer-mined) are the informative ones.
* `clean_of_training` -- the trainer's eval gate removed only the 779 ids
  that are queries or golds of the CROSS-LINGUAL set, so members of
  monolingual clusters could be trained on. 61
  of the non-exact clusters have no member in ANY training pair file of the
  campaign; 38 have at least one (`trained_in` lists the files).

PRIMARY slice = non-exact AND clean of training: 125
queries (115 English). Leaked and exact-text
queries are reported separately, never mixed in.

Metric: R@1 of the best-ranked gold (`overall_any_gold`) and of the best
gold in the query's own language (`same_language_gold`), which coincide
except where a cluster carries a member with unknown language. Evaluate
with `scripts/eval_crosslingual.py --eval-dir data/samelang_eval`; the
registered readout is in `scripts/samelang_eval.slurm` and is computed by
`scripts/samelang_verdict.py`.

License: derived from the public MathNet corpus (CC-BY-4.0); likewise CC-BY-4.0.
