# Hard-Tier Human Audit — Annotator Guide

**Goal.** Judge whether the MathNet-Retrieve *hard* tier is valid: are the "equivalent"
golds truly equivalent to their query, are the near-miss negatives truly *non*-equivalent,
and can a human distinguish them at all (every model scores ~0% R@1 here)?
You will see 100 items; the **first 60 are the core sample** — any prefix is analyzable.

## Equivalence definition (apply exactly this)

Mark a candidate **Equivalent** iff it has the **same mathematical content** as the query in
the MathNet paper's *Invariance* sense: the same problem under renaming of objects/variables,
algebraic reformulation, or re-characterization of the setup (different story, same constraints,
same answer set / same thing to prove). Superficial changes to numbers or story are fine **as
long as they do not change the mathematics**.

Mark **Not equivalent** if any mathematical content differs — changed constants, weakened or
strengthened bounds, different quantifiers, a different set to count, a different claim —
*even if* the topic, method, and phrasing are nearly identical. Same topic or same solution
recipe is **not** equivalence.

Mark **Ill-formed / cannot judge** if the text is broken/truncated/not a well-posed problem, or
you are genuinely unsure whether a change alters the answer set. **When unsure, choose
cannot-judge rather than guessing** — a guessed label is worse than a missing one.

## Per-item workflow (~15 min/item; 60 items ≈ 15–20 h)

1. Read the QUERY carefully; identify its exact mathematical content (objects, constraints,
   quantifiers, what is asked). Solving fully is not required — extracting the content is.
2. For each of candidates A–D independently: diff its content against the query. Look
   especially for changed constants/exponents, swapped quantifiers, and subtly different
   conclusions (near-misses are built to differ minimally). Pick one of the three labels.
   Expect *usually* 1 equivalent and 3 not — but do NOT force that pattern; genuine label
   errors (0 or 2+ equivalents) are exactly what this audit measures.
3. Set **Disguise difficulty** (1–5): could an expert see the equivalence (of whichever
   candidate is equivalent) *without solving*? 1 = transparent rewording … 5 = only by fully
   solving both.
4. Optional note: suspected label errors, why cannot-judge, interesting disguise tricks.
5. Next. Every click autosaves to the browser's localStorage; closing the tab loses nothing
   (same browser + machine; don't use private/incognito mode).

**Stop anytime.** Items are ordered so any prefix is ~balanced across strata (recipe-hit vs
non-hit) and the four domains; a partial export is fully analyzable. Do NOT open
`audit_items.json` while annotating — it reveals which candidate is the gold.

## Export & ingest

- In the sheet, click **Export answers JSON** (works for partial progress) →
  downloads `hard_audit_answers.json`. Export occasionally as a backup; the latest
  export supersedes earlier ones.
- Then, from the repository root:

```bash
conda activate mathnet
python scripts/audit_ingest.py path/to/hard_audit_answers.json
# -> prints the stats table and writes results/hard_audit_stats.json
```

Open the sheet by double-clicking `audit_sheet.html` in any normal browser with internet
(MathJax loads from CDN; everything else is embedded — no server needed).
