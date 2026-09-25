#!/usr/bin/env python3
"""Output-schema checker for the LLM-judged supervision arm (InvarEmbed).

Validates the flat one-row-per-candidate JSONL that generate_llm_pairs.py
emits (pair_id / anchor_text / candidate_text / label / verification / ...).
NOTE: generate_cas_pairs.py does NOT use this schema -- it writes the grouped
trainer format (source_id / positive_text / negatives) that
train_invarembed.py consumes directly; LLM-arm output is bridged into that
same trainer format by scripts/convert_llm_pairs.py (which filters on
verification.verified == true). Import validate_row(), or run on a file:
    python3 validate_pairs_schema.py data/pairs/llm_pairs.jsonl
"""
import json, sys

LABELS = {"positive", "hard_negative"}
ARMS = {"cas", "llm"}
METHODS = {"cas", "llm_judge", "metadata", "none"}
STR_KEYS = ("pair_id", "source_id", "anchor_text", "candidate_text", "channel")


def validate_row(row, idx=0):
    """Return a list of error strings (empty list = row is schema-valid)."""
    e, p = [], f"row {idx}: "
    if not isinstance(row, dict):
        return [p + "not a JSON object"]
    for k in STR_KEYS:
        if not isinstance(row.get(k), str) or not row[k].strip():
            e.append(p + f"'{k}' must be a non-empty string")
    if row.get("label") not in LABELS:
        e.append(p + f"'label' must be one of {sorted(LABELS)}")
    if row.get("arm") not in ARMS:
        e.append(p + f"'arm' must be one of {sorted(ARMS)}")
    v = row.get("verification")
    if not isinstance(v, dict) or v.get("method") not in METHODS \
            or not isinstance(v.get("verified"), bool) or "evidence" not in v:
        e.append(p + "'verification' needs method in %s, verified: bool, evidence" % sorted(METHODS))
    if not (isinstance(row.get("transformation_tags"), list)
            and all(isinstance(t, str) for t in row["transformation_tags"])):
        e.append(p + "'transformation_tags' must be a list of strings")
    if not isinstance(row.get("meta"), dict):
        e.append(p + "'meta' must be an object")
    return e


if __name__ == "__main__":
    errs, n, seen = [], 0, set()
    for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
        row = json.loads(line)
        errs += validate_row(row, n)
        pid = row.get("pair_id") if isinstance(row, dict) else None
        if pid in seen:
            errs.append(f"row {n}: duplicate pair_id {pid!r}")
        seen.add(pid)
    print("\n".join(errs[:50]) or f"OK: {n} rows, all schema-valid")
    sys.exit(1 if errs else 0)
