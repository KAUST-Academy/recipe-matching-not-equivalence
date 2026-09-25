#!/usr/bin/env python
"""Headless validation of data/hard_audit/audit_sheet.html.

Checks (no browser available on the login node):
  1. HTML parses; non-void tags are balanced.
  2. Embedded audit data (<script id="audit-data">) is valid JSON with the
     expected shape (100 items, 4 candidates each, matching audit_items.json).
  3. NO provenance leakage into the HTML: no doc ids, roles, strata, domains.
  4. Core JS (state normalization, localStorage save/load round-trip,
     completion counting, export payload) runs correctly under dukpy
     (Duktape, ES5). Real-browser rendering + MathJax remain untested here;
     see AUDIT_GUIDE.md.

Run: python scripts/test_audit_sheet.py
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "data/hard_audit/audit_sheet.html"
ITEMS_PATH = ROOT / "data/hard_audit/audit_items.json"

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}


class BalanceChecker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"closing </{tag}> with empty stack")
        elif self.stack[-1] != tag:
            self.errors.append(f"mismatch: expected </{self.stack[-1]}>, got </{tag}>")
        else:
            self.stack.pop()


def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    items_doc = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    failures = []

    # 1. parse + balance
    p = BalanceChecker()
    p.feed(html)
    p.close()
    if p.errors:
        failures.append(f"tag balance: {p.errors[:5]}")
    if p.stack:
        failures.append(f"unclosed tags at EOF: {p.stack}")

    # 2. embedded JSON
    m = re.search(
        r'<script id="audit-data" type="application/json">\s*(.*?)\s*</script>',
        html, re.S)
    if not m:
        failures.append("audit-data script block not found")
        data = []
    else:
        data = json.loads(m.group(1))  # raises if invalid
        if len(data) != len(items_doc["items"]):
            failures.append(f"embedded item count {len(data)} != {len(items_doc['items'])}")
        for it, prov in zip(data, items_doc["items"]):
            if it["item_index"] != prov["item_index"] or it["anchor_id"] != prov["anchor_id"]:
                failures.append(f"item order mismatch at {it['item_index']}")
                break
            if sorted(it["candidates"].keys()) != ["A", "B", "C", "D"]:
                failures.append(f"item {it['item_index']}: bad candidate letters")
                break
            for L in "ABCD":
                if set(it["candidates"][L].keys()) != {"text"}:
                    failures.append(f"item {it['item_index']}{L}: extra keys leak provenance")
                    break
                if it["candidates"][L]["text"] != prov["candidates"][L]["text"]:
                    failures.append(f"item {it['item_index']}{L}: text mismatch vs provenance")
                    break
            expected_keys = {"item_index", "anchor_id", "query_text", "candidates"}
            if set(it.keys()) != expected_keys:
                failures.append(f"item {it['item_index']}: unexpected top-level keys {set(it.keys()) - expected_keys}")
                break

    # 3. leakage: none of these strings may appear anywhere in the HTML
    for needle in ["::eq::", "::nm::", "recipe_hit", "non_hit", '"role"',
                   '"gold_doc_id"', '"stratum"', '"domain"', '"permutation"']:
        if needle in html:
            failures.append(f"provenance leak: {needle!r} found in HTML")

    # MathJax CDN present
    if "cdn.jsdelivr.net/npm/mathjax@3" not in html:
        failures.append("MathJax CDN script tag missing")

    # 4. core JS under dukpy
    m = re.search(r"<!--CORE_JS_START-->\s*<script>(.*?)</script>\s*<!--CORE_JS_END-->",
                  html, re.S)
    if not m:
        failures.append("CORE_JS block not found")
    else:
        core_js = m.group(1)
        try:
            import dukpy
        except ImportError:
            print("WARNING: dukpy unavailable; core JS untested")
            dukpy = None
        if dukpy:
            test_js = core_js + r"""
var C = AuditCore;
var items = [
  {item_index: 1, anchor_id: 'a1'},
  {item_index: 2, anchor_id: 'a2'},
  {item_index: 3, anchor_id: 'a3'}
];
var store = {};
var mockLS = {
  getItem: function (k) { return store.hasOwnProperty(k) ? store[k] : null; },
  setItem: function (k, v) { store[k] = String(v); }
};
var out = {};

// fresh load -> defaults
var s = C.loadState(mockLS, 'k', items);
out.fresh_cur = s.cur;                          // 1
out.fresh_n_answers = Object.keys(s.answers).length;  // 0
out.fresh_complete = C.countComplete(s, 3);     // 0

// answer item 1 fully, item 2 partially; save; reload
s.answers[1] = {anchor_id: 'a1',
  cand: {A: 'equiv', B: 'not_equiv', C: 'not_equiv', D: 'cannot_judge'},
  disguise: 4, note: 'hi'};
s.answers[2] = {anchor_id: 'a2',
  cand: {A: 'equiv', B: null, C: null, D: null}, disguise: null, note: ''};
s.cur = 2;
C.saveState(mockLS, 'k', s);
var s2 = C.loadState(mockLS, 'k', items);
out.reload_cur = s2.cur;                        // 2
out.reload_complete = C.countComplete(s2, 3);   // 1
out.reload_note = s2.answers[1].note;           // 'hi'
out.reload_disguise = s2.answers[1].disguise;   // 4
out.first_incomplete = C.firstIncompleteIndex(s2, 3);  // 2

// corrupted / hostile state is sanitized
store['bad'] = '{"cur": 99, "answers": {"1": {"cand": {"A": "EVIL"}, "disguise": 77, "note": 5}, "7": {}}}';
var s3 = C.loadState(mockLS, 'bad', items);
out.bad_cur = s3.cur;                           // 1 (99 out of range)
out.bad_candA = String(s3.answers[1].cand.A);   // 'null' (EVIL rejected)
out.bad_disguise = String(s3.answers[1].disguise); // 'null' (77 rejected)
out.bad_has7 = String(s3.answers[7] === undefined ? undefined : 'present'); // undefined
store['junk'] = 'not json{{';
out.junk_cur = C.loadState(mockLS, 'junk', items).cur;  // 1

// export: includes touched only (item 1 complete + item 2 partial)
var exp = C.buildExport(s2, 3, '2026-07-30T00:00:00Z');
out.exp_schema = exp.schema;
out.exp_n_complete = exp.n_items_complete;      // 1
out.exp_keys = Object.keys(exp.answers).sort().join(',');  // '1,2'
out.exp_a1_A = exp.answers[1].cand.A;           // 'equiv'
JSON.stringify(out);
"""
            res = json.loads(dukpy.evaljs(test_js))
            expect = {
                "fresh_cur": 1, "fresh_n_answers": 0, "fresh_complete": 0,
                "reload_cur": 2, "reload_complete": 1, "reload_note": "hi",
                "reload_disguise": 4, "first_incomplete": 2,
                "bad_cur": 1, "bad_candA": "null", "bad_disguise": "null",
                "bad_has7": "undefined", "junk_cur": 1,
                "exp_schema": "hard_audit_answers_v1", "exp_n_complete": 1,
                "exp_keys": "1,2", "exp_a1_A": "equiv",
            }
            for k, v in expect.items():
                if res.get(k) != v:
                    failures.append(f"core JS: {k} = {res.get(k)!r}, expected {v!r}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"ALL audit_sheet TESTS PASSED "
          f"(html {len(html)/1024:.0f} KB, {len(data)} items, core JS verified in dukpy)")


if __name__ == "__main__":
    main()
