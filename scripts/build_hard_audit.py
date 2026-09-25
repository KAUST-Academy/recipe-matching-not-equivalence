#!/usr/bin/env python
"""Build the hard-tier human-audit sample and annotation sheet.

Produces:
  data/hard_audit/audit_items.json  - full provenance (anchor ids, doc ids,
                                      stratum, domain, candidate permutation).
                                      NEVER open this while annotating.
  data/hard_audit/audit_sheet.html  - single self-contained annotation tool
                                      (MathJax from CDN; everything else inline).

Design:
  N=100 items, seed 42, ordered so that ANY prefix is approximately
  stratified: items cycle through 8 cells = {recipe-hit, non-hit} x
  {Algebra, Discrete Mathematics, Geometry, Number Theory}, alternating
  hit/non-hit on every consecutive item. First 60 items = core sample
  (exactly 30/30 hit/non-hit, 15 per domain).

  Per item the annotator sees the anchor query plus 4 candidates (the
  ::eq::hard gold and its 3 ::nm:: near-misses) shuffled into positions
  A-D. The permutation lives only in audit_items.json, not in the HTML.

Usage:
  conda activate mathnet
  python scripts/build_hard_audit.py
"""
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_ITEMS = 100
CORE_N = 60
MAJOR_DOMAINS = ["Algebra", "Discrete Mathematics", "Geometry", "Number Theory"]
# Cell visitation order: alternates hit/rest every item; each cycle of 8
# touches every domain exactly twice (once per stratum).
CELL_ORDER = [
    ("hit", "Algebra"),
    ("rest", "Geometry"),
    ("hit", "Discrete Mathematics"),
    ("rest", "Number Theory"),
    ("hit", "Geometry"),
    ("rest", "Algebra"),
    ("hit", "Number Theory"),
    ("rest", "Discrete Mathematics"),
]

QUERIES = ROOT / "data/retrieve/hard/queries.jsonl"
CORPUS = ROOT / "data/retrieve/hard/corpus.jsonl"
PERQUERY = ROOT / "results/analyze_hits_perquery_hard.jsonl"
MAPPING = ROOT / "anchor_to_corpus_mapping.json"
PARQUET = ROOT / "data/mathnet_corpus.parquet"
OUT_DIR = ROOT / "data/hard_audit"


def load_inputs():
    queries = {}
    with open(QUERIES) as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]

    corpus = {}
    with open(CORPUS) as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = d["text"]

    groups = {}
    with open(PERQUERY) as f:
        for line in f:
            d = json.loads(line)
            groups[d["qid"]] = d["group"]  # 'hit' or 'rest'

    import pandas as pd

    df = pd.read_parquet(PARQUET, columns=["id", "topics_flat"])
    id2topics = dict(zip(df["id"], df["topics_flat"]))
    with open(MAPPING) as f:
        m = json.load(f)
    anchor_domain = {}
    for e in m["mapping"]:
        topics = id2topics.get(e["corpus_id"])
        if topics is None or len(topics) == 0:
            continue
        anchor_domain[e["anchor_id"]] = topics[0].split(">")[0].strip()
    return queries, corpus, groups, anchor_domain


def build_sample(queries, corpus, groups, anchor_domain):
    rng = random.Random(SEED)

    # Eligible anchors per cell (sorted for determinism before shuffling).
    cells = {c: [] for c in CELL_ORDER}
    for aid in sorted(anchor_domain):
        dom = anchor_domain[aid]
        g = groups.get(aid)
        if dom not in MAJOR_DOMAINS or g not in ("hit", "rest"):
            continue
        if aid not in queries:
            continue
        need = [f"{aid}::eq::hard", f"{aid}::nm::0", f"{aid}::nm::1", f"{aid}::nm::2"]
        if not all(n in corpus for n in need):
            continue
        cells[(g, dom)].append(aid)

    for c in CELL_ORDER:
        rng.shuffle(cells[c])

    # Quotas: 12 per cell + 1 extra for the first 4 cells in CELL_ORDER -> 100.
    quotas = {c: 12 for c in CELL_ORDER}
    for c in CELL_ORDER[:4]:
        quotas[c] += 1

    # Emit items by cycling CELL_ORDER until quotas are exhausted.
    picked = {c: 0 for c in CELL_ORDER}
    order = []
    while len(order) < N_ITEMS:
        for c in CELL_ORDER:
            if picked[c] < quotas[c]:
                order.append((c, cells[c][picked[c]]))
                picked[c] += 1
                if len(order) == N_ITEMS:
                    break

    items = []
    for idx, ((stratum, dom), aid) in enumerate(order, start=1):
        docs = [
            {"doc_id": f"{aid}::eq::hard", "role": "gold"},
            {"doc_id": f"{aid}::nm::0", "role": "nm0"},
            {"doc_id": f"{aid}::nm::1", "role": "nm1"},
            {"doc_id": f"{aid}::nm::2", "role": "nm2"},
        ]
        perm = rng.sample(range(4), 4)  # perm[slot] = index into docs
        letters = ["A", "B", "C", "D"]
        candidates = {}
        for slot, letter in enumerate(letters):
            d = docs[perm[slot]]
            candidates[letter] = {
                "doc_id": d["doc_id"],
                "role": d["role"],
                "text": corpus[d["doc_id"]],
            }
        items.append(
            {
                "item_index": idx,
                "anchor_id": aid,
                "stratum": "recipe_hit" if stratum == "hit" else "non_hit",
                "domain": dom,
                "core_sample": idx <= CORE_N,
                "query_text": queries[aid],
                "gold_doc_id": f"{aid}::eq::hard",
                "nm_doc_ids": [f"{aid}::nm::{k}" for k in range(3)],
                "permutation": perm,
                "candidates": candidates,
            }
        )
    return items, {str(c): len(v) for c, v in cells.items()}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hard-Tier Human Audit</title>
<script>
window.MathJax = {
  tex: {inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']],
        processEscapes: true},
  options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: Georgia, 'Times New Roman', serif; margin: 0; background: #f4f2ee; color: #1d1d1f; }
  .wrap { max-width: 880px; margin: 0 auto; padding: 16px 20px 80px; }
  header { position: sticky; top: 0; background: #f4f2ee; padding: 10px 0 6px; z-index: 5;
           border-bottom: 1px solid #d8d4cc; }
  .progress-text { font-size: 14px; color: #555; font-family: Helvetica, Arial, sans-serif; }
  .bar { height: 8px; background: #ddd8cf; border-radius: 4px; margin: 6px 0; overflow: hidden; }
  .bar-fill { height: 100%; background: #2f6f4f; width: 0%; transition: width .2s; }
  .bar-core { position: relative; }
  .core-mark { position: absolute; left: 60%; top: -3px; bottom: -3px; width: 2px; background: #a33; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .itemhead { display: flex; justify-content: space-between; align-items: baseline; margin-top: 14px; }
  .itemhead h2 { font-size: 16px; margin: 0; }
  .query, .cand { background: #fff; border: 1px solid #d8d4cc; border-radius: 6px;
                  padding: 12px 14px; margin: 10px 0; white-space: pre-wrap; line-height: 1.45; }
  .query { border-left: 4px solid #2f6f4f; }
  .cand { border-left: 4px solid #888; }
  .cand.judged-equiv { border-left-color: #2f6f4f; }
  .cand.judged-not { border-left-color: #a33; }
  .cand.judged-cj { border-left-color: #c90; }
  .candlabel { font-weight: bold; font-family: Helvetica, Arial, sans-serif; font-size: 13px;
               color: #444; margin-bottom: 6px; }
  .choices { margin-top: 10px; font-family: Helvetica, Arial, sans-serif; font-size: 13.5px; }
  .choices label { margin-right: 18px; cursor: pointer; }
  .meta { background: #fbfaf7; border: 1px dashed #c9c4ba; border-radius: 6px; padding: 12px 14px;
          margin: 14px 0; font-family: Helvetica, Arial, sans-serif; font-size: 13.5px; }
  .meta label { cursor: pointer; margin-right: 10px; }
  textarea { width: 100%; min-height: 52px; margin-top: 8px; font: 13px/1.4 Helvetica, Arial, sans-serif;
             border: 1px solid #c9c4ba; border-radius: 4px; padding: 6px; }
  .nav { display: flex; gap: 10px; margin: 18px 0; font-family: Helvetica, Arial, sans-serif; }
  button { font: 14px Helvetica, Arial, sans-serif; padding: 8px 16px; border-radius: 5px;
           border: 1px solid #999; background: #fff; cursor: pointer; }
  button:hover { background: #eee; }
  button.primary { background: #2f6f4f; color: #fff; border-color: #2f6f4f; }
  button.export { background: #37507a; color: #fff; border-color: #37507a; }
  .spacer { flex: 1; }
  .hint { font-size: 12px; color: #777; font-family: Helvetica, Arial, sans-serif; }
  .done { color: #2f6f4f; font-weight: bold; }
  .jump { font-size: 12.5px; font-family: Helvetica, Arial, sans-serif; }
  .jump a { color: #37507a; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Hard-Tier Human Audit &mdash; Recipe-Matching, Not Equivalence</h1>
  <div class="progress-text" id="progressText"></div>
  <div class="bar bar-core"><div class="bar-fill" id="barFill"></div><div class="core-mark" title="core sample boundary (60)"></div></div>
  <div class="jump">Autosaves to this browser (localStorage). <a href="#" id="firstIncomplete">Go to first incomplete item</a></div>
</header>
<div id="itemArea"></div>
<div class="nav">
  <button id="prevBtn">&larr; Prev</button>
  <button id="nextBtn" class="primary">Next &rarr;</button>
  <span class="spacer"></span>
  <button id="exportBtn" class="export">Export answers JSON</button>
</div>
<p class="hint">For each candidate: is it <em>mathematically equivalent</em> to the query
(same content under renaming / algebraic reformulation / re-characterization &mdash; not merely same
topic or method)? If unsure whether a change alters the answer set, choose &ldquo;Ill-formed / cannot
judge&rdquo; rather than guessing. Export works at any time and includes partial progress.</p>
</div>

<script id="audit-data" type="application/json">
__DATA_JSON__
</script>

<!--CORE_JS_START-->
<script>
(function (global) {
  'use strict';
  var CAND_KEYS = ['A', 'B', 'C', 'D'];
  var CAND_VALUES = ['equiv', 'not_equiv', 'cannot_judge'];

  function emptyAnswer(anchorId) {
    return { anchor_id: anchorId,
             cand: { A: null, B: null, C: null, D: null },
             disguise: null, note: '' };
  }

  function normalizeState(raw, items) {
    var state = { cur: 1, answers: {} };
    var byIndex = {};
    items.forEach(function (it) { byIndex[it.item_index] = it; });
    if (raw && typeof raw === 'object') {
      if (typeof raw.cur === 'number' && raw.cur >= 1 && raw.cur <= items.length) {
        state.cur = Math.floor(raw.cur);
      }
      var answers = (raw.answers && typeof raw.answers === 'object') ? raw.answers : {};
      Object.keys(answers).forEach(function (k) {
        var idx = parseInt(k, 10);
        if (!byIndex[idx]) return;
        var a = answers[k] || {};
        var clean = emptyAnswer(byIndex[idx].anchor_id);
        CAND_KEYS.forEach(function (c) {
          if (a.cand && CAND_VALUES.indexOf(a.cand[c]) !== -1) clean.cand[c] = a.cand[c];
        });
        if (typeof a.disguise === 'number' && a.disguise >= 1 && a.disguise <= 5) {
          clean.disguise = Math.floor(a.disguise);
        }
        if (typeof a.note === 'string') clean.note = a.note;
        state.answers[idx] = clean;
      });
    }
    return state;
  }

  function loadState(storage, key, items) {
    var raw = null;
    try { raw = JSON.parse(storage.getItem(key) || 'null'); } catch (e) { raw = null; }
    return normalizeState(raw, items);
  }

  function saveState(storage, key, state) {
    storage.setItem(key, JSON.stringify(state));
  }

  function isItemComplete(answer) {
    if (!answer || !answer.cand) return false;
    return CAND_KEYS.every(function (c) {
      return CAND_VALUES.indexOf(answer.cand[c]) !== -1;
    });
  }

  function countComplete(state, nItems) {
    var k = 0;
    for (var i = 1; i <= nItems; i++) {
      if (isItemComplete(state.answers[i])) k++;
    }
    return k;
  }

  function firstIncompleteIndex(state, nItems) {
    for (var i = 1; i <= nItems; i++) {
      if (!isItemComplete(state.answers[i])) return i;
    }
    return nItems;
  }

  function buildExport(state, nItems, nowIso) {
    var answers = {};
    Object.keys(state.answers).forEach(function (k) {
      var a = state.answers[k];
      var touched = CAND_KEYS.some(function (c) { return a.cand[c] !== null; }) ||
                    a.disguise !== null || (a.note && a.note.length > 0);
      if (touched) answers[k] = a;
    });
    return {
      schema: 'hard_audit_answers_v1',
      exported_at: nowIso,
      n_items_total: nItems,
      n_items_complete: countComplete(state, nItems),
      answers: answers
    };
  }

  var core = {
    CAND_KEYS: CAND_KEYS, CAND_VALUES: CAND_VALUES,
    emptyAnswer: emptyAnswer, normalizeState: normalizeState,
    loadState: loadState, saveState: saveState,
    isItemComplete: isItemComplete, countComplete: countComplete,
    firstIncompleteIndex: firstIncompleteIndex, buildExport: buildExport
  };
  if (typeof module !== 'undefined' && module.exports) { module.exports = core; }
  global.AuditCore = core;
})(typeof window !== 'undefined' ? window :
   (typeof globalThis !== 'undefined' ? globalThis : this));
</script>
<!--CORE_JS_END-->

<script>
(function () {
  'use strict';
  var C = window.AuditCore;
  var ITEMS = JSON.parse(document.getElementById('audit-data').textContent);
  var N = ITEMS.length;
  var LS_KEY = 'hard_audit_v1';
  var state = C.loadState(window.localStorage, LS_KEY, ITEMS);

  var area = document.getElementById('itemArea');

  function persist() { C.saveState(window.localStorage, LS_KEY, state); }

  function getAnswer(idx) {
    if (!state.answers[idx]) state.answers[idx] = C.emptyAnswer(ITEMS[idx - 1].anchor_id);
    return state.answers[idx];
  }

  function updateProgress() {
    var k = C.countComplete(state, N);
    var t = document.getElementById('progressText');
    t.innerHTML = 'Item <b>' + state.cur + '</b> of ' + N + ' &mdash; <b>' + k + ' / ' + N +
      '</b> answered (first 60 = core sample)' +
      (k >= N ? ' <span class="done">ALL DONE - export now</span>' : '');
    document.getElementById('barFill').style.width = (100 * k / N) + '%';
  }

  function candClass(v) {
    if (v === 'equiv') return ' judged-equiv';
    if (v === 'not_equiv') return ' judged-not';
    if (v === 'cannot_judge') return ' judged-cj';
    return '';
  }

  function render() {
    var it = ITEMS[state.cur - 1];
    var ans = getAnswer(state.cur);
    area.innerHTML = '';

    var head = document.createElement('div');
    head.className = 'itemhead';
    head.innerHTML = '<h2>Item ' + it.item_index + (it.item_index <= 60 ? ' (core)' : '') +
      '</h2><span class="hint">judge each candidate against the query</span>';
    area.appendChild(head);

    var q = document.createElement('div');
    q.className = 'query';
    var ql = document.createElement('div');
    ql.className = 'candlabel';
    ql.textContent = 'QUERY';
    var qt = document.createElement('div');
    qt.textContent = it.query_text;
    q.appendChild(ql); q.appendChild(qt);
    area.appendChild(q);

    C.CAND_KEYS.forEach(function (letter) {
      var cd = it.candidates[letter];
      var box = document.createElement('div');
      box.className = 'cand' + candClass(ans.cand[letter]);
      var lab = document.createElement('div');
      lab.className = 'candlabel';
      lab.textContent = 'CANDIDATE ' + letter;
      var txt = document.createElement('div');
      txt.textContent = cd.text;
      var ch = document.createElement('div');
      ch.className = 'choices';
      [['equiv', 'Equivalent'], ['not_equiv', 'Not equivalent'],
       ['cannot_judge', 'Ill-formed / cannot judge']].forEach(function (opt) {
        var l = document.createElement('label');
        var r = document.createElement('input');
        r.type = 'radio';
        r.name = 'cand_' + letter;
        r.value = opt[0];
        r.checked = (ans.cand[letter] === opt[0]);
        r.addEventListener('change', function () {
          ans.cand[letter] = opt[0];
          box.className = 'cand' + candClass(opt[0]);
          persist(); updateProgress();
        });
        l.appendChild(r);
        l.appendChild(document.createTextNode(' ' + opt[1]));
        ch.appendChild(l);
      });
      box.appendChild(lab); box.appendChild(txt); box.appendChild(ch);
      area.appendChild(box);
    });

    var meta = document.createElement('div');
    meta.className = 'meta';
    var dl = document.createElement('div');
    dl.innerHTML = '<b>Disguise difficulty</b> &mdash; could an expert see the equivalence ' +
      'without solving? (1 = transparent &hellip; 5 = only by fully solving both)&nbsp; ';
    [1, 2, 3, 4, 5].forEach(function (v) {
      var l = document.createElement('label');
      var r = document.createElement('input');
      r.type = 'radio'; r.name = 'disguise'; r.value = String(v);
      r.checked = (ans.disguise === v);
      r.addEventListener('change', function () { ans.disguise = v; persist(); });
      l.appendChild(r);
      l.appendChild(document.createTextNode(' ' + v));
      dl.appendChild(l);
    });
    var ta = document.createElement('textarea');
    ta.placeholder = 'Optional note (why cannot-judge, suspected label error, interesting disguise trick, ...)';
    ta.value = ans.note;
    ta.addEventListener('input', function () { ans.note = ta.value; persist(); });
    meta.appendChild(dl);
    meta.appendChild(ta);
    area.appendChild(meta);

    updateProgress();
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([area]).catch(function () {});
    }
    window.scrollTo(0, 0);
  }

  function goto(idx) {
    if (idx < 1 || idx > N) return;
    state.cur = idx;
    persist();
    render();
  }

  document.getElementById('prevBtn').addEventListener('click', function () { goto(state.cur - 1); });
  document.getElementById('nextBtn').addEventListener('click', function () { goto(state.cur + 1); });
  document.getElementById('firstIncomplete').addEventListener('click', function (e) {
    e.preventDefault();
    goto(C.firstIncompleteIndex(state, N));
  });
  document.addEventListener('keydown', function (e) {
    var tag = (e.target && e.target.tagName) || '';
    if (tag === 'TEXTAREA' || tag === 'INPUT') return;
    if (e.key === 'ArrowLeft') goto(state.cur - 1);
    if (e.key === 'ArrowRight') goto(state.cur + 1);
  });
  document.getElementById('exportBtn').addEventListener('click', function () {
    var payload = C.buildExport(state, N, new Date().toISOString());
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'hard_audit_answers.json';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  });

  render();
})();
</script>
</body>
</html>
"""


def build_html(items):
    # Strip provenance before embedding: the annotator must not see roles,
    # doc ids, stratum, or domain.
    public = []
    for it in items:
        public.append(
            {
                "item_index": it["item_index"],
                "anchor_id": it["anchor_id"],
                "query_text": it["query_text"],
                "candidates": {
                    L: {"text": c["text"]} for L, c in it["candidates"].items()
                },
            }
        )
    data = json.dumps(public, ensure_ascii=False)
    # Guard against '</script>' and '<!--' sequences inside embedded strings,
    # using escapes that remain valid JSON ('\/' and '!').
    data = data.replace("</", "<\\/").replace("<!--", "<\\u0021--")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data)


def main():
    queries, corpus, groups, anchor_domain = load_inputs()
    items, cell_sizes = build_sample(queries, corpus, groups, anchor_domain)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    provenance = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_items": len(items),
        "core_sample_n": CORE_N,
        "cell_order": [list(c) for c in CELL_ORDER],
        "eligible_pool_sizes": cell_sizes,
        "sources": {
            "queries": str(QUERIES.relative_to(ROOT)),
            "corpus": str(CORPUS.relative_to(ROOT)),
            "strata": str(PERQUERY.relative_to(ROOT)),
            "domains": f"{MAPPING.name} + {PARQUET.relative_to(ROOT)} (topics_flat[0] top level)",
        },
        "items": items,
    }
    items_path = OUT_DIR / "audit_items.json"
    with open(items_path, "w") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=1)

    html_path = OUT_DIR / "audit_sheet.html"
    html_path.write_text(build_html(items), encoding="utf-8")

    # Summary
    from collections import Counter

    strata = Counter(it["stratum"] for it in items)
    doms = Counter(it["domain"] for it in items)
    core = items[:CORE_N]
    strata60 = Counter(it["stratum"] for it in core)
    doms60 = Counter(it["domain"] for it in core)
    qlens = [len(it["query_text"]) for it in items]
    clens = [len(c["text"]) for it in items for c in it["candidates"].values()]
    print(f"Wrote {items_path} ({len(items)} items) and {html_path}")
    print(f"Strata (all {len(items)}): {dict(strata)}; domains: {dict(doms)}")
    print(f"Strata (first {CORE_N}): {dict(strata60)}; domains: {dict(doms60)}")
    print(
        f"Query chars: mean {statistics.mean(qlens):.0f} median {statistics.median(qlens):.0f}; "
        f"candidate chars: mean {statistics.mean(clens):.0f} median {statistics.median(clens):.0f}"
    )
    print(f"Eligible pools: {cell_sizes}")


if __name__ == "__main__":
    main()
