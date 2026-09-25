#!/usr/bin/env python3
"""E-R9 (2026-09-04): back-translated positives, the non-LLM deep-paraphrase
control: it separates paraphrase depth from LLM authorship of the positives.

For every source problem of the reference cell (data/llm_pairs_unrelated_casnegs,
the 6,768 D4 sources) the SOURCE text is round-tripped through NLLB-200-distilled-
600M, a neural MT model that is not a generative LLM: source -> pivot -> source.
The pivot is French for English sources and English for every other language.
Every LaTeX span ($...$, $$...$$, \\(...\\), \\[...\\], \\begin{env}...\\end{env})
and every markdown image link is replaced by a placeholder token before
translation and restored afterwards, so the mathematics is never translated.
Translation is sentence by sentence. A sentence whose placeholders do not all
come back exactly once is re-translated in FALLBACK mode: only the prose runs
between its formulas are translated, each on its own, and the formulas are put
back in their original positions, so the mathematics is preserved at the cost of
some fluency (the share of sentences and rows that needed the fallback is
reported). Segments that carry no prose (a displayed formula, an image, an OCR
header such as "Problem:") pass through untouched. A row is kept only if the
round trip is non-empty, its length stays within [0.5, 2.0] of the source, and it
is not identical to the source after whitespace normalisation. Everything dropped
is counted in results/bt_pairs_build.json together with the paraphrase-depth
ledger of E-R2 (char-3-gram Jaccard, token Jaccard, length ratio) so the
control can be placed beside the verified arm (0.865) and the LLM rungs
(0.40--0.46) and disclosed as what it is: an NMT round trip, shallower than a
free restatement. No LLM judge is applied (the point of the control is that no
LLM writes or filters the positive).

Source language is decided per line (bilingual sources mix scripts): the corpus
`language` tag when it names one language whose script matches the line;
otherwise the line's script, with letter-based sub-detection for Cyrillic
(Serbian, Macedonian, Ukrainian, Belarusian, Kazakh, Mongolian, Bulgarian,
Russian) and Chinese (Traditional vs Simplified), and for Latin script a
stop-word vote over twelve languages with English as the default.

  --pilot N       round-trip N stratified anchors with every placeholder style,
                  print them, write results/bt_pilot.json, write no pairs
  --style S       placeholder style for the full run (default from the pilot)
  --pivots A,B,C  E-R13 (2026-09-05; multi-hop back-translation, to extend
                  the non-LLM paraphrase depth beyond a single round trip):
                  SEQUENTIAL round trips through several pivot languages, each
                  hop the full validated machinery above applied to the previous
                  hop's output (source -> A -> source -> B -> source -> C ->
                  source); a hop whose pivot equals the source language uses
                  English instead. A row survives only if every hop does. Depth
                  is measured against the ORIGINAL source. Use with --out-dir
                  data/bt2_pairs --stats results/bt2_pairs_build.json.
  --out-dir / --stats   output locations (defaults: the E-R9 ones)
Writes data/bt_pairs/pairs.jsonl rows {source_id, positive_text, positive_from,
src_lang, pivot, char3_jaccard}; then `python scripts/build_review2_cells.py
--which r9` builds the cell and `sbatch --export=ALL,CELL=btcasnegs
scripts/factorial_cells.slurm` trains it (registered reading R9 in that header).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PROJECT, "data", "llm_pairs_unrelated_casnegs", "pairs.jsonl")
CORPUS = os.path.join(PROJECT, "data", "mathnet_corpus.parquet")
OUT_DIR = os.path.join(PROJECT, "data", "bt_pairs")
STATS = os.path.join(PROJECT, "results", "bt_pairs_build.json")
PILOT = os.path.join(PROJECT, "results", "bt_pilot.json")
MODEL = "facebook/nllb-200-distilled-600M"

TAGS = {
    "english": "eng_Latn", "spanish": "spa_Latn", "español": "spa_Latn",
    "russian": "rus_Cyrl", "chinese": "zho_Hans", "chinese (traditional)": "zho_Hant",
    "chinese (simplified)": "zho_Hans", "mongolian": "khk_Cyrl", "arabic": "arb_Arab",
    "macedonian": "mkd_Cyrl", "german": "deu_Latn", "afrikaans": "afr_Latn",
    "french": "fra_Latn", "portuguese": "por_Latn", "italian": "ita_Latn",
    "romanian": "ron_Latn", "dutch": "nld_Latn", "hungarian": "hun_Latn",
    "slovenian": "slv_Latn", "slovene": "slv_Latn", "serbian": "srp_Cyrl",
    "bulgarian": "bul_Cyrl", "ukrainian": "ukr_Cyrl", "greek": "ell_Grek",
    "turkish": "tur_Latn", "polish": "pol_Latn", "czech": "ces_Latn",
    "vietnamese": "vie_Latn", "korean": "kor_Hang", "japanese": "jpn_Jpan",
    "persian": "pes_Arab", "farsi": "pes_Arab", "hebrew": "heb_Hebr",
    "hindi": "hin_Deva", "indonesian": "ind_Latn", "swedish": "swe_Latn",
    "norwegian": "nob_Latn", "danish": "dan_Latn", "finnish": "fin_Latn",
    "croatian": "hrv_Latn", "slovak": "slk_Latn", "catalan": "cat_Latn",
    "thai": "tha_Thai", "kazakh": "kaz_Cyrl", "estonian": "est_Latn",
    "latvian": "lvs_Latn", "lithuanian": "lit_Latn", "bosnian": "bos_Latn",
    "belarusian": "bel_Cyrl", "georgian": "kat_Geor", "armenian": "hye_Armn",
}
SCRIPT_DEFAULT = {"cyr": "rus_Cyrl", "cjk": "zho_Hans", "arab": "arb_Arab",
                  "greek": "ell_Grek", "latin": "eng_Latn"}
STOPWORDS = {
    "eng_Latn": r"\b(the|and|of|that|is|are|which|each|find|prove|let|with|all|such|show|for)\b",
    "spa_Latn": r"\b(el|la|los|las|que|de|un|una|sea|sean|cada|todos|demuestra|demostrar|hallar|calcula|calcular|para|con)\b",
    "fra_Latn": r"\b(le|la|les|des|que|est|soit|soient|tous|montrer|trouver|une|dans|pour|avec|déterminer)\b",
    "deu_Latn": r"\b(der|die|das|und|ist|sind|sei|seien|alle|zeige|man|mit|für|eine|einer|bestimme|beweise)\b",
    "por_Latn": r"\b(o|os|as|que|um|uma|seja|sejam|todos|mostre|encontre|com|para|determine|prove)\b",
    "ita_Latn": r"\b(il|gli|che|di|un|una|sia|siano|tutti|dimostrare|trovare|con|per|determinare)\b",
    "ron_Latn": r"\b(și|este|sunt|fie|toate|arătați|aflați|cu|pentru|care|să|determinați|demonstrați)\b",
    "nld_Latn": r"\b(de|het|een|dat|zijn|is|alle|bewijs|bepaal|met|voor|waarvoor|zodat)\b",
    "hun_Latn": r"\b(és|egy|hogy|az|legyen|minden|bizonyítsuk|határozzuk|meg|van|amely)\b",
    "slv_Latn": r"\b(in|je|so|naj|vsa|dokaži|določi|za|ki|da|število)\b",
    "pol_Latn": r"\b(i|jest|są|niech|wszystkie|udowodnij|wyznacz|dla|które|że|liczba)\b",
    "tur_Latn": r"\b(ve|bir|olsun|olduğunu|tüm|gösteriniz|bulunuz|için|ile|sayı)\b",
}
STYLES = {"MATH": lambda i: f"MATH{i}", "X": lambda i: f"X{i}X", "BR": lambda i: f"[{i}]"}
STYLE_RX = {"MATH": r"MATH(\d+)", "X": r"X(\d+)X", "BR": r"\[(\d+)\]"}

MASK_RX = [
    r"!\[[^\]]*\]\([^)]*\)",                       # markdown images
    r"\$\$[\s\S]*?\$\$",                            # display math
    r"\\\[[\s\S]*?\\\]",                            # \[ ... \]
    r"\\\([\s\S]*?\\\)",                            # \( ... \)
    r"\\begin\{([a-zA-Z*]+)\}[\s\S]*?\\end\{\1\}",  # environments
    r"\$[^$\n]*?\$",                                # inline math (single line)
    r"\$[\s\S]*?\$",                                # inline math spanning lines
]


def script_of(t):
    if re.search(r"[\u0400-\u04ff]", t):
        return "cyr"
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t):
        return "cjk"
    if re.search(r"[\u0600-\u06ff]", t):
        return "arab"
    if re.search(r"[\u0370-\u03ff]{3,}", t):
        return "greek"
    return "latin"


TRAD_ONLY = set("實數記試個們這為與於對點線圓證題邊長積質級從國滿總體條讓誰幾義將對應該結構圖說")
SIMP_ONLY = set("实数记试个们这为与于对点线圆证题边长积质级从国满总体条让谁几义将应该结构图说")


def cjk_variant(text, default):
    t = sum(ch in TRAD_ONLY for ch in text)
    si = sum(ch in SIMP_ONLY for ch in text)
    if t > si:
        return "zho_Hant"
    if si > t:
        return "zho_Hans"
    return default


def cyrillic_lang(text):
    low = text.lower()
    if re.search(r"[ѓќѕ]", low):
        return "mkd_Cyrl"
    if re.search(r"[ђћџ]", low) or "ј" in low:
        return "srp_Cyrl"
    if re.search(r"[іїєґ]", low):
        return "ukr_Cyrl"
    if "ў" in low:
        return "bel_Cyrl"
    if re.search(r"[әғқңұһ]", low):
        return "kaz_Cyrl"
    if re.search(r"[өү]", low):
        return "khk_Cyrl"
    if re.search(r"[ыэё]", low):
        return "rus_Cyrl"
    if low.count("ъ") >= 2 and "ы" not in low:
        return "bul_Cyrl"
    return "rus_Cyrl"


def lang_of(text, tag):
    """NLLB code for the text: tag when it fits the script, else script + stop-words."""
    plain = re.sub(r"\$[^$]*\$", " ", text)
    sc = script_of(plain)
    cands = []
    if isinstance(tag, str) and tag.strip():
        for part in re.split(r"[;,/]| and ", tag.lower()):
            code = TAGS.get(part.strip())
            if code:
                cands.append(code)
    fits = [c for c in cands if
            (sc == "cyr" and c.endswith("Cyrl")) or (sc == "cjk" and c[-4:] in ("Hans", "Hant", "Hang", "Jpan"))
            or (sc == "arab" and c.endswith("Arab")) or (sc == "greek" and c.endswith("Grek"))
            or (sc == "latin" and c.endswith("Latn"))]
    if len(fits) == 1:
        return fits[0], "tag"
    if sc != "latin":
        if fits:
            code = fits[0]
            if code.startswith("zho"):
                code = cjk_variant(plain, code)
            return code, "tag"
        if sc == "cyr":
            return cyrillic_lang(plain), "script"
        if sc == "cjk":
            return cjk_variant(plain, "zho_Hans"), "script"
        return SCRIPT_DEFAULT[sc], "script"
    low = plain.lower()
    votes = {c: len(re.findall(p, low)) for c, p in STOPWORDS.items()}
    if fits:
        votes = {c: votes.get(c, 0) for c in fits}
    best = max(votes, key=votes.get)
    if votes[best] >= 2:
        return best, ("tag" if fits else "stopwords")
    return (fits[0] if fits else "eng_Latn"), ("tag" if fits else "default")


def mask(text, style):
    spans, out, i = [], text, 0
    for rx in MASK_RX:
        def rep(m):
            nonlocal i
            spans.append(m.group(0))
            tok = STYLES[style](i)
            i += 1
            return f" {tok} "
        out = re.sub(rx, rep, out)
    out = re.sub(r"[ \t]+", " ", out)
    return out, spans


def unmask(text, spans, style):
    rx = STYLE_RX[style]
    found = [int(x) for x in re.findall(rx, text)]
    ok = sorted(found) == list(range(len(spans)))
    if not ok:
        return None, found
    def rep(m):
        return spans[int(m.group(1))]
    out = re.sub(rx, rep, text)
    out = re.sub(r" ?([.,;:!?)]) ?", r"\1 ", out)
    out = re.sub(r"\( ", "(", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" +\n", "\n", out)
    out = re.sub(r"\n +", "\n", out)
    return out.strip(), found


PH_ONLY_MIN_WORDS = 2


def sentences(line):
    s = line.strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?;:])\s+(?=\S)", s) if p.strip()]
    return parts or [s]


def prose_words(seg, style):
    plain = re.sub(STYLE_RX[style], " ", seg)
    return re.findall(r"[^\W\d_]{2,}", plain)


def has_all_placeholders(text, expected, style):
    found = [int(x) for x in re.findall(STYLE_RX[style], text)]
    return sorted(found) == sorted(expected)


class Translator:
    def __init__(self, device, beams=4, batch=32):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
        self.device, self.beams, self.batch = device, beams, batch

    def translate(self, texts, src, tgt):
        """texts: list[str] all in language src -> list[str] in tgt (order kept)."""
        if not texts:
            return []
        self.tok.src_lang = src
        bos = self.tok.convert_tokens_to_ids(tgt)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            max_new = min(600, int(enc["input_ids"].shape[1] * 1.6) + 20)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=self.beams,
                                          max_new_tokens=max_new, repetition_penalty=1.1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True)
            for i, t in zip(idx, dec):
                out[i] = t
        return out


def char_ngrams(t, n=3):
    t = " ".join((t or "").split())
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jacc(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def norm(t):
    return " ".join(t.split())


def default_pivot(lang):
    return "fra_Latn" if lang == "eng_Latn" else "eng_Latn"


def round_trip(tr, items, style, pivot_of=default_pivot):
    """items: list of dicts {source_id, text, lang, tag}. Sentence-level round trip with a
    prose-only fallback for sentences whose placeholders do not survive.
    pivot_of(lang) names the pivot language (E-R13 passes a per-hop choice)."""
    docs = []
    for it in items:
        masked, spans = mask(it["text"], style)
        lines = masked.split("\n")
        units = []          # (line_index, sentence, lang, expected placeholder ids, mode)
        for li, line in enumerate(lines):
            lang, how = lang_of(line, it.get("tag")) if line.strip() else (it["lang"], it["lang_from"])
            if script_of(re.sub(STYLE_RX[style], " ", line)) == "latin" and script_of(it["text"]) == "latin":
                lang = it["lang"]      # document-level Latin-script vote is more reliable than one line
            for sent in sentences(line):
                ids = [int(x) for x in re.findall(STYLE_RX[style], sent)]
                mode = "pass" if len(prose_words(sent, style)) < PH_ONLY_MIN_WORDS else "sent"
                units.append({"li": li, "src": sent, "lang": lang, "ids": ids, "mode": mode,
                              "pivot": pivot_of(lang)})
        docs.append({"masked": masked, "spans": spans, "nlines": len(lines), "units": units, **it})

    def translate_units(entries, key_src, key_out):
        """entries: list of (unit_dict, text, src, tgt); writes unit[key_out]."""
        groups = defaultdict(list)
        for e in entries:
            groups[(e[2], e[3])].append(e)
        for (src, tgt), lst in groups.items():
            t0 = time.time()
            outs = tr.translate([e[1] for e in lst], src, tgt)
            for e, o in zip(lst, outs):
                e[0][key_out] = o
            print(f"[{key_out}] {src}->{tgt}: {len(lst)} segments in {time.time() - t0:.0f}s", flush=True)

    # pass 1: whole sentences, forward then back
    sent_units = [u for d in docs for u in d["units"] if u["mode"] == "sent"]
    translate_units([(u, u["src"], u["lang"], u["pivot"]) for u in sent_units], "src", "fwd")
    translate_units([(u, u["fwd"], u["pivot"], u["lang"]) for u in sent_units], "fwd", "back")
    # pass 2: fallback for sentences whose placeholders did not survive
    chunks = []
    for u in sent_units:
        if has_all_placeholders(u["back"], u["ids"], style):
            continue
        u["mode"] = "fallback"
        parts = re.split(f"({STYLE_RX[style]})", u["src"])   # keeps placeholders (and their digit group)
        u["parts"] = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if re.fullmatch(STYLE_RX[style], p or ""):
                u["parts"].append({"text": p, "kind": "ph"})
                i += 2                                       # skip the captured digit group
            else:
                if len(prose_words(p, style)) >= 1:
                    c = {"text": p.strip(), "kind": "prose"}
                    chunks.append(c)
                else:
                    c = {"text": p, "kind": "keep"}
                u["parts"].append(c)
                i += 1
        for c in u["parts"]:
            if c["kind"] == "prose":
                c["lang"], c["pivot"] = u["lang"], u["pivot"]
    translate_units([(c, c["text"], c["lang"], c["pivot"]) for c in chunks], "text", "fwd")
    translate_units([(c, c["fwd"], c["pivot"], c["lang"]) for c in chunks], "fwd", "back")

    results = []
    for d in docs:
        lines_out = defaultdict(list)
        n_fb = 0
        for u in d["units"]:
            if u["mode"] == "pass":
                out = u["src"]
            elif u["mode"] == "sent":
                out = u["back"]
            else:
                n_fb += 1
                out = " ".join(c["back"] if c["kind"] == "prose" else c["text"] for c in u["parts"])
            lines_out[u["li"]].append(out)
        joined = "\n".join(" ".join(lines_out[li]) if li in lines_out else "" for li in range(d["nlines"]))
        joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
        restored, found = unmask(joined, d["spans"], style)
        pivot_text = "\n".join(" ".join((u.get("fwd") or u["src"]) if u["mode"] != "fallback"
                                        else " ".join(c.get("fwd", c["text"]) for c in u["parts"])
                                        for u in d["units"] if u["li"] == li)
                               for li in range(d["nlines"])).strip()
        status = "ok"
        if restored is None:
            status = "placeholder_lost_or_duplicated"
        elif not restored.strip():
            status = "empty"
        else:
            ratio = len(restored) / max(1, len(d["text"]))
            if ratio < 0.5 or ratio > 2.0:
                status = "length_ratio_out_of_range"
            elif norm(restored) == norm(d["text"]):
                status = "identical_to_source"
        langs = Counter(u["lang"] for u in d["units"] if u["mode"] != "pass")
        results.append({"source_id": d["source_id"], "src_lang": d["lang"], "lang_from": d["lang_from"],
                        "line_langs": dict(langs), "pivot": pivot_of(d["lang"]),
                        "status": status, "n_spans": len(d["spans"]),
                        "n_sentences": sum(u["mode"] != "pass" for u in d["units"]), "n_fallback": n_fb,
                        "pivot_text": pivot_text, "positive_text": restored,
                        "char3_jaccard": round(jacc(char_ngrams(d["text"]), char_ngrams(restored)), 4) if restored else None,
                        "token_jaccard": round(jacc(set(d["text"].split()), set(restored.split())), 4) if restored else None,
                        "len_ratio": round(len(restored) / max(1, len(d["text"])), 3) if restored else None})
    return results


def load_sources():
    ids = [json.loads(l)["source_id"] for l in open(REF, encoding="utf-8")]
    df = pd.read_parquet(CORPUS).set_index("id")
    items = []
    for s in ids:
        text = str(df.at[s, "problem_markdown"])
        tag = df.at[s, "language"]
        lang, how = lang_of(text, tag if isinstance(tag, str) else None)
        items.append({"source_id": s, "text": text, "lang": lang, "lang_from": how,
                      "tag": tag if isinstance(tag, str) else None})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--style", default="BR", choices=list(STYLES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N sources")
    ap.add_argument("--pivots", default=None,
                    help="E-R13: comma-separated NLLB codes for sequential round trips")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--stats", default=STATS)
    args = ap.parse_args()
    out_dir, stats_path = args.out_dir, args.stats
    import torch
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    items = load_sources()
    langs = Counter(i["lang"] for i in items)
    print(f"[data] {len(items)} sources; languages {dict(langs.most_common())}", flush=True)
    print(f"[data] language from: {dict(Counter(i['lang_from'] for i in items))}", flush=True)
    tr = Translator(device, beams=args.beams, batch=args.batch)

    if args.pilot:
        rng = np.random.default_rng(0)
        strata = {"eng_tag": lambda i: i["lang"] == "eng_Latn" and i["lang_from"] == "tag",
                  "eng_untagged": lambda i: i["lang"] == "eng_Latn" and i["lang_from"] != "tag",
                  "spa": lambda i: i["lang"] == "spa_Latn", "fra": lambda i: i["lang"] == "fra_Latn",
                  "deu_por_ita_ron": lambda i: i["lang"] in ("deu_Latn", "por_Latn", "ita_Latn", "ron_Latn"),
                  "cyr": lambda i: i["lang"].endswith("Cyrl"), "cjk": lambda i: i["lang"].startswith("zho")}
        quota = {"eng_tag": 3, "eng_untagged": 3, "spa": 1, "fra": 1, "deu_por_ita_ron": 1, "cyr": 2, "cjk": 2}
        picked = []
        for k, f in strata.items():
            pool = [i for i in items if f(i)]
            n = min(quota.get(k, 1), len(pool))
            for j in rng.choice(len(pool), n, replace=False):
                picked.append(pool[int(j)])
        picked = picked[:args.pilot] if args.pilot < len(picked) else picked
        report = {"model": MODEL, "device": device, "beams": args.beams, "n": len(picked), "styles": {}}
        for style in STYLES:
            res = round_trip(tr, picked, style)
            stat = Counter(r["status"] for r in res)
            ok = [r for r in res if r["status"] == "ok"]
            report["styles"][style] = {
                "status": dict(stat),
                "sentences": int(sum(r["n_sentences"] for r in res)),
                "sentences_fallback": int(sum(r["n_fallback"] for r in res)),
                "char3_jaccard_mean_ok": round(float(np.mean([r["char3_jaccard"] for r in ok])), 3) if ok else None,
                "rows": [{k: v for k, v in r.items()} for r in res]}
            print(f"\n===== style {style}: {dict(stat)} =====")
            for it, r in zip(picked, res):
                print(f"\n--- {r['source_id']} [{r['src_lang']} via {r['pivot']}, {r['lang_from']}; lines {r['line_langs']}] "
                      f"status={r['status']} spans={r['n_spans']} sentences={r['n_sentences']} "
                      f"fallback={r['n_fallback']} c3J={r['char3_jaccard']}")
                print("SRC :", it["text"][:700].replace("\n", " | "))
                print("PIV :", (r["pivot_text"] or "")[:400].replace("\n", " | "))
                print("BACK:", (r["positive_text"] or "")[:700].replace("\n", " | "))
        with open(PILOT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        print(f"\n[done] pilot report -> {PILOT}")
        return

    if args.limit:
        items = items[:args.limit]
    t0 = time.time()
    hops = [h.strip() for h in args.pivots.split(",")] if args.pivots else None
    if hops:
        # E-R13: sequential round trips; every hop re-applies the full single-hop
        # machinery to the previous hop's output, so masks and validation hold
        # at every step. A row that fails any hop is dropped with that status.
        original = {it["source_id"]: it["text"] for it in items}
        cur, hop_log = list(items), defaultdict(list)
        for h in hops:
            pivot_of = (lambda lang, h=h: h if h != lang else "eng_Latn")
            r_h = round_trip(tr, cur, args.style, pivot_of)
            nxt = []
            for it, r in zip(cur, r_h):
                hop_log[it["source_id"]].append({"pivot": r["pivot"], "status": r["status"],
                                                 "n_fallback": r["n_fallback"]})
                if r["status"] == "ok":
                    nxt.append(dict(it, text=r["positive_text"]))
            print(f"[hop {h}] {len(cur)} in -> {len(nxt)} ok", flush=True)
            cur = nxt
        res = []
        for it in items:
            log = hop_log[it["source_id"]]
            ok = len(log) == len(hops) and all(l["status"] == "ok" for l in log)
            final = next((c["text"] for c in cur if c["source_id"] == it["source_id"]), None) if ok else None
            src = original[it["source_id"]]
            res.append({"source_id": it["source_id"], "src_lang": it["lang"], "lang_from": it["lang_from"],
                        "pivot": "->".join(l["pivot"] for l in log),
                        "status": "ok" if ok else ("hop_failed:" + next(l["status"] for l in log if l["status"] != "ok")),
                        "n_sentences": 0, "n_fallback": sum(l["n_fallback"] for l in log),
                        "positive_text": final,
                        "char3_jaccard": round(jacc(char_ngrams(src), char_ngrams(final)), 4) if final else None,
                        "token_jaccard": round(jacc(set(src.split()), set(final.split())), 4) if final else None,
                        "len_ratio": round(len(final) / max(1, len(src)), 3) if final else None})
        # a multi-hop result identical to the source is dropped like a single hop's
        for r in res:
            if r["status"] == "ok" and norm(r["positive_text"]) == norm(original[r["source_id"]]):
                r["status"] = "identical_to_source"
        tag = "E-R13 multi-hop " + "->".join(hops)
    else:
        res = round_trip(tr, items, args.style)
        tag = "E-R9"
    os.makedirs(out_dir, exist_ok=True)
    kept = [r for r in res if r["status"] == "ok"]
    with open(os.path.join(out_dir, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({
                "source_id": r["source_id"], "positive_text": r["positive_text"],
                "positive_from": f"{MODEL} round trip {r['src_lang']}->{r['pivot']}->{r['src_lang']}, "
                                 f"LaTeX spans masked, no LLM judge, {tag}",
                "src_lang": r["src_lang"], "pivot": r["pivot"], "char3_jaccard": r["char3_jaccard"]},
                ensure_ascii=False) + "\n")
    def depth(key):
        v = np.array([r[key] for r in kept], float)
        return {k: round(float(x), 3) for k, x in (("mean", v.mean()), ("median", np.median(v)),
                                                   ("q25", np.percentile(v, 25)), ("q75", np.percentile(v, 75)))}
    stats = {"generated_by": "scripts/backtranslate_anchors.py", "model": MODEL, "device": device,
             "beams": args.beams, "style": args.style, "seconds": round(time.time() - t0),
             "n_sources": len(items), "n_written": len(kept),
             "status_counts": dict(Counter(r["status"] for r in res)),
             "sentences_total": int(sum(r["n_sentences"] for r in res)),
             "sentences_fallback": int(sum(r["n_fallback"] for r in res)),
             "rows_with_any_fallback": int(sum(r["n_fallback"] > 0 for r in res)),
             "rows_with_any_fallback_kept": int(sum(r["n_fallback"] > 0 for r in kept)),
             "languages": dict(Counter(r["src_lang"] for r in res).most_common()),
             "language_from": dict(Counter(i["lang_from"] for i in items)),
             "pivots": dict(Counter(r["pivot"] for r in res)),
             "depth_vs_source_kept_rows": {"char3_jaccard": depth("char3_jaccard"),
                                           "token_jaccard": depth("token_jaccard"),
                                           "len_ratio": depth("len_ratio")},
             "depth_reference_E_R2": {"ctrl-CAS char3": 0.865, "D1 char3": 0.463,
                                      "note": "results/paraphrase_depth.json is the authority"},
             "pivots_chain": hops,
             "output": os.path.join(out_dir, "pairs.jsonl")}
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "languages"}, indent=1))
    print(f"[done] wrote {len(kept)} rows -> {os.path.join(out_dir, 'pairs.jsonl')} and {stats_path}")


if __name__ == "__main__":
    main()
