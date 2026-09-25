#!/usr/bin/env python3
"""
Mine cross-language / cross-booklet DUPLICATE problem candidates from the
MathNet corpus parquet (27,817 rows).

Duplicates = the same competition problem appearing in more than one source
row (different national booklet, different country, or different language).
These become *verified* equivalence pairs for InvarEmbed training
(same official problem => provably equivalent).

Architecture: candidate GENERATORS (high recall) + a uniform VERIFICATION
layer (high precision).  Calibrated on a manually-labeled sample: true
cross-language duplicates share their LaTeX formulas nearly verbatim
(math 4-gram Jaccard >= 0.6-0.8), while thematically-similar non-duplicates
rarely exceed 0.45.

Generators:
  g_exact    identical aggressively-normalized problem text
  g_lsh_text MinHash-LSH over word 3-shingles (same-language near-dups)
  g_lsh_math MinHash-LSH over LaTeX-formula char 4-grams (cross-language!)
  g_raregram inverted index over rare formula 4-grams (corpus df 2..10),
             pairs sharing >= 3 rare grams
  g_tag      same canonical international competition (IMO/BMO/JBMO/...)
             across different booklets, pre-gated by answer/number overlap
  g_numfp    identical nontrivial integer-multiset fingerprint, or >= 2
             shared rare integers with high multiset Jaccard

Verification (any one rule accepts a pair; exact text needs none).
Key quantities, all over formula 4-grams with corpus df <= 20 ("rare"):
  rshared   = IDF mass of rare grams present on BOTH sides
  rcont_min = rshared / (rshared + min-side exclusive rare mass)
  rcont_sym = rshared / (rshared + MAX-side exclusive rare mass)
A true duplicate keeps its ENTIRE math content verbatim => high rshared,
near-zero exclusive rare mass on both sides (rcont_sym high).
A deliberate minimal variation (e.g. same FE with ^3 -> ^4) has rare
grams exclusive to BOTH sides => low rcont_min ("suspect").
Boilerplate matches (\\triangle ABC, digit lists, subscript label runs
E_1,E_2,...) either have rshared ~ 0 or fail rcont_sym because the rest
of each problem's math differs.

  V1_text        word-shingle Jaccard >= 0.55 (>= 0.70 when no shared
                 rare math), and not suspect                (high conf)
  V2_rare_math   rshared >= 25 AND rcont_min >= 0.75 AND
                 rcont_sym >= 0.65 AND all-gram symmetric
                 containment >= 0.50                        (high conf)
  V4_answer_math distinctive shared final answer (corpus df <= 4) AND
                 raw math Jaccard >= 0.35 (sigs >= 8), not suspect
                                                            (medium conf)
  V5_answer_tag  distinctive answer AND same intl tag AND compatible
                 years AND number-multiset Jaccard >= 0.6   (medium conf)

All thresholds calibrated on ~80 hand-labeled candidate pairs across
three sampling rounds: the final V2 rule keeps 32/32 math-verifiable
true pairs and admits 1/19 math-bearing false pairs (a shared-prefix
formula sqrt(a)+sqrt(b)=sqrt(...) with different right-hand sides).
Unbiased random-sample precision (20 fresh accepted pairs, round 3):
17/20 before the last two fixes; 17/18 after they removed two of the
three observed false positives.

Outputs:
  results/duplicate_candidates.jsonl   one JSON object per accepted pair
  results/duplicate_mining_summary.json

Runtime: ~2-4 min single process on a login node.
Usage: python3 scripts/mine_duplicates.py
"""

import hashlib
import itertools
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

import duckdb

BASE = "/ibex/user/habiam0b/MathNet_Follow_Up"
PARQUET = f"{BASE}/data/mathnet_corpus.parquet"
OUT_DIR = f"{BASE}/results"
ANCHOR_MAP = f"{BASE}/anchor_to_corpus_mapping.json"

# --------------------------------------------------------------------------
# 1. Canonical international-competition tags + edition years
# --------------------------------------------------------------------------
INTL_PATTERNS = [
    ("IMO",       r"\bIMO\b|international mathematical olympiad"),
    ("JBMO",      r"\bJBMO\b|\bOJBM\b|junior balkan"),
    ("BMO",       r"\bBMO\b|balkan mathematical olympiad"),
    ("APMO",      r"\bAPMO\b|asia\s*pacific"),
    ("EGMO",      r"\bEGMO\b|european girls"),
    ("IBERO",     r"iber(o|oamerican|o-american)"),
    ("BALTIC",    r"baltic way"),
    ("ZHAUTYKOV", r"zhautykov|\bIZhO\b"),
    ("RMM",       r"\bRMM\b|romanian master"),
    ("MEMO",      r"\bMEMO\b|middle european"),
    ("BXMO",      r"\bBxMO\b|benelux"),
    ("SILKROAD",  r"silk road"),
    ("NORDIC",    r"nordic mathematical"),
    ("CPS",       r"czech.{0,3}(polish|slovak)|cesko|caps match"),
    ("OMCPLP",    r"OMCPLP|países de língua portuguesa"),
]
# competition-string markers meaning "TST/training row ABOUT the competition,
# not a problem OF it" -- such rows still generate candidates via g_tag
# (booklets often reprint the real problems) but never contribute years.
NOT_ACTUAL = re.compile(
    r"tst|selection|selektion|selekcija|selectietoets|training|preparation|"
    r"préparation|booklet|mock|practice|team selection|izborno|auswahl",
    re.I)

FOUNDING = {"IMO": 1959, "BMO": 1984, "JBMO": 1997, "APMO": 1989,
            "EGMO": 2012, "IBERO": 1985, "BALTIC": 1990, "ZHAUTYKOV": 2005,
            "RMM": 2008, "MEMO": 2007, "BXMO": 2009, "SILKROAD": 2002,
            "NORDIC": 1987}

ROMAN_RE = re.compile(r"\b([IVXLC]{2,7}|[VXL])\b")
ORDINAL_RE = re.compile(r"\b(\d{1,3})\s*[-]?\s*(?:st|nd|rd|th|a|e)?\b")
ORDINAL_STRICT_RE = re.compile(r"\b(\d{1,3})\s*[-]?\s*(?:st|nd|rd|th)\b", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(s):
        v = vals.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def detect_tag(country, competition):
    """Return (tag, tag_from_country).  tag_from_country=True means MathNet
    itself grouped the row under the international competition (trusted)."""
    c = f"{country or ''}"
    comp = f"{competition or ''}"
    for tag, pat in INTL_PATTERNS:
        if re.search(pat, c, re.I):
            return tag, True
        if re.search(pat, comp, re.I):
            return tag, False
    return None, False


def extract_year(competition, tag, tag_from_country):
    """Year from explicit 4-digit anywhere; edition ordinals/romans are
    trusted ONLY when the row's country IS the international competition
    (for national TST rows an ordinal is usually the test number)."""
    comp = f"{competition or ''}"
    m = YEAR_RE.search(comp)
    if m:
        return int(m.group(1)), "explicit"
    if tag in FOUNDING and tag_from_country and not NOT_ACTUAL.search(comp):
        m = ORDINAL_STRICT_RE.search(comp)
        n = int(m.group(1)) if m else None
        if n is None:
            rm = ROMAN_RE.search(comp.upper())
            if rm:
                n = roman_to_int(rm.group(1))
        if n and 1 <= n <= 80:
            return FOUNDING[tag] + n - 1, "ordinal"
    return None, None


# --------------------------------------------------------------------------
# 2. Text normalization, math signatures, language detection
# --------------------------------------------------------------------------
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
NONALNUM_RE = re.compile(r"[^0-9a-zÀ-ɏͰ-ϿЀ-ӿ一-鿿]+")


def norm_text(t):
    """Aggressive normalization for exact-duplicate hashing."""
    t = IMG_RE.sub(" ", t or "")
    t = unicodedata.normalize("NFKC", t).lower()
    t = LATEX_CMD_RE.sub(lambda m: m.group(0)[1:], t)  # \frac -> frac
    return NONALNUM_RE.sub("", t)


def word_tokens(t):
    t = IMG_RE.sub(" ", t or "").lower()
    return re.findall(r"[0-9a-zÀ-ɏЀ-ӿ一-鿿]+", t)


def shingles(t, k=3):
    w = word_tokens(t)
    return {hash(" ".join(w[i:i + k])) & 0xFFFFFFFFFFFF
            for i in range(max(1, len(w) - k + 1))}


MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
MATH_STRIP = re.compile(
    r"\\(left|right|,|;|!|quad|qquad|\s)|[{}\s~]|\\mathrm|\\text\w*|"
    r"\\operatorname|\\mathbb|\\mathcal|\\overline|\\underline|"
    r"\\displaystyle|\\limits"
    # list-bullet / layout macros that appear inside $...$ in some booklets
    # (French prep uses $\triangleright$ bullets; tables use $\square$) --
    # they are rare-but-meaningless and caused false positives:
    r"|\\triangleright|\\triangleleft|\\blacksquare|\\square|\\bullet")


def math_sig(t, n=4):
    """Set of char n-grams over whitespace/format-stripped LaTeX formulas.
    Language-invariant: translations keep formulas nearly verbatim."""
    grams = set()
    for m in MATH_RE.finditer(t or ""):
        s = MATH_STRIP.sub("", m.group(1) or m.group(2) or "")
        for i in range(len(s) - n + 1):
            grams.add(s[i:i + n])
    return grams


NUM_RE = re.compile(r"(?<![\d.])(\d{1,9})(?![\d.])")


def num_fingerprint(t):
    nums = [int(x) for x in NUM_RE.findall(t or "")]
    return tuple(sorted(n for n in nums if n >= 3))


def fp_nontrivial(fp):
    if len(fp) < 3 or len(set(fp)) < 2:
        return False
    return max(fp) >= 7 or len(fp) >= 5


def multiset_jaccard(a, b):
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 0.0


def set_jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# compact stopword language ID -- approximate, for reporting/blocking only
LANG_MARKERS = {
    "en": "the of and is are that which find prove let each every number all such",
    "fr": "le la les des une est que dans pour soit tous nombre montrer sont chaque déterminer",
    "es": "el los las una que para sea todos cada número demuestre entero sean positivos",
    "pt": "uma não são também seja mostre número inteiro cada quais é os em",
    "it": "il gli è che per sia tutti ogni numero dimostrare siano interi positivi",
    "de": "der die das und ist dass eine für sei alle zahl zeige jede sind man",
    "nl": "de het een en is dat voor zij alle getal bewijs elke zijn worden",
    "ro": "și în este care să pentru fie toate număr arătați fiecare numerele sunt",
    "sl": "in je da za naj vsak število dokaži velja tako kjer katerih ki so",
    "hr": "je da za su koji broj dokažite svaki neka ako tada vrijedi realni",
    "sr": "je da za su koji broj dokazati svaki neka ako tada važi realni",
    "et": "ja on et iga arv leia tõesta kõik olgu ning kus mille korral",
    "vi": "và là các của cho số chứng minh với một những rằng",
    "tr": "ve bir için olduğunu sayı her ile bu olan tüm",
    "hu": "és egy hogy minden szám bizonyítsa amely az ha akkor",
    "pl": "i jest że dla każdy liczba udowodnij niech oraz takich",
    "cs": "je že pro každé číslo dokažte nechť jsou právě všechna",
    "af": "die van en is dat vir alle getal bewys elke wat",
    "ru": "что это как все или для быть на числа число докажите найдите пусть если",
    "bg": "че са като който на да се числа число докажете намерете нека ако",
    "uk": "що як усі або для на числа число доведіть знайдіть нехай якщо",
    "mk": "што сите или за на броеви број докажете најдете нека ако",
    "mn": "нь бол бүх тоо байг тоог батал ол гэж бүр",
}
LANG_SETS = {k: set(v.split()) for k, v in LANG_MARKERS.items()}
CYR = ("ru", "bg", "uk", "mk", "mn")


def detect_lang(t):
    head = (t or "")[:1500]
    if sum(1 for ch in head if 0x4E00 <= ord(ch) < 0xA000) > 10:
        return "zh"
    words = set(word_tokens(head))
    if sum(1 for ch in head if 0x400 <= ord(ch) < 0x530) > 20:
        if "ө" in head or "ү" in head:
            return "mn"
        best = max(CYR, key=lambda l: len(words & LANG_SETS[l]))
        return best if len(words & LANG_SETS[best]) >= 1 else "ru"
    scores = {l: len(words & LANG_SETS[l]) for l in LANG_SETS
              if l not in CYR}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "unknown"


TRIVIAL_ANSWERS = {"yes", "no", "none", "true", "false", "a", "b", "c",
                   "d", "e", ""}


def norm_answer(a):
    if a is None:
        return ""
    s = unicodedata.normalize("NFKC", str(a)).lower()
    return re.sub(r"[\s$\\{}~,;:.]+", "", s)


# --------------------------------------------------------------------------
# 3. Load corpus rows with features
# --------------------------------------------------------------------------
def load_rows():
    df = duckdb.connect().execute(f"""
        select id, problem_markdown, country, competition, final_answer
        from '{PARQUET}'
    """).df()
    rows = []
    for r in df.itertuples(index=False):
        tag, from_country = detect_tag(r.country, r.competition)
        year, ysrc = extract_year(r.competition, tag, from_country)
        text = r.problem_markdown or ""
        rows.append({
            "id": r.id, "text": text,
            "country": r.country if isinstance(r.country, str) else "",
            "competition": (r.competition
                            if isinstance(r.competition, str) else ""),
            "tag": tag, "year": year, "year_src": ysrc,
            "lang": detect_lang(text),
            "hash": hashlib.md5(norm_text(text).encode()).hexdigest(),
            "normlen": len(norm_text(text)),
            "fp": num_fingerprint(text),
            "msig": math_sig(text),
            "sh": None,  # word shingles, filled lazily
            "nans": norm_answer(r.final_answer),
        })
    return rows


def get_sh(r):
    if r["sh"] is None:
        r["sh"] = shingles(r["text"])
    return r["sh"]


def different_source(a, b):
    return (a["country"] != b["country"]) or (a["competition"] != b["competition"])


# --------------------------------------------------------------------------
# 4. Candidate generators (each returns set of (id_a,id_b) with id_a<id_b)
# --------------------------------------------------------------------------
def g_exact(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["normlen"] >= 40:
            groups[r["hash"]].append(r)
    out = set()
    for g in groups.values():
        if 2 <= len(g) <= 12:
            for a, b in itertools.combinations(g, 2):
                if different_source(a, b):
                    out.add(tuple(sorted((a["id"], b["id"]))))
    return out


def _minhash_lsh(items, n_perm=64, bands=16, cap=50, seed=13):
    """items: {id: set-of-int-hashes}. Returns candidate id pairs."""
    rnd = random.Random(seed)
    perms = [(rnd.randrange(1, 2**61 - 1), rnd.randrange(0, 2**61 - 1))
             for _ in range(n_perm)]
    rpb = n_perm // bands
    buckets = defaultdict(list)
    for rid, s in items.items():
        sig = [min(((a * h + b) % 2305843009213693951) for h in s)
               for a, b in perms]
        for band in range(bands):
            buckets[(band, tuple(sig[band * rpb:(band + 1) * rpb]))].append(rid)
    cand = set()
    for ids in buckets.values():
        if 2 <= len(ids) <= cap:
            for a, b in itertools.combinations(sorted(ids), 2):
                cand.add((a, b))
    return cand


def g_lsh_text(rows):
    items = {}
    for r in rows:
        sh = {h & 0xFFFFFFFFFFFF for h in get_sh(r)}
        if len(sh) >= 15:
            items[r["id"]] = sh
    return _minhash_lsh(items, seed=13)


def g_lsh_math(rows):
    items = {}
    for r in rows:
        if len(r["msig"]) >= 10:
            items[r["id"]] = {hash(g) & 0xFFFFFFFFFFFF for g in r["msig"]}
    return _minhash_lsh(items, seed=31)


def g_raregram(rows, gram_df, lo=2, hi=10, min_shared=3):
    """Inverted index over rare formula grams: catches cross-language pairs
    whose overall math Jaccard is diluted by boilerplate formulas."""
    inv = defaultdict(list)
    for r in rows:
        for g in r["msig"]:
            if lo <= gram_df[g] <= hi:
                inv[g].append(r["id"])
    cnt = Counter()
    for ids in inv.values():
        for a, b in itertools.combinations(sorted(ids), 2):
            cnt[(a, b)] += 1
    return {p for p, c in cnt.items() if c >= min_shared}


def g_tag(rows, answer_df):
    """Same intl competition, pre-gated by distinctive answer or number
    overlap (full verification happens later)."""
    bytag = defaultdict(list)
    for r in rows:
        if r["tag"]:
            bytag[r["tag"]].append(r)
    out = set()
    for grp in bytag.values():
        for a, b in itertools.combinations(grp, 2):
            if not different_source(a, b):
                continue
            ya, yb = a["year"], b["year"]
            if ya is not None and yb is not None and abs(ya - yb) > 1:
                continue
            distinctive = (a["nans"] and a["nans"] == b["nans"]
                           and a["nans"] not in TRIVIAL_ANSWERS
                           and answer_df.get(a["nans"], 99) <= 4)
            if distinctive:
                out.add(tuple(sorted((a["id"], b["id"]))))
                continue
            ca, cb = Counter(a["fp"]), Counter(b["fp"])
            shared = sum((ca & cb).values())
            if shared >= 2 and multiset_jaccard(a["fp"], b["fp"]) >= 0.4:
                out.add(tuple(sorted((a["id"], b["id"]))))
    return out


def g_numfp(rows):
    out = set()
    byid = {r["id"]: r for r in rows}
    # exact nontrivial fingerprint groups
    groups = defaultdict(list)
    for r in rows:
        if fp_nontrivial(r["fp"]):
            groups[r["fp"]].append(r["id"])
    for ids in groups.values():
        if 2 <= len(ids) <= 8:
            for a, b in itertools.combinations(sorted(ids), 2):
                if different_source(byid[a], byid[b]):
                    out.add((a, b))
    # rare-number blocking
    docfreq = Counter()
    for r in rows:
        for n in set(r["fp"]):
            docfreq[n] += 1
    inv = defaultdict(list)
    for r in rows:
        if len(r["fp"]) >= 4:
            for n in set(r["fp"]):
                if n >= 10 and docfreq[n] <= 25:
                    inv[n].append(r["id"])
    cand = Counter()
    for ids in inv.values():
        for a, b in itertools.combinations(sorted(ids), 2):
            cand[(a, b)] += 1
    for (a, b), nshared in cand.items():
        if nshared >= 2 and different_source(byid[a], byid[b]) and \
                multiset_jaccard(byid[a]["fp"], byid[b]["fp"]) >= 0.5:
            out.add((a, b))
    return out


# --------------------------------------------------------------------------
# 5. Verification layer (thresholds calibrated on 39 hand-labeled pairs)
# --------------------------------------------------------------------------
RARE_DF = 20        # a formula gram is "rare" if its corpus df <= 20
RSHARED_MIN = 25    # min shared rare-gram IDF mass for V2
RCONTAIN_MIN = 0.75  # min-side rare containment
RSYM_MIN = 0.65     # symmetric rare containment: BOTH sides nearly covered
ASYM_MIN = 0.50     # symmetric containment over ALL grams (idf-weighted)


def rare_stats(ra, rb, gram_df, idf):
    """(rshared, rexcl_min, rcont_min, rcont_sym, allcont_sym).
    rcont_sym uses the MAX exclusive side: a true translation matches the
    entire math content, so exclusive rare mass must be small on BOTH
    sides.  Generic-but-rare gram runs (subscripted label lists, digit
    sequences) fail this because the rest of each problem's math differs.
    allcont_sym repeats the symmetric test over ALL grams (idf-weighted):
    it catches pairs whose only rare gram happens to be shared (e.g. both
    contain '\\angle B < \\angle C') while the bulk of the math differs."""
    A = {g for g in ra["msig"] if gram_df[g] <= RARE_DF}
    B = {g for g in rb["msig"] if gram_df[g] <= RARE_DF}
    rshared = sum(idf[g] for g in A & B)
    ex_a = sum(idf[g] for g in A - B)
    ex_b = sum(idf[g] for g in B - A)
    lo, hi = min(ex_a, ex_b), max(ex_a, ex_b)
    rcont = rshared / (rshared + lo) if (rshared + lo) > 0 else 0.0
    rsym = rshared / (rshared + hi) if (rshared + hi) > 0 else 0.0
    FA, FB = ra["msig"], rb["msig"]
    ash = sum(idf[g] for g in FA & FB)
    aex = max(sum(idf[g] for g in FA - FB), sum(idf[g] for g in FB - FA))
    asym = ash / (ash + aex) if (ash + aex) > 0 else 0.0
    return rshared, lo, rcont, rsym, asym


def verify(ra, rb, answer_df, gram_df, idf):
    """Return (rule, confidence, evidence) or None."""
    tj = set_jaccard(get_sh(ra), get_sh(rb))
    ms = min(len(ra["msig"]), len(rb["msig"]))
    mj = set_jaccard(ra["msig"], rb["msig"]) if ms else 0.0
    nj = multiset_jaccard(ra["fp"], rb["fp"])
    rshared, rexcl, rcont, rsym, asym = rare_stats(ra, rb, gram_df, idf)
    # "suspect" = both sides carry substantial exclusive rare math:
    # the signature of a deliberate minimal variation, not a translation
    suspect = rexcl >= 12 and rcont < 0.7
    distinctive = (ra["nans"] and ra["nans"] == rb["nans"]
                   and ra["nans"] not in TRIVIAL_ANSWERS
                   and answer_df.get(ra["nans"], 99) <= 4)
    same_tag = ra["tag"] is not None and ra["tag"] == rb["tag"]
    ya, yb = ra["year"], rb["year"]
    years_ok = ya is None or yb is None or abs(ya - yb) <= 1

    ev = {"text_jaccard": round(tj, 3), "math_jaccard": round(mj, 3),
          "math_sig_min": ms, "num_jaccard": round(nj, 3),
          "rare_shared_mass": round(rshared, 1),
          "rare_excl_mass": round(rexcl, 1),
          "rare_containment": round(rcont, 3),
          "rare_containment_sym": round(rsym, 3),
          "allgram_containment_sym": round(asym, 3),
          "minimal_pair_suspect": suspect,
          "answer_distinctive": distinctive,
          "tag": ra["tag"] if same_tag else None,
          "year_a": ya, "year_b": yb}

    if not suspect and (tj >= 0.70 or (tj >= 0.55 and rshared >= 15)):
        return "V1_text", "high", ev
    if (rshared >= RSHARED_MIN and rcont >= RCONTAIN_MIN
            and rsym >= RSYM_MIN and asym >= ASYM_MIN):
        return "V2_rare_math", "high", ev
    if distinctive and not suspect and mj >= 0.35 and ms >= 8:
        return "V4_answer_math", "medium", ev
    if distinctive and same_tag and years_ok and nj >= 0.6:
        return "V5_answer_tag", "medium", ev
    return None


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    byid = {r["id"]: r for r in rows}
    answer_df = Counter(r["nans"] for r in rows if r["nans"])
    print(f"loaded {len(rows)} rows; "
          f"{sum(1 for r in rows if r['tag'])} intl-tagged; "
          f"{sum(1 for r in rows if r['year'])} with year; "
          f"{sum(1 for r in rows if len(r['msig']) >= 10)} with math sig")
    lang_dist = Counter(r["lang"] for r in rows)
    print("lang distribution:", dict(lang_dist.most_common(15)))

    # corpus df / idf of formula grams (needed by generator + verifier)
    import math as _math
    gram_df = Counter()
    for r in rows:
        for g in r["msig"]:
            gram_df[g] += 1
    N = len(rows)
    idf = {g: _math.log(N / c) for g, c in gram_df.items()}

    gens = {
        "exact": g_exact(rows),
        "lsh_text": g_lsh_text(rows),
        "lsh_math": g_lsh_math(rows),
        "raregram": g_raregram(rows, gram_df),
        "tag": g_tag(rows, answer_df),
        "numfp": g_numfp(rows),
    }
    for name, s in gens.items():
        print(f"generator {name}: {len(s)} candidate pairs")

    # union candidates with their generator provenance
    cand = defaultdict(set)
    for name, s in gens.items():
        for key in s:
            cand[key].add(name)
    print(f"union candidates: {len(cand)}")

    exact_set = gens["exact"]
    accepted = {}
    for key, sources in cand.items():
        a, b = key
        ra, rb = byid[a], byid[b]
        if not different_source(ra, rb):
            continue
        if key in exact_set:
            accepted[key] = ("exact_text", "high",
                             {"text_jaccard": 1.0}, sorted(sources))
            continue
        v = verify(ra, rb, answer_df, gram_df, idf)
        if v:
            rule, conf, ev = v
            accepted[key] = (rule, conf, ev, sorted(sources))

    anchors = set(json.load(open(ANCHOR_MAP))["exclude_corpus_ids"])
    out_path = f"{OUT_DIR}/duplicate_candidates.jsonl"
    n_cross_lang = n_cross_country = 0
    rule_counts, conf_counts = Counter(), Counter()
    lang_pair_counts = Counter()
    with open(out_path, "w") as f:
        for (a, b), (rule, conf, ev, sources) in sorted(accepted.items()):
            ra, rb = byid[a], byid[b]
            cross_lang = (ra["lang"] != rb["lang"]
                          and "unknown" not in (ra["lang"], rb["lang"]))
            n_cross_lang += cross_lang
            n_cross_country += ra["country"] != rb["country"]
            rule_counts[rule] += 1
            conf_counts[conf] += 1
            if cross_lang:
                lang_pair_counts[tuple(sorted((ra["lang"], rb["lang"])))] += 1
            f.write(json.dumps({
                "id_a": a, "id_b": b,
                "rule": rule, "confidence": conf,
                "generators": sources, "evidence": ev,
                "country_a": ra["country"], "country_b": rb["country"],
                "competition_a": ra["competition"],
                "competition_b": rb["competition"],
                "lang_a": ra["lang"], "lang_b": rb["lang"],
                "cross_language": cross_lang,
                "cross_country": ra["country"] != rb["country"],
                "in_retrieve_anchors": (a in anchors) or (b in anchors),
            }, ensure_ascii=False) + "\n")

    summary = {
        "generated": "2026-07-29",
        "corpus": PARQUET,
        "n_corpus_rows": len(rows),
        "n_rows_intl_tagged": sum(1 for r in rows if r["tag"]),
        "n_rows_with_year": sum(1 for r in rows if r["year"]),
        "language_distribution": dict(lang_dist.most_common()),
        "generator_candidate_counts": {k: len(v) for k, v in gens.items()},
        "union_candidates_prefilter": len(cand),
        "total_accepted_pairs": len(accepted),
        "by_rule": dict(rule_counts),
        "by_confidence": dict(conf_counts),
        "cross_language_pairs": n_cross_lang,
        "cross_language_pair_breakdown":
            {f"{a}-{b}": c for (a, b), c in lang_pair_counts.most_common()},
        "cross_country_pairs": n_cross_country,
        "pairs_touching_retrieve_anchors": sum(
            1 for (a, b) in accepted if a in anchors or b in anchors),
        "notes": ("Generators produce candidates; verification rules "
                  "accept. Core signal: IDF mass of shared RARE LaTeX "
                  "formula 4-grams (corpus df<=20) with two containment "
                  "tests: rcont_min rejects deliberate minimal variations "
                  "(same formula, one exponent changed) and rcont_sym "
                  "requires the ENTIRE math content to match on both sides "
                  "(kills generic-rare-gram matches: subscript label runs, "
                  "digit lists) plus an all-gram symmetric containment "
                  "gate. Thresholds calibrated on ~80 hand-labeled pairs "
                  "over three sampling rounds: 32/32 math-verifiable trues "
                  "kept, 1/19 math-bearing falses admitted. Unbiased "
                  "random-sample precision ~85-90percent (17/20 pre-fix, "
                  "17/18 after the final two fixes). 'high' confidence = "
                  "text/math-verified; 'medium' = distinctive-answer "
                  "corroboration. lang-ID is stopword-based, approximate "
                  "for close languages (hr/sr/sl, es/pt). Known limits: "
                  "formula-free problems (pure-text combinatorics) are "
                  "under-recalled - cross-language duplicates without "
                  "LaTeX cannot be verified by this pipeline; problems "
                  "reprinted with an extra sub-question can fail "
                  "rcont_sym; V1 at tj>=0.7 could still admit a minimal "
                  "pair whose only change is outside rare math grams."),
    }
    with open(f"{OUT_DIR}/duplicate_mining_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("language_distribution",
                                   "cross_language_pair_breakdown")},
                     indent=2))


if __name__ == "__main__":
    main()
