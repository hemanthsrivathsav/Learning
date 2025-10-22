import re
import math
import numpy as np

from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz, distance
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False

from .config import (
    LONG_TEXT_CHAR_THRESHOLD, LONG_TEXT_TOKEN_THRESHOLD,
    SHORT_TEXT_CHAR_THRESHOLD, SHORT_TEXT_TOKEN_THRESHOLD,
    MIN_PRECISION_FOR_LONG_TEXT, MIN_PARTIAL_FOR_LONG_TEXT, MIN_TFIDF_FOR_LONG_TEXT,
    DIRECT_MATCH_SCORE, WRAPPED_LIST_MATCH_SCORE, LISTLIKE_DELIMITED_MATCH_SCORE,
    CONTAINER_PENALTY, TAG_THRESHOLDS,
    W_FP, W_FR, W_LEV, W_TFIDF,
    TFIDF_ANALYZER, TFIDF_NGRAM_RANGE,
    BM25_SQUASH_SCALE,
)


# ---------- Tokenization / normalization ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(str(text))]

def normalize_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    s = s.strip("{}[]()\"' ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_listy(s: str) -> List[str]:
    """Split wrapped {a,b}, [a,b] or (a,b) else single normalized element."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    raw = str(s).strip()
    if not raw:
        return []
    first, last = raw[0], raw[-1]
    if (first == "{" and last == "}") or (first == "[" and last == "]") or (first == "(" and last == ")"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [normalize_text(p) for p in parts if p]
    return [normalize_text(raw)]

def is_long_text_value(s: str) -> bool:
    s = normalize_text(s)
    if not s:
        return False
    return len(s) >= LONG_TEXT_CHAR_THRESHOLD or len(s.split()) >= LONG_TEXT_TOKEN_THRESHOLD

def is_short_text_value(s: str) -> bool:
    s = normalize_text(s)
    if not s:
        return True
    return len(s) <= SHORT_TEXT_CHAR_THRESHOLD or len(s.split()) <= SHORT_TEXT_TOKEN_THRESHOLD

def boundary_substring(a: str, b: str) -> bool:
    A = re.escape(normalize_text(a))
    B = normalize_text(b)
    if not A or not B:
        return False
    return re.search(rf"\b{A}\b", B) is not None

def overlap_proportions(a: str, b: str) -> Tuple[float, float]:
    A = normalize_text(a)
    B = normalize_text(b)
    if not A or not B:
        return (0.0, 0.0)
    if A in B:
        return (1.0, len(A) / len(B))
    m = re.search(rf"\b{re.escape(A)}\b", B)
    if m:
        matched = m.end() - m.start()
        return (matched / len(A), matched / len(B))
    return (0.0, 0.0)

# ---------- Container detection ----------
DELIMS_SIMPLE = r"[,\|;/]"  # detect unwrapped delimited strings

def is_wrapped_list(raw: object) -> bool:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return False
    text = str(raw).strip()
    if not text:
        return False
    first, last = text[0], text[-1]
    return (first == "{" and last == "}") or (first == "[" and last == "]") or (first == "(" and last == ")")

def wrapped_list_elements(raw: object) -> List[str]:
    if not is_wrapped_list(raw):
        return []
    text = str(raw).strip()
    inner = text[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [normalize_text(p) for p in parts if p]

def source_container_type(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "empty"
    text = str(raw).strip()
    if not text:
        return "empty"
    if is_wrapped_list(text):
        return "wrapped_list"
    if re.search(DELIMS_SIMPLE, text):
        return "delimited_list"
    return "single"

# ---------- BM25 ----------
def build_bm25_stats_for_source(src_df) -> Dict[str, Dict]:
    stats = {}
    for col in src_df.columns:
        vals = src_df[col].fillna("").astype(str).tolist()
        N = len(vals) if len(vals) > 0 else 1
        dfs = {}
        doc_lens = []
        tokenized = []
        for v in vals:
            toks = tokenize(v)
            tokenized.append(set(toks))
            doc_lens.append(len(toks))
        for toks in tokenized:
            for t in toks:
                dfs[t] = dfs.get(t, 0) + 1
        idf = {t: math.log((N - df + 0.5) / (df + 0.5) + 1e-12) for t, df in dfs.items()}
        avgdl = (sum(doc_lens) / N) if N > 0 else 1.0
        stats[col] = {"N": N, "avgdl": float(avgdl), "idf": idf}
    return stats

def bm25_score(query_tokens: List[str], doc_tokens: List[str], idf_map: Dict[str, float], avgdl: float, k1: float = 1.2, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0
    tf = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for q in query_tokens:
        idf = idf_map.get(q, 0.0)
        f = tf.get(q, 0)
        if f == 0 or idf <= 0.0:
            continue
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * ((f * (k1 + 1)) / denom)
    return score

def squash_bm25(raw_score: float, scale: float = BM25_SQUASH_SCALE) -> float:
    x = raw_score / max(1e-6, scale)
    return 1.0 / (1.0 + math.exp(-x)) - 0.5  # ~ (0..0.5)


# ---------- Fuzzy + TF-IDF ----------
def tfidf_cosine(a: str, b: str, tfidf_vec: TfidfVectorizer) -> float:
    if not a or not b:
        return 0.0
    va = tfidf_vec.transform([normalize_text(a)])
    vb = tfidf_vec.transform([normalize_text(b)])
    sim = cosine_similarity(va, vb)[0][0]
    return float(sim) if not np.isnan(sim) else 0.0

def fuzzy_scores(a: str, b: str) -> Tuple[float, float, float]:
    if HAVE_RAPIDFUZZ:
        fr = fuzz.ratio(a, b) / 100.0
        fp = fuzz.partial_ratio(a, b) / 100.0
        lev = distance.Levenshtein.normalized_similarity(a, b)
    else:
        import difflib
        fr = difflib.SequenceMatcher(a=a, b=b).ratio()
        fp = fr
        lev = fr
    return fr, fp, lev

def compute_similarity(a: str, b: str, tfidf_vec: TfidfVectorizer) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0

    is_exact = (a == b)
    is_boundary_sub = boundary_substring(a, b)
    recall, precision = overlap_proportions(a, b)
    long_text_b = is_long_text_value(b)
    short_text_a = is_short_text_value(a)

    fr, fp, lev = fuzzy_scores(a, b)
    tfidf_score = tfidf_cosine(a, b, tfidf_vec)

    base = W_FP * fp + W_FR * fr + W_LEV * lev + W_TFIDF * tfidf_score
    boost = 0.0

    if is_exact:
        if not long_text_b:
            boost += 0.40
        else:
            if short_text_a:
                if (precision >= MIN_PRECISION_FOR_LONG_TEXT or
                    tfidf_score >= MIN_TFIDF_FOR_LONG_TEXT or
                    fp >= MIN_PARTIAL_FOR_LONG_TEXT):
                    boost += 0.25
                else:
                    boost -= 0.20
            else:
                if (precision >= MIN_PRECISION_FOR_LONG_TEXT/2 or
                    tfidf_score >= MIN_TFIDF_FOR_LONG_TEXT/2 or
                    fp >= 0.85):
                    boost += 0.20
                else:
                    boost -= 0.10

    if is_boundary_sub and not is_exact:
        if not long_text_b:
            boost += 0.25
        else:
            if short_text_a:
                if (precision >= MIN_PRECISION_FOR_LONG_TEXT or
                    tfidf_score >= MIN_TFIDF_FOR_LONG_TEXT or
                    fp >= MIN_PARTIAL_FOR_LONG_TEXT):
                    boost += 0.15
                else:
                    boost -= 0.15
            else:
                if (precision >= MIN_PRECISION_FOR_LONG_TEXT/2 or
                    tfidf_score >= MIN_TFIDF_FOR_LONG_TEXT/2 or
                    fp >= 0.80):
                    boost += 0.10
                else:
                    boost -= 0.10

    return min(0.999, max(0.0, base + boost))


# ---------- Fast-path dispatcher ----------
def compare_elements(dash_val: str, source_val: str, tfidf_vec: TfidfVectorizer, bm25_sc_stats: Optional[dict]) -> Tuple[float, str]:
    """
    1) Direct string == string                 -> 0.99, 'direct'
    2) Wrapped list membership                 -> 0.965, 'list'
    2.5) Delimited whole-item membership       -> 0.94, 'listlike'
    3) BM25 fallback (length & IDF normalized) -> <= ~0.5, 'bm25'
    4) Hybrid similarity (TF-IDF + fuzzy)      -> 'similarity'
    Returns (score, tag) choosing the stronger of BM25 vs similarity (no early return).
    """
    d_norm = normalize_text(dash_val)
    s_norm = normalize_text(source_val)
    if not d_norm or not s_norm:
        return 0.0, "none"

    # detect container once
    src_type = source_container_type(source_val)
    penalty = CONTAINER_PENALTY.get(src_type, 1.0)

    # ---- Step 1: direct equality on single source value
    if src_type == "single" and d_norm == s_norm:
        return DIRECT_MATCH_SCORE, "direct"

    # ---- Step 2: wrapped list membership
    if src_type == "wrapped_list":
        elems = wrapped_list_elements(source_val)  # normalized elements
        if d_norm in elems:
            return WRAPPED_LIST_MATCH_SCORE, "list"

    # ---- Step 2.5: delimited whole-item membership (unwrapped comma/pipe/semicolon/slash lists)
    if src_type == "delimited_list":
        token = re.escape(d_norm)
        pattern = rf"(?:^|[,\|;/])\s*{token}\s*(?=[,\|;/]|$)"
        if re.search(pattern, s_norm):
            return LISTLIKE_DELIMITED_MATCH_SCORE, "listlike"

    # ---- Step 3: BM25 (compute but DO NOT early-return)
    bm25_final = 0.0
    if bm25_sc_stats is not None:
        q_toks = tokenize(d_norm)
        d_toks = tokenize(s_norm)
        bm25_raw = bm25_score(q_toks, d_toks, bm25_sc_stats["idf"], bm25_sc_stats["avgdl"])
        bm25_norm = squash_bm25(bm25_raw)      # ~0..~0.5
        bm25_final = bm25_norm * penalty       # apply container penalty

    # ---- Step 4: Hybrid similarity (always compute)
    d_elems = split_listy(dash_val) or [d_norm]
    s_elems = split_listy(source_val) or [s_norm]
    sim_best = 0.0
    for d in d_elems:
        for s in s_elems:
            sim_best = max(sim_best, compute_similarity(d, s, tfidf_vec))
    sim_final = sim_best * penalty             # apply same penalty once

    # ---- Choose the stronger signal
    if bm25_final >= sim_final:
        return bm25_final, "bm25"
    else:
        return sim_final, "similarity"


# ---------- Vectorizer factory ----------
def make_tfidf_vectorizer():
    return TfidfVectorizer(analyzer=TFIDF_ANALYZER, ngram_range=TFIDF_NGRAM_RANGE)
