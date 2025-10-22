#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Mapper v2 — FastPath + Long-Text Guard + Hybrid Similarity + Parallel + Progress
---------------------------------------------------------------------------------------
- Row-wise, ID-aligned comparison between dashboard and source rows
- Handles duplicate IDs by grouping (concatenates duplicate rows by ID)
- Fast path order:
    1) Direct string == string            -> tag='direct', highest score
    2) Wrapped-list membership            -> tag='list', slightly lower
    2.5) Delimited whole-item membership  -> tag='listlike', lower than wrapped
    3) Hybrid similarity (TF-IDF + fuzzy) -> tag='similarity' (last resort)
- Parallel execution with chunked tqdm progress
- Outputs:
    * column_mapping.csv  (best per dashboard column)
    * all_candidates.csv  (all column pairs with metrics + hit-kind counts)
    * per_cell_audit.csv  (optional; --audit) with per-row score + tag

Usage:
python hybrid_mapper_single.py \
  --source source.csv \
  --dashboard dashboard.csv \
  --id-source "Source ID Column" \
  --id-dash "Dashboard ID Column" \
  --out-dir ./out \
  --n-jobs 6 \
  --sample-rows 1000 \
  --min-ratio 0.70 \
  --audit
"""

# ======================
# ======= CONFIG =======
# ======================
# --- Global row-level default (can override via --min-ratio)
MIN_RATIO = 0.70

# --- Heuristics for short/long text (guarding substring/“exact” inside long paragraphs)
LONG_TEXT_CHAR_THRESHOLD   = 95
LONG_TEXT_TOKEN_THRESHOLD  = 15
SHORT_TEXT_CHAR_THRESHOLD  = 35
SHORT_TEXT_TOKEN_THRESHOLD = 4

# When dashboard short is inside source long, require one of these:
MIN_PRECISION_FOR_LONG_TEXT = 0.40   # matched_len / len(source_val)
MIN_PARTIAL_FOR_LONG_TEXT   = 0.90   # partial_ratio
MIN_TFIDF_FOR_LONG_TEXT     = 0.45   # TF-IDF cosine

# --- Fast-path “points” (scores) for early exits
DIRECT_MATCH_SCORE              = 0.99   # pure “single == single”
WRAPPED_LIST_MATCH_SCORE        = 0.94   # member of [A,B] / {A,B} / (A,B)
LISTLIKE_DELIMITED_MATCH_SCORE  = 0.90   # member of unwrapped “A, B, C”

# --- Container penalty applied to similarity fallback
CONTAINER_PENALTY = {
    "single":         1.00,
    "wrapped_list":   0.94,
    "delimited_list": 0.90,
    "empty":          0.00,
}

# --- Hybrid similarity blend weights (only in fallback step)
# base score = w_fp*partial_ratio + w_fr*ratio + w_lev*lev + w_tfidf*tfidf
W_FP    = 0.25
W_FR    = 0.25
W_LEV   = 0.20
W_TFIDF = 0.30

# --- TF-IDF vectorizer params (char-ngrams is robust to typos)
TFIDF_ANALYZER    = "char"
TFIDF_NGRAM_RANGE = (3, 5)

# --- Suggestion labels (for best mapping per dashboard col)
HIGH_RATIO = 0.85
HIGH_SCORE = 0.95


# ======================
# ====== IMPORTS =======
# ======================
import argparse
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps with safe fallbacks
try:
    from rapidfuzz import fuzz, distance
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False


# ==============================
# ====== TEXT UTILITIES ========
# ==============================
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(str(text))]

def normalize_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    # trim common wrappers/quotes if present (only extremes)
    s = s.strip("{}[]()\"' ")
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_listy(s: str) -> List[str]:
    """
    Split wrapped values {a,b}, [a,b], (a,b) into elements;
    otherwise return one normalized element.
    """
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
    """Word-boundary substring test."""
    A = re.escape(normalize_text(a))
    B = normalize_text(b)
    if not A or not B:
        return False
    return re.search(rf"\b{A}\b", B) is not None

def overlap_proportions(a: str, b: str) -> Tuple[float, float]:
    """
    Returns (recall, precision):
    - recall    = |match| / len(a)
    - precision = |match| / len(b)
    """
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


# ==================================
# ====== CONTAINER DETECTION =======
# ==================================
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


# =================================
# ====== SIMILARITY PRIMITIVES ====
# =================================
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
    """
    Hybrid similarity with long-text guards + boundary awareness.
    """
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0

    is_exact = (a == b)
    is_boundary_sub = boundary_substring(a, b)
    _, precision = overlap_proportions(a, b)
    long_text_b = is_long_text_value(b)
    short_text_a = is_short_text_value(a)

    fr, fp, lev = fuzzy_scores(a, b)
    tfidf_score = tfidf_cosine(a, b, tfidf_vec)

    base = W_FP * fp + W_FR * fr + W_LEV * lev + W_TFIDF * tfidf_score
    boost = 0.0

    # EXACT equality: strong, but still gated for long-text
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

    # SUBSTRING (word boundary) boost, also gated
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


def make_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=TFIDF_ANALYZER, ngram_range=TFIDF_NGRAM_RANGE)


# =======================================
# ====== FAST-PATH DISPATCH (TAGS) ======
# =======================================
def compare_elements(dash_val: str, source_val: str, tfidf_vec: TfidfVectorizer) -> Tuple[float, str]:
    """
    1) Direct string == string                 -> 0.99, 'direct'
    2) Wrapped list membership                 -> 0.94, 'list'
    2.5) Delimited whole-item membership       -> 0.90, 'listlike'
    3) Hybrid similarity                       -> 'similarity'
    Returns (score, tag).
    """
    d_norm = normalize_text(dash_val)
    s_norm = normalize_text(source_val)
    if not d_norm or not s_norm:
        return 0.0, "none"

    src_type = source_container_type(source_val)
    penalty = CONTAINER_PENALTY.get(src_type, 1.0)

    # Step 1: direct equality on single source value
    if src_type == "single" and d_norm == s_norm:
        return DIRECT_MATCH_SCORE, "direct"

    # Step 2: wrapped list membership
    if src_type == "wrapped_list":
        elems = wrapped_list_elements(source_val)
        if d_norm in elems:
            return WRAPPED_LIST_MATCH_SCORE, "list"

    # Step 2.5: delimited whole-item membership (unwrapped comma/pipe/semicolon/slash lists)
    if src_type == "delimited_list":
        token = re.escape(d_norm)
        pattern = rf"(?:^|[,\|;/])\s*{token}\s*(?=[,\|;/]|$)"
        if re.search(pattern, s_norm):
            return LISTLIKE_DELIMITED_MATCH_SCORE, "listlike"

    # Step 3: hybrid similarity fallback
    d_elems = split_listy(dash_val) or [d_norm]
    s_elems = split_listy(source_val) or [s_norm]
    sim_best = 0.0
    for d in d_elems:
        for s in s_elems:
            sim_best = max(sim_best, compute_similarity(d, s, tfidf_vec))
    sim_final = sim_best * penalty
    return sim_final, "similarity"


# ====================================
# ====== GROUP DUPLICATE ROW IDs =====
# ====================================
def merge_group(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    If an ID appears multiple times, concatenate values across rows per column.
    This preserves all data for comparison without breaking index uniqueness.
    Works for duplicates in BOTH source and dashboard.
    """
    def join_values(x: pd.Series) -> str:
        vals = [str(v) for v in x if pd.notna(v) and str(v).strip()]
        return " | ".join(vals)
    return df.groupby(id_col, dropna=False).agg(join_values).reset_index()


# ===========================
# ====== CORE MAPPING =======
# ===========================
def map_columns(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    sample_rows: Optional[int] = None,
    n_jobs: int = 1,
    progress: bool = True,
    min_ratio: float = MIN_RATIO
):
    # Merge duplicates by concatenating values under each ID
    source_df = merge_group(source_df, id_source)
    dash_df   = merge_group(dash_df, id_dash)

    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        raise ValueError("No matching IDs found between source and dashboard!")

    if sample_rows:
        common_ids = common_ids[:sample_rows]

    # Build TF-IDF once (over all values)
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = make_tfidf_vectorizer().fit([normalize_text(v) for v in all_values if v])

    dash_cols = list(dst.columns)
    src_cols  = list(src.columns)
    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]
    if not pairs:
        return pd.DataFrame(), pd.DataFrame()

    def compare_pair(dc: str, sc: str):
        hits = 0
        total_pairs = 0
        kept_scores = []
        direct_hits = 0
        list_hits = 0
        listlike_hits = 0
        similarity_hits = 0

        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            total_pairs += 1

            score, tag = compare_elements(dv, sv, tfidf_vec)
            if score >= min_ratio:
                hits += 1
                kept_scores.append(score)
                if tag == "direct":
                    direct_hits += 1
                elif tag == "list":
                    list_hits += 1
                elif tag == "listlike":
                    listlike_hits += 1
                else:
                    similarity_hits += 1

        if total_pairs == 0 or hits == 0:
            return None

        avg_score = float(np.mean(kept_scores)) if kept_scores else 0.0
        max_score = float(np.max(kept_scores)) if kept_scores else 0.0
        hit_rate  = hits / total_pairs
        weighted_score = avg_score * hits
        score_pct = (weighted_score / total_pairs) * 100.0

        return dict(
            dashboard_col=dc,
            source_col=sc,
            hits=hits,
            total_pairs=total_pairs,
            hit_rate=hit_rate,
            avg_score=avg_score,
            max_score=max_score,
            weighted_score=weighted_score,
            score_pct=score_pct,
            direct_hits=direct_hits,
            list_hits=list_hits,
            listlike_hits=listlike_hits,
            similarity_hits=similarity_hits
        )

    use_parallel = HAVE_JOBLIB and (n_jobs is not None) and (n_jobs != 1)
    results = []

    if use_parallel:
        total = len(pairs)
        # ~50 progress bar updates (chunked scheduling)
        target_updates = 50
        chunk_size = max(1, total // target_updates)
        pbar = tqdm(total=total, desc="Mapping columns (parallel & weighted)") if (HAVE_TQDM and progress) else None

        for i in range(0, total, chunk_size):
            chunk = pairs[i:i + chunk_size]
            mapped = Parallel(n_jobs=n_jobs)(
                delayed(compare_pair)(dc, sc) for dc, sc in chunk
            )
            results.extend([m for m in mapped if m])
            if pbar:
                pbar.update(len(chunk))
        if pbar:
            pbar.close()
    else:
        it = tqdm(pairs, desc="Mapping columns (weighted)") if (progress and HAVE_TQDM) else pairs
        for dc, sc in it:
            r = compare_pair(dc, sc)
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rank
    df = df.sort_values(
        ["weighted_score", "avg_score", "hits", "hit_rate"],
        ascending=False
    ).reset_index(drop=True)

    # Best per dashboard column (by weighted_score)
    best_idx = df.groupby("dashboard_col")["weighted_score"].idxmax()
    best = df.loc[best_idx].copy()

    # Friendly suggestion label
    def suggest(r):
        if r["hit_rate"] >= 0.90 and r["avg_score"] >= HIGH_RATIO:
            return "exact/very likely (strong coverage)"
        if r["hit_rate"] >= 0.75 and r["avg_score"] >= min_ratio:
            return "likely (good coverage)"
        if r["hit_rate"] >= 0.50 and r["avg_score"] >= min_ratio:
            return "possible (review)"
        return "uncertain"

    best["suggestion"] = best.apply(suggest, axis=1)
    best = best.sort_values(
        ["weighted_score", "avg_score", "hits", "hit_rate"],
        ascending=False
    ).reset_index(drop=True)

    return best, df


# ===========================
# ====== PER-CELL AUDIT =====
# ===========================
def per_cell_audit(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    pairs: List[Tuple[str, str]],
    max_rows: int = 1000,
    min_ratio: float = MIN_RATIO
) -> pd.DataFrame:
    source_df = merge_group(source_df, id_source)
    dash_df   = merge_group(dash_df, id_dash)

    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        return pd.DataFrame()

    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = make_tfidf_vectorizer().fit([normalize_text(v) for v in all_values if v])

    rows = []
    for dc, sc in pairs:
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            score, tag = compare_elements(dv, sv, tfidf_vec)
            if score >= min_ratio:  # only log “hit” rows if you prefer; or remove this guard to log all
                rows.append(
                    dict(
                        id=rid,
                        dashboard_col=dc,
                        source_col=sc,
                        dashboard_val=dv,
                        source_val=sv,
                        score=score,
                        tag=tag
                    )
                )
            if len(rows) >= max_rows:
                break

    return pd.DataFrame(rows)


# ===================
# ====== CLI ========
# ===================
def load_dataframes(source_path: str, dashboard_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    read_src = pd.read_excel if source_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    read_dash = pd.read_excel if dashboard_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    s_df = read_src(source_path)
    d_df = read_dash(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df


def main():
    p = argparse.ArgumentParser(description="Hybrid Mapper v2 (FastPath + Long-Text Guard + Similarity, Parallel, Progress)")
    p.add_argument("--source", required=True)
    p.add_argument("--dashboard", required=True)
    p.add_argument("--id-source", required=True)
    p.add_argument("--id-dash", required=True)
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--sample-rows", type=int, default=None)
    p.add_argument("--min-ratio", type=float, default=MIN_RATIO,
                   help="Row-level similarity threshold to count as a hit (default 0.70)")
    args = p.parse_args()

    s_df, d_df = load_dataframes(args.source, args.dashboard)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    best, full = map_columns(
        s_df, d_df,
        id_source=args.id_source,
        id_dash=args.id_dash,
        sample_rows=args.sample_rows,
        n_jobs=args.n_jobs,
        progress=True,
        min_ratio=args.min_ratio
    )

    # Save outputs
    best_path = out / "column_mapping.csv"
    best.to_csv(best_path, index=False)
    print(f"[✔] Wrote {best_path}")

    full_path = out / "all_candidates.csv"
    full.to_csv(full_path, index=False)
    print(f"[✔] Wrote {full_path}")

    if args.audit and not best.empty:
        pairs = list(zip(best["dashboard_col"], best["source_col"]))
        audit = per_cell_audit(
            s_df, d_df, args.id_source, args.id_dash, pairs,
            max_rows=1000, min_ratio=args.min_ratio
        )
        audit_path = out / "per_cell_audit.csv"
        audit.to_csv(audit_path, index=False)
        print(f"[✔] Wrote {audit_path}")


if __name__ == "__main__":
    main()
