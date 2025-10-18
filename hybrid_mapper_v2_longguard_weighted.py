#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Mapper v2 — Long-Text Guard + Parallel + Hit-Weighted Scoring
--------------------------------------------------------------------
- Row-wise, ID-aligned comparison between dashboard and source rows
- Handles duplicate IDs by grouping and concatenating values
- Length-aware, boundary-aware substring gating (also applied to "exact" equality)
- Similarity metrics blended: RapidFuzz (or difflib fallback) + TF-IDF + Levenshtein-like
- Outputs:
    * column_mapping.csv (best per dashboard column)
    * all_candidates.csv (all (dashboard_col, source_col) pairs with metrics)
    * per_cell_audit.csv (optional; --audit)

Scoring:
- For each (dashboard_col, source_col) pair:
    hits          = number of row pairs with similarity >= MIN_RATIO
    total_pairs   = number of comparable non-null row pairs
    avg_score     = mean similarity over *hit* rows only
    weighted_score= avg_score * hits
    score_pct     = (weighted_score / total_pairs) * 100

Ranking:
- We choose the best source column for each dashboard column by
  sorting candidates on: weighted_score DESC, avg_score DESC, hits DESC, hit_rate DESC.

Run:
python hybrid_mapper_v2_longguard_weighted.py \
  --source source.csv \
  --dashboard dashboard.csv \
  --id-source "Source ID Column" \
  --id-dash "Dashboard ID Column" \
  --out-dir ./out \
  --audit \
  --n-jobs 6 \
  --sample-rows 1000
"""

import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple

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


# ----------------------
# Thresholds / Settings
# ----------------------
# Your original thresholds (kept)
HIGH_SCORE = 0.95
HIGH_RATIO = 0.85
MIN_RATIO  = 0.70

# Long-text guard thresholds (generic)
LONG_TEXT_CHAR_THRESHOLD   = 80    # source considered "long" if len >= this OR tokens >= 15
LONG_TEXT_TOKEN_THRESHOLD  = 15
SHORT_TEXT_CHAR_THRESHOLD  = 24    # dashboard considered "short" if len <= this OR tokens <= 4
SHORT_TEXT_TOKEN_THRESHOLD = 4

# When (dashboard short) inside (source long), require one of these:
MIN_PRECISION_FOR_LONG_TEXT = 0.40   # matched_len / len(source_val)
MIN_PARTIAL_FOR_LONG_TEXT   = 0.90   # RapidFuzz partial_ratio
MIN_TFIDF_FOR_LONG_TEXT     = 0.50   # TF-IDF similarity

DELIMS = r"[,\|;/]+"


# ----------------------
# Normalization helpers
# ----------------------
def normalize_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x)
    s = s.strip()
    # strip common wrappers and quotes
    s = s.strip("{}[]()\"' ")
    # collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def split_listy(s: str) -> List[str]:
    """
    Split list-like cells such as:
    - "{A, B, C}" or "A, B, C" or "A|B|C" etc.
    """
    s = normalize_text(s)
    if not s:
        return []
    if re.search(DELIMS, s):
        parts = [normalize_text(p) for p in re.split(DELIMS, s)]
    else:
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        parts = [normalize_text(p) for p in s.split(",")]
    return [p for p in parts if p]


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
    """Word-boundary substring test: 'april' must appear as a whole word in b."""
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
    Uses boundary match if present; otherwise falls back to simple containment.
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


# ----------------------
# Similarity primitives
# ----------------------
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


# ----------------------
# Similarity with guards
# ----------------------
def compute_similarity(a: str, b: str, tfidf_vec: TfidfVectorizer) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0

    # Flags & proportions
    is_exact = (a == b)
    is_boundary_sub = boundary_substring(a, b)
    recall, precision = overlap_proportions(a, b)
    long_text_b = is_long_text_value(b)
    short_text_a = is_short_text_value(a)

    # Fuzzy + TF-IDF
    fr, fp, lev = fuzzy_scores(a, b)
    tfidf_score = tfidf_cosine(a, b, tfidf_vec)

    # Base blended score
    base = 0.25 * fp + 0.25 * fr + 0.20 * lev + 0.30 * tfidf_score
    boost = 0.0

    # EXACT equality is strong, but still gated for long-text
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

    score = base + boost
    # keep within [0, 1), never return perfect 1.0 from gated boosts
    score = min(0.999, max(0.0, score))
    return score


def compare_elements(dash_val: str, source_val: str, tfidf_vec: TfidfVectorizer) -> float:
    """
    Handle multi-value cells by splitting both sides and taking the max pairwise score.
    """
    d_elems = split_listy(dash_val) or [normalize_text(dash_val)]
    s_elems = split_listy(source_val) or [normalize_text(source_val)]
    scores = [compute_similarity(d, s, tfidf_vec) for d in d_elems for s in s_elems]
    return max(scores) if scores else 0.0


# ----------------------
# Group-merge duplicates
# ----------------------
def merge_group(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    If an ID appears multiple times, concatenate values across rows per column.
    This preserves all data for comparison without breaking index uniqueness.
    """
    def join_values(x: pd.Series) -> str:
        vals = [str(v) for v in x if pd.notna(v) and str(v).strip()]
        return " | ".join(vals)
    return df.groupby(id_col, dropna=False).agg(join_values).reset_index()


# ----------------------
# Core mapping
# ----------------------
def map_columns(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    sample_rows: int = None,
    n_jobs: int = 1,
    progress: bool = True
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

    # Build TF-IDF over all values (once)
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(
        [normalize_text(v) for v in all_values if v]
    )

    dash_cols = list(dst.columns)
    src_cols  = list(src.columns)
    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]

    def compare_pair(dc: str, sc: str):
        hits = 0
        total_pairs = 0
        kept_scores = []  # scores that pass MIN_RATIO

        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            total_pairs += 1
            score = compare_elements(dv, sv, tfidf_vec)
            if score >= MIN_RATIO:
                hits += 1
                kept_scores.append(score)

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
            score_pct=score_pct
        )

    use_parallel = HAVE_JOBLIB and (n_jobs is not None) and (n_jobs != 1)
    iterator = pairs

    if use_parallel:
        mapped = Parallel(n_jobs=n_jobs)(
            delayed(compare_pair)(dc, sc) for dc, sc in iterator
        )
        results = [m for m in mapped if m]
    else:
        results = []
        it = tqdm(pairs, desc="Mapping columns (weighted)") if (progress and HAVE_TQDM) else pairs
        for dc, sc in it:
            r = compare_pair(dc, sc)
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rank by weighted_score, then avg_score, then hits, then hit_rate
    df = df.sort_values(
        ["weighted_score", "avg_score", "hits", "hit_rate"],
        ascending=False
    ).reset_index(drop=True)

    # Choose best source per dashboard column
    best_idx = df.groupby("dashboard_col")["weighted_score"].idxmax()
    best = df.loc[best_idx].copy()

    # Friendly suggestion label (uses both coverage and strength)
    def suggest(r):
        if r["hit_rate"] >= 0.9 and r["avg_score"] >= HIGH_RATIO:
            return "exact/very likely (strong coverage)"
        if r["hit_rate"] >= 0.75 and r["avg_score"] >= MIN_RATIO:
            return "likely (good coverage)"
        if r["hit_rate"] >= 0.5 and r["avg_score"] >= MIN_RATIO:
            return "possible (review)"
        return "uncertain"

    best["suggestion"] = best.apply(suggest, axis=1)
    best = best.sort_values(
        ["weighted_score", "avg_score", "hits", "hit_rate"],
        ascending=False
    ).reset_index(drop=True)

    return best, df


# ----------------------
# Audit
# ----------------------
def per_cell_audit(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    pairs: List[Tuple[str, str]],
    max_rows: int = 1000
) -> pd.DataFrame:
    # Merge dups
    source_df = merge_group(source_df, id_source)
    dash_df   = merge_group(dash_df, id_dash)

    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        return pd.DataFrame()

    # Fit TF-IDF once
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit(
        [normalize_text(v) for v in all_values if v]
    )

    rows = []
    for dc, sc in pairs:
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            score = compare_elements(dv, sv, tfidf_vec)
            rows.append(
                dict(
                    id=rid,
                    dashboard_col=dc,
                    source_col=sc,
                    dashboard_val=dv,
                    source_val=sv,
                    score=score
                )
            )
            if len(rows) >= max_rows:
                break

    return pd.DataFrame(rows)


# ----------------------
# CLI
# ----------------------
def load_dataframes(source_path: str, dashboard_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    read_src = pd.read_excel if source_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    read_dash = pd.read_excel if dashboard_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    s_df = read_src(source_path)
    d_df = read_dash(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df


def main():
    p = argparse.ArgumentParser(description="Hybrid Mapper v2 (Long-Text Guard, Parallel, Hit-Weighted)")
    p.add_argument("--source", required=True)
    p.add_argument("--dashboard", required=True)
    p.add_argument("--id-source", required=True)
    p.add_argument("--id-dash", required=True)
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--sample-rows", type=int, default=None)
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
        progress=True
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
        audit = per_cell_audit(s_df, d_df, args.id_source, args.id_dash, pairs)
        audit_path = out / "per_cell_audit.csv"
        audit.to_csv(audit_path, index=False)
        print(f"[✔] Wrote {audit_path}")


if __name__ == "__main__":
    main()
