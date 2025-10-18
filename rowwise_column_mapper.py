#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Row-wise Column Mapper
----------------------
Maps dashboard columns to source columns by comparing their row-wise values
(using shared IDs), computing multi-metric similarity (substring, list-item,
RapidFuzz, TF-IDF, etc.), and aggregating the scores per column pair.

Outputs:
- column_mapping.csv: best-matched source column for each dashboard column
- per_cell_audit.csv: optional detailed comparison log (use --audit)

Usage Example:
python rowwise_column_mapper.py \
  --source source.csv \
  --dashboard dashboard.csv \
  --id-source "Source ID Column" \
  --id-dash "Dashboard ID Column" \
  --out-dir ./out \
  --audit \
  --n-jobs 6
"""

import argparse
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# -----------------------------
# Text normalization utilities
# -----------------------------
DELIMS = r"[,\|;/]+"

def normalize_text(x: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip().strip("{}[]()\"' ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_listy(s: str) -> List[str]:
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

# -----------------------------
# Similarity metrics
# -----------------------------
def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize_text(a) == normalize_text(b) and normalize_text(a) != "" else 0.0

def substring_match(a: str, b: str) -> float:
    A, B = normalize_text(a), normalize_text(b)
    return 1.0 if (A and B and A in B) else 0.0

def list_item_hit(a: str, b: str) -> float:
    A = normalize_text(a)
    items = set(split_listy(b))
    return 1.0 if A and A in items else 0.0

def rf_partial(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib
        return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return fuzz.partial_ratio(normalize_text(a), normalize_text(b)) / 100.0

def rf_ratio(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib
        return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return fuzz.ratio(normalize_text(a), normalize_text(b)) / 100.0

def rf_levenshtein(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib
        return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return distance.Levenshtein.normalized_similarity(normalize_text(a), normalize_text(b))

# -----------------------------
# TF-IDF cache
# -----------------------------
class TFIDFCache:
    def __init__(self, values, analyzer='char', ngram_range=(3,5)):
        uniq = list({normalize_text(v) for v in values if str(v).strip().lower() != 'nan'})
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        if uniq:
            self.vectorizer.fit(uniq)
        self.cache = {}

    def transform(self, s: str):
        key = normalize_text(s)
        if key in self.cache:
            return self.cache[key]
        vec = self.vectorizer.transform([key])
        self.cache[key] = vec
        return vec

def tfidf_cosine(a: str, b: str, tfidf_cache: TFIDFCache) -> float:
    if not a or not b:
        return 0.0
    va, vb = tfidf_cache.transform(a), tfidf_cache.transform(b)
    sim = cosine_similarity(va, vb)[0][0]
    return float(sim) if not np.isnan(sim) else 0.0

# -----------------------------
# Combined similarity (weighted)
# -----------------------------
def combined_score(a: str, b: str, tfidf_cache: TFIDFCache) -> float:
    weights = {
        "exact": 0.3,
        "substring": 0.3,
        "list": 0.2,
        "rf_partial": 0.1,
        "rf_ratio": 0.05,
        "tfidf": 0.05
    }
    s_exact = exact_match(a, b)
    s_sub = substring_match(a, b)
    s_list = list_item_hit(a, b)
    s_rf = rf_partial(a, b)
    s_ratio = rf_ratio(a, b)
    s_tfidf = tfidf_cosine(a, b, tfidf_cache)
    score = (
        weights["exact"] * s_exact +
        weights["substring"] * s_sub +
        weights["list"] * s_list +
        weights["rf_partial"] * s_rf +
        weights["rf_ratio"] * s_ratio +
        weights["tfidf"] * s_tfidf
    )
    return max(0.0, min(1.0, score))

# -----------------------------
# Main mapping logic
# -----------------------------
def map_columns_rowwise(source_df, dash_df, id_source, id_dash, sample_rows=None, n_jobs=1, progress=True):
    # Align rows by ID values
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        raise ValueError("No matching IDs between source and dashboard!")
    src, dst = src.loc[common_ids], dst.loc[common_ids]
    if sample_rows:
        common_ids = common_ids[:sample_rows]

    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_cache = TFIDFCache(all_values)

    dash_cols = [c for c in dst.columns if c != id_dash]
    src_cols = [c for c in src.columns if c != id_source]

    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]
    results = []

    iterator = pairs
    if progress and HAVE_TQDM:
        iterator = tqdm(pairs, desc="Comparing column pairs row-wise")

    def compute_pair(dc, sc):
        scores = []
        for rid in common_ids:
            a = dst.at[rid, dc]
            b = src.at[rid, sc]
            if pd.isna(a) or pd.isna(b):
                continue
            s = combined_score(str(a), str(b), tfidf_cache)
            scores.append(s)
        avg = float(np.mean(scores)) if scores else 0.0
        return {"dashboard_col": dc, "source_col": sc, "avg_score": avg, "support": len(scores)}

    if HAVE_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_pair)(dc, sc) for dc, sc in iterator)
    else:
        results = [compute_pair(dc, sc) for dc, sc in iterator]

    df = pd.DataFrame(results)
    df["rank_in_dashboard"] = df.groupby("dashboard_col")["avg_score"].rank(ascending=False, method="first")
    best = df[df["rank_in_dashboard"] == 1].copy()
    best = best.sort_values(["avg_score", "support"], ascending=False).reset_index(drop=True)
    best["suggestion"] = best["avg_score"].apply(
        lambda x: "exact/substring" if x >= 0.95 else
                  "very likely" if x >= 0.85 else
                  "likely (review)" if x >= 0.7 else "uncertain"
    )
    return best

# -----------------------------
# Per-cell audit
# -----------------------------
def per_cell_audit(source_df, dash_df, id_source, id_dash, pairs, max_rows=1000):
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        return pd.DataFrame()
    all_vals = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_cache = TFIDFCache(all_vals)
    rows = []
    for dc, sc in pairs:
        for rid in common_ids:
            a, b = dst.at[rid, dc], src.at[rid, sc]
            if pd.isna(a) or pd.isna(b):
                continue
            score = combined_score(str(a), str(b), tfidf_cache)
            rows.append({
                "id": rid, "dashboard_col": dc, "source_col": sc,
                "dashboard_val": a, "source_val": b, "score": score
            })
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)

# -----------------------------
# CLI
# -----------------------------
def load_dataframes(source_path, dashboard_path):
    read_src = pd.read_excel if source_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    read_dash = pd.read_excel if dashboard_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    s_df = read_src(source_path)
    d_df = read_dash(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df

def main():
    parser = argparse.ArgumentParser(description="Row-wise Column Mapper using multi-metric similarity.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dashboard", required=True)
    parser.add_argument("--id-source", required=True)
    parser.add_argument("--id-dash", required=True)
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-rows", type=int, default=1000)
    args = parser.parse_args()

    s_df, d_df = load_dataframes(args.source, args.dashboard)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = map_columns_rowwise(s_df, d_df, args.id_source, args.id_dash,
                                  sample_rows=args.sample_rows, n_jobs=args.n_jobs, progress=True)
    map_path = out / "column_mapping.csv"
    mapping.to_csv(map_path, index=False)
    print(f"Wrote: {map_path}")

    if args.audit:
        pairs = list(zip(mapping["dashboard_col"], mapping["source_col"]))
        audit = per_cell_audit(s_df, d_df, args.id_source, args.id_dash, pairs, max_rows=args.audit_rows)
        audit_path = out / "per_cell_audit.csv"
        audit.to_csv(audit_path, index=False)
        print(f"Wrote: {audit_path}")

if __name__ == "__main__":
    main()
