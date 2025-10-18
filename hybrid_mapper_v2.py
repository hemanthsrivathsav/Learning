#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Mapper v2 — Row-wise, Multi-metric, Domain-tuned Column Mapper
----------------------------------------------------------------------
This script combines:
- Your original row-by-row comparison logic
- Multiple similarity metrics (substring, fuzzy, TF-IDF, Levenshtein)
- Multi-value cell parsing
- Aggregation of similarity scores across rows
- Configurable thresholds (HIGH_SCORE, HIGH_RATIO, MIN_RATIO)

Usage:
python hybrid_mapper_v2.py \
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
from typing import List, Dict, Tuple

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


# -----------------------------------------------------------
# Configurable thresholds
# -----------------------------------------------------------
HIGH_SCORE = 0.95
HIGH_RATIO = 0.85
MIN_RATIO = 0.7

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------
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

def tfidf_cosine(a: str, b: str, tfidf_vec) -> float:
    if not a or not b:
        return 0.0
    va = tfidf_vec.transform([normalize_text(a)])
    vb = tfidf_vec.transform([normalize_text(b)])
    return float(cosine_similarity(va, vb)[0][0])

def compute_similarity(a: str, b: str, tfidf_vec) -> float:
    """
    Compute similarity between two text values using multiple methods,
    prioritizing substring and element containment.
    """
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0

    # Basic exact / substring
    if a == b:
        return 1.0
    if a in b:
        return 0.95

    # Fuzzy / partial ratios
    if HAVE_RAPIDFUZZ:
        fr = fuzz.ratio(a, b) / 100
        fp = fuzz.partial_ratio(a, b) / 100
        lev = distance.Levenshtein.normalized_similarity(a, b)
    else:
        import difflib
        fr = difflib.SequenceMatcher(a=a, b=b).ratio()
        fp = fr
        lev = fr

    tfidf_score = tfidf_cosine(a, b, tfidf_vec)

    # Weighted score
    final_score = (
        0.25 * fp +   # partial match
        0.25 * fr +   # overall fuzzy
        0.2  * lev +  # levenshtein
        0.3  * tfidf_score  # tf-idf
    )

    return min(1.0, final_score)

def compare_elements(dash_val, source_val, tfidf_vec) -> float:
    """
    Handle multi-value cells: compare each element and take the max score.
    """
    dash_elems = split_listy(dash_val) or [normalize_text(dash_val)]
    src_elems = split_listy(source_val) or [normalize_text(source_val)]
    scores = []
    for d in dash_elems:
        for s in src_elems:
            scores.append(compute_similarity(d, s, tfidf_vec))
    return max(scores) if scores else 0.0

# -----------------------------------------------------------
# Core Mapping Logic
# -----------------------------------------------------------
def map_columns(source_df, dash_df, id_source, id_dash, sample_rows=None, n_jobs=1, progress=True):
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        raise ValueError("No matching IDs found between source and dashboard!")
    src, dst = src.loc[common_ids], dst.loc[common_ids]
    if sample_rows:
        common_ids = common_ids[:sample_rows]

    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5)).fit([normalize_text(v) for v in all_values if v])

    dash_cols = [c for c in dst.columns if c != id_dash]
    src_cols = [c for c in src.columns if c != id_source]

    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]
    results = []

    iterator = pairs
    if progress and HAVE_TQDM:
        iterator = tqdm(pairs, desc="Computing column mappings (row-wise)")

    def compare_pair(dc, sc):
        scores = []
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            score = compare_elements(dv, sv, tfidf_vec)
            if score >= MIN_RATIO:
                scores.append(score)
        if not scores:
            return None
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        return {
            "dashboard_col": dc,
            "source_col": sc,
            "avg_score": avg_score,
            "max_score": max_score,
            "support": len(scores)
        }

    if HAVE_JOBLIB and n_jobs != 1:
        mapped = Parallel(n_jobs=n_jobs)(
            delayed(compare_pair)(dc, sc) for dc, sc in iterator
        )
        results = [r for r in mapped if r]
    else:
        for dc, sc in iterator:
            r = compare_pair(dc, sc)
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Pick best match per dashboard column
    df["rank"] = df.groupby("dashboard_col")["avg_score"].rank(ascending=False, method="first")
    best = df[df["rank"] == 1].copy()

    def suggest_label(r):
        if r["max_score"] >= HIGH_SCORE:
            return "exact match"
        elif r["avg_score"] >= HIGH_RATIO:
            return "very likely match"
        elif r["avg_score"] >= MIN_RATIO:
            return "likely match"
        return "uncertain"

    best["suggestion"] = best.apply(suggest_label, axis=1)
    best = best.sort_values(["avg_score", "support"], ascending=False).reset_index(drop=True)
    return best

# -----------------------------------------------------------
# Audit Function
# -----------------------------------------------------------
def per_cell_audit(source_df, dash_df, id_source, id_dash, pairs, max_rows=1000):
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        return pd.DataFrame()

    all_vals = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5)).fit([normalize_text(v) for v in all_vals if v])

    rows = []
    for dc, sc in pairs:
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            score = compare_elements(dv, sv, tfidf_vec)
            rows.append({
                "id": rid,
                "dashboard_col": dc,
                "source_col": sc,
                "dashboard_val": dv,
                "source_val": sv,
                "score": score
            })
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def load_dataframes(source_path, dashboard_path):
    read_src = pd.read_excel if source_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    read_dash = pd.read_excel if dashboard_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    s_df = read_src(source_path)
    d_df = read_dash(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df

def main():
    parser = argparse.ArgumentParser(description="Hybrid Mapper v2 — row-wise, domain-tuned column matcher.")
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

    mapping = map_columns(s_df, d_df, args.id_source, args.id_dash,
                          sample_rows=args.sample_rows, n_jobs=args.n_jobs, progress=True)
    map_path = out / "column_mapping.csv"
    mapping.to_csv(map_path, index=False)
    print(f"[✔] Wrote column mapping: {map_path}")

    if args.audit:
        pairs = list(zip(mapping["dashboard_col"], mapping["source_col"]))
        audit = per_cell_audit(s_df, d_df, args.id_source, args.id_dash, pairs, max_rows=args.audit_rows)
        audit_path = out / "per_cell_audit.csv"
        audit.to_csv(audit_path, index=False)
        print(f"[✔] Wrote per-cell audit: {audit_path}")

if __name__ == "__main__":
    main()
