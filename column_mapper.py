
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Column Mapper: containment-first, asymmetric similarity between dashboard and source columns.
Usage (CLI):
    python column_mapper.py --source source.xlsx --dashboard dashboard.xlsx --id id_c \
        --source-sheet Sheet1 --dash-sheet Sheet1 --out-dir ./out

You can also import the functions in a notebook:
    from column_mapper import map_columns, load_dataframes
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Try rapidfuzz if present; otherwise fall back to difflib
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib
    HAVE_RAPIDFUZZ = False


# -----------------------------
# Text normalization utilities
# -----------------------------

DELIMS = r"[,\|;/]+"  # common delimiters for "exploded" list-like cells


def normalize_text(x: str) -> str:
    if x is None:
        return ""
    s = str(x)
    # remove surrounding braces/brackets/quotes and normalize whitespace/case
    s = s.strip().strip("{}[]()\"' ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def split_listy(s: str) -> List[str]:
    """Split a cell that may contain multiple items like 'Account, Bank Operation, Closure'.
       Returns unique, normalized tokens/items, preserving multiword items.
    """
    s = normalize_text(s)
    if not s:
        return []
    parts = re.split(DELIMS, s)
    parts = [normalize_text(p) for p in parts if normalize_text(p)]
    # de-dup while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out or ([s] if s else [])


def word_tokens(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    # keep alphanumerics as words
    toks = re.findall(r"[a-z0-9]+", s)
    return toks


def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)] if len(s) >= n else [s]


# -----------------------------
# Asymmetric similarity metrics
# -----------------------------

def contains_substring(a: str, b: str) -> bool:
    """Return True if normalized a is a substring of normalized b (case-insensitive)."""
    A, B = normalize_text(a), normalize_text(b)
    return A in B if A and B else False


def containment_ratio(dash: str, src: str) -> float:
    """Asymmetric token containment: proportion of dashboard tokens found in source tokens."""
    dtoks = set(word_tokens(dash))
    stoks = set(word_tokens(src))
    if not dtoks:
        return 0.0
    return len(dtoks & stoks) / len(dtoks)


def list_item_hit(dash: str, src: str) -> float:
    """Asymmetric: returns 1.0 if the entire dashboard value is one of the items in the source list-like cell."""
    d = normalize_text(dash)
    items = set(split_listy(src))
    if not d or not items:
        return 0.0
    return 1.0 if d in items else 0.0


def char_ngram_containment(dash: str, src: str, n: int = 3) -> float:
    """Asymmetric: character n-gram containment of dash in src."""
    dg = set(char_ngrams(dash, n))
    sg = set(char_ngrams(src, n))
    if not dg:
        return 0.0
    return len(dg & sg) / len(dg)


def lcs_like(dash: str, src: str) -> float:
    """Approx LCS normalized by dashboard length. Uses difflib SequenceMatcher for fallback behavior."""
    d = normalize_text(dash)
    s = normalize_text(src)
    if not d or not s:
        return 0.0
    if HAVE_RAPIDFUZZ:
        # rapidfuzz partial_ratio is close to asymmetric containment of substrings
        return fuzz.partial_ratio(d, s) / 100.0
    else:
        matcher = difflib.SequenceMatcher(a=d, b=s)
        return matcher.ratio()  # symmetric; acceptable fallback


def exact_match(dash: str, src: str) -> float:
    return 1.0 if normalize_text(dash) == normalize_text(src) and dash != "" else 0.0


def substring_score(dash: str, src: str) -> float:
    d = normalize_text(dash)
    s = normalize_text(src)
    if not d or not s:
        return 0.0
    return 1.0 if d in s else 0.0


def combined_asymmetric_score(dash: str, src: str) -> float:
    """Weighted combination prioritizing exact/substring and list-item containment.
       Tuned for cases where dashboard value should be found *inside* the source value.
    """
    w_exact = 0.35
    w_sub   = 0.30
    w_list  = 0.20
    w_tok   = 0.10
    w_char  = 0.05

    s_exact = exact_match(dash, src)
    s_sub   = substring_score(dash, src)
    s_list  = list_item_hit(dash, src)
    s_tok   = containment_ratio(dash, src)
    s_char  = char_ngram_containment(dash, src, n=3)

    score = (w_exact * s_exact +
             w_sub   * s_sub   +
             w_list  * s_list  +
             w_tok   * s_tok   +
             w_char  * s_char)
    # Cap to [0,1]
    return max(0.0, min(1.0, score))


# -----------------------------
# Core mapping logic
# -----------------------------

def align_on_id(source_df: pd.DataFrame, dash_df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ensure id column exists
    if id_col not in source_df.columns or id_col not in dash_df.columns:
        raise ValueError(f"'{id_col}' must exist in both dataframes.")
    # Index on id for fast lookup
    s = source_df.set_index(id_col, drop=False)
    d = dash_df.set_index(id_col, drop=False)
    # Intersect IDs
    common_ids = s.index.intersection(d.index)
    if len(common_ids) == 0:
        raise ValueError(f"No overlapping '{id_col}' values between source and dashboard.")
    s2 = s.loc[common_ids].copy()
    d2 = d.loc[common_ids].copy()
    return s2, d2


def score_column_pair(source_col: str, dash_col: str, s_df: pd.DataFrame, d_df: pd.DataFrame, sample_rows: Optional[int] = None) -> Dict[str, float]:
    """Compute aggregate asymmetric scores for a candidate (dash_col -> source_col) mapping."""
    ids = d_df.index
    if sample_rows is not None and sample_rows < len(ids):
        ids = ids[:sample_rows]

    scores = []
    non_null_pairs = 0
    for _id in ids:
        dv = d_df.at[_id, dash_col] if dash_col in d_df.columns else None
        sv = s_df.at[_id, source_col] if source_col in s_df.columns else None
        if pd.isna(dv) or pd.isna(sv):
            continue
        non_null_pairs += 1
        scores.append(combined_asymmetric_score(str(dv), str(sv)))

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    coverage = non_null_pairs / max(1, len(ids))
    return {
        "avg_score": avg_score,
        "coverage": coverage,
        "support": non_null_pairs
    }


def map_columns(source_df: pd.DataFrame,
                dash_df: pd.DataFrame,
                id_col: str = "id_c",
                min_support: int = 10,
                sample_rows: Optional[int] = None,
                one_to_one: bool = True) -> pd.DataFrame:
    """Compute a mapping table dash_col -> best source_col with diagnostics."""
    s_df, d_df = align_on_id(source_df, dash_df, id_col)

    dash_cols = [c for c in d_df.columns if c != id_col]
    source_cols = [c for c in s_df.columns if c != id_col]

    records = []
    for dc in dash_cols:
        for sc in source_cols:
            stats = score_column_pair(sc, dc, s_df, d_df, sample_rows=sample_rows)
            rec = {
                "dashboard_col": dc,
                "source_col": sc,
                **stats
            }
            records.append(rec)

    result = pd.DataFrame.from_records(records)

    # Filter for enough evidence
    result = result[result["support"] >= min_support].copy()

    # Choose best source col per dashboard col
    result["rank_in_dashboard"] = result.groupby("dashboard_col")["avg_score"].rank(ascending=False, method="first")
    best = result[result["rank_in_dashboard"] == 1].copy()

    if one_to_one:
        # Enforce one-to-one via greedy selection by score
        best = best.sort_values("avg_score", ascending=False)
        taken_src = set()
        final_rows = []
        for _, row in best.iterrows():
            if row["source_col"] in taken_src:
                continue
            final_rows.append(row)
            taken_src.add(row["source_col"])
        best = pd.DataFrame(final_rows)

    # Add a suggested label based on thresholds
    def label_row(r):
        if r["avg_score"] >= 0.95:
            return "exact/substring match"
        if r["avg_score"] >= 0.85:
            return "very likely match"
        if r["avg_score"] >= 0.75:
            return "likely match (review)"
        return "uncertain"

    best["suggestion"] = best.apply(label_row, axis=1)
    best = best.sort_values(["avg_score", "coverage", "support"], ascending=False).reset_index(drop=True)
    return best


# -----------------------------
# Optional: per-cell audit
# -----------------------------

def per_cell_audit(source_df: pd.DataFrame,
                   dash_df: pd.DataFrame,
                   id_col: str,
                   pairs: List[Tuple[str, str]],
                   max_rows: int = 1000) -> pd.DataFrame:
    """Return a row-wise audit for selected (dash_col, source_col) pairs."""
    s_df, d_df = align_on_id(source_df, dash_df, id_col)
    rows = []
    count = 0
    for dc, sc in pairs:
        for _id in d_df.index:
            if count >= max_rows:
                break
            dv = d_df.at[_id, dc] if dc in d_df.columns else None
            sv = s_df.at[_id, sc] if sc in s_df.columns else None
            if pd.isna(dv) or pd.isna(sv):
                continue
            score = combined_asymmetric_score(str(dv), str(sv))
            rows.append({
                "id_c": _id,
                "dashboard_col": dc,
                "source_col": sc,
                "dashboard_val": dv,
                "source_val": sv,
                "score": score,
                "substring_hit": substring_score(str(dv), str(sv)),
                "token_containment": containment_ratio(str(dv), str(sv)),
                "list_item_hit": list_item_hit(str(dv), str(sv)),
            })
            count += 1
    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------

def load_dataframes(source_path: str, dashboard_path: str, source_sheet: Optional[str], dash_sheet: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s_df = pd.read_excel(source_path, sheet_name=source_sheet) if source_path.lower().endswith(".xlsx") else pd.read_csv(source_path)
    d_df = pd.read_excel(dashboard_path, sheet_name=dash_sheet) if dashboard_path.lower().endswith(".xlsx") else pd.read_csv(dashboard_path)
    return s_df, d_df


def main():
    parser = argparse.ArgumentParser(description="Map dashboard columns to source columns using asymmetric containment-first similarity.")
    parser.add_argument("--source", required=True, help="Path to source.xlsx/.csv")
    parser.add_argument("--dashboard", required=True, help="Path to dashboard.xlsx/.csv")
    parser.add_argument("--id", default="id_c", help="Primary key column name present in both files (default: id_c)")
    parser.add_argument("--source-sheet", default=None, help="Excel sheet name for source (optional)")
    parser.add_argument("--dash-sheet", default=None, help="Excel sheet name for dashboard (optional)")
    parser.add_argument("--min-support", type=int, default=10, help="Minimum overlapping non-null pairs per mapping (default: 10)")
    parser.add_argument("--sample-rows", type=int, default=None, help="Sample first N aligned rows for faster scoring (optional)")
    parser.add_argument("--one-to-one", action="store_true", help="Enforce one-to-one column mapping")
    parser.add_argument("--out-dir", default="./out", help="Output directory (default: ./out)")
    parser.add_argument("--audit", action="store_true", help="Write per-cell audit CSV for top mappings")
    parser.add_argument("--audit-rows", type=int, default=500, help="Max rows in the audit CSV (default: 500)")

    args = parser.parse_args()

    s_df, d_df = load_dataframes(args.source, args.dashboard, args.source_sheet, args.dash_sheet)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = map_columns(
        source_df=s_df,
        dash_df=d_df,
        id_col=args.id,
        min_support=args.min_support,
        sample_rows=args.sample_rows,
        one_to_one=args.one_to_one
    )

    map_path = out_dir / "column_mapping.csv"
    mapping_df.to_csv(map_path, index=False)

    print(f"Wrote mapping table: {map_path}")

    if args.audit and len(mapping_df):
        # Build pairs for audit using top K results
        pairs = list(zip(mapping_df["dashboard_col"].tolist(), mapping_df["source_col"].tolist()))
        audit_df = per_cell_audit(
            source_df=s_df,
            dash_df=d_df,
            id_col=args.id,
            pairs=pairs,
            max_rows=args.audit_rows
        )
        audit_path = out_dir / "per_cell_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        print(f"Wrote per-cell audit: {audit_path}")


if __name__ == "__main__":
    main()
