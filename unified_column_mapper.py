
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Column Mapper (Containment + Fuzzy + TF-IDF) — with Audit & Parallelization
See header in previous cell for full description.
"""
import argparse, re, sys, math
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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

DELIMS = r"[,\|;/]+"

def normalize_text(x: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x)
    s = s.strip().strip("{}[]()\"' ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_listy(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    if re.search(DELIMS, s):
        parts = [normalize_text(p) for p in re.split(DELIMS, s)]
    else:
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1]; parts = [normalize_text(p) for p in re.split(DELIMS, inner)]
        elif s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]; parts = [normalize_text(p.strip().strip("'\"")) for p in re.split(DELIMS, inner)]
        elif s.startswith("{") and s.endswith("}"):
            inner = s[1:-1]; parts = [normalize_text(p) for p in re.split(DELIMS, inner)]
        else:
            parts = [s]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def word_tokens(s: str) -> List[str]:
    s = normalize_text(s)
    if not s: return []
    return re.findall(r"[a-z0-9]+", s)

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = normalize_text(s)
    if not s: return []
    return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else [s]

def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize_text(a) == normalize_text(b) and normalize_text(a) != "" else 0.0

def substring_match(a: str, b: str) -> float:
    A, B = normalize_text(a), normalize_text(b)
    return 1.0 if (A and B and A in B) else 0.0

def list_item_hit(a: str, b: str) -> float:
    A = normalize_text(a)
    items = set(split_listy(b))
    return 1.0 if A and A in items else 0.0

def token_containment(a: str, b: str) -> float:
    ta, tb = set(word_tokens(a)), set(word_tokens(b))
    if not ta: return 0.0
    return len(ta & tb)/len(ta)

def char_ngram_containment(a: str, b: str, n: int = 3) -> float:
    ga, gb = set(char_ngrams(a, n)), set(char_ngrams(b, n))
    if not ga: return 0.0
    return len(ga & gb)/len(ga)

def rf_partial_ratio(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib; return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return fuzz.partial_ratio(normalize_text(a), normalize_text(b))/100.0

def rf_ratio(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib; return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return fuzz.ratio(normalize_text(a), normalize_text(b))/100.0

def rf_levenshtein(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        import difflib; return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()
    return distance.Levenshtein.normalized_similarity(normalize_text(a), normalize_text(b))

class TFIDFCache:
    def __init__(self, values, analyzer='char', ngram_range=(3,5)):
        uniq = list({normalize_text(v) for v in values if str(v).strip().lower() != 'nan'})
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        if uniq:
            self.vectorizer.fit(uniq)
        self.cache = {}
    def transform(self, s: str):
        key = normalize_text(s)
        if key in self.cache: return self.cache[key]
        vec = self.vectorizer.transform([key]); self.cache[key] = vec; return vec

def tfidf_cosine(a: str, b: str, tfidf: TFIDFCache) -> float:
    if not a or not b: return 0.0
    va, vb = tfidf.transform(a), tfidf.transform(b)
    sim = cosine_similarity(va, vb)[0][0]
    return float(sim) if not np.isnan(sim) else 0.0

def combined_score(a: str, b: str, tfidf_cache: TFIDFCache=None) -> dict:
    metrics = {
        "exact": exact_match(a,b),
        "substring": substring_match(a,b),
        "list_item": list_item_hit(a,b),
        "token_containment": token_containment(a,b),
        "char3_containment": char_ngram_containment(a,b,3),
        "rf_partial": rf_partial_ratio(a,b),
        "rf_ratio": rf_ratio(a,b),
        "levenshtein": rf_levenshtein(a,b),
        "tfidf": tfidf_cosine(a,b, tfidf_cache) if tfidf_cache is not None else 0.0
    }
    w = {"exact":0.25,"substring":0.25,"list_item":0.20,"token_containment":0.10,
         "char3_containment":0.05,"rf_partial":0.10,"rf_ratio":0.03,"levenshtein":0.01,"tfidf":0.01}
    final = sum(metrics[k]*w[k] for k in w)
    metrics["final"] = max(0.0, min(1.0, final))
    return metrics

def align_on_id(source_df, dash_df, id_col):
    if id_col not in source_df.columns or id_col not in dash_df.columns:
        raise ValueError(f"'{id_col}' must exist in both dataframes.")
    s = source_df.set_index(id_col, drop=False)
    d = dash_df.set_index(id_col, drop=False)
    common = s.index.intersection(d.index)
    if len(common) == 0: raise ValueError(f"No overlapping '{id_col}' between inputs.")
    return s.loc[common].copy(), d.loc[common].copy()

def build_tfidf_cache(source_df, dash_df):
    all_vals = []
    for df in (source_df, dash_df):
        vals = df.replace(np.nan, "", regex=False).values.flatten().tolist()
        all_vals.extend(vals)
    return TFIDFCache(all_vals, analyzer='char', ngram_range=(3,5))

def score_column_pair(source_col, dash_col, s_df, d_df, tfidf_cache, max_rows=None):
    ids = d_df.index
    if max_rows is not None and max_rows < len(ids):
        ids = ids[:max_rows]
    from collections import defaultdict
    metrics_accum = defaultdict(list); support = 0
    for _id in ids:
        a = d_df.at[_id, dash_col] if dash_col in d_df.columns else None
        b = s_df.at[_id, source_col] if source_col in s_df.columns else None
        if pd.isna(a) or pd.isna(b): continue
        a, b = str(a), str(b)
        m = combined_score(a,b, tfidf_cache)
        for k,v in m.items(): metrics_accum[k].append(v)
        support += 1
    if support == 0:
        avg_metrics = {k:0.0 for k in ["final","exact","substring","list_item","token_containment","char3_containment","rf_partial","rf_ratio","levenshtein","tfidf"]}
    else:
        avg_metrics = {k: float(np.mean(v)) for k,v in metrics_accum.items()}
    avg_metrics.update({"dashboard_col": dash_col, "source_col": source_col, "support": support, "coverage": support/max(1,len(d_df.index))})
    return avg_metrics

def map_columns(source_df, dash_df, id_col="id_c", min_support=10, sample_rows=None, one_to_one=True, n_jobs=1, progress=True):
    s_df, d_df = align_on_id(source_df, dash_df, id_col)
    tfidf_cache = build_tfidf_cache(s_df, d_df)
    dash_cols = [c for c in d_df.columns if c != id_col]
    source_cols = [c for c in s_df.columns if c != id_col]
    tasks = [(sc, dc) for dc in dash_cols for sc in source_cols]

    results = []
    iterator = tasks
    if progress and HAVE_TQDM:
        iterator = tqdm(tasks, desc="Scoring column pairs")

    if HAVE_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(score_column_pair)(sc, dc, s_df, d_df, tfidf_cache, sample_rows) for (sc,dc) in iterator)
    else:
        for sc, dc in iterator:
            results.append(score_column_pair(sc, dc, s_df, d_df, tfidf_cache, sample_rows))

    df = pd.DataFrame(results)
    if df.empty: return df
    df = df[df["support"] >= min_support].copy()
    if df.empty: return df
    df["rank_in_dashboard"] = df.groupby("dashboard_col")["final"].rank(ascending=False, method="first")
    best = df[df["rank_in_dashboard"] == 1].copy()
    if one_to_one:
        best = best.sort_values("final", ascending=False)
        used_src=set(); rows=[]
        for _, r in best.iterrows():
            if r["source_col"] in used_src: continue
            rows.append(r); used_src.add(r["source_col"])
        best = pd.DataFrame(rows)
    def label_row(r):
        if r["final"] >= 0.95: return "exact/substring match"
        if r["final"] >= 0.85: return "very likely match"
        if r["final"] >= 0.75: return "likely match (review)"
        return "uncertain"
    best["suggestion"] = best.apply(label_row, axis=1)
    best = best.sort_values(["final","coverage","support"], ascending=False).reset_index(drop=True)
    return best

def per_cell_audit(source_df, dash_df, id_col, pairs, max_rows=1000):
    s_df, d_df = align_on_id(source_df, dash_df, id_col)
    tfidf_cache = build_tfidf_cache(s_df, d_df)
    rows = []; count=0
    for dc, sc in pairs:
        for _id in d_df.index:
            if count >= max_rows: break
            a = d_df.at[_id, dc] if dc in d_df.columns else None
            b = s_df.at[_id, sc] if sc in s_df.columns else None
            if pd.isna(a) or pd.isna(b): continue
            a, b = str(a), str(b)
            m = combined_score(a,b, tfidf_cache)
            m.update({"id_c": _id, "dashboard_col": dc, "source_col": sc, "dashboard_val": a, "source_val": b})
            rows.append(m); count += 1
    return pd.DataFrame(rows)

def load_dataframes(source_path, dashboard_path, source_sheet=None, dash_sheet=None):
    read_src = pd.read_excel if source_path.lower().endswith((".xlsx",".xls")) else pd.read_csv
    read_dst = pd.read_excel if dashboard_path.lower().endswith((".xlsx",".xls")) else pd.read_csv
    s_df = read_src(source_path, sheet_name=source_sheet) if source_path.lower().endswith((".xlsx",".xls")) else read_src(source_path)
    d_df = read_dst(dashboard_path, sheet_name=dash_sheet) if dashboard_path.lower().endswith((".xlsx",".xls")) else read_dst(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Column Mapper (containment + fuzzy + tfidf).")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dashboard", required=True)
    parser.add_argument("--id", default="id_c")
    parser.add_argument("--source-sheet", default=None)
    parser.add_argument("--dash-sheet", default=None)
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--one-to-one", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-rows", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--preview-metrics", action="store_true")
    args = parser.parse_args()

    s_df, d_df = load_dataframes(args.source, args.dashboard, args.source_sheet, args.dash_sheet)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    mapping = map_columns(s_df, d_df, id_col=args.id, min_support=args.min_support, sample_rows=args.sample_rows, one_to_one=args.one_to_one, n_jobs=args.n_jobs, progress=True)
    map_path = out / "column_mapping.csv"; mapping.to_csv(map_path, index=False); print(f"Wrote: {map_path}")

    if args.audit and not mapping.empty:
        pairs = list(zip(mapping["dashboard_col"].tolist(), mapping["source_col"].tolist()))
        audit = per_cell_audit(s_df, d_df, id_col=args.id, pairs=pairs, max_rows=args.audit_rows)
        audit_path = out / "per_cell_audit.csv"; audit.to_csv(audit_path, index=False); print(f"Wrote: {audit_path}")

    if args.preview_metrics and not mapping.empty:
        dc, sc = mapping.iloc[0]["dashboard_col"], mapping.iloc[0]["source_col"]
        sample = per_cell_audit(s_df, d_df, id_col=args.id, pairs=[(dc, sc)], max_rows=200)
        prev_path = out / "metrics_preview.csv"; sample.to_csv(prev_path, index=False); print(f"Wrote: {prev_path}")

if __name__ == "__main__":
    main()
