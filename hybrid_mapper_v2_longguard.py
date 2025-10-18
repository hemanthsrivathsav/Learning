#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Hybrid Mapper v2 (Long-Text Guarded)
# ------------------------------------
# This version prevents spurious matches where short values like "April"
# appear inside unrelated long comment fields.
# Includes length-aware, boundary-aware substring logic and TF-IDF weighting.

import argparse, re, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from tqdm import tqdm
from rapidfuzz import fuzz, distance

HIGH_SCORE = 0.95
HIGH_RATIO = 0.85
MIN_RATIO = 0.70

LONG_TEXT_CHAR_THRESHOLD = 80
MIN_PRECISION_FOR_LONG_TEXT = 0.30
MIN_TOKEN_CONTAINMENT_FOR_LONG = 0.70
MIN_TFIDF_FOR_LONG_TEXT = 0.30
MONTH_WORDS = set([
    "january","february","march","april","may","june","july",
    "august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec"
])

DELIMS = r"[,\|;/]+"

def normalize_text(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip().strip("{}[]()"' ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def split_listy(s):
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

def is_long_text_value(s):
    s = normalize_text(s)
    if not s: return False
    return len(s) >= LONG_TEXT_CHAR_THRESHOLD or len(s.split()) >= 15

def boundary_substring(a, b):
    A = re.escape(normalize_text(a))
    B = normalize_text(b)
    if not A or not B: return False
    return re.search(rf"\b{A}\b", B) is not None

def overlap_proportions(a, b):
    A = normalize_text(a)
    B = normalize_text(b)
    if not A or not B: return (0.0, 0.0)
    if A in B:
        return (1.0, len(A)/len(B))
    m = re.search(rf"\b{re.escape(A)}\b", B)
    if m:
        matched = m.end() - m.start()
        return (matched/len(A), matched/len(B))
    return (0.0, 0.0)

def tfidf_cosine(a, b, tfidf_vec):
    if not a or not b:
        return 0.0
    va = tfidf_vec.transform([normalize_text(a)])
    vb = tfidf_vec.transform([normalize_text(b)])
    sim = cosine_similarity(va, vb)[0][0]
    return float(sim) if not np.isnan(sim) else 0.0

def fuzzy_scores(a, b):
    fr = fuzz.ratio(a, b) / 100
    fp = fuzz.partial_ratio(a, b) / 100
    lev = distance.Levenshtein.normalized_similarity(a, b)
    return fr, fp, lev

def compute_similarity(a, b, tfidf_vec):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    is_boundary_sub = boundary_substring(a, b)
    recall, precision = overlap_proportions(a, b)
    long_text_b = is_long_text_value(b)
    fr, fp, lev = fuzzy_scores(a, b)
    tfidf_score = tfidf_cosine(a, b, tfidf_vec)
    base = 0.25*fp + 0.25*fr + 0.20*lev + 0.30*tfidf_score
    substring_boost = 0.0
    if is_boundary_sub:
        if not long_text_b:
            substring_boost = 0.25
        else:
            if (precision >= MIN_PRECISION_FOR_LONG_TEXT or
                tfidf_score >= MIN_TFIDF_FOR_LONG_TEXT or
                fp >= MIN_TOKEN_CONTAINMENT_FOR_LONG):
                substring_boost = 0.15
            else:
                substring_boost = -0.10
    if long_text_b and a in MONTH_WORDS and is_boundary_sub and tfidf_score < 0.25 and fp < 0.8:
        substring_boost = -0.20
    score = base + substring_boost
    return max(0.0, min(1.0, score))

def compare_elements(dash_val, source_val, tfidf_vec):
    d_elems = split_listy(dash_val) or [normalize_text(dash_val)]
    s_elems = split_listy(source_val) or [normalize_text(source_val)]
    scores = [compute_similarity(d, s, tfidf_vec) for d in d_elems for s in s_elems]
    return max(scores) if scores else 0.0

def merge_group(df, id_col):
    def join_values(x):
        vals = [str(v) for v in x if pd.notna(v) and str(v).strip()]
        return " | ".join(vals)
    return df.groupby(id_col, dropna=False).agg(join_values).reset_index()

def map_columns(source_df, dash_df, id_source, id_dash, sample_rows=None, n_jobs=1):
    source_df = merge_group(source_df, id_source)
    dash_df = merge_group(dash_df, id_dash)
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        raise ValueError("No matching IDs found between source and dashboard!")
    if sample_rows:
        common_ids = common_ids[:sample_rows]
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5)).fit([normalize_text(v) for v in all_values if v])
    dash_cols = list(dst.columns)
    src_cols = list(src.columns)
    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]
    results = []
    for dc, sc in tqdm(pairs, desc="Mapping columns"):
        scores = []
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv): continue
            score = compare_elements(dv, sv, tfidf_vec)
            if score >= MIN_RATIO:
                scores.append(score)
        if not scores: continue
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        results.append(dict(dashboard_col=dc, source_col=sc, avg_score=avg_score, max_score=max_score, support=len(scores)))
    df = pd.DataFrame(results)
    if df.empty: return df
    df["rank"] = df.groupby("dashboard_col")["avg_score"].rank(ascending=False, method="first")
    best = df[df["rank"]==1].copy()
    def suggest(r):
        if r["max_score"]>=HIGH_SCORE: return "exact match"
        if r["avg_score"]>=HIGH_RATIO: return "very likely match"
        if r["avg_score"]>=MIN_RATIO:  return "likely match"
        return "uncertain"
    best["suggestion"]=best.apply(suggest,axis=1)
    return best.sort_values(["avg_score","support"],ascending=False).reset_index(drop=True)

def per_cell_audit(source_df,dash_df,id_source,id_dash,pairs,max_rows=1000):
    source_df = merge_group(source_df, id_source)
    dash_df = merge_group(dash_df, id_dash)
    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids): return pd.DataFrame()
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5)).fit([normalize_text(v) for v in all_values if v])
    rows=[]
    for dc,sc in pairs:
        for rid in common_ids:
            dv=dst.at[rid,dc]; sv=src.at[rid,sc]
            if pd.isna(dv) or pd.isna(sv): continue
            score=compare_elements(dv,sv,tfidf_vec)
            rows.append(dict(id=rid,dashboard_col=dc,source_col=sc,dashboard_val=dv,source_val=sv,score=score))
            if len(rows)>=max_rows: break
    return pd.DataFrame(rows)

def load_dataframes(source_path, dashboard_path):
    read_src = pd.read_excel if source_path.lower().endswith(('.xlsx','.xls')) else pd.read_csv
    read_dash= pd.read_excel if dashboard_path.lower().endswith(('.xlsx','.xls')) else pd.read_csv
    s_df=read_src(source_path); d_df=read_dash(dashboard_path)
    s_df.columns=[str(c).strip() for c in s_df.columns]
    d_df.columns=[str(c).strip() for c in d_df.columns]
    return s_df,d_df

def main():
    p=argparse.ArgumentParser(description='Hybrid Mapper v2 (Long-Text Guarded)')
    p.add_argument('--source',required=True); p.add_argument('--dashboard',required=True)
    p.add_argument('--id-source',required=True); p.add_argument('--id-dash',required=True)
    p.add_argument('--out-dir',default='./out'); p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    s_df,d_df=load_dataframes(args.source,args.dashboard)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    mapping=map_columns(s_df,d_df,args.id_source,args.id_dash)
    map_path=out/'column_mapping.csv'; mapping.to_csv(map_path,index=False)
    print(f'[✔] Wrote {map_path}')
    if args.audit and not mapping.empty:
        pairs=list(zip(mapping.dashboard_col,mapping.source_col))
        audit=per_cell_audit(s_df,d_df,args.id_source,args.id_dash,pairs)
        a_path=out/'per_cell_audit.csv'; audit.to_csv(a_path,index=False)
        print(f'[✔] Wrote {a_path}')

if __name__=='__main__': main()
