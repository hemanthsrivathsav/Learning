from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

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

from .config import (
    MIN_RATIO, HIGH_RATIO,
)
from .scoring import (
    normalize_text, make_tfidf_vectorizer, compare_elements,
    split_listy
)

# --------- helpers ----------
def merge_group(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    def join_values(x: pd.Series) -> str:
        vals = [str(v) for v in x if pd.notna(v) and str(v).strip()]
        return " | ".join(vals)
    return df.groupby(id_col, dropna=False).agg(join_values).reset_index()

def load_dataframes(source_path: str, dashboard_path: str):
    read_src  = pd.read_excel if source_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    read_dash = pd.read_excel if dashboard_path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    s_df = read_src(source_path)
    d_df = read_dash(dashboard_path)
    s_df.columns = [str(c).strip() for c in s_df.columns]
    d_df.columns = [str(c).strip() for c in d_df.columns]
    return s_df, d_df

# --------- main mapping ----------
def map_columns(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    sample_rows: Optional[int] = None,
    n_jobs: int = 1,
    progress: bool = True,
    tag_thresholds: Optional[dict] = None,
):
    # prepare
    source_df = merge_group(source_df, id_source)
    dash_df   = merge_group(dash_df, id_dash)

    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        raise ValueError("No matching IDs found between source and dashboard!")

    if sample_rows:
        common_ids = common_ids[:sample_rows]

    # TF-IDF once
    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    tfidf_vec = make_tfidf_vectorizer().fit([normalize_text(v) for v in all_values if v])


    dash_cols = list(dst.columns)
    src_cols  = list(src.columns)
    pairs = [(dc, sc) for dc in dash_cols for sc in src_cols]
    if not pairs:
        return pd.DataFrame(), pd.DataFrame()

    thresholds = tag_thresholds or {}
    def threshold_for(tag: str, default: float = MIN_RATIO) -> float:
        return thresholds.get(tag, default)

    def compare_pair(dc: str, sc: str):
        hits = 0
        total_pairs = 0
        kept_scores = []
        direct_hits = list_hits = listlike_hits = similarity_hits = 0


        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            total_pairs += 1

            score, tag = compare_elements(dv, sv, tfidf_vec)
            # per-tag threshold
            tag_thresh = threshold_for(tag)
            if score >= tag_thresh:
                hits += 1
                kept_scores.append(score)
                if tag == "direct":     direct_hits += 1
                elif tag == "list":     list_hits += 1
                elif tag == "listlike": listlike_hits += 1
                elif tag == "similarity": similarity_hits += 1

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
        target_updates = 50
        chunk_size = max(1, total // target_updates)
        pbar = tqdm(total=total, desc="Mapping columns (parallel & weighted)") if (HAVE_TQDM and progress) else None
        for i in range(0, total, chunk_size):
            chunk = pairs[i:i + chunk_size]
            mapped = Parallel(n_jobs=n_jobs)(delayed(compare_pair)(dc, sc) for dc, sc in chunk)
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

    # Rank & pick best per dashboard column
    df = df.sort_values(["weighted_score", "avg_score", "hits", "hit_rate"], ascending=False).reset_index(drop=True)
    best_idx = df.groupby("dashboard_col")["weighted_score"].idxmax()
    best = df.loc[best_idx].copy()

    # friendly suggestion label
    def suggest(r):
        if r["hit_rate"] >= 0.90 and r["avg_score"] >= HIGH_RATIO:
            return "exact/very likely (strong coverage)"
        if r["hit_rate"] >= 0.75 and r["avg_score"] >= MIN_RATIO:
            return "likely (good coverage)"
        if r["hit_rate"] >= 0.50 and r["avg_score"] >= MIN_RATIO:
            return "possible (review)"
        return "uncertain"

    best["suggestion"] = best.apply(suggest, axis=1)
    best = best.sort_values(["weighted_score", "avg_score", "hits", "hit_rate"], ascending=False).reset_index(drop=True)

    return best, df

# --------- Audit ----------
def per_cell_audit(
    source_df: pd.DataFrame,
    dash_df: pd.DataFrame,
    id_source: str,
    id_dash: str,
    pairs: List[Tuple[str, str]],
    max_rows: int = 1000
) -> pd.DataFrame:
    source_df = merge_group(source_df, id_source)
    dash_df   = merge_group(dash_df, id_dash)

    src = source_df.set_index(id_source)
    dst = dash_df.set_index(id_dash)
    common_ids = src.index.intersection(dst.index)
    if not len(common_ids):
        return pd.DataFrame()

    all_values = pd.concat([src, dst], axis=1).replace(np.nan, "", regex=False).values.flatten().tolist()
    from .scoring import make_tfidf_vectorizer, compare_elements, normalize_text
    tfidf_vec = make_tfidf_vectorizer().fit([normalize_text(v) for v in all_values if v])

    rows = []
    for dc, sc in pairs:
        for rid in common_ids:
            dv = dst.at[rid, dc]
            sv = src.at[rid, sc]
            if pd.isna(dv) or pd.isna(sv):
                continue
            score, tag = compare_elements(dv, sv, tfidf_vec)
            rows.append(dict(
                id=rid,
                dashboard_col=dc,
                source_col=sc,
                dashboard_val=dv,
                source_val=sv,
                score=score,
                tag=tag
            ))
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)
