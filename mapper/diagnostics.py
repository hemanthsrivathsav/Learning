"""
Diagnostics for Hybrid Mapper
-----------------------------
1) Hit distribution by tag (from all_candidates.csv)
2) Threshold sensitivity sweep using per_cell_audit.csv
"""

from typing import Dict, List, Iterable, Optional
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Part A: quick summary from all_candidates.csv ----------

def analyze_threshold_sensitivity(df_all_candidates: pd.DataFrame,
                                  tag_thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Summarize current hit distribution by tag type using all_candidates.csv.

    Parameters
    ----------
    df_all_candidates : DataFrame with columns:
        ['direct_hits','list_hits','listlike_hits','bm25_hits','similarity_hits']
    tag_thresholds : dict, e.g. {"direct":0.70,"list":0.70,"listlike":0.70,"bm25":0.35,"similarity":0.60}

    Returns
    -------
    DataFrame with columns: ['Tag','Total_Hits','Threshold','Relative_%']
    """
    tag_cols = ["direct_hits", "list_hits", "listlike_hits", "bm25_hits", "similarity_hits"]
    # Guard for missing columns (older runs)
    tag_cols = [c for c in tag_cols if c in df_all_candidates.columns]
    if not tag_cols:
        raise ValueError("No tag hit columns found in all_candidates.csv.")

    df_summary = df_all_candidates[tag_cols].sum().reset_index()
    df_summary.columns = ["Tag", "Total_Hits"]
    df_summary["Threshold"] = df_summary["Tag"].map(lambda t: tag_thresholds.get(t.replace("_hits", ""), None))
    total = float(df_summary["Total_Hits"].sum()) or 1.0
    df_summary["Relative_%"] = (df_summary["Total_Hits"] / total * 100).round(2)
    return df_summary


def plot_hit_distribution(df_summary: pd.DataFrame) -> None:
    """
    Bar chart of total hits per tag (from all_candidates.csv).
    """
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df_summary["Tag"], df_summary["Total_Hits"])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 f"{df_summary['Relative_%'].iloc[i]}%",
                 ha='center', va='bottom', fontsize=10)
    plt.title("Hit Distribution by Match Type")
    plt.ylabel("Total Hits Across All Column Pairs")
    plt.xlabel("Match Tag")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ---------- Part B: sensitivity sweep from per_cell_audit.csv ----------

def default_sweep_grid(base_thresholds: Dict[str, float]) -> Dict[str, List[float]]:
    """
    Build sensible sweep ranges around your current thresholds.
    - direct/list/listlike are already very high-confidence → narrow sweep
    - bm25 benefits from wider exploration
    - similarity moderate range

    Returns dict like:
      {"direct":[0.65,0.70,0.75], "bm25":[0.25,0.30,0.35,0.40,0.45], ...}
    """
    def arange(start, stop, step):
        vals = []
        x = start
        while x <= stop + 1e-9:
            vals.append(round(x, 3))
            x += step
        return vals

    grid = {}
    grid["direct"]     = arange(max(0.5, base_thresholds.get("direct", 0.70) - 0.05),
                                min(0.99, base_thresholds.get("direct", 0.70) + 0.05),
                                0.025)
    grid["list"]       = arange(max(0.5, base_thresholds.get("list", 0.70) - 0.05),
                                min(0.99, base_thresholds.get("list", 0.70) + 0.05),
                                0.025)
    grid["listlike"]   = arange(max(0.5, base_thresholds.get("listlike", 0.70) - 0.05),
                                min(0.99, base_thresholds.get("listlike", 0.70) + 0.05),
                                0.025)
    grid["bm25"]       = arange(max(0.05, base_thresholds.get("bm25", 0.35) - 0.15),
                                min(0.95, base_thresholds.get("bm25", 0.35) + 0.20),
                                0.025)
    grid["similarity"] = arange(max(0.3, base_thresholds.get("similarity", 0.60) - 0.15),
                                min(0.95, base_thresholds.get("similarity", 0.60) + 0.15),
                                0.025)
    return grid


def simulate_threshold_sweep_from_audit(audit_df: pd.DataFrame,
                                        base_thresholds: Dict[str, float],
                                        sweep_grid: Optional[Dict[str, List[float]]] = None
                                        ) -> pd.DataFrame:
    """
    Recompute hit counts under different threshold settings using per_cell_audit rows.

    Parameters
    ----------
    audit_df : DataFrame with columns at least ['dashboard_col','source_col','score','tag']
               (output of per_cell_audit.csv)
    base_thresholds : current TAG_THRESHOLDS dict
    sweep_grid : dict mapping tag -> list of thresholds to test.
                 If None, uses default_sweep_grid(base_thresholds).

    Returns
    -------
    DataFrame with columns:
      ['tag','threshold','hits','rows_considered','hit_rate','avg_score_of_hits']
    (Aggregated across all pairs; you can also group by (dashboard_col, source_col) if you want per-pair curves.)
    """
    required = {"score", "tag"}
    missing = required - set(audit_df.columns)
    if missing:
        raise ValueError(f"audit_df is missing required columns: {missing}")

    if sweep_grid is None:
        sweep_grid = default_sweep_grid(base_thresholds)

    rows = []
    # For each tag, sweep thresholds independently — counts are computed only over rows with that tag
    for tag, thresholds in sweep_grid.items():
        tag_mask = (audit_df["tag"] == tag)
        tag_rows = audit_df.loc[tag_mask, ["score"]].copy()
        if tag_rows.empty:
            # still record zero-hit lines so plots align
            for thr in thresholds:
                rows.append(dict(tag=tag, threshold=thr, hits=0, rows_considered=0,
                                 hit_rate=0.0, avg_score_of_hits=0.0))
            continue

        # sort once for efficient cumulative counting
        scores_sorted = tag_rows["score"].sort_values().values
        n = len(scores_sorted)

        for thr in thresholds:
            # number of hits where score >= thr
            # binary search:
            import bisect
            idx = bisect.bisect_left(scores_sorted, thr)
            hits = n - idx
            if hits > 0:
                avg_score = float(scores_sorted[idx:].mean())
            else:
                avg_score = 0.0
            hit_rate = hits / n if n else 0.0
            rows.append(dict(tag=tag, threshold=round(thr, 3), hits=int(hits),
                             rows_considered=int(n), hit_rate=round(hit_rate, 4),
                             avg_score_of_hits=round(avg_score, 4)))

    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep_df: pd.DataFrame) -> None:
    """
    Line plot: hits vs threshold for each tag (one chart).
    """
    if sweep_df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(9, 5))
    tags = sorted(sweep_df["tag"].unique())

    for tag in tags:
        sub = sweep_df[sweep_df["tag"] == tag].sort_values("threshold")
        plt.plot(sub["threshold"], sub["hits"], marker='o', label=tag)

    plt.title("Threshold Sensitivity: Hits vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Total Hits (from audit sample)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
