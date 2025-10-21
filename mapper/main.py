import argparse
from pathlib import Path
from .config import MIN_RATIO, TAG_THRESHOLDS
from .core import load_dataframes, map_columns, per_cell_audit

def main():
    global MIN_RATIO  # allow override via CLI

    p = argparse.ArgumentParser(description="Hybrid Mapper v2 (modular)")
    p.add_argument("--source", required=True)
    p.add_argument("--dashboard", required=True)
    p.add_argument("--id-source", required=True)
    p.add_argument("--id-dash", required=True)
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--sample-rows", type=int, default=None)
    p.add_argument("--min-ratio", type=float, default=MIN_RATIO,
                   help="Default row-level threshold; per-tag thresholds still apply.")
    args = p.parse_args()

    # Let CLI default override the global used in suggestion labels
    from . import config as _cfg
    _cfg.MIN_RATIO = args.min_ratio

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
        tag_thresholds=TAG_THRESHOLDS,   # you can also pass a custom dict here
    )

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
