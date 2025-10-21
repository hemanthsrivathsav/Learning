"""
Threshold Sweep CLI for Hybrid Mapper
-------------------------------------
- Loads tag thresholds from hybrid_mapper.config.TAG_THRESHOLDS
- Summarizes tag hit distribution from out/all_candidates.csv
- If out/per_cell_audit.csv exists, runs a threshold sweep to show
  how hits change as you vary thresholds per tag.

Usage:
  python tools/threshold_sweep.py \
    --out-dir ./out \
    --module-root . \
    --save-plots \
    --no-show

Outputs (in --out-dir):
  - threshold_summary_all_candidates.csv
  - threshold_sweep_from_audit.csv  (only if audit file is present)
  - plots saved as PNG when --save-plots is set
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Hybrid Mapper threshold diagnostics")
    ap.add_argument("--out-dir", default="./out", help="Directory containing all_candidates.csv (and per_cell_audit.csv if present).")
    ap.add_argument("--module-root", default=".", help="Project root to add to sys.path so 'hybrid_mapper' can be imported.")
    ap.add_argument("--save-plots", action="store_true", help="Save charts as PNGs next to CSVs.")
    ap.add_argument("--no-show", action="store_true", help="Do not display plots interactively (useful on headless runs).")
    args = ap.parse_args()

    # Make sure we can import hybrid_mapper.*
    sys.path.insert(0, str(Path(args.module_root).resolve()))

    # Lazy import after sys.path tweak
    try:
        from config import TAG_THRESHOLDS
        from diagnostics import (
            analyze_threshold_sensitivity,
            plot_hit_distribution,
            default_sweep_grid,
            simulate_threshold_sweep_from_audit,
            plot_threshold_sweep,
        )
    except Exception as e:
        print(f"[!] Could not import hybrid_mapper modules. Check --module-root. Error: {e}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Part 1: all_candidates summary ----
    candidates_path = out_dir / "all_candidates.csv"
    if not candidates_path.exists():
        print(f"[!] {candidates_path} not found. Run the mapper first.")
        sys.exit(1)

    df_all = pd.read_csv(candidates_path)
    try:
        summary = analyze_threshold_sensitivity(df_all, TAG_THRESHOLDS)
    except ValueError as ve:
        print(f"[!] {ve}")
        sys.exit(1)

    print("\n=== Hit Distribution by Match Type (current thresholds) ===")
    print(summary.to_string(index=False))

    summary_path = out_dir / "threshold_summary_all_candidates.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[✔] Wrote {summary_path}")

    # Plot bar chart
    if not args.no_show or args.save_plots:
        import matplotlib.pyplot as plt
        plot_hit_distribution(summary)
        if args.save_plots:
            plt.savefig(out_dir / "hit_distribution_by_tag.png", dpi=140, bbox_inches="tight")
            print(f"[✔] Saved {out_dir / 'hit_distribution_by_tag.png'}")
        if args.no_show:
            plt.close()

    # ---- Part 2: threshold sweep from audit (optional) ----
    audit_path = out_dir / "per_cell_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if not {"score", "tag"}.issubset(audit.columns):
            print(f"[!] {audit_path} is missing 'score' or 'tag' columns. Re-run mapper with --audit to refresh.")
        else:
            grid = default_sweep_grid(TAG_THRESHOLDS)
            sweep = simulate_threshold_sweep_from_audit(audit, TAG_THRESHOLDS, grid)
            sweep_path = out_dir / "threshold_sweep_from_audit.csv"
            sweep.to_csv(sweep_path, index=False)
            print(f"[✔] Wrote {sweep_path}")

            # Plot sweep
            if not args.no_show or args.save_plots:
                import matplotlib.pyplot as plt
                plot_threshold_sweep(sweep)
                if args.save_plots:
                    plt.savefig(out_dir / "threshold_sweep_hits_vs_threshold.png", dpi=140, bbox_inches="tight")
                    print(f"[✔] Saved {out_dir / 'threshold_sweep_hits_vs_threshold.png'}")
                if args.no_show:
                    plt.close()
    else:
        print(f"[i] {audit_path} not found. For the sensitivity sweep, run your mapper with --audit to generate it.")

    print("\nDone.")

if __name__ == "__main__":
    main()
