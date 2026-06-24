"""
Build all static thesis charts from saved metrics CSVs.

Usage:
    python3 scripts/build_charts.py

On each run this script will:
  1. Check whether the metrics CSVs are within the cache TTL defined in
     src/config.py. If they are stale or missing, build_dataset.py is
     invoked automatically to refresh the data before charting.
  2. Clear charts/png/ and charts/pdf/ so no stale outputs accumulate.
  3. Build all charts and save PNGs to charts/png/, PDFs to charts/pdf/.

Reads:
    data/combined_metrics.csv
    data/concentration_metrics.csv

Writes:
    charts/png/chart*.png
    charts/pdf/chart*.pdf
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import CACHE_TTL_SECONDS
from src.bronze.charts import (
    chart1_total_cav,
    chart2_cav_by_asset_class,
    chart3_holder_growth,
    chart4_turnover_ratio,
    chart5_scorecard,
    chart6_ex_repo_total_cav,
    chart7_ex_repo_by_class,
    chart8_latest_composition,
    chart9_cav_share_over_time,
    chart10_before_after_repo,
    chart11_latest_market_map,
    chart12_median_turnover,
    chart13_total_holders,
    chart14_ex_repo_hhi,
    chart15_ex_repo_rolling_growth,
)

DATA_DIR = REPO_ROOT / "data"
CHARTS_DIR = REPO_ROOT / "charts"

COMBINED_PATH = DATA_DIR / "combined_metrics.csv"
CONCENTRATION_PATH = DATA_DIR / "concentration_metrics.csv"


def _ensure_fresh_data() -> None:
    """Re-run build_dataset.py if either metrics file is missing or stale."""
    sentinel_files = [COMBINED_PATH, CONCENTRATION_PATH]
    needs_refresh = False

    for path in sentinel_files:
        if not path.exists():
            print(f"[data] {path.name} not found — will run build_dataset.py")
            needs_refresh = True
            break
        age_seconds = time.time() - path.stat().st_mtime
        age_hours = age_seconds / 3600
        ttl_hours = CACHE_TTL_SECONDS / 3600
        if age_seconds > CACHE_TTL_SECONDS:
            print(
                f"[data] {path.name} is {age_hours:.1f}h old "
                f"(TTL: {ttl_hours:.0f}h) — refreshing data..."
            )
            needs_refresh = True
            break
        else:
            print(f"[data] {path.name} is fresh ({age_hours:.1f}h old, TTL: {ttl_hours:.0f}h)")

    if needs_refresh:
        print("[data] Running build_dataset.py to fetch fresh data...")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_dataset.py")],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print("ERROR: build_dataset.py failed. Cannot build charts with stale data.")
            sys.exit(1)
        print("[data] Data refresh complete.")


def _clear_chart_dirs() -> None:
    """Delete all files in charts/png/ and charts/pdf/ before regenerating."""
    for subdir in ("png", "pdf"):
        d = CHARTS_DIR / subdir
        if d.exists():
            deleted = 0
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
                    deleted += 1
            if deleted:
                print(f"[charts] Cleared {deleted} file(s) from {d.relative_to(REPO_ROOT)}/")
        else:
            d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(COMBINED_PATH)
    concentration = pd.read_csv(CONCENTRATION_PATH)
    return combined, concentration


def main() -> None:
    print("=== build_charts.py ===")
    print()

    print("Checking data freshness...")
    _ensure_fresh_data()
    print()

    print("Clearing output directories...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_chart_dirs()
    print()

    print("Loading metrics data...")
    combined, concentration = load_data()
    print(f"  combined_metrics:      {len(combined)} rows, {combined['date'].min()} → {combined['date'].max()}")
    print(f"  concentration_metrics: {len(concentration)} rows, {concentration['date'].min()} → {concentration['date'].max()}")
    print()

    results: dict[str, Path | None] = {}

    print("Building core charts (1–5)...")
    results["chart1"]  = chart1_total_cav(concentration, CHARTS_DIR)
    results["chart2"]  = chart2_cav_by_asset_class(combined, CHARTS_DIR)
    results["chart3"]  = chart3_holder_growth(combined, CHARTS_DIR)
    results["chart4"]  = chart4_turnover_ratio(combined, CHARTS_DIR)
    results["chart5"]  = chart5_scorecard(concentration, combined, CHARTS_DIR)

    print("Building clarity/ex-repo variants (6–15)...")
    results["chart6"] = chart6_ex_repo_total_cav(combined, CHARTS_DIR)
    results["chart7"] = chart7_ex_repo_by_class(combined, CHARTS_DIR)
    results["chart8"] = chart8_latest_composition(combined, CHARTS_DIR)
    results["chart9"] = chart9_cav_share_over_time(combined, CHARTS_DIR)
    results["chart10"] = chart10_before_after_repo(combined, CHARTS_DIR)
    results["chart11"] = chart11_latest_market_map(combined, CHARTS_DIR)
    results["chart12"] = chart12_median_turnover(combined, CHARTS_DIR)
    results["chart13"] = chart13_total_holders(combined, CHARTS_DIR)
    results["chart14"] = chart14_ex_repo_hhi(combined, CHARTS_DIR)
    results["chart15"] = chart15_ex_repo_rolling_growth(combined, CHARTS_DIR)

    print()
    print("=" * 60)
    produced = [p for p in results.values() if p is not None]
    skipped  = [k for k, p in results.items() if p is None]

    print(f"Charts produced: {len(produced)}")
    for path in sorted(produced, key=lambda p: p.name):
        size_kb = path.stat().st_size / 1024
        print(f"  ✓  {path.name}  ({size_kb:.0f} KB)")

    if skipped:
        print(f"\nCharts skipped ({len(skipped)}): {', '.join(skipped)}")

    core_charts = {"chart1", "chart2", "chart3", "chart4", "chart5"}
    missing_core = core_charts - {k for k, p in results.items() if p is not None}
    if missing_core:
        print(f"\nWARNING: Core charts not produced: {missing_core}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
