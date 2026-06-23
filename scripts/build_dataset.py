#!/usr/bin/env python3
# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Build the combined monthly RWA dataset and Bronze metrics layer.

Phase 1 — Fetches (or loads from 24-hour cache) rwa.xyz API data, transforms
           it into a clean monthly dataset, and saves combined_monthly.csv.
Phase 2 — Computes all derived metrics and saves:
             combined_metrics.csv      (per-asset-class metrics)
             concentration_metrics.csv (market-level concentration metrics)

Usage (from repo root):
    python scripts/build_dataset.py
"""

import sys
import os

# Ensure the repo root is on sys.path when the script is invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bronze.api_client import (
    cached_request,
    get_cav_by_asset_class,
    get_holders_by_asset_class,
    get_transfer_volume_by_asset_class,
)
from src.bronze.data_processing import (
    build_combined_dataset,
    validate_dataset,
    save_combined_dataset,
    print_data_quality_summary,
)
from src.bronze.metrics import build_all_metrics, save_metrics
from src.config import ASSET_CLASSES_IN_SCOPE, RESULTS_DIR


def _fetch_all():
    """Fetch all three measures, using cached data where available."""
    cav     = cached_request(get_cav_by_asset_class,             "cav_by_asset_class")
    holders = cached_request(get_holders_by_asset_class,         "holders_by_asset_class")
    volume  = cached_request(get_transfer_volume_by_asset_class, "volume_by_asset_class")
    return cav, holders, volume


def _log_missing_classes(df):
    """Warn clearly about any in-scope asset class that has no rows."""
    present = set(df["asset_class"].unique())
    for ac in ASSET_CLASSES_IN_SCOPE:
        if ac not in present:
            print(f"[build_dataset] WARNING: No data for in-scope class '{ac}'")


def main():
    print("[build_dataset] Building Bronze dataset and metrics layer")
    print(f"[build_dataset] Output directory: {RESULTS_DIR}/")

    # ── Phase 1: Fetch + clean ────────────────────────────────────────────────
    cav, holders, volume = _fetch_all()

    print("[build_dataset] Building combined monthly dataset...")
    df = build_combined_dataset(cav, holders, volume)

    print_data_quality_summary(df)
    _log_missing_classes(df)

    errors = validate_dataset(df)
    if errors:
        print("[build_dataset] VALIDATION ERRORS — dataset will not be saved:")
        for err in errors:
            print(f"  ERROR: {err}")
        sys.exit(1)

    print("[build_dataset] All validation checks passed.")

    combined_monthly_path = save_combined_dataset(df, RESULTS_DIR)
    print(f"[build_dataset] Saved {len(df):,} rows → {combined_monthly_path}")

    # ── Phase 2: Metrics ──────────────────────────────────────────────────────
    print("[build_dataset] Computing Bronze metrics layer...")
    combined_metrics_df, concentration_df = build_all_metrics(df)

    metrics_path, concentration_path = save_metrics(
        combined_metrics_df, concentration_df, RESULTS_DIR
    )
    print(f"[build_dataset] Saved {len(combined_metrics_df):,} rows → {metrics_path}")
    print(f"[build_dataset] Saved {len(concentration_df):,} rows → {concentration_path}")

    print("[build_dataset] Done.")


if __name__ == "__main__":
    main()
