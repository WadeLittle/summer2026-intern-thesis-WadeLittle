#!/usr/bin/env python3
# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Build the combined monthly RWA dataset.

Fetches (or loads from 24-hour cache) rwa.xyz API data, transforms it into a
clean monthly dataset, validates it, and saves to results/combined_monthly.csv.

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
    print("[build_dataset] Phase 1 — Bronze: building combined monthly dataset")
    print(f"[build_dataset] Output directory: {RESULTS_DIR}/")

    # ── Fetch ────────────────────────────────────────────────────────────────
    cav, holders, volume = _fetch_all()

    # ── Transform ────────────────────────────────────────────────────────────
    print("[build_dataset] Building combined monthly dataset...")
    df = build_combined_dataset(cav, holders, volume)

    # ── Data quality summary ─────────────────────────────────────────────────
    print_data_quality_summary(df)
    _log_missing_classes(df)

    # ── Validate ─────────────────────────────────────────────────────────────
    errors = validate_dataset(df)
    if errors:
        print("[build_dataset] VALIDATION ERRORS — dataset will not be saved:")
        for err in errors:
            print(f"  ERROR: {err}")
        sys.exit(1)

    print("[build_dataset] All validation checks passed.")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = save_combined_dataset(df, RESULTS_DIR)
    print(f"[build_dataset] Saved {len(df):,} rows to {output_path}")
    print("[build_dataset] Done.")


if __name__ == "__main__":
    main()
