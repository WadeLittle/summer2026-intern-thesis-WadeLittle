# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Raw-to-clean transformation: converts rwa.xyz API responses into a single
cleaned, long-format monthly DataFrame ready for the metrics layer.

Pipeline: raw JSON (per measure) -> long DataFrame (per measure)
          -> merged combined DataFrame -> validated/windowed DataFrame -> CSV

Derived metric logic lives in src/bronze/metrics.py.
"""

import os
import numpy as np
import pandas as pd

from src.config import ANALYSIS_START_DATE, ASSET_CLASSES_IN_SCOPE, RESULTS_DIR

# Re-export baseline constants so existing callers don't break.
from src.bronze.metrics_config import MIN_BASELINE_CAV, MIN_BASELINE_HOLDERS  # noqa: F401

_REQUIRED_COLUMNS = {"date", "asset_class", "cav", "holders", "volume"}
_NUMERIC_COLUMNS = ["cav", "holders", "volume"]


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_long_df(api_response, value_name):
    """
    Converts one timeseries API response (grouped by asset_class) into
    a long-format DataFrame: columns = date, asset_class, <value_name>.

    Long format (one row per date/asset_class/value) is what pandas'
    groupby, pivot_table, and merge expect.
    """
    rows = []
    for series in api_response["results"]:
        asset_class = series["group"]["name"]
        for date_str, value in series["points"]:
            rows.append({
                "date": date_str,
                "asset_class": asset_class,
                value_name: value
            })

    if not rows:
        print(f"[extract_long_df] No rows for '{value_name}'. "
              f"API result keys: {list(api_response.keys())}, "
              f"result count: {len(api_response.get('results', []))}")
        return pd.DataFrame(columns=["date", "asset_class", value_name])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────
# Resampling strategies
# ─────────────────────────────────────────────

def _resample_stock(df, value_col):
    """
    Collapses daily stock-mode data to monthly by taking the last observed
    value per asset class per month (end-of-month snapshot).
    Used for holders — a count where the end-of-period value is the right
    representation of participation at that point in time.
    """
    return (
        df.groupby(["asset_class", pd.Grouper(key="date", freq="MS")])[value_col]
        .last()
        .reset_index()
    )


def _resample_monthly_avg(df, value_col):
    """
    Collapses daily stock-mode data to monthly by taking the mean across all
    daily values per asset class per month. Dates are labeled as month-start.
    Used for CAV — monthly average is more stable than a single end-of-month
    snapshot and is the correct denominator for the turnover ratio per the spec.
    """
    return (
        df.groupby(["asset_class", pd.Grouper(key="date", freq="MS")])[value_col]
        .mean()
        .reset_index()
    )


def _resample_flow(df, value_col):
    """
    Collapses daily flow-mode data to monthly by summing all daily values
    per asset class per month. Dates are labeled as month-start.

    'sum' is correct here: flow measures (volume) represent activity that
    accumulates over a period, so monthly total = sum of daily totals.
    """
    return (
        df.groupby(["asset_class", pd.Grouper(key="date", freq="MS")])[value_col]
        .sum()
        .reset_index()
    )


# ─────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────

def build_combined_dataset(aum_data, holders_data, volume_data):
    """
    Merges CAV, holders, and volume into one long-format monthly DataFrame,
    applies the analysis date window, drops the partial current month,
    removes negative CAV rows, and fills missing activity values with 0.

    Daily API data is resampled here to monthly using the correct strategy
    per measure type (mean for CAV stock, last for holders stock, sum for volume flow).

    Returns columns: date, asset_class, cav, holders, volume
    """
    cav_long = extract_long_df(aum_data, "cav")
    cav_df = _resample_monthly_avg(cav_long, "cav")
    holders_df = _resample_stock(extract_long_df(holders_data, "holders"), "holders")
    volume_df = _resample_flow(extract_long_df(volume_data, "volume"), "volume")

    # Outer merge: keep a (date, asset_class) row even if only one
    # measure has data for it — gaps are handled explicitly below.
    combined = cav_df.merge(holders_df, on=["date", "asset_class"], how="outer")
    combined = combined.merge(volume_df, on=["date", "asset_class"], how="outer")

    # Scope to traditional RWA asset classes (see config.py for rationale).
    combined = combined[combined["asset_class"].isin(ASSET_CLASSES_IN_SCOPE)]

    # Apply analysis window.
    combined = combined[combined["date"] >= ANALYSIS_START_DATE]

    # Drop the most recent month — rwa.xyz data for the current month is a
    # partial-month snapshot and would distort trend/growth calculations.
    if not combined.empty:
        latest_month = combined["date"].max()
        combined = combined[combined["date"] < latest_month]

    # Remove negative CAV rows — these indicate data anomalies, not real values.
    neg_mask = combined["cav"] < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        print(f"[data_processing] WARNING: Dropping {n_neg} row(s) with negative CAV.")
        combined = combined[~neg_mask]

    # Missing holders/volume for a (date, asset_class) means no token activity
    # was recorded that month — a real zero, not missing data. CAV gaps are
    # left as NaN; we cannot assume zero CAV when data is simply absent.
    combined["holders"] = combined["holders"].fillna(0)
    combined["volume"] = combined["volume"].fillna(0)

    combined = combined.sort_values(["asset_class", "date"]).reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_dataset(df):
    """
    Validates the combined dataset against spec requirements.
    Returns a list of error strings. An empty list means all checks passed.

    Checks:
      - All required columns are present
      - Numeric columns are actually numeric types
      - Date range starts no earlier than ANALYSIS_START_DATE
      - No rows with negative CAV remain
    """
    errors = []

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return errors  # can't proceed with further column-dependent checks

    for col in _NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric (dtype: {df[col].dtype})")

    if not df.empty:
        min_date = df["date"].min()
        if min_date < pd.Timestamp(ANALYSIS_START_DATE):
            errors.append(
                f"Date range starts before {ANALYSIS_START_DATE}: "
                f"earliest row is {min_date.date()}"
            )

        n_neg_cav = int((df["cav"] < 0).sum())
        if n_neg_cav > 0:
            errors.append(f"{n_neg_cav} row(s) with negative CAV remain after processing")

    return errors


# validate_growth_index moved to src/bronze/metrics.py; re-exported for backward compat.
from src.bronze.metrics import validate_growth_index  # noqa: F401, E402


# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────

def save_combined_dataset(df, results_dir=RESULTS_DIR):
    """Saves the combined monthly dataset to <results_dir>/combined_monthly.csv."""
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "combined_monthly.csv")
    df.to_csv(output_path, index=False)
    return output_path


def print_data_quality_summary(df):
    """Prints a human-readable data quality report for the combined dataset."""
    W = 60
    print("\n" + "=" * W)
    print("  DATA QUALITY SUMMARY")
    print("=" * W)

    if df.empty:
        print("  WARNING: Dataset is empty.")
        print("=" * W + "\n")
        return

    date_min = df["date"].min().strftime("%Y-%m")
    date_max = df["date"].max().strftime("%Y-%m")
    n_months = df["date"].nunique()
    n_classes = df["asset_class"].nunique()

    print(f"  Rows:          {len(df):,}")
    print(f"  Date range:    {date_min} to {date_max}  ({n_months} months)")
    print(f"  Asset classes: {n_classes}")

    print(f"\n  Missing values per column:")
    for col in _NUMERIC_COLUMNS:
        n_null = df[col].isna().sum()
        pct = n_null / len(df) * 100
        flag = "  <-- review" if col == "cav" and n_null > 0 else ""
        print(f"    {col:<10}: {n_null:>5} rows  ({pct:5.1f}%){flag}")

    print(f"\n  In-scope asset class coverage:")
    for ac in ASSET_CLASSES_IN_SCOPE:
        subset = df[df["asset_class"] == ac]
        if subset.empty:
            print(f"    {'[NO DATA]':<12} {ac}")
        else:
            n_cav_null = subset["cav"].isna().sum()
            gap_note = f"  ({n_cav_null} null CAV months)" if n_cav_null else ""
            print(f"    {len(subset):>3} months    {ac}{gap_note}")

    missing_classes = [
        ac for ac in ASSET_CLASSES_IN_SCOPE if df[df["asset_class"] == ac].empty
    ]
    if missing_classes:
        print(f"\n  WARNING: No data found for {len(missing_classes)} class(es):")
        for ac in missing_classes:
            print(f"    - {ac}")

    print("=" * W + "\n")
