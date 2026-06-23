# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Transforms raw rwa.xyz API responses into a single cleaned,
long-format DataFrame ready for analysis.

Pipeline: raw JSON (per measure) -> long DataFrame (per measure)
          -> merged combined DataFrame -> validated/windowed DataFrame -> CSV
"""

import os
import numpy as np
import pandas as pd

from src.config import ANALYSIS_START_DATE, ASSET_CLASSES_IN_SCOPE, RESULTS_DIR

_REQUIRED_COLUMNS = {"date", "asset_class", "cav", "holders", "volume"}
_NUMERIC_COLUMNS = ["cav", "holders", "volume"]

# Thresholds below which a baseline value is flagged as potentially unreliable.
# Low baselines cause indexed growth to look exaggerated relative to real-world scale.
MIN_BASELINE_CAV = 1_000_000        # $1M minimum credible CAV baseline
MIN_BASELINE_HOLDERS = 10           # 10 holders minimum credible baseline


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
# Downstream metric builders
# ─────────────────────────────────────────────

def build_composition_shares(df):
    """
    Pillar 1: Each asset class's % share of total CAV per month.
    Adds 'cav_share' column (0–1).
    """
    df = df.copy()
    monthly_total = df.groupby("date")["cav"].transform("sum")
    df["cav_share"] = df["cav"] / monthly_total
    return df


def build_relative_growth_index(df):
    """
    Pillar 2: Within-asset-class relative growth index.

    Each asset class is indexed to its own first valid monthly observation = 100.
    This shows relative growth *within* each class, not absolute adoption size.
    Small or late baselines may exaggerate growth percentages; always interpret
    the index alongside cav_baseline, holders_baseline, and the baseline flags.

    Added columns:
      cav_index                           — CAV relative to first valid month (100 = baseline)
      holders_index                       — holders relative to first valid month (100 = baseline)
      participation_ratio                 — holders_index / cav_index (directional only)
      cav_baseline                        — absolute CAV value at the baseline month
      holders_baseline                    — absolute holders value at the baseline month
      cav_baseline_date                   — date the CAV baseline was taken
      holders_baseline_date               — date the holders baseline was taken
      cav_absolute_change_from_baseline   — cav - cav_baseline
      holders_absolute_change_from_baseline — holders - holders_baseline
      low_cav_baseline_flag               — True if cav_baseline < MIN_BASELINE_CAV
      low_holders_baseline_flag           — True if holders_baseline < MIN_BASELINE_HOLDERS
      late_start_flag                     — True if the first valid month is after ANALYSIS_START_DATE

    Index values are NaN before the first valid baseline month for each class.
    No index is calculated when the baseline is null, zero, or negative.

    participation_ratio > 1: holders grew faster than CAV relative to their shared baseline.
    participation_ratio < 1: CAV grew faster than holders relative to their shared baseline.
    This is directional only and does not prove decentralization, unique user growth,
    or absolute adoption strength.
    """
    df = df.copy()
    analysis_start = pd.Timestamp(ANALYSIS_START_DATE)

    group_frames = []
    for asset_class, group in df.groupby("asset_class", sort=False):
        g = group.sort_values("date").copy()

        # First valid (positive, non-null) CAV baseline.
        cav_valid = g[g["cav"].notna() & (g["cav"] > 0)]
        cav_base_val = cav_valid["cav"].iloc[0] if not cav_valid.empty else None
        cav_base_date = cav_valid["date"].iloc[0] if not cav_valid.empty else None

        # First valid (positive) holders baseline.
        holders_valid = g[g["holders"].notna() & (g["holders"] > 0)]
        holders_base_val = holders_valid["holders"].iloc[0] if not holders_valid.empty else None
        holders_base_date = holders_valid["date"].iloc[0] if not holders_valid.empty else None

        # Baseline-quality flags.
        g["cav_baseline"] = cav_base_val
        g["cav_baseline_date"] = cav_base_date
        g["holders_baseline"] = holders_base_val
        g["holders_baseline_date"] = holders_base_date

        g["low_cav_baseline_flag"] = bool(
            cav_base_val is not None and cav_base_val < MIN_BASELINE_CAV
        )
        g["low_holders_baseline_flag"] = bool(
            holders_base_val is not None and holders_base_val < MIN_BASELINE_HOLDERS
        )

        # late_start_flag: first valid month is after the global analysis start.
        valid_dates = [d for d in [cav_base_date, holders_base_date] if d is not None]
        first_valid = min(valid_dates) if valid_dates else None
        g["late_start_flag"] = bool(first_valid is not None and first_valid > analysis_start)

        # CAV index — NaN before baseline date; requires valid baseline.
        g["cav_index"] = np.nan
        g["cav_absolute_change_from_baseline"] = np.nan
        if cav_base_val is not None and cav_base_date is not None:
            after_base = g["date"] >= cav_base_date
            has_cav = g["cav"].notna()
            mask = after_base & has_cav
            g.loc[mask, "cav_index"] = g.loc[mask, "cav"] / cav_base_val * 100
            g.loc[mask, "cav_absolute_change_from_baseline"] = g.loc[mask, "cav"] - cav_base_val

        # Holders index — NaN before baseline date; requires valid baseline.
        g["holders_index"] = np.nan
        g["holders_absolute_change_from_baseline"] = np.nan
        if holders_base_val is not None and holders_base_date is not None:
            after_base = g["date"] >= holders_base_date
            has_holders = g["holders"].notna()
            mask = after_base & has_holders
            g.loc[mask, "holders_index"] = g.loc[mask, "holders"] / holders_base_val * 100
            g.loc[mask, "holders_absolute_change_from_baseline"] = (
                g.loc[mask, "holders"] - holders_base_val
            )

        group_frames.append(g)

    if not group_frames:
        return df

    out = pd.concat(group_frames, ignore_index=True)

    # participation_ratio: only where both indexes are valid and positive.
    both_valid = (
        out["cav_index"].notna() & out["holders_index"].notna()
        & (out["cav_index"] > 0) & (out["holders_index"] > 0)
    )
    out["participation_ratio"] = np.nan
    out.loc[both_valid, "participation_ratio"] = (
        out.loc[both_valid, "holders_index"] / out.loc[both_valid, "cav_index"]
    )

    return out


def build_adoption_index(df):
    """
    Backward-compatible alias for build_relative_growth_index.

    Prefer calling build_relative_growth_index directly in new code.
    The index measures within-asset-class relative growth from each class's
    own first valid observation = 100. It is not an absolute adoption ranking.
    """
    return build_relative_growth_index(df)


def build_avg_position_size(df):
    """
    Pillar 2: Average dollar value held per wallet per asset class per month.
    avg_position = CAV / holders. Declining trend = new smaller participants
    entering (broadening adoption). Rising trend = existing holders adding
    capital (concentration).
    """
    df = df.copy()
    df["avg_position"] = df["cav"] / df["holders"].replace(0, float("nan"))
    return df


def build_turnover_ratio(df):
    """
    Pillar 3: Turnover ratio = monthly transfer volume / average monthly CAV.
    Adds turnover_ratio and turnover_3m (3-month rolling average per asset class) columns.
    """
    df = df.copy()
    df["turnover_ratio"] = df["volume"] / df["cav"]
    df["turnover_3m"] = (
        df.sort_values("date")
        .groupby("asset_class")["turnover_ratio"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    return df


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


def validate_growth_index(df):
    """
    Additional validation for DataFrames produced by build_relative_growth_index.
    Returns a list of error strings. Checks:
      - Baseline flag columns exist and are boolean
      - No index value is calculated from a zero or negative baseline
      - Rows before the baseline month have null index values
      - participation_ratio is null unless both indexes are valid and positive
    """
    errors = []

    flag_cols = ["low_cav_baseline_flag", "low_holders_baseline_flag", "late_start_flag"]
    for col in flag_cols:
        if col not in df.columns:
            errors.append(f"Missing baseline flag column: '{col}'")
        elif not pd.api.types.is_bool_dtype(df[col]):
            errors.append(f"Column '{col}' should be boolean (dtype: {df[col].dtype})")

    if "cav_index" in df.columns and "cav_baseline" in df.columns:
        bad = df["cav_index"].notna() & df["cav_baseline"].notna() & (df["cav_baseline"] <= 0)
        if bad.any():
            errors.append(f"{bad.sum()} row(s) have cav_index calculated from a non-positive baseline")

    if "holders_index" in df.columns and "holders_baseline" in df.columns:
        bad = df["holders_index"].notna() & df["holders_baseline"].notna() & (df["holders_baseline"] <= 0)
        if bad.any():
            errors.append(f"{bad.sum()} row(s) have holders_index calculated from a non-positive baseline")

    if all(c in df.columns for c in ["cav_index", "cav_baseline_date", "date"]):
        pre_base = df.merge(
            df.groupby("asset_class")["cav_baseline_date"].first().reset_index(),
            on="asset_class", suffixes=("", "_min")
        )
        before_base = pre_base["date"] < pre_base["cav_baseline_date_min"]
        leaked = before_base & pre_base["cav_index"].notna()
        if leaked.any():
            errors.append(f"{leaked.sum()} row(s) have cav_index set before the baseline date")

    if "participation_ratio" in df.columns:
        cav_ok = df["cav_index"].notna() & (df["cav_index"] > 0)
        holders_ok = df["holders_index"].notna() & (df["holders_index"] > 0)
        should_be_null = ~(cav_ok & holders_ok)
        spurious = should_be_null & df["participation_ratio"].notna()
        if spurious.any():
            errors.append(
                f"{spurious.sum()} row(s) have participation_ratio set "
                f"where one or both indexes are invalid/non-positive"
            )

    return errors


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
