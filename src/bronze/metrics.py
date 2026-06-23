# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Bronze metrics layer: all derived fields computed from combined_monthly.csv.

Design principles
-----------------
* Every public function is a pure transformation: takes a DataFrame, returns
  a new DataFrame with one or more columns added.  No side-effects.
* Functions are composable: each adds a single logical group of columns so
  callers can mix and match for tests or ad-hoc analysis.
* Division-by-zero safety: replace zeros/nulls in denominators with NaN so
  that results propagate NaN rather than raising ZeroDivisionError or
  silently producing inf values.
* Sorting responsibility: callers pass unsorted data; each function sorts
  internally where order matters (rolling windows, pct_change).

Entry point
-----------
    combined_df, concentration_df = build_all_metrics(df)
"""

import os

import numpy as np
import pandas as pd

from src.bronze.metrics_config import (
    ACTIVE_CAV_THRESHOLD,
    MIN_BASELINE_CAV,
    MIN_BASELINE_HOLDERS,
    MIN_PERIODS_ROLLING,
    ROLLING_3M,
    ROLLING_6M,
)
from src.config import ANALYSIS_START_DATE, RESULTS_DIR

# Re-export so downstream code that previously imported these from
# data_processing still works without changes.
__all__ = [
    "add_cav_share",
    "add_cav_index",
    "add_asset_class_cav_growth",
    "add_holders_index",
    "add_monthly_holder_growth",
    "add_holders_per_million_cav",
    "add_avg_position",
    "add_turnover_ratio",
    "add_turnover_3m",
    "add_monthly_volume_growth",
    "build_asset_class_metrics",
    "build_concentration_metrics",
    "build_all_metrics",
    "save_metrics",
    # Legacy analysis helpers (moved from data_processing.py)
    "build_composition_shares",
    "build_relative_growth_index",
    "build_adoption_index",
    "build_avg_position_size",
    "build_turnover_ratio_legacy",
    "validate_growth_index",
    # Constants re-exported for backward compatibility
    "MIN_BASELINE_CAV",
    "MIN_BASELINE_HOLDERS",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pct_change(series: pd.Series) -> pd.Series:
    """
    pct_change that converts inf to NaN.

    inf arises when the previous value is 0 — there is no meaningful
    percentage base.  NaN (no previous value or null previous) is
    preserved as-is.  Negative-to-zero transitions correctly return -100%.
    Coerces to float first so np.isinf works on object-dtype inputs.
    """
    pct = series.astype(float).pct_change(fill_method=None) * 100.0
    return pct.where(~np.isinf(pct))


def _index_to_100(series: pd.Series) -> pd.Series:
    """
    Index an already-sorted series to 100 at its first non-null, positive value.

    Rows before that first valid observation remain NaN.
    Subsequent NaN values within the series also remain NaN (no forward-fill).
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = series.notna() & (series > 0)
    if not valid_mask.any():
        return result
    first_valid_iloc = int(np.argmax(valid_mask.values))
    base_val = float(series.iloc[first_valid_iloc])
    after_base = np.zeros(len(series), dtype=bool)
    after_base[first_valid_iloc:] = True
    apply_mask = after_base & series.notna().values
    result.values[apply_mask] = series.values[apply_mask] / base_val * 100.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 1 — Growth (per asset class)
# ─────────────────────────────────────────────────────────────────────────────

def add_cav_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index each asset class's CAV to 100 at its first non-null, positive month.
    Adds column: cav_index.
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["cav_index"] = (
        df.groupby("asset_class", sort=False)["cav"]
        .transform(_index_to_100)
    )
    return df


def add_asset_class_cav_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month % change in CAV per asset class.
    Adds column: asset_class_cav_growth.
    Returns NaN for the first month, when the previous CAV is null,
    or when the previous CAV is 0 (no meaningful base for %).
    Correctly returns -100% when CAV falls to 0 from a positive value.
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["asset_class_cav_growth"] = (
        df.groupby("asset_class", sort=False)["cav"]
        .transform(_safe_pct_change)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Composition (per asset class)
# ─────────────────────────────────────────────────────────────────────────────

def add_cav_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each asset class's share of total CAV per month (0–1 range).
    Null CAV months contribute 0 to the total so that the denominator
    reflects only months with actual reported values.
    Adds column: cav_share.
    """
    df = df.copy()
    monthly_total = df.groupby("date")["cav"].transform("sum")
    df["cav_share"] = df["cav"] / monthly_total.where(monthly_total > 0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Adoption (per asset class)
# ─────────────────────────────────────────────────────────────────────────────

def add_holders_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index each asset class's holder count to 100 at its first non-null,
    positive month.
    Adds column: holders_index.
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["holders_index"] = (
        df.groupby("asset_class", sort=False)["holders"]
        .transform(_index_to_100)
    )
    return df


def add_monthly_holder_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month % change in holder count per asset class.
    Adds column: monthly_holder_growth.
    Returns -100% when holders fall to 0 from a positive value.
    Returns NaN when the previous month had 0 holders (undefined base).
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["monthly_holder_growth"] = (
        df.groupby("asset_class", sort=False)["holders"]
        .transform(_safe_pct_change)
    )
    return df


def add_holders_per_million_cav(df: pd.DataFrame) -> pd.DataFrame:
    """
    Holder count per $1 M of CAV — a normalised adoption density metric.
    Zero or null CAV produces NaN (not inf).
    Adds column: holders_per_million_cav.
    """
    df = df.copy()
    cav_in_millions = df["cav"].where(df["cav"] > 0) / 1_000_000.0
    df["holders_per_million_cav"] = df["holders"] / cav_in_millions
    return df


def add_avg_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average dollar value held per wallet (CAV / holders).
    Zero holders produces NaN.
    Adds column: avg_position.
    """
    df = df.copy()
    df["avg_position"] = df["cav"] / df["holders"].where(df["holders"] > 0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Liquidity (per asset class)
# ─────────────────────────────────────────────────────────────────────────────

def add_turnover_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly transfer volume / average monthly CAV.
    Zero or null CAV produces NaN.
    Adds column: turnover_ratio.
    """
    df = df.copy()
    df["turnover_ratio"] = df["volume"] / df["cav"].where(df["cav"] > 0)
    return df


def add_turnover_3m(df: pd.DataFrame) -> pd.DataFrame:
    """
    3-month rolling average of turnover_ratio per asset class.
    Requires turnover_ratio to already be present (call add_turnover_ratio first).
    Adds column: turnover_3m.
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["turnover_3m"] = (
        df.groupby("asset_class", sort=False)["turnover_ratio"]
        .transform(
            lambda s: s.rolling(ROLLING_3M, min_periods=MIN_PERIODS_ROLLING).mean()
        )
    )
    return df


def add_monthly_volume_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month % change in transfer volume per asset class.
    Adds column: monthly_volume_growth.
    Returns -100% when volume falls to 0 from a positive value.
    Returns NaN when the previous month had 0 volume (undefined base).
    """
    df = df.copy().sort_values(["asset_class", "date"])
    df["monthly_volume_growth"] = (
        df.groupby("asset_class", sort=False)["volume"]
        .transform(_safe_pct_change)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Market-level (Concentration) metrics
# ─────────────────────────────────────────────────────────────────────────────

def _hhi_for_group(group: pd.DataFrame) -> float:
    """Compute HHI for one date's cross-section of asset classes."""
    total = group["cav"].sum(skipna=True)
    if total <= 0:
        return np.nan
    shares = group["cav"].fillna(0.0) / total
    return float((shares ** 2).sum())


def _top5_share_for_group(group: pd.DataFrame) -> float:
    """Sum of the top-5 CAV shares for one date."""
    total = group["cav"].sum(skipna=True)
    if total <= 0:
        return np.nan
    shares = group["cav"].fillna(0.0) / total
    return float(shares.nlargest(5).sum())


def build_concentration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds market-level concentration metrics indexed by date.

    Returns columns:
        date, total_cav,
        monthly_cav_growth, rolling_3m_cav_growth, rolling_6m_cav_growth,
        hhi, top_5_share,
        asset_class_count, active_asset_class_count
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Total CAV per date ────────────────────────────────────────────────────
    totals = (
        df.groupby("date")["cav"]
        .sum(min_count=1)          # NaN if all CAV is null for that month
        .reset_index(name="total_cav")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ── Growth metrics ────────────────────────────────────────────────────────
    totals["monthly_cav_growth"] = _safe_pct_change(totals["total_cav"])
    totals["rolling_3m_cav_growth"] = (
        totals["monthly_cav_growth"]
        .rolling(ROLLING_3M, min_periods=MIN_PERIODS_ROLLING)
        .mean()
    )
    totals["rolling_6m_cav_growth"] = (
        totals["monthly_cav_growth"]
        .rolling(ROLLING_6M, min_periods=MIN_PERIODS_ROLLING)
        .mean()
    )

    # ── HHI ──────────────────────────────────────────────────────────────────
    hhi = (
        df.groupby("date")
        .apply(_hhi_for_group, include_groups=False)
        .reset_index(name="hhi")
    )

    # ── Top-5 share ───────────────────────────────────────────────────────────
    top5 = (
        df.groupby("date")
        .apply(_top5_share_for_group, include_groups=False)
        .reset_index(name="top_5_share")
    )

    # ── Asset class counts ────────────────────────────────────────────────────
    has_any_cav = df[df["cav"].notna() & (df["cav"] > 0)]
    ac_count = (
        has_any_cav
        .groupby("date")["asset_class"]
        .nunique()
        .reset_index(name="asset_class_count")
    )

    is_active = df[df["cav"].notna() & (df["cav"] > ACTIVE_CAV_THRESHOLD)]
    active_ac_count = (
        is_active
        .groupby("date")["asset_class"]
        .nunique()
        .reset_index(name="active_asset_class_count")
    )

    # ── Merge all ─────────────────────────────────────────────────────────────
    result = totals
    for frame in (hhi, top5, ac_count, active_ac_count):
        result = result.merge(frame, on="date", how="left")

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-asset-class orchestration
# ─────────────────────────────────────────────────────────────────────────────

_COMBINED_METRICS_COLS = [
    "date", "asset_class", "cav", "holders", "volume",
    "cav_share", "cav_index", "asset_class_cav_growth",
    "holders_index", "monthly_holder_growth", "holders_per_million_cav", "avg_position",
    "turnover_ratio", "turnover_3m", "monthly_volume_growth",
]


def build_asset_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all per-asset-class metric functions in dependency order and
    returns a DataFrame with exactly the columns listed in the spec.
    """
    out = df.copy()

    # Pillar 2 — Composition
    out = add_cav_share(out)

    # Pillar 1 — Growth
    out = add_cav_index(out)
    out = add_asset_class_cav_growth(out)

    # Pillar 3 — Adoption
    out = add_holders_index(out)
    out = add_monthly_holder_growth(out)
    out = add_holders_per_million_cav(out)
    out = add_avg_position(out)

    # Pillar 4 — Liquidity (turnover_3m depends on turnover_ratio)
    out = add_turnover_ratio(out)
    out = add_turnover_3m(out)
    out = add_monthly_volume_growth(out)

    return (
        out[_COMBINED_METRICS_COLS]
        .sort_values(["asset_class", "date"])
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_all_metrics(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all Phase-2 metrics from the cleaned combined monthly DataFrame.

    Returns
    -------
    combined_df      : per-(date, asset_class) metrics  → combined_metrics.csv
    concentration_df : per-date market-level metrics    → concentration_metrics.csv
    """
    combined_df = build_asset_class_metrics(df)
    concentration_df = build_concentration_metrics(df)
    return combined_df, concentration_df


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(
    combined_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
) -> tuple[str, str]:
    """
    Saves both metric DataFrames to CSV.

    Returns
    -------
    (combined_path, concentration_path)
    """
    os.makedirs(results_dir, exist_ok=True)
    combined_path = os.path.join(results_dir, "combined_metrics.csv")
    concentration_path = os.path.join(results_dir, "concentration_metrics.csv")
    combined_df.to_csv(combined_path, index=False)
    concentration_df.to_csv(concentration_path, index=False)
    return combined_path, concentration_path


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers (moved from data_processing.py)
# These preserve the richer analytical output for ad-hoc analysis notebooks.
# build_all_metrics() uses the simpler add_* functions above for the spec CSVs.
# ─────────────────────────────────────────────────────────────────────────────

def build_composition_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: adds cav_share column. Prefer add_cav_share for new code."""
    return add_cav_share(df)


def build_relative_growth_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pillar 1+3 combined: within-asset-class relative growth index.

    Each asset class is indexed to its own first valid monthly observation = 100
    for both CAV and holders.  Adds cav_index, holders_index,
    participation_ratio, baseline diagnostic columns, and quality flags.

    This is the richer analytical version that carries baseline metadata and
    quality flags.  The spec CSVs produced by build_all_metrics() only need
    cav_index and holders_index; use add_cav_index / add_holders_index there.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    analysis_start = pd.Timestamp(ANALYSIS_START_DATE)

    group_frames = []
    for ac_name, group in df.groupby("asset_class", sort=False):
        g = group.sort_values("date").copy()

        cav_valid = g[g["cav"].notna() & (g["cav"] > 0)]
        cav_base_val = cav_valid["cav"].iloc[0] if not cav_valid.empty else None
        cav_base_date = cav_valid["date"].iloc[0] if not cav_valid.empty else None

        holders_valid = g[g["holders"].notna() & (g["holders"] > 0)]
        holders_base_val = (
            holders_valid["holders"].iloc[0] if not holders_valid.empty else None
        )
        holders_base_date = (
            holders_valid["date"].iloc[0] if not holders_valid.empty else None
        )

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

        valid_dates = [d for d in [cav_base_date, holders_base_date] if d is not None]
        first_valid = min(valid_dates) if valid_dates else None
        g["late_start_flag"] = bool(
            first_valid is not None and first_valid > analysis_start
        )

        g["cav_index"] = np.nan
        g["cav_absolute_change_from_baseline"] = np.nan
        if cav_base_val is not None and cav_base_date is not None:
            mask = (g["date"] >= cav_base_date) & g["cav"].notna()
            g.loc[mask, "cav_index"] = g.loc[mask, "cav"] / cav_base_val * 100.0
            g.loc[mask, "cav_absolute_change_from_baseline"] = (
                g.loc[mask, "cav"] - cav_base_val
            )

        g["holders_index"] = np.nan
        g["holders_absolute_change_from_baseline"] = np.nan
        if holders_base_val is not None and holders_base_date is not None:
            mask = (g["date"] >= holders_base_date) & g["holders"].notna()
            g.loc[mask, "holders_index"] = (
                g.loc[mask, "holders"] / holders_base_val * 100.0
            )
            g.loc[mask, "holders_absolute_change_from_baseline"] = (
                g.loc[mask, "holders"] - holders_base_val
            )

        group_frames.append(g)

    if not group_frames:
        return df

    out = pd.concat(group_frames, ignore_index=True)

    both_valid = (
        out["cav_index"].notna() & out["holders_index"].notna()
        & (out["cav_index"] > 0) & (out["holders_index"] > 0)
    )
    out["participation_ratio"] = np.nan
    out.loc[both_valid, "participation_ratio"] = (
        out.loc[both_valid, "holders_index"] / out.loc[both_valid, "cav_index"]
    )

    return out


def build_adoption_index(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for build_relative_growth_index."""
    return build_relative_growth_index(df)


def build_avg_position_size(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: adds avg_position column. Prefer add_avg_position for new code."""
    return add_avg_position(df)


def build_turnover_ratio_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy: adds turnover_ratio and turnover_3m columns.
    Prefer add_turnover_ratio + add_turnover_3m for new code.
    """
    out = add_turnover_ratio(df)
    return add_turnover_3m(out)


def validate_growth_index(df: pd.DataFrame) -> list[str]:
    """
    Validates DataFrames produced by build_relative_growth_index.
    Returns a list of error strings; empty list means all checks passed.
    """
    errors = []

    flag_cols = ["low_cav_baseline_flag", "low_holders_baseline_flag", "late_start_flag"]
    for col in flag_cols:
        if col not in df.columns:
            errors.append(f"Missing baseline flag column: '{col}'")
        elif not pd.api.types.is_bool_dtype(df[col]):
            errors.append(
                f"Column '{col}' should be boolean (dtype: {df[col].dtype})"
            )

    if "cav_index" in df.columns and "cav_baseline" in df.columns:
        bad = (
            df["cav_index"].notna()
            & df["cav_baseline"].notna()
            & (df["cav_baseline"] <= 0)
        )
        if bad.any():
            errors.append(
                f"{bad.sum()} row(s) have cav_index calculated from a non-positive baseline"
            )

    if "holders_index" in df.columns and "holders_baseline" in df.columns:
        bad = (
            df["holders_index"].notna()
            & df["holders_baseline"].notna()
            & (df["holders_baseline"] <= 0)
        )
        if bad.any():
            errors.append(
                f"{bad.sum()} row(s) have holders_index calculated from a non-positive baseline"
            )

    if all(c in df.columns for c in ["cav_index", "cav_baseline_date", "date"]):
        pre_base = df.merge(
            df.groupby("asset_class")["cav_baseline_date"]
            .first()
            .reset_index(),
            on="asset_class",
            suffixes=("", "_min"),
        )
        before_base = pre_base["date"] < pre_base["cav_baseline_date_min"]
        leaked = before_base & pre_base["cav_index"].notna()
        if leaked.any():
            errors.append(
                f"{leaked.sum()} row(s) have cav_index set before the baseline date"
            )

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
