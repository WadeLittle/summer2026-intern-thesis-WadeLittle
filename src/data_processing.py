# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Transforms raw rwa.xyz API responses into a single cleaned,
long-format DataFrame ready for analysis.

Pipeline: raw JSON (per measure) -> long DataFrame (per measure)
          -> merged combined DataFrame -> cleaned/windowed DataFrame
"""

import pandas

from config import ANALYSIS_START_DATE, ASSET_CLASSES_IN_SCOPE


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
        return pandas.DataFrame(columns=["date", "asset_class", value_name])

    dataframe = pandas.DataFrame(rows)
    dataframe["date"] = pandas.to_datetime(dataframe["date"])
    return dataframe


def _resample_stock(df, value_col):
    """
    Collapses daily stock-mode data to monthly by taking the last observed
    value per asset class per month (end-of-month snapshot).
    Used for holders — a count where the end-of-period value is the right
    representation of participation at that point in time.

    Note: CAV previously used this function but now uses _resample_monthly_avg,
    which is more stable as a market-size denominator.
    """
    return (
        df.groupby(["asset_class", pandas.Grouper(key="date", freq="MS")])[value_col]
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
        df.groupby(["asset_class", pandas.Grouper(key="date", freq="MS")])[value_col]
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
        df.groupby(["asset_class", pandas.Grouper(key="date", freq="MS")])[value_col]
        .sum()
        .reset_index()
    )


def build_combined_dataset(aum_data, holders_data, volume_data):
    """
    Merges CAV, holders, and volume into one long-format monthly DataFrame,
    applies the analysis date window, drops the partial current month,
    and fills missing activity values with 0.

    Daily API data is resampled here to monthly using the correct strategy
    per measure type (last-of-month for stock, sum for flow).

    Returns columns: date, asset_class, cav, holders, volume
    """
    cav_long = extract_long_df(aum_data, "cav")
    cav_dataframe = _resample_monthly_avg(cav_long, "cav")
    holders_dataframe = _resample_stock(extract_long_df(holders_data, "holders"), "holders")
    volume_dataframe = _resample_flow(extract_long_df(volume_data, "volume"), "volume")

    # Outer merge: keep a (date, asset_class) row even if only one
    # measure has data for it -- we decide what to do with gaps next.
    combined = cav_dataframe.merge(holders_dataframe, on=["date", "asset_class"], how="outer")
    combined = combined.merge(volume_dataframe, on=["date", "asset_class"], how="outer")

    # Scope to traditional RWA asset classes (see config.py for rationale)
    combined = combined[combined["asset_class"].isin(ASSET_CLASSES_IN_SCOPE)]

    # Apply analysis window (see config.py for rationale on this date)
    combined = combined[combined["date"] >= ANALYSIS_START_DATE]

    # Drop the most recent month -- rwa.xyz data for the current month
    # is a partial-month snapshot, not a complete period, and would
    # distort trend/growth calculations.
    latest_month = combined["date"].max()
    combined = combined[combined["date"] < latest_month]

    # Missing holders/volume for a (date, asset_class) means no
    # token activity was recorded that month -- a real zero, not
    # missing data. CAV gaps are left as NaN; you can't assume an
    # asset class had zero CAV just because this measure's response
    # didn't include it for that month.
    combined["holders"] = combined["holders"].fillna(0)
    combined["volume"] = combined["volume"].fillna(0)

    combined = combined.sort_values(["asset_class", "date"]).reset_index(drop=True)
    return combined


def build_composition_shares(df):
    """
    Pillar 1: Each asset class's % share of total CAV per month.
    Adds 'cav_share' column (0–1).
    """
    df = df.copy()
    monthly_total = df.groupby("date")["cav"].transform("sum")
    df["cav_share"] = df["cav"] / monthly_total
    return df


def build_adoption_index(df):
    """
    Pillar 2: Index CAV and holders to ANALYSIS_START_DATE = 100 per asset class.
    Adds cav_index, holders_index, and participation_ratio columns.
    participation_ratio > 1: holder growth outpaces value growth (broadening participation).
    participation_ratio < 1: value growing faster than participation (concentration).
    """
    df = df.copy()
    base = (
        df[df["date"] == pandas.Timestamp(ANALYSIS_START_DATE)]
        [["asset_class", "cav", "holders"]]
        .rename(columns={"cav": "cav_base", "holders": "holders_base"})
    )
    df = df.merge(base, on="asset_class", how="left")
    df["cav_index"] = df["cav"] / df["cav_base"] * 100
    df["holders_index"] = df["holders"] / df["holders_base"] * 100
    df["participation_ratio"] = df["holders_index"] / df["cav_index"]
    df = df.drop(columns=["cav_base", "holders_base"])
    return df


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

