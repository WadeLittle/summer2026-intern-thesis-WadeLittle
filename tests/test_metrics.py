# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Unit tests for src/bronze/metrics.py

All tests use synthetic DataFrames — no API calls, no file I/O.

Coverage by pillar:
  Pillar 1 — Growth  : add_cav_index, add_asset_class_cav_growth
  Pillar 2 — Composition: add_cav_share, build_concentration_metrics (hhi, top_5_share, counts)
  Pillar 3 — Adoption: add_holders_index, add_monthly_holder_growth,
                       add_holders_per_million_cav, add_avg_position
  Pillar 4 — Liquidity: add_turnover_ratio, add_turnover_3m, add_monthly_volume_growth
  Orchestration       : build_asset_class_metrics, build_all_metrics output columns
"""

import numpy as np
import pandas as pd
import pytest

from src.bronze.metrics import (
    add_asset_class_cav_growth,
    add_avg_position,
    add_cav_index,
    add_cav_share,
    add_holders_index,
    add_holders_per_million_cav,
    add_monthly_holder_growth,
    add_monthly_volume_growth,
    add_turnover_3m,
    add_turnover_ratio,
    build_all_metrics,
    build_asset_class_metrics,
    build_concentration_metrics,
)
from src.bronze.metrics_config import ACTIVE_CAV_THRESHOLD, ROLLING_3M


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(asset_class, dates, cavs, holders=None, volumes=None):
    """Build a minimal combined-style DataFrame for one asset class."""
    n = len(dates)
    return pd.DataFrame({
        "date":        pd.to_datetime(dates),
        "asset_class": asset_class,
        "cav":         cavs,
        "holders":     holders if holders is not None else [100.0] * n,
        "volume":      volumes if volumes is not None else [0.0] * n,
    })


def _two_class_df():
    """Two asset classes, three months each, clean data."""
    a = _make_df(
        "US Treasury Debt",
        ["2023-01-01", "2023-02-01", "2023-03-01"],
        cavs=[10_000_000, 20_000_000, 30_000_000],
        holders=[100, 200, 300],
        volumes=[500_000, 1_000_000, 1_500_000],
    )
    b = _make_df(
        "Stocks",
        ["2023-01-01", "2023-02-01", "2023-03-01"],
        cavs=[10_000_000, 10_000_000, 10_000_000],
        holders=[50, 50, 50],
        volumes=[100_000, 200_000, 300_000],
    )
    return pd.concat([a, b], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 1 — Growth
# ─────────────────────────────────────────────────────────────────────────────

class TestAddCavIndex:
    def test_baseline_month_is_100(self):
        df = _make_df("US Treasury Debt", ["2023-01-01", "2023-02-01"],
                      cavs=[5_000_000, 10_000_000])
        out = add_cav_index(df).sort_values("date")
        assert out["cav_index"].iloc[0] == pytest.approx(100.0)

    def test_doubles_to_200(self):
        df = _make_df("US Treasury Debt", ["2023-01-01", "2023-02-01"],
                      cavs=[5_000_000, 10_000_000])
        out = add_cav_index(df).sort_values("date")
        assert out["cav_index"].iloc[1] == pytest.approx(200.0)

    def test_nan_before_first_valid_month(self):
        df = _make_df("US Treasury Debt",
                      ["2023-01-01", "2023-02-01", "2023-03-01"],
                      cavs=[None, 5_000_000, 10_000_000])
        out = add_cav_index(df).sort_values("date")
        assert pd.isna(out["cav_index"].iloc[0])
        assert out["cav_index"].iloc[1] == pytest.approx(100.0)

    def test_all_nan_when_no_positive_cav(self):
        df = _make_df("Venture Capital", ["2023-01-01", "2023-02-01"],
                      cavs=[None, None])
        out = add_cav_index(df)
        assert out["cav_index"].isna().all()

    def test_zero_cav_not_used_as_baseline(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01", "2023-03-01"],
                      cavs=[0, 5_000_000, 10_000_000])
        out = add_cav_index(df).sort_values("date")
        # First non-zero month is idx=1 → should be 100
        assert pd.isna(out["cav_index"].iloc[0])
        assert out["cav_index"].iloc[1] == pytest.approx(100.0)

    def test_independent_per_asset_class(self):
        df = _two_class_df()
        out = add_cav_index(df)
        for ac in ["US Treasury Debt", "Stocks"]:
            first = out[out["asset_class"] == ac].sort_values("date").iloc[0]
            assert first["cav_index"] == pytest.approx(100.0)


class TestAddAssetClassCavGrowth:
    def test_first_row_is_nan(self):
        df = _make_df("US Treasury Debt", ["2023-01-01", "2023-02-01"],
                      cavs=[10_000_000, 20_000_000])
        out = add_asset_class_cav_growth(df).sort_values("date")
        assert pd.isna(out["asset_class_cav_growth"].iloc[0])

    def test_100_percent_growth(self):
        df = _make_df("US Treasury Debt", ["2023-01-01", "2023-02-01"],
                      cavs=[10_000_000, 20_000_000])
        out = add_asset_class_cav_growth(df).sort_values("date")
        assert out["asset_class_cav_growth"].iloc[1] == pytest.approx(100.0)

    def test_negative_growth(self):
        df = _make_df("Commodities", ["2023-01-01", "2023-02-01"],
                      cavs=[20_000_000, 10_000_000])
        out = add_asset_class_cav_growth(df).sort_values("date")
        assert out["asset_class_cav_growth"].iloc[1] == pytest.approx(-50.0)

    def test_independent_per_asset_class(self):
        df = _two_class_df()
        out = add_asset_class_cav_growth(df)
        for ac in out["asset_class"].unique():
            rows = out[out["asset_class"] == ac].sort_values("date")
            assert pd.isna(rows["asset_class_cav_growth"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Composition
# ─────────────────────────────────────────────────────────────────────────────

class TestAddCavShare:
    def test_shares_sum_to_1_per_date(self):
        df = _two_class_df()
        out = add_cav_share(df)
        for date, grp in out.groupby("date"):
            total = grp["cav_share"].sum()
            assert total == pytest.approx(1.0), f"Share sum != 1 for {date}: {total}"

    def test_equal_cav_gives_equal_shares(self):
        df = _two_class_df()  # both classes have 10M in Jan
        out = add_cav_share(df)
        jan = out[out["date"] == pd.Timestamp("2023-01-01")]
        assert jan["cav_share"].iloc[0] == pytest.approx(0.5)
        assert jan["cav_share"].iloc[1] == pytest.approx(0.5)

    def test_cav_share_null_when_total_zero(self):
        df = _make_df("US Treasury Debt", ["2023-01-01"], cavs=[0])
        out = add_cav_share(df)
        assert out["cav_share"].isna().all()

    def test_share_range_0_to_1(self):
        df = _two_class_df()
        out = add_cav_share(df)
        assert (out["cav_share"].dropna() >= 0).all()
        assert (out["cav_share"].dropna() <= 1).all()


class TestBuildConcentrationMetrics:
    def setup_method(self):
        self.df = _two_class_df()
        self.conc = build_concentration_metrics(self.df)

    def test_total_cav_correct(self):
        jan = self.conc[self.conc["date"] == pd.Timestamp("2023-01-01")].iloc[0]
        assert jan["total_cav"] == pytest.approx(20_000_000)

    def test_monthly_cav_growth_first_row_nan(self):
        sorted_conc = self.conc.sort_values("date")
        assert pd.isna(sorted_conc["monthly_cav_growth"].iloc[0])

    def test_monthly_cav_growth_correct(self):
        # Jan total = 20M, Feb total = 30M → 50% growth
        feb = self.conc[self.conc["date"] == pd.Timestamp("2023-02-01")].iloc[0]
        assert feb["monthly_cav_growth"] == pytest.approx(50.0)

    def test_hhi_between_0_and_1(self):
        assert (self.conc["hhi"].dropna() > 0).all()
        assert (self.conc["hhi"].dropna() <= 1).all()

    def test_hhi_equal_shares(self):
        # In Jan both classes have equal CAV → HHI = 0.5^2 + 0.5^2 = 0.5
        jan = self.conc[self.conc["date"] == pd.Timestamp("2023-01-01")].iloc[0]
        assert jan["hhi"] == pytest.approx(0.5)

    def test_top_5_share_lte_1(self):
        assert (self.conc["top_5_share"].dropna() <= 1.0 + 1e-9).all()

    def test_top_5_share_equals_total_with_two_classes(self):
        # Only 2 classes — top-5 share should equal 1.0 for every date
        assert self.conc["top_5_share"].apply(lambda v: pytest.approx(1.0) == v).all()

    def test_asset_class_count(self):
        assert (self.conc["asset_class_count"] == 2).all()

    def test_active_asset_class_count(self):
        # Both classes have CAV well above ACTIVE_CAV_THRESHOLD ($1M)
        assert (self.conc["active_asset_class_count"] == 2).all()

    def test_rolling_3m_cav_growth_present(self):
        assert "rolling_3m_cav_growth" in self.conc.columns

    def test_rolling_6m_cav_growth_present(self):
        assert "rolling_6m_cav_growth" in self.conc.columns

    def test_output_columns(self):
        expected = {
            "date", "total_cav",
            "monthly_cav_growth", "rolling_3m_cav_growth", "rolling_6m_cav_growth",
            "hhi", "top_5_share",
            "asset_class_count", "active_asset_class_count",
        }
        assert expected.issubset(set(self.conc.columns))


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Adoption
# ─────────────────────────────────────────────────────────────────────────────

class TestAddHoldersIndex:
    def test_baseline_is_100(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[5_000_000, 5_000_000], holders=[200, 400])
        out = add_holders_index(df).sort_values("date")
        assert out["holders_index"].iloc[0] == pytest.approx(100.0)

    def test_doubles_to_200(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[5_000_000, 5_000_000], holders=[200, 400])
        out = add_holders_index(df).sort_values("date")
        assert out["holders_index"].iloc[1] == pytest.approx(200.0)

    def test_zero_holders_not_baseline(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01", "2023-03-01"],
                      cavs=[1_000_000] * 3, holders=[0, 100, 200])
        out = add_holders_index(df).sort_values("date")
        assert pd.isna(out["holders_index"].iloc[0])
        assert out["holders_index"].iloc[1] == pytest.approx(100.0)

    def test_all_nan_when_no_positive_holders(self):
        df = _make_df("Real Estate", ["2023-01-01"], cavs=[1_000_000], holders=[0])
        out = add_holders_index(df)
        assert out["holders_index"].isna().all()


class TestAddMonthlyHolderGrowth:
    def test_first_row_nan(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[1_000_000] * 2, holders=[100, 200])
        out = add_monthly_holder_growth(df).sort_values("date")
        assert pd.isna(out["monthly_holder_growth"].iloc[0])

    def test_100_percent_growth(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[1_000_000] * 2, holders=[100, 200])
        out = add_monthly_holder_growth(df).sort_values("date")
        assert out["monthly_holder_growth"].iloc[1] == pytest.approx(100.0)


class TestAddHoldersPerMillionCav:
    def test_basic_calculation(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[2_000_000], holders=[100])
        out = add_holders_per_million_cav(df)
        # 100 / (2_000_000 / 1_000_000) = 100 / 2 = 50
        assert out["holders_per_million_cav"].iloc[0] == pytest.approx(50.0)

    def test_zero_cav_gives_nan(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[0], holders=[100])
        out = add_holders_per_million_cav(df)
        assert pd.isna(out["holders_per_million_cav"].iloc[0])

    def test_null_cav_gives_nan(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[None], holders=[100])
        out = add_holders_per_million_cav(df)
        assert pd.isna(out["holders_per_million_cav"].iloc[0])


class TestAddAvgPosition:
    def test_basic_calculation(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[10_000_000], holders=[100])
        out = add_avg_position(df)
        assert out["avg_position"].iloc[0] == pytest.approx(100_000.0)

    def test_zero_holders_gives_nan(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[10_000_000], holders=[0])
        out = add_avg_position(df)
        assert pd.isna(out["avg_position"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Liquidity
# ─────────────────────────────────────────────────────────────────────────────

class TestAddTurnoverRatio:
    def test_basic_calculation(self):
        df = _make_df("Stocks", ["2023-01-01"],
                      cavs=[10_000_000], volumes=[1_000_000])
        out = add_turnover_ratio(df)
        assert out["turnover_ratio"].iloc[0] == pytest.approx(0.1)

    def test_zero_cav_gives_nan(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[0], volumes=[1_000_000])
        out = add_turnover_ratio(df)
        assert pd.isna(out["turnover_ratio"].iloc[0])

    def test_null_cav_gives_nan(self):
        df = _make_df("Stocks", ["2023-01-01"], cavs=[None], volumes=[500_000])
        out = add_turnover_ratio(df)
        assert pd.isna(out["turnover_ratio"].iloc[0])


class TestAddTurnover3m:
    def setup_method(self):
        df = _make_df(
            "Stocks",
            ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
            cavs=[10_000_000] * 4,
            volumes=[1_000_000, 2_000_000, 3_000_000, 4_000_000],
        )
        df = add_turnover_ratio(df)
        self.out = add_turnover_3m(df).sort_values("date")

    def test_first_month_equals_itself(self):
        # min_periods=1 so first row gets a value equal to its own turnover_ratio
        assert self.out["turnover_3m"].iloc[0] == pytest.approx(
            self.out["turnover_ratio"].iloc[0]
        )

    def test_third_month_is_3_month_average(self):
        # turnover ratios: 0.1, 0.2, 0.3 → mean = 0.2
        assert self.out["turnover_3m"].iloc[2] == pytest.approx(0.2)

    def test_rolling_window_slides(self):
        # Month 4: turnover ratios 0.2, 0.3, 0.4 → mean = 0.3
        assert self.out["turnover_3m"].iloc[3] == pytest.approx(0.3)


class TestAddMonthlyVolumeGrowth:
    def test_first_row_nan(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[1_000_000] * 2, volumes=[1_000_000, 2_000_000])
        out = add_monthly_volume_growth(df).sort_values("date")
        assert pd.isna(out["monthly_volume_growth"].iloc[0])

    def test_100_percent_growth(self):
        df = _make_df("Stocks", ["2023-01-01", "2023-02-01"],
                      cavs=[1_000_000] * 2, volumes=[1_000_000, 2_000_000])
        out = add_monthly_volume_growth(df).sort_values("date")
        assert out["monthly_volume_growth"].iloc[1] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — build_asset_class_metrics & build_all_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAssetClassMetrics:
    EXPECTED_COLS = [
        "date", "asset_class", "cav", "holders", "volume",
        "cav_share", "cav_index", "asset_class_cav_growth",
        "holders_index", "monthly_holder_growth", "holders_per_million_cav", "avg_position",
        "turnover_ratio", "turnover_3m", "monthly_volume_growth",
    ]

    def setup_method(self):
        self.df = _two_class_df()
        self.out = build_asset_class_metrics(self.df)

    def test_output_columns_match_spec(self):
        assert list(self.out.columns) == self.EXPECTED_COLS

    def test_no_extra_columns(self):
        assert set(self.out.columns) == set(self.EXPECTED_COLS)

    def test_row_count_preserved(self):
        assert len(self.out) == len(self.df)

    def test_cav_share_sums_to_1_per_date(self):
        for date, grp in self.out.groupby("date"):
            assert grp["cav_share"].sum() == pytest.approx(1.0)

    def test_cav_index_baseline_is_100(self):
        for ac, grp in self.out.groupby("asset_class"):
            first_valid = grp.sort_values("date").dropna(subset=["cav_index"]).iloc[0]
            assert first_valid["cav_index"] == pytest.approx(100.0)

    def test_no_inf_values(self):
        numeric = self.out.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Inf values found in output"


class TestBuildAllMetrics:
    def setup_method(self):
        self.df = _two_class_df()
        self.combined, self.concentration = build_all_metrics(self.df)

    def test_returns_two_dataframes(self):
        assert isinstance(self.combined, pd.DataFrame)
        assert isinstance(self.concentration, pd.DataFrame)

    def test_combined_has_expected_columns(self):
        expected = {
            "date", "asset_class", "cav", "holders", "volume",
            "cav_share", "cav_index", "asset_class_cav_growth",
            "holders_index", "monthly_holder_growth", "holders_per_million_cav",
            "avg_position", "turnover_ratio", "turnover_3m", "monthly_volume_growth",
        }
        assert expected == set(self.combined.columns)

    def test_concentration_has_expected_columns(self):
        expected = {
            "date", "total_cav",
            "monthly_cav_growth", "rolling_3m_cav_growth", "rolling_6m_cav_growth",
            "hhi", "top_5_share",
            "asset_class_count", "active_asset_class_count",
        }
        assert expected == set(self.concentration.columns)

    def test_concentration_one_row_per_date(self):
        assert self.concentration["date"].nunique() == len(self.concentration)

    def test_combined_no_division_by_zero_with_zero_cav(self):
        df = _make_df("Stocks", ["2023-01-01"],
                      cavs=[0], holders=[0], volumes=[0])
        combined, _ = build_all_metrics(df)
        numeric = combined.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()

    def test_combined_no_division_by_zero_with_null_cav(self):
        df = _make_df("Stocks", ["2023-01-01"],
                      cavs=[None], holders=[0], volumes=[0])
        combined, _ = build_all_metrics(df)
        numeric = combined.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_row_dataset(self):
        df = _make_df("US Treasury Debt", ["2023-01-01"],
                      cavs=[5_000_000], holders=[50], volumes=[100_000])
        combined, concentration = build_all_metrics(df)
        assert len(combined) == 1
        assert len(concentration) == 1

    def test_single_asset_class(self):
        df = _make_df("US Treasury Debt",
                      ["2023-01-01", "2023-02-01"],
                      cavs=[5_000_000, 10_000_000], holders=[50, 100])
        combined, concentration = build_all_metrics(df)
        # Single class owns 100% of CAV share
        assert np.allclose(combined["cav_share"].dropna(), 1.0)

    def test_cav_with_gap_in_middle(self):
        df = _make_df("Commodities",
                      ["2023-01-01", "2023-02-01", "2023-03-01"],
                      cavs=[5_000_000, None, 15_000_000])
        out = add_cav_index(df).sort_values("date")
        # Baseline at month 1 = 100, month 2 gap stays NaN, month 3 = 300
        assert out["cav_index"].iloc[0] == pytest.approx(100.0)
        assert pd.isna(out["cav_index"].iloc[1])
        assert out["cav_index"].iloc[2] == pytest.approx(300.0)

    def test_active_asset_class_count_excludes_below_threshold(self):
        small = _make_df("Venture Capital", ["2023-01-01"],
                         cavs=[500_000])  # below $1M threshold
        large = _make_df("US Treasury Debt", ["2023-01-01"],
                         cavs=[50_000_000])
        df = pd.concat([small, large], ignore_index=True)
        _, concentration = build_all_metrics(df)
        assert concentration.iloc[0]["asset_class_count"] == 2
        assert concentration.iloc[0]["active_asset_class_count"] == 1
