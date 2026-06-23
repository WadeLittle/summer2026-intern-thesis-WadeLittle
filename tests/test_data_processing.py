# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Unit tests for src/bronze/data_processing.py

Run from the repo root:
    pytest tests/test_data_processing.py -v

Covers four scenarios per the spec:
  1. Normal valid data — index and flags behave correctly.
  2. Late-starting asset class — late_start_flag is True; rows before baseline are NaN.
  3. Low holder baseline — low_holders_baseline_flag is True.
  4. Missing/zero baseline — no index is calculated; all index columns remain NaN.
"""

import numpy as np
import pandas as pd
import pytest

from src.bronze.data_processing import validate_dataset
from src.bronze.metrics import (
    build_relative_growth_index,
    validate_growth_index,
    MIN_BASELINE_CAV,
    MIN_BASELINE_HOLDERS,
)
from src.config import ANALYSIS_START_DATE


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_df(asset_class, dates, cavs, holders, volumes=None):
    """Build a minimal combined-style DataFrame for one asset class."""
    n = len(dates)
    return pd.DataFrame({
        "date":        pd.to_datetime(dates),
        "asset_class": asset_class,
        "cav":         cavs,
        "holders":     holders if holders is not None else [0.0] * n,
        "volume":      volumes if volumes is not None else [0.0] * n,
    })


# ─────────────────────────────────────────────
# Scenario 1: Normal valid data
# ─────────────────────────────────────────────

class TestNormalValidData:
    def setup_method(self):
        dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        self.df = _make_df(
            "US Treasury Debt",
            dates,
            cavs=[10_000_000.0, 20_000_000.0, 30_000_000.0],
            holders=[100.0, 200.0, 300.0],
        )
        self.out = build_relative_growth_index(self.df)

    def test_cav_index_at_baseline_is_100(self):
        first = self.out.sort_values("date").iloc[0]
        assert first["cav_index"] == pytest.approx(100.0)

    def test_cav_index_grows_proportionally(self):
        rows = self.out.sort_values("date")
        assert rows["cav_index"].iloc[1] == pytest.approx(200.0)
        assert rows["cav_index"].iloc[2] == pytest.approx(300.0)

    def test_holders_index_at_baseline_is_100(self):
        first = self.out.sort_values("date").iloc[0]
        assert first["holders_index"] == pytest.approx(100.0)

    def test_participation_ratio_is_valid(self):
        rows = self.out.sort_values("date")
        assert rows["participation_ratio"].notna().all()
        # holders and CAV grow at the same rate, so ratio stays 1.0
        assert np.allclose(rows["participation_ratio"], 1.0)

    def test_low_flags_are_false(self):
        assert not self.out["low_cav_baseline_flag"].any()
        assert not self.out["low_holders_baseline_flag"].any()

    def test_late_start_flag_is_false(self):
        assert not self.out["late_start_flag"].any()

    def test_absolute_change_columns_exist(self):
        assert "cav_absolute_change_from_baseline" in self.out.columns
        assert "holders_absolute_change_from_baseline" in self.out.columns

    def test_baseline_columns_populated(self):
        assert self.out["cav_baseline"].notna().all()
        assert self.out["holders_baseline"].notna().all()
        assert self.out["cav_baseline_date"].notna().all()
        assert self.out["holders_baseline_date"].notna().all()

    def test_validate_growth_index_passes(self):
        errors = validate_growth_index(self.out)
        assert errors == [], f"Unexpected validation errors: {errors}"


# ─────────────────────────────────────────────
# Scenario 2: Late-starting asset class
# ─────────────────────────────────────────────

class TestLateStart:
    def setup_method(self):
        # ANALYSIS_START_DATE is 2023-01-01; this class starts 2024-01-01
        dates_early = ["2023-01-01", "2023-06-01"]
        dates_late  = ["2024-01-01", "2024-06-01"]

        early = _make_df("US Treasury Debt", dates_early,
                         cavs=[10_000_000.0, 11_000_000.0], holders=[100.0, 110.0])
        late  = _make_df("Stocks", dates_late,
                         cavs=[5_000_000.0, 8_000_000.0], holders=[50.0, 80.0])
        self.df  = pd.concat([early, late], ignore_index=True)
        self.out = build_relative_growth_index(self.df)

    def test_late_start_flag_true_for_late_class(self):
        late_rows = self.out[self.out["asset_class"] == "Stocks"]
        assert late_rows["late_start_flag"].all()

    def test_late_start_flag_false_for_early_class(self):
        early_rows = self.out[self.out["asset_class"] == "US Treasury Debt"]
        assert not early_rows["late_start_flag"].any()

    def test_no_index_before_baseline_for_late_class(self):
        # Stocks only has rows from 2024 onward, so all should have index set
        late_rows = self.out[self.out["asset_class"] == "Stocks"].sort_values("date")
        assert late_rows["cav_index"].notna().all()
        assert late_rows["cav_index"].iloc[0] == pytest.approx(100.0)


# ─────────────────────────────────────────────
# Scenario 3: Low holder baseline
# ─────────────────────────────────────────────

class TestLowHolderBaseline:
    def setup_method(self):
        dates = ["2023-01-01", "2023-02-01"]
        self.df = _make_df(
            "Venture Capital",
            dates,
            cavs=[2_000_000.0, 4_000_000.0],
            holders=[3.0, 6.0],  # below MIN_BASELINE_HOLDERS (10)
        )
        self.out = build_relative_growth_index(self.df)

    def test_low_holders_baseline_flag_is_true(self):
        assert self.out["low_holders_baseline_flag"].all()

    def test_low_cav_baseline_flag_is_false(self):
        # CAV baseline is $2M, above MIN_BASELINE_CAV ($1M)
        assert not self.out["low_cav_baseline_flag"].any()

    def test_index_still_computed_despite_low_baseline(self):
        # We still compute the index; the flag is a warning, not a suppressor
        assert self.out["holders_index"].notna().all()
        assert self.out["holders_index"].iloc[0] == pytest.approx(100.0)


# ─────────────────────────────────────────────
# Scenario 4: Missing / zero baseline
# ─────────────────────────────────────────────

class TestMissingOrZeroBaseline:
    def setup_method(self):
        dates = ["2023-01-01", "2023-02-01", "2023-03-01"]
        # All CAV values are null/zero — no valid baseline exists
        self.df = _make_df(
            "Private Equity",
            dates,
            cavs=[None, None, None],
            holders=[0.0, 0.0, 0.0],
        )
        self.out = build_relative_growth_index(self.df)

    def test_cav_index_all_nan_when_no_baseline(self):
        assert self.out["cav_index"].isna().all()

    def test_holders_index_all_nan_when_no_holders(self):
        assert self.out["holders_index"].isna().all()

    def test_participation_ratio_all_nan(self):
        assert self.out["participation_ratio"].isna().all()

    def test_baseline_columns_are_null(self):
        assert self.out["cav_baseline"].isna().all()
        assert self.out["holders_baseline"].isna().all()

    def test_validate_growth_index_passes(self):
        # No index was calculated, so there should be no violations
        errors = validate_growth_index(self.out)
        assert errors == [], f"Unexpected validation errors: {errors}"


# ─────────────────────────────────────────────
# validate_dataset tests
# ─────────────────────────────────────────────

class TestValidateDataset:
    def _base_df(self):
        return pd.DataFrame({
            "date":        [pd.Timestamp("2023-01-01")],
            "asset_class": ["US Treasury Debt"],
            "cav":         [1_000_000.0],
            "holders":     [100.0],
            "volume":      [50_000.0],
        })

    def test_passes_clean_df(self):
        assert validate_dataset(self._base_df()) == []

    def test_flags_missing_column(self):
        df = self._base_df().drop(columns=["volume"])
        errors = validate_dataset(df)
        assert any("volume" in e for e in errors)

    def test_flags_negative_cav(self):
        df = self._base_df()
        df.loc[0, "cav"] = -500.0
        errors = validate_dataset(df)
        assert any("negative CAV" in e for e in errors)

    def test_flags_date_before_analysis_start(self):
        df = self._base_df()
        df.loc[0, "date"] = pd.Timestamp("2020-01-01")
        errors = validate_dataset(df)
        assert any(ANALYSIS_START_DATE in e for e in errors)
