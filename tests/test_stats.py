# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Unit tests for src/bronze/stats.py

All tests use synthetic DataFrames — no API calls, no file I/O.
Tests cover:
  Core stats  : _ols_trend, _ols_log_log, _spearman, _structural_break_ols
  Data helpers: _build_market_aggregates, _build_hhi_series, _repo_entry_idx
  Pillars 1-4 : output structure, key flags, insufficient-data handling
  Conclusion  : no causality language, produces non-empty text
  Serialization: _make_serializable handles NaN, inf, numpy types
  Entry point : run_analysis returns all required top-level keys
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.bronze.stats import (
    MIN_OBS_OLS,
    MIN_OBS_CORR,
    REPO_ENTRY_DATE,
    _ols_trend,
    _ols_log_log,
    _spearman,
    _structural_break_ols,
    _build_market_aggregates,
    _build_hhi_series,
    _make_serializable,
    pillar1_growth,
    pillar2_composition,
    pillar3_adoption,
    pillar4_liquidity,
    build_conclusion,
    run_analysis,
)
from src.config import REPO_CLASS


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data builders
# ─────────────────────────────────────────────────────────────────────────────

def _dates(n: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


def _make_combined(
    n_months: int = 20,
    asset_classes: list[str] | None = None,
    include_repo: bool = False,
    cav_growth_rate: float = 0.05,   # 5%/month
    holder_growth_rate: float = 0.03,
    volume_fraction: float = 0.10,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic combined_metrics-style DataFrame.
    Each asset class grows at constant compound rates; volume = fraction of CAV.
    """
    rng = np.random.default_rng(rng_seed)
    if asset_classes is None:
        asset_classes = ["US Treasury Debt", "Commodities", "Real Estate"]

    rows = []
    dates = _dates(n_months)

    for ac in asset_classes:
        base_cav     = rng.uniform(1e8, 5e8)
        base_holders = rng.integers(1000, 10000)

        for i, dt in enumerate(dates):
            cav     = base_cav     * ((1 + cav_growth_rate)     ** i)
            holders = base_holders * ((1 + holder_growth_rate)  ** i)
            volume  = cav * volume_fraction * rng.uniform(0.5, 1.5)
            rows.append({
                "date":                 dt,
                "asset_class":          ac,
                "cav":                  cav,
                "holders":              float(int(holders)),
                "volume":               volume,
                "cav_share":            np.nan,    # not needed for stats tests
                "cav_index":            np.nan,
                "asset_class_cav_growth": np.nan,
                "holders_index":        np.nan,
                "monthly_holder_growth": np.nan,
                "holders_per_million_cav": np.nan,
                "avg_position":         np.nan,
                "turnover_ratio":       volume / cav,
                "turnover_3m":          volume / cav,    # simplified (no rolling)
                "monthly_volume_growth": np.nan,
            })

    if include_repo:
        # Repo: large CAV, zero holders, zero volume, enters at REPO_ENTRY_DATE
        repo_dates = [d for d in dates if d >= REPO_ENTRY_DATE]
        for dt in repo_dates:
            rows.append({
                "date":        dt,
                "asset_class": REPO_CLASS,
                "cav":         2e11,      # 200B — dominates total CAV
                "holders":     0.0,
                "volume":      0.0,
                "cav_share":   np.nan,
                "cav_index":   np.nan,
                "asset_class_cav_growth": np.nan,
                "holders_index": np.nan,
                "monthly_holder_growth": np.nan,
                "holders_per_million_cav": 0.0,
                "avg_position": np.nan,
                "turnover_ratio": 0.0,
                "turnover_3m":    0.0,
                "monthly_volume_growth": np.nan,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_concentration(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Build a simplified concentration_metrics DataFrame from a combined DataFrame."""
    df = combined_df.copy()
    totals = df.groupby("date")["cav"].sum().reset_index(name="total_cav").sort_values("date")
    totals["monthly_cav_growth"] = totals["total_cav"].pct_change() * 100
    totals["rolling_3m_cav_growth"] = totals["monthly_cav_growth"].rolling(3, min_periods=1).mean()
    totals["rolling_6m_cav_growth"] = totals["monthly_cav_growth"].rolling(6, min_periods=1).mean()

    def _hhi(g):
        total = g["cav"].sum()
        if total <= 0:
            return np.nan
        shares = g["cav"].fillna(0) / total
        return float((shares ** 2).sum())

    hhi = df.groupby("date").apply(_hhi, include_groups=False).rename("hhi").reset_index()
    top5 = (
        df.groupby("date")
        .apply(lambda g: float(g.nlargest(5, "cav")["cav"].sum() / g["cav"].sum() if g["cav"].sum() > 0 else np.nan), include_groups=False)
        .rename("top_5_share")
        .reset_index()
    )
    active = (
        df[df["cav"] > 1_000_000]
        .groupby("date")["asset_class"]
        .nunique()
        .rename("active_asset_class_count")
        .reset_index()
    )

    conc = (
        totals
        .merge(hhi, on="date", how="left")
        .merge(top5, on="date", how="left")
        .merge(active, on="date", how="left")
    )
    conc["asset_class_count"] = conc["active_asset_class_count"]
    return conc.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# _ols_trend
# ─────────────────────────────────────────────────────────────────────────────

class TestOlsTrend:
    def test_significant_positive_slope(self):
        # y = 2*t + small noise → beta ≈ 2, p < 0.05
        t = np.arange(30, dtype=float)
        y = pd.Series(2.0 * t + np.random.default_rng(0).normal(0, 0.1, 30))
        r = _ols_trend(y)
        assert r["status"] == "ok"
        assert r["n"] == 30
        assert abs(r["beta"] - 2.0) < 0.5
        assert r["p_value"] < 0.05
        assert r["confidence_interval"][0] < r["beta"] < r["confidence_interval"][1]

    def test_flat_series_not_significant(self):
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(5.0, 3.0, 30))
        r = _ols_trend(y)
        assert r["status"] == "ok"
        assert r["p_value"] > 0.05

    def test_insufficient_data(self):
        y = pd.Series([1.0, 2.0, 3.0])   # n=3 < MIN_OBS_OLS
        r = _ols_trend(y)
        assert r["status"] == "insufficient_data"
        assert r["n"] == 3

    def test_exactly_min_obs(self):
        y = pd.Series(range(MIN_OBS_OLS), dtype=float)
        r = _ols_trend(y)
        assert r["status"] == "ok"
        assert r["n"] == MIN_OBS_OLS

    def test_nan_values_dropped(self):
        y = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
        r = _ols_trend(y)
        assert r["status"] == "ok"
        assert r["n"] == 6   # two NaN dropped


# ─────────────────────────────────────────────────────────────────────────────
# _ols_log_log
# ─────────────────────────────────────────────────────────────────────────────

class TestOlsLogLog:
    def test_unit_elastic(self):
        # y = x (elasticity = 1) with tiny noise
        rng = np.random.default_rng(10)
        x = pd.Series(np.exp(np.linspace(1, 5, 30)))
        y = pd.Series(x * np.exp(rng.normal(0, 0.01, 30)))
        r = _ols_log_log(y, x)
        assert r["status"] == "ok"
        assert abs(r["beta"] - 1.0) < 0.15

    def test_insufficient_data(self):
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([1.0, 2.0, 3.0])
        r = _ols_log_log(y, x)
        assert r["status"] == "insufficient_data"

    def test_non_positive_filtered_out(self):
        x = pd.Series([0, -1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = pd.Series([0, 1,  1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        r = _ols_log_log(y, x)
        # Only 8 positive pairs remain
        assert r["status"] == "ok"
        assert r["n"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# _spearman
# ─────────────────────────────────────────────────────────────────────────────

class TestSpearman:
    def test_positive_correlation(self):
        x = pd.Series(range(20), dtype=float)
        y = pd.Series(range(20), dtype=float)
        r = _spearman(x, y)
        assert r["status"] == "ok"
        assert r["rho"] > 0.9
        assert r["p_value"] < 0.01

    def test_no_correlation(self):
        rng = np.random.default_rng(5)
        x = pd.Series(rng.random(30))
        y = pd.Series(rng.random(30))
        r = _spearman(x, y)
        assert r["status"] == "ok"

    def test_insufficient_data(self):
        r = _spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
        assert r["status"] == "insufficient_data"

    def test_nan_dropped(self):
        x = pd.Series([np.nan, 1, 2, 3, 4, 5, 6])
        y = pd.Series([1, np.nan, 2, 3, 4, 5, 6])
        r = _spearman(x, y)
        assert r["status"] == "ok"
        assert r["n"] == 5   # two pairs with at least one NaN dropped


# ─────────────────────────────────────────────────────────────────────────────
# _structural_break_ols
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralBreak:
    def test_detects_acceleration(self):
        # Pre-break: slow growth; post-break: faster growth
        n = 30
        t = np.arange(n, dtype=float)
        break_at = 15
        y = np.where(t < break_at, t * 0.1, (t - break_at) * 0.5 + break_at * 0.1)
        r = _structural_break_ols(pd.Series(y), break_at)
        assert r["status"] == "ok"
        # beta_t_D should be positive (slope increased post-break)
        assert r["beta_t_D"] > 0

    def test_insufficient_data(self):
        y = pd.Series(range(10), dtype=float)
        r = _structural_break_ols(y, 5)
        assert r["status"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# _build_market_aggregates
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMarketAggregates:
    def test_no_repo(self):
        df = _make_combined(n_months=12, include_repo=False)
        mkt = _build_market_aggregates(df)
        assert "ex_repo_cav" in mkt.columns
        assert "repo_cav" in mkt.columns
        # repo_cav should be NaN when no repos in dataset
        assert mkt["repo_cav"].isna().all() or (mkt["repo_cav"].fillna(0) == 0).all()

    def test_repo_excluded_from_holders(self):
        # Even with repos at 0 holders, total holders = ex_repo_holders
        df = _make_combined(
            n_months=30,
            include_repo=True,
            asset_classes=["US Treasury Debt", "Commodities"],
        )
        mkt = _build_market_aggregates(df)
        # Confirm total holders are from ex-repo classes (repos add 0)
        ex_h = (
            df[df["asset_class"] != REPO_CLASS]
            .groupby("date")["holders"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        pd.testing.assert_series_equal(
            mkt["ex_repo_holders"].reset_index(drop=True),
            ex_h["holders"].reset_index(drop=True),
            check_names=False,
        )

    def test_repo_share_computed(self):
        df = _make_combined(n_months=30, include_repo=True,
                            asset_classes=["US Treasury Debt"])
        mkt = _build_market_aggregates(df)
        repo_rows = mkt[mkt["repo_present"]]
        assert not repo_rows.empty
        # Repo CAV is huge (~200B) vs small synthetic ex-repo → share > 90%
        assert (repo_rows["repo_share"] > 0.9).all()


# ─────────────────────────────────────────────────────────────────────────────
# _build_hhi_series
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHhiSeries:
    def test_monopoly(self):
        df = _make_combined(n_months=6, asset_classes=["US Treasury Debt"])
        hhi = _build_hhi_series(df, exclude_repo=True)
        assert (hhi.dropna() == 1.0).all()

    def test_two_equal_classes(self):
        # Two classes same CAV → HHI = 0.5
        dates = _dates(6)
        rows = []
        for dt in dates:
            for ac in ["Class A", "Class B"]:
                rows.append({"date": dt, "asset_class": ac, "cav": 1e8,
                             "holders": 100.0, "volume": 1e6})
        df = pd.DataFrame(rows)
        hhi = _build_hhi_series(df, exclude_repo=False)
        assert all(abs(v - 0.5) < 1e-6 for v in hhi.dropna())

    def test_excludes_repo(self):
        df = _make_combined(n_months=10, include_repo=True,
                            asset_classes=["US Treasury Debt"])
        hhi_ex  = _build_hhi_series(df, exclude_repo=True)
        hhi_all = _build_hhi_series(df, exclude_repo=False)
        # Repo entry dominates: full-market HHI should be higher in repo months
        repo_months = df[df["asset_class"] == REPO_CLASS]["date"].unique()
        if len(repo_months) > 0:
            first_repo = pd.Timestamp(repo_months.min())
            hhi_ex_post  = hhi_ex[hhi_ex.index  >= first_repo]
            hhi_all_post = hhi_all[hhi_all.index >= first_repo]
            # Ex-repo HHI should be lower (single class, but no repo swamping)
            assert (hhi_all_post.dropna() >= hhi_ex_post.dropna().values).all()


# ─────────────────────────────────────────────────────────────────────────────
# _make_serializable
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSerializable:
    def test_nan_becomes_none(self):
        r = _make_serializable({"x": float("nan")})
        assert r["x"] is None

    def test_inf_becomes_none(self):
        r = _make_serializable({"x": float("inf")})
        assert r["x"] is None

    def test_numpy_float(self):
        r = _make_serializable(np.float64(3.14))
        assert isinstance(r, float)
        assert abs(r - 3.14) < 1e-6

    def test_numpy_bool(self):
        r = _make_serializable(np.bool_(True))
        assert r is True
        assert isinstance(r, bool)

    def test_nested_dict(self):
        obj = {"a": {"b": np.float32(1.0), "c": float("nan")}}
        r = _make_serializable(obj)
        assert isinstance(r["a"]["b"], float)
        assert r["a"]["c"] is None

    def test_list_of_arrays(self):
        obj = [np.array([1.0, np.nan])]
        r = _make_serializable(obj)
        assert r[0] == [1.0, None]


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 1 — Growth
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar1Growth:
    def _run(self, n_months=30, include_repo=False, growth=0.08):
        df   = _make_combined(n_months=n_months, include_repo=include_repo,
                               cav_growth_rate=growth)
        conc = _make_concentration(df)
        return pillar1_growth(df, conc)

    def test_growing_market_detected(self):
        r = self._run(growth=0.08)
        # With 5%/month growth over 30 months, OLS should find significant positive beta
        t = r["log_cav_trend_ex_repo"]
        assert t["status"] == "ok"
        assert t["beta"] > 0
        assert t["annualized_growth_rate"] > 0

    def test_summary_keys_present(self):
        r = self._run()
        s = r["summary"]
        for key in ("is_growing", "is_accelerating", "is_decelerating",
                    "ex_repo_break_detected", "interpretation"):
            assert key in s, f"Missing summary key: {key}"

    def test_required_output_keys(self):
        r = self._run()
        for key in ("log_cav_trend_ex_repo", "log_cav_trend_full_market",
                    "growth_acceleration_ex_repo", "rolling_growth_trend_ex_repo",
                    "structural_break_at_repo_entry", "repo_context", "summary"):
            assert key in r, f"Missing pillar1 key: {key}"

    def test_repo_context_present(self):
        r = self._run(include_repo=True, n_months=30)
        ctx = r["repo_context"]
        assert ctx["repo_holders_in_data"] == 0
        assert ctx["repo_volume_in_data"]  == 0

    def test_no_data_insufficient(self):
        df   = _make_combined(n_months=4)
        conc = _make_concentration(df)
        r    = pillar1_growth(df, conc)
        assert r["log_cav_trend_ex_repo"]["status"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Composition
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar2Composition:
    def _run(self, n_months=30, asset_classes=None, include_repo=False):
        df   = _make_combined(n_months=n_months,
                               asset_classes=asset_classes or ["US Treasury Debt", "Commodities", "Real Estate"],
                               include_repo=include_repo)
        conc = _make_concentration(df)
        return pillar2_composition(df, conc)

    def test_required_keys(self):
        r = self._run()
        for key in ("hhi_trend_ex_repo", "hhi_trend_full_market",
                    "top_5_share_trend_pre_repo", "active_class_count_trend_ex_repo",
                    "hhi_snapshots", "summary"):
            assert key in r, f"Missing pillar2 key: {key}"

    def test_summary_flags(self):
        r = self._run()
        s = r["summary"]
        for key in ("is_diversifying_ex_repo", "active_classes_growing",
                    "is_diversifying", "interpretation"):
            assert key in s, f"Missing summary key: {key}"

    def test_hhi_snapshots_present(self):
        r = self._run()
        snap = r["hhi_snapshots"]
        assert snap["hhi_ex_repo_earliest"] is not None
        assert snap["hhi_ex_repo_latest"]   is not None

    def test_diversifying_market(self):
        # Build a market that starts concentrated (one big class) and diversifies
        # by making two equal-sized classes that both grow at the same rate
        dates = _dates(25)
        rows = []
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "asset_class": "Class A", "cav": 1e9,
                         "holders": 1000.0, "volume": 1e7,
                         "cav_share": np.nan, "cav_index": np.nan,
                         "asset_class_cav_growth": np.nan, "holders_index": np.nan,
                         "monthly_holder_growth": np.nan, "holders_per_million_cav": np.nan,
                         "avg_position": np.nan,
                         "turnover_ratio": 0.01, "turnover_3m": 0.01,
                         "monthly_volume_growth": np.nan})
            # Class B starts tiny and grows to match Class A over time
            cav_b = 1e7 * (1.15 ** i)  # fast growth
            rows.append({"date": dt, "asset_class": "Class B", "cav": cav_b,
                         "holders": 500.0, "volume": cav_b * 0.01,
                         "cav_share": np.nan, "cav_index": np.nan,
                         "asset_class_cav_growth": np.nan, "holders_index": np.nan,
                         "monthly_holder_growth": np.nan, "holders_per_million_cav": np.nan,
                         "avg_position": np.nan,
                         "turnover_ratio": 0.01, "turnover_3m": 0.01,
                         "monthly_volume_growth": np.nan})
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        conc = _make_concentration(df)
        r = pillar2_composition(df, conc)
        # HHI should be declining as Class B grows
        assert r["hhi_trend_ex_repo"]["status"] == "ok"
        assert r["hhi_trend_ex_repo"]["beta"] < 0   # HHI falling


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Adoption
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar3Adoption:
    def _run(self, n_months=25, holder_growth=0.04, cav_growth=0.05, include_repo=False):
        df = _make_combined(n_months=n_months, include_repo=include_repo,
                             cav_growth_rate=cav_growth,
                             holder_growth_rate=holder_growth)
        return pillar3_adoption(df)

    def test_required_keys(self):
        r = self._run()
        for key in ("holder_trend_ex_repo", "log_log_cav_holders_ex_repo",
                    "spearman_cav_holder_growth", "asset_class_regressions",
                    "cav_weighted_adoption_score", "summary"):
            assert key in r, f"Missing pillar3 key: {key}"

    def test_holders_growing_detected(self):
        r = self._run(holder_growth=0.05, n_months=30)
        t = r["holder_trend_ex_repo"]
        if t["status"] == "ok":
            assert t["beta"] > 0

    def test_repo_excluded(self):
        r = self._run(include_repo=True)
        # Repos are excluded; asset_class_regressions should not contain REPO_CLASS
        assert REPO_CLASS not in r["asset_class_regressions"]

    def test_cav_weighted_score_range(self):
        r = self._run(n_months=30)
        score = r["cav_weighted_adoption_score"]
        if score is not None:
            assert 0.0 <= score <= 1.0

    def test_summary_flags(self):
        r = self._run()
        for key in ("holders_growing", "adoption_tracks_cav", "n_broadening_classes",
                    "n_classes_tested", "interpretation"):
            assert key in r["summary"], f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Liquidity
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar4Liquidity:
    def _run(self, n_months=25, volume_fraction=0.10, include_repo=False):
        df = _make_combined(n_months=n_months, include_repo=include_repo,
                             volume_fraction=volume_fraction)
        return pillar4_liquidity(df)

    def test_required_keys(self):
        r = self._run()
        for key in ("turnover_trend_ex_repo", "log_log_cav_turnover_ex_repo",
                    "spearman_cav_volume_growth", "asset_class_regressions",
                    "cav_weighted_liquidity_score", "repo_exclusion_note", "summary"):
            assert key in r, f"Missing pillar4 key: {key}"

    def test_repo_excluded_from_regressions(self):
        r = self._run(include_repo=True)
        assert REPO_CLASS not in r["asset_class_regressions"]

    def test_repo_exclusion_note_present(self):
        r = self._run(include_repo=True)
        assert len(r["repo_exclusion_note"]) > 20

    def test_cav_weighted_score_range(self):
        r = self._run(n_months=30)
        score = r["cav_weighted_liquidity_score"]
        if score is not None:
            assert 0.0 <= score <= 1.0

    def test_summary_flags(self):
        r = self._run()
        for key in ("liquidity_growing", "liquidity_tracks_cav",
                    "n_improving_classes", "n_classes_tested", "interpretation"):
            assert key in r["summary"], f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# build_conclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildConclusion:
    def _make_pillars(self, growing=True, diversifying=True,
                       holders_growing=True, liquidity_growing=True):
        p1 = {"summary": {
            "is_growing": growing, "is_accelerating": False,
            "is_decelerating": False, "ex_repo_break_detected": False,
            "annualized_growth_rate_ex_repo": 0.5,
            "interpretation": "test",
        }}
        p2 = {"summary": {
            "is_diversifying_ex_repo": diversifying,
            "active_classes_growing": True,
            "top5_falling_pre_repo": True,
            "is_diversifying": diversifying,
            "interpretation": "test",
        }, "hhi_snapshots": {
            "hhi_ex_repo_earliest": 0.6,
            "hhi_ex_repo_latest": 0.35,
            "hhi_full_pre_repo": 0.6,
            "hhi_full_latest": 0.85,
        }}
        p3 = {"summary": {
            "holders_growing": holders_growing,
            "adoption_tracks_cav": True,
            "adoption_elastic": True,
            "n_broadening_classes": 3,
            "n_classes_tested": 4,
            "cav_weighted_adoption_score": 0.65,
            "interpretation": "test",
        }}
        p4 = {"summary": {
            "liquidity_growing": liquidity_growing,
            "liquidity_tracks_cav": True,
            "n_improving_classes": 2,
            "n_classes_tested": 4,
            "cav_weighted_liquidity_score": 0.4,
            "interpretation": "test",
        }}
        return p1, p2, p3, p4

    def test_produces_non_empty_string(self):
        p1, p2, p3, p4 = self._make_pillars()
        text = build_conclusion(p1, p2, p3, p4)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_no_causal_language(self):
        p1, p2, p3, p4 = self._make_pillars()
        text = build_conclusion(p1, p2, p3, p4).lower()
        # These exact phrases would imply causation
        for forbidden in ("causes", "caused by", "is causing", "proves that"):
            assert forbidden not in text, f"Found causal language: '{forbidden}'"

    def test_positive_verdict_when_all_signals_green(self):
        p1, p2, p3, p4 = self._make_pillars(
            growing=True, diversifying=True,
            holders_growing=True, liquidity_growing=True
        )
        text = build_conclusion(p1, p2, p3, p4)
        assert "structural shift" in text.lower()

    def test_weak_verdict_when_mixed_signals(self):
        p1, p2, p3, p4 = self._make_pillars(
            growing=True, diversifying=False,
            holders_growing=False, liquidity_growing=False
        )
        text = build_conclusion(p1, p2, p3, p4)
        assert "mixed" in text.lower() or "early" in text.lower()

    def test_repo_context_mentioned(self):
        p1, p2, p3, p4 = self._make_pillars()
        text = build_conclusion(p1, p2, p3, p4)
        assert "repo" in text.lower() or "repurchase" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# run_analysis — integration / structure tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAnalysis:
    def _run(self, n_months=20):
        df   = _make_combined(n_months=n_months, include_repo=True,
                               asset_classes=["US Treasury Debt", "Commodities", "Real Estate"])
        conc = _make_concentration(df)
        return run_analysis(df, conc)

    def test_top_level_keys(self):
        r = self._run()
        for key in ("pillar1_growth", "pillar2_composition", "pillar3_adoption",
                    "pillar4_liquidity", "conclusion", "generated_at",
                    "data_window", "limitations"):
            assert key in r, f"Missing top-level key: {key}"

    def test_data_window_populated(self):
        r = self._run()
        dw = r["data_window"]
        assert dw["start"] is not None
        assert dw["end"]   is not None
        assert dw["n_months_total"] > 0
        assert dw["repo_entry_date"] == str(REPO_ENTRY_DATE.date())

    def test_limitations_list(self):
        r = self._run()
        assert isinstance(r["limitations"], list)
        assert len(r["limitations"]) > 0

    def test_conclusion_non_empty(self):
        r = self._run()
        assert isinstance(r["conclusion"], str)
        assert len(r["conclusion"]) > 100

    def test_json_serializable(self):
        import json
        r = self._run()
        dumped = json.dumps(r)   # must not raise
        loaded = json.loads(dumped)
        assert "pillar1_growth" in loaded

    def test_no_nan_in_output(self):
        import json
        r = self._run()
        text = json.dumps(r)
        # JSON-serialized NaN would appear as literal NaN (not a string)
        # After _make_serializable, all NaN → None → "null" in JSON
        assert "NaN" not in text

    def test_small_dataset_graceful(self):
        """Very short dataset → insufficient_data flags, not exceptions."""
        df   = _make_combined(n_months=4)
        conc = _make_concentration(df)
        r    = run_analysis(df, conc)
        assert r["pillar1_growth"]["log_cav_trend_ex_repo"]["status"] == "insufficient_data"
