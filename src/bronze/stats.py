# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Statistical analysis for the RWA tokenization thesis — four-pillar framework.

Repurchase Agreements (repos) — data limitation
-------------------------------------------------
Repos entered the rwa.xyz dataset in June 2025, contributing ~$282B CAV
(~92% of total market value at the time). Two facts govern their treatment:

  1. Holders = 0 for every repo observation (rwa.xyz does not track them).
  2. Volume  = 0 for every repo observation (on-chain transfers not tracked).

Consequence for each pillar:
  - Pillar 1 (Growth):   full-market CAV has an artificial 1208% spike in
    June 2025. Ex-repo CAV is the primary growth metric.
  - Pillar 2 (Compos.): full-market HHI spikes from ~0.35 → ~0.85 at the
    same event. Ex-repo HHI is the primary concentration metric.
  - Pillar 3 (Adoption): repos have 0 holders → excluded entirely.
  - Pillar 4 (Liquidity): repos have 0 volume → excluded entirely.
    Including repo CAV in the turnover denominator would *suppress* measured
    liquidity (large CAV denominator, zero volume numerator).

All primary statistical tests use ex-repo data. Full-market figures are
provided as context with explicit caveats. Repos are documented as a known
data limitation that does not change the thesis conclusion.

Public API
----------
    run_analysis(combined_df, concentration_df) -> dict
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm

from src.config import REPO_CLASS

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds / constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_OBS_OLS = 6
MIN_OBS_CORR = 5
MIN_OBS_STRUCTURAL_BREAK = 20

# The month repos entered the rwa.xyz dataset — this is a data-inclusion event,
# not an organic market development.
REPO_ENTRY_DATE = pd.Timestamp("2025-06-01")

LIMITATIONS = [
    (
        "Repurchase Agreements (repos) entered the rwa.xyz dataset in June 2025 with "
        "~$282B CAV, immediately representing ~92% of total market value. Repos report "
        "zero holders and zero on-chain volume throughout. The full-market concentration "
        "metrics (HHI, CAV growth) contain an artificial discontinuity at June 2025. "
        "All primary thesis analyses use ex-repo data; this limitation does not alter "
        "the thesis conclusion on the non-repo RWA market."
    ),
    (
        "Holder counts are on-chain address proxies, not verified unique participants. "
        "One institution may control multiple wallets; one wallet may represent multiple "
        "underlying investors. Market-level holder totals sum across asset classes and "
        "double-count wallets that hold multiple classes."
    ),
    (
        "Volume data reflects on-chain transfer events and may include intra-protocol "
        "or custodial movements, not solely secondary-market trades."
    ),
    (
        "OLS regressions assume linear trends. Non-linear growth phases and the "
        "June 2025 repo entry event may not be fully captured."
    ),
    (
        "All findings are associational. Correlation between CAV growth and holder or "
        "liquidity growth does not imply causation."
    ),
    (
        "The ex-repo sample window of ~41 months limits statistical power for "
        "long-run structural-break tests."
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Statistical primitives
# ─────────────────────────────────────────────────────────────────────────────

def _ols_trend(y: pd.Series) -> dict:
    """
    Fit y = alpha + beta*t where t = 0, 1, … over non-null observations.

    Returns a result dict with status, n, beta, p_value, 95% confidence_interval,
    r_squared, and alpha.
    Returns {"status": "insufficient_data"} when n < MIN_OBS_OLS.
    """
    clean = y.dropna().astype(float)
    n = int(len(clean))
    if n < MIN_OBS_OLS:
        return {"status": "insufficient_data", "n": n, "min_required": MIN_OBS_OLS}

    t = pd.Series(np.arange(n, dtype=float), index=clean.index)
    X = sm.add_constant(t)
    model = sm.OLS(clean, X).fit()

    beta = float(model.params.iloc[1])
    p = float(model.pvalues.iloc[1])
    ci = model.conf_int()

    return {
        "status": "ok",
        "n": n,
        "beta": beta,
        "p_value": p,
        "confidence_interval": [float(ci.iloc[1, 0]), float(ci.iloc[1, 1])],
        "r_squared": float(model.rsquared),
        "alpha": float(model.params.iloc[0]),
    }


def _ols_log_log(y: pd.Series, x: pd.Series) -> dict:
    """
    Fit ln(y) = alpha + beta*ln(x) after removing non-positive and NaN pairs.

    WARNING: only use this on *stationary* series or as a structural description
    of long-run scale relationships.  If both y and x share a common upward trend
    (non-stationary), OLS in levels produces spurious significance.  Use
    _ols_growth_on_growth for per-class co-movement tests instead.

    beta interpretation:
      > 1  →  y grows faster than x (e.g. holders outpace CAV)
      = 1  →  proportional growth
      < 1  →  y grows slower than x (e.g. CAV outpaces holders)
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    df = df[(df["y"] > 0) & (df["x"] > 0)]
    n = int(len(df))
    if n < MIN_OBS_OLS:
        return {"status": "insufficient_data", "n": n, "min_required": MIN_OBS_OLS}

    log_y = np.log(df["y"])
    log_x = np.log(df["x"])
    X = sm.add_constant(log_x)
    model = sm.OLS(log_y, X).fit()

    from statsmodels.stats.stattools import durbin_watson
    dw = float(durbin_watson(model.resid))

    beta = float(model.params.iloc[1])
    p = float(model.pvalues.iloc[1])
    ci = model.conf_int()

    return {
        "status": "ok",
        "n": n,
        "beta": beta,
        "p_value": p,
        "confidence_interval": [float(ci.iloc[1, 0]), float(ci.iloc[1, 1])],
        "r_squared": float(model.rsquared),
        "alpha": float(model.params.iloc[0]),
        "durbin_watson": dw,
        "spurious_regression_warning": dw < 1.0,
    }


def _ols_growth_on_growth(pct_y: pd.Series, pct_x: pd.Series) -> dict:
    """
    Fit Δy_pct = alpha + beta * Δx_pct using monthly growth rates (first differences).

    This is the statistically valid co-movement test for non-stationary time series.
    Regressing levels on levels when both trend upward produces spurious significance
    (low Durbin-Watson, inflated t-stats). First differences remove the common trend.

    beta interpretation:
      > 0  →  y growth co-moves positively with x growth
      beta magnitude: a 1pp increase in x growth rate → beta pp increase in y growth
    """
    df = pd.DataFrame({"y": pct_y, "x": pct_x}).dropna()
    n = int(len(df))
    if n < MIN_OBS_OLS:
        return {"status": "insufficient_data", "n": n, "min_required": MIN_OBS_OLS}

    X = sm.add_constant(df["x"].astype(float))
    model = sm.OLS(df["y"].astype(float), X).fit()

    beta = float(model.params.iloc[1])
    p = float(model.pvalues.iloc[1])
    ci = model.conf_int()

    return {
        "status": "ok",
        "n": n,
        "beta": beta,
        "p_value": p,
        "confidence_interval": [float(ci.iloc[1, 0]), float(ci.iloc[1, 1])],
        "r_squared": float(model.rsquared),
        "alpha": float(model.params.iloc[0]),
    }


def _spearman(x: pd.Series, y: pd.Series) -> dict:
    """Spearman rank correlation after dropping joint NaNs."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(df))
    if n < MIN_OBS_CORR:
        return {"status": "insufficient_data", "n": n, "min_required": MIN_OBS_CORR}
    rho, p = scipy_stats.spearmanr(df["x"], df["y"])
    return {"status": "ok", "n": n, "rho": float(rho), "p_value": float(p)}


def _structural_break_ols(y: pd.Series, break_idx: int) -> dict:
    """
    Chow dummy-variable structural break test.
    Model: y = α + β₁t + β₂D + β₃(t·D)
    D = 1 for t ≥ break_idx, else 0.

    β₃ (p_t_D): slope change after the break.
      Positive → acceleration post-break.
      Negative → deceleration post-break.
    """
    clean = y.dropna().astype(float)
    n = int(len(clean))
    if n < MIN_OBS_STRUCTURAL_BREAK:
        return {"status": "insufficient_data", "n": n, "min_required": MIN_OBS_STRUCTURAL_BREAK}

    t = np.arange(n, dtype=float)
    D = (t >= break_idx).astype(float)
    X = pd.DataFrame(
        {"const": 1.0, "t": t, "D": D, "t_D": t * D},
        index=clean.index,
    )
    model = sm.OLS(clean, X).fit()

    def _p(k): return float(model.pvalues.get(k, np.nan))
    def _b(k): return float(model.params.get(k, np.nan))
    def _ci(k):
        ci = model.conf_int()
        return [float(ci.loc[k, 0]), float(ci.loc[k, 1])] if k in ci.index else [None, None]

    return {
        "status": "ok",
        "n": n,
        "break_idx": int(break_idx),
        "beta_t":   _b("t"),
        "beta_D":   _b("D"),
        "beta_t_D": _b("t_D"),
        "p_t":      _p("t"),
        "p_D":      _p("D"),
        "p_t_D":    _p("t_D"),
        "ci_D":     _ci("D"),
        "ci_t_D":   _ci("t_D"),
        "r_squared": float(model.rsquared),
    }


def _annualized_growth(beta_log_per_month: float) -> float:
    """Convert log-linear monthly slope to annualized compound growth rate."""
    return float(math.exp(12 * beta_log_per_month) - 1)


def _sig_label(p: float) -> str:
    if p < 0.01:
        return "significant (p<0.01)"
    if p < 0.05:
        return "significant (p<0.05)"
    if p < 0.10:
        return "marginal (p<0.10)"
    return "not significant"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pct_change(series: pd.Series) -> pd.Series:
    pct = series.astype(float).pct_change(fill_method=None) * 100.0
    return pct.where(~np.isinf(pct))


def _build_market_aggregates(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(date, asset_class) data to monthly ex-repo market totals.

    Since repos have zero holders and zero volume, market-level holder/volume
    totals are identical whether or not repos are included. We compute them on
    the ex-repo slice to be explicit and to avoid CAV distortion in ratio metrics.

    Returns columns:
        date, ex_repo_cav, ex_repo_holders, ex_repo_volume,
        total_cav (including repos, for context),
        repo_cav, repo_share,
        ex_repo_cav_growth, ex_repo_holders_growth, ex_repo_volume_growth,
        repo_present (bool flag)
    """
    df = combined_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    full_cav = (
        df.groupby("date")["cav"]
        .sum(min_count=1)
        .rename("total_cav")
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    repo_cav = (
        df[df["asset_class"] == REPO_CLASS]
        .groupby("date")["cav"]
        .sum(min_count=1)
        .rename("repo_cav")
        .reset_index()
    )

    ex_agg = (
        df[df["asset_class"] != REPO_CLASS]
        .groupby("date")[["cav", "holders", "volume"]]
        .sum(min_count=1)
        .rename(columns={
            "cav":     "ex_repo_cav",
            "holders": "ex_repo_holders",
            "volume":  "ex_repo_volume",
        })
        .reset_index()
    )

    mkt = (
        full_cav
        .merge(repo_cav, on="date", how="left")
        .merge(ex_agg, on="date", how="left")
    )
    mkt["repo_share"] = mkt["repo_cav"] / mkt["total_cav"].where(mkt["total_cav"] > 0)
    mkt["repo_present"] = mkt["date"] >= REPO_ENTRY_DATE

    for col in ["ex_repo_cav", "ex_repo_holders", "ex_repo_volume"]:
        mkt[f"{col}_growth"] = _safe_pct_change(mkt[col])

    return mkt.reset_index(drop=True)


def _build_hhi_series(combined_df: pd.DataFrame, exclude_repo: bool = True) -> pd.Series:
    """
    Compute monthly HHI from combined_df.
    By default excludes repos (primary metric); pass exclude_repo=False for
    the full-market version (contains the June 2025 artificial spike).
    """
    df = combined_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if exclude_repo:
        df = df[df["asset_class"] != REPO_CLASS]

    def _hhi(g):
        total = g["cav"].sum(skipna=True)
        if total <= 0:
            return np.nan
        shares = g["cav"].fillna(0.0) / total
        return float((shares ** 2).sum())

    return df.groupby("date").apply(_hhi, include_groups=False).sort_index()


def _repo_entry_idx(mkt: pd.DataFrame) -> int | None:
    """Positional index of the first month repos appear in the dataset."""
    mask = mkt["repo_present"]
    return int(np.argmax(mask.values)) if mask.any() else None


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 1 — Growth / Rate of Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def pillar1_growth(combined_df: pd.DataFrame, concentration_df: pd.DataFrame) -> dict:
    """
    Pillar 1: Is tokenized RWA value growing? Is growth accelerating or slowing?

    Primary metric: ex-repo CAV (41 months, clean continuous series).
    Full-market CAV provided as context — contains a 1208% artificial spike in
    June 2025 when repos entered the dataset.

    Note: concentration_df monthly_cav_growth column is NOT used for acceleration
    tests because it contains the June 2025 repo spike. Growth rates are computed
    from the ex-repo CAV series in combined_df instead.
    """
    conc = concentration_df.copy()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values("date").reset_index(drop=True)
    mkt = _build_market_aggregates(combined_df)

    # ── Primary: log-linear CAV trend, ex-repo ────────────────────────────────
    log_cav_ex = mkt["ex_repo_cav"].where(mkt["ex_repo_cav"] > 0).pipe(np.log)
    t_ex = _ols_trend(log_cav_ex)
    if t_ex["status"] == "ok":
        t_ex["annualized_growth_rate"] = _annualized_growth(t_ex["beta"])
        t_ex["growth_direction"] = "growing" if t_ex["beta"] > 0 else "declining"
        t_ex["interpretation"] = (
            f"Ex-repo CAV {t_ex['growth_direction']} at "
            f"{t_ex['annualized_growth_rate']:.1%}/yr annualized "
            f"({_sig_label(t_ex['p_value'])})."
        )

    # ── Context: log-linear CAV trend, full market ───────────────────────────
    log_cav_full = conc["total_cav"].where(conc["total_cav"] > 0).pipe(np.log)
    t_full = _ols_trend(log_cav_full)
    if t_full["status"] == "ok":
        t_full["annualized_growth_rate"] = _annualized_growth(t_full["beta"])
        t_full["data_caveat"] = (
            "Full-market series contains an artificial step-change in June 2025 "
            "when Repurchase Agreements (~$282B) entered the rwa.xyz dataset. "
            "This inflates the estimated growth rate. Ex-repo is the primary measure."
        )

    # ── Growth acceleration: monthly ex-repo growth ~ t ──────────────────────
    t_accel = _ols_trend(mkt["ex_repo_cav_growth"])
    if t_accel["status"] == "ok":
        direction = "accelerating" if t_accel["beta"] > 0 else "decelerating"
        t_accel["acceleration_direction"] = direction
        t_accel["interpretation"] = (
            f"Ex-repo monthly CAV growth is {direction} by "
            f"{t_accel['beta']:+.3f} pp/month ({_sig_label(t_accel['p_value'])})."
        )

    # ── Rolling 3m growth trend (ex-repo) ────────────────────────────────────
    ex_rolling_3m = mkt["ex_repo_cav_growth"].rolling(3, min_periods=1).mean()
    t_roll = _ols_trend(ex_rolling_3m)
    if t_roll["status"] == "ok":
        direction = "accelerating" if t_roll["beta"] > 0 else "decelerating"
        t_roll["acceleration_direction"] = direction
        t_roll["interpretation"] = (
            f"Ex-repo 3m rolling growth {direction} ({_sig_label(t_roll['p_value'])})."
        )

    # ── Structural break: does ex-repo CAV also accelerate at repo entry? ────
    # If yes, this suggests organic growth acceleration separate from the repo event.
    break_idx = _repo_entry_idx(mkt)
    t_break: dict = {"status": "insufficient_data", "note": "No repo entry date detected."}
    if break_idx is not None and break_idx >= 6:
        t_break = _structural_break_ols(log_cav_ex, break_idx)
        t_break["break_date"] = str(REPO_ENTRY_DATE.date())
        t_break["note"] = (
            "Break date is June 2025 (repo entry). A significant positive beta_t_D "
            "means ex-repo CAV also accelerated organically around this period, "
            "independent of the repo inclusion event."
        )
        if t_break["status"] == "ok":
            t_break["interpretation"] = (
                f"Ex-repo CAV post-break slope change: {t_break['beta_t_D']:+.4f}/month "
                f"({_sig_label(t_break['p_t_D'])})."
            )

    # ── Repo context ──────────────────────────────────────────────────────────
    latest_repo_share = None
    valid = mkt["repo_share"].dropna()
    if not valid.empty:
        latest_repo_share = float(valid.iloc[-1])

    # ── Summary ───────────────────────────────────────────────────────────────
    ex_growing = (
        t_ex["status"] == "ok"
        and t_ex["beta"] > 0
        and t_ex["p_value"] < 0.05
    )
    is_accelerating = (
        t_accel["status"] == "ok"
        and t_accel["beta"] > 0
        and t_accel["p_value"] < 0.05
    )
    is_decelerating = (
        t_accel["status"] == "ok"
        and t_accel["beta"] < 0
        and t_accel["p_value"] < 0.05
    )
    ex_repo_break_detected = (
        t_break.get("status") == "ok"
        and t_break.get("p_t_D", 1.0) < 0.05
        and t_break.get("beta_t_D", 0) > 0
    )

    parts = []
    if ex_growing:
        parts.append(
            f"Ex-repo RWA CAV is growing significantly "
            f"(~{t_ex.get('annualized_growth_rate', 0):.1%}/yr annualized)."
        )
    else:
        parts.append("Ex-repo RWA CAV growth is not statistically significant.")
    if is_decelerating:
        parts.append("Monthly growth is decelerating — pace of expansion may be moderating.")
    elif is_accelerating:
        parts.append("Monthly growth is accelerating.")
    if ex_repo_break_detected:
        parts.append(
            "Ex-repo CAV shows a significant slope increase around June 2025, "
            "suggesting organic acceleration independent of the repo entry event."
        )

    return {
        "log_cav_trend_ex_repo":        t_ex,
        "log_cav_trend_full_market":     t_full,
        "growth_acceleration_ex_repo":   t_accel,
        "rolling_growth_trend_ex_repo":  t_roll,
        "structural_break_at_repo_entry": t_break,
        "repo_context": {
            "repo_entry_date":               str(REPO_ENTRY_DATE.date()),
            "latest_repo_share_of_total_cav": latest_repo_share,
            "repo_holders_in_data":          0,
            "repo_volume_in_data":           0,
            "note": (
                "Repos were added to the rwa.xyz dataset in June 2025 (~$282B CAV). "
                "They report zero holders and zero on-chain volume throughout. "
                "Full-market CAV metrics are distorted by this entry event; "
                "ex-repo metrics are the primary analytical series."
            ),
        },
        "summary": {
            "is_growing":               ex_growing,
            "is_accelerating":          is_accelerating,
            "is_decelerating":          is_decelerating,
            "ex_repo_break_detected":   ex_repo_break_detected,
            "annualized_growth_rate_ex_repo": t_ex.get("annualized_growth_rate"),
            "interpretation": " ".join(parts),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2 — Composition / Diversification
# ─────────────────────────────────────────────────────────────────────────────

def pillar2_composition(combined_df: pd.DataFrame, concentration_df: pd.DataFrame) -> dict:
    """
    Pillar 2: Is the market becoming more or less concentrated across asset classes?

    Primary metric: ex-repo HHI — the repo entry in June 2025 caused HHI to
    jump from ~0.35 to ~0.85 artificially, making full-market HHI uninformative
    for trend analysis.

    concentration_df HHI column is NOT used as primary input for the same reason.
    Ex-repo HHI is recomputed from combined_df.
    """
    conc = concentration_df.copy()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values("date").reset_index(drop=True)

    hhi_ex_series = pd.Series(
        _build_hhi_series(combined_df, exclude_repo=True).values, dtype=float
    )
    hhi_full_series = pd.Series(conc["hhi"].values, dtype=float)

    # ── Primary: HHI ex-repo ──────────────────────────────────────────────────
    t_hhi_ex = _ols_trend(hhi_ex_series)
    if t_hhi_ex["status"] == "ok":
        direction = "falling (diversifying)" if t_hhi_ex["beta"] < 0 else "rising (concentrating)"
        t_hhi_ex["concentration_direction"] = direction
        t_hhi_ex["interpretation"] = (
            f"Ex-repo HHI {direction}: β={t_hhi_ex['beta']:+.5f}/month "
            f"({_sig_label(t_hhi_ex['p_value'])})."
        )

    # ── Context: full-market HHI ──────────────────────────────────────────────
    t_hhi_full = _ols_trend(hhi_full_series)
    if t_hhi_full["status"] == "ok":
        t_hhi_full["data_caveat"] = (
            "Full-market HHI jumped from ~0.35 to ~0.85 in June 2025 due to repo entry "
            "(single asset class at 92% market share). Trend is not interpretable. "
            "Ex-repo HHI is the primary diversification measure."
        )

    # ── HHI snapshots ─────────────────────────────────────────────────────────
    hhi_ex_earliest  = float(hhi_ex_series.dropna().iloc[0])  if not hhi_ex_series.dropna().empty else None
    hhi_ex_latest    = float(hhi_ex_series.dropna().iloc[-1]) if not hhi_ex_series.dropna().empty else None
    hhi_full_pre     = float(hhi_full_series[:29].dropna().iloc[-1]) if len(hhi_full_series) >= 29 else None
    hhi_full_latest  = float(hhi_full_series.dropna().iloc[-1]) if not hhi_full_series.dropna().empty else None

    # ── Top-5 share (pre-repo window only for clean read) ─────────────────────
    # Pre-repo = first 29 months (Jan 2023 – May 2025)
    n_pre = int((mkt := _build_market_aggregates(combined_df))["repo_present"].values.argmax())
    if not mkt["repo_present"].any():
        n_pre = len(conc)

    t_top5_pre = _ols_trend(pd.Series(conc["top_5_share"].values[:n_pre], dtype=float))
    if t_top5_pre["status"] == "ok":
        direction = "falling" if t_top5_pre["beta"] < 0 else "rising"
        t_top5_pre["concentration_direction"] = direction
        t_top5_pre["interpretation"] = (
            f"Pre-repo top-5 CAV share {direction}: β={t_top5_pre['beta']:+.5f}/month "
            f"({_sig_label(t_top5_pre['p_value'])})."
        )

    # Full-period top-5 (context only — distorted post-June 2025)
    t_top5_full = _ols_trend(pd.Series(conc["top_5_share"].values, dtype=float))
    if t_top5_full["status"] == "ok":
        t_top5_full["data_caveat"] = (
            "Top-5 share post-June 2025 is dominated by repos in first position. "
            "Pre-repo window is the cleaner trend signal."
        )

    # ── Ex-repo active class count ────────────────────────────────────────────
    df_ex = combined_df[combined_df["asset_class"] != REPO_CLASS].copy()
    df_ex["date"] = pd.to_datetime(df_ex["date"])
    active_ex = (
        df_ex[df_ex["cav"] > 1_000_000]
        .groupby("date")["asset_class"]
        .nunique()
        .sort_index()
    )
    t_active_ex = _ols_trend(pd.Series(active_ex.values, dtype=float))
    if t_active_ex["status"] == "ok":
        direction = "rising" if t_active_ex["beta"] > 0 else "falling"
        t_active_ex["interpretation"] = (
            f"Ex-repo active class count {direction}: β={t_active_ex['beta']:+.4f}/month "
            f"({_sig_label(t_active_ex['p_value'])})."
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    is_diversifying = (
        t_hhi_ex.get("status") == "ok"
        and t_hhi_ex["beta"] < 0
        and t_hhi_ex["p_value"] < 0.05
    )
    active_rising = (
        t_active_ex.get("status") == "ok"
        and t_active_ex["beta"] > 0
        and t_active_ex["p_value"] < 0.05
    )
    top5_falling_pre = (
        t_top5_pre.get("status") == "ok"
        and t_top5_pre["beta"] < 0
        and t_top5_pre["p_value"] < 0.05
    )

    parts = []
    if is_diversifying:
        parts.append(
            f"Ex-repo HHI is declining significantly "
            f"({hhi_ex_earliest:.3f} → {hhi_ex_latest:.3f})."
        )
    else:
        parts.append("Ex-repo HHI shows no significant declining trend.")
    if hhi_full_latest and hhi_ex_latest:
        repo_distortion = hhi_full_latest - hhi_ex_latest
        if repo_distortion > 0.05:
            parts.append(
                f"Repos add {repo_distortion:.3f} to HHI (latest: full={hhi_full_latest:.3f} "
                f"vs ex-repo={hhi_ex_latest:.3f})."
            )
    if active_rising:
        parts.append("Ex-repo active class count is rising significantly.")
    if top5_falling_pre:
        parts.append("Pre-repo top-5 CAV share was declining — diversification pre-dates the repo event.")

    return {
        "hhi_trend_ex_repo":           t_hhi_ex,
        "hhi_trend_full_market":       t_hhi_full,
        "top_5_share_trend_pre_repo":  t_top5_pre,
        "top_5_share_trend_full":      t_top5_full,
        "active_class_count_trend_ex_repo": t_active_ex,
        "hhi_snapshots": {
            "hhi_ex_repo_earliest":   hhi_ex_earliest,
            "hhi_ex_repo_latest":     hhi_ex_latest,
            "hhi_full_pre_repo":      hhi_full_pre,
            "hhi_full_latest":        hhi_full_latest,
            "note": (
                "HHI scale: 1.0 = one class holds all value; 1/n ≈ equal distribution. "
                "The June 2025 repo entry caused full-market HHI to jump ~0.35 → ~0.85 "
                "in a single month — a data artifact."
            ),
        },
        "summary": {
            "is_diversifying_ex_repo":    is_diversifying,
            "active_classes_growing":     active_rising,
            "top5_falling_pre_repo":      top5_falling_pre,
            "is_diversifying":            is_diversifying,
            "interpretation": " ".join(parts) if parts else "No significant diversification trend detected.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Adoption vs Asset Growth
# ─────────────────────────────────────────────────────────────────────────────

def pillar3_adoption(combined_df: pd.DataFrame) -> dict:
    """
    Pillar 3: Is growth in tokenized value accompanied by growth in on-chain holders?

    Repos are excluded entirely — they report zero holders in every observation.
    Including them would not change any holder metric (since 0 + something = something)
    but would distort CAV-based ratios by adding a large denominator with no signal.

    Holder counts are on-chain address proxies. Market-level totals sum across
    asset classes; the same wallet holding two classes is counted twice.
    """
    mkt = _build_market_aggregates(combined_df)
    df  = combined_df[combined_df["asset_class"] != REPO_CLASS].copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Log-linear holder trend ───────────────────────────────────────────────
    log_h = mkt["ex_repo_holders"].where(mkt["ex_repo_holders"] > 0).pipe(np.log)
    t_holder = _ols_trend(log_h)
    if t_holder["status"] == "ok":
        t_holder["annualized_growth_rate"] = _annualized_growth(t_holder["beta"])
        t_holder["interpretation"] = (
            f"Ex-repo holder count growing at "
            f"{t_holder['annualized_growth_rate']:.1%}/yr annualized "
            f"({_sig_label(t_holder['p_value'])})."
        )

    # ── OLS (growth on growth): monthly holder growth ~ monthly CAV growth ───
    # This is the primary co-movement test. Regressing levels on levels when both
    # series trend upward produces spurious significance (Durbin-Watson ≈ 0.3-0.5).
    # First differences (growth rates) are stationary and not subject to this.
    ll_market = _ols_growth_on_growth(mkt["ex_repo_holders_growth"], mkt["ex_repo_cav_growth"])
    if ll_market["status"] == "ok":
        beta = ll_market["beta"]
        ll_market["interpretation"] = (
            f"1 pp CAV monthly growth → {beta:.3f} pp holder monthly growth "
            f"({_sig_label(ll_market['p_value'])})."
        )
        ll_market["method_note"] = (
            "OLS on monthly growth rates (first differences of log levels). "
            "Stationary test; avoids spurious regression from trending levels."
        )

    # ── Spearman: monthly CAV growth vs holder growth ─────────────────────────
    sp = _spearman(mkt["ex_repo_cav_growth"], mkt["ex_repo_holders_growth"])
    if sp["status"] == "ok":
        sp["interpretation"] = (
            f"Monthly ex-repo CAV growth vs holder growth: ρ={sp['rho']:+.3f} "
            f"({_sig_label(sp['p_value'])})."
        )

    # ── Per-asset-class: monthly holder growth ~ monthly CAV growth ──────────
    # Uses first differences (growth rates) to avoid spurious regression.
    # Regressing ln(holders) on ln(cav) in levels produces DW ≈ 0.3-0.5 across
    # all 12 classes — a hallmark of spurious regression from shared time trend.
    cav_weights = df.sort_values("date").groupby("asset_class")["cav"].last()
    total_weight = float(cav_weights.sum())

    asset_class_results: dict[str, dict] = {}
    weighted_positive_sig = 0.0
    significant_adoption_classes: list[str] = []

    for ac, group in df.groupby("asset_class"):
        g = group.sort_values("date")
        r = _ols_growth_on_growth(g["monthly_holder_growth"], g["asset_class_cav_growth"])
        if r["status"] == "ok":
            r["interpretation"] = (
                f"1 pp CAV monthly growth → {r['beta']:.3f} pp holder monthly growth "
                f"({_sig_label(r['p_value'])})."
            )
            if r["beta"] > 0 and r["p_value"] < 0.05:
                weighted_positive_sig += float(cav_weights.get(ac, 0))
                significant_adoption_classes.append(ac)
        asset_class_results[ac] = r

    cav_weighted_score = (
        float(weighted_positive_sig / total_weight) if total_weight > 0 else None
    )

    n_broadening = len(significant_adoption_classes)
    n_tested = sum(1 for r in asset_class_results.values() if r.get("status") == "ok")

    # ── Summary ───────────────────────────────────────────────────────────────
    holders_growing = (
        t_holder.get("status") == "ok"
        and t_holder["beta"] > 0
        and t_holder["p_value"] < 0.05
    )
    adoption_tracks_cav = (
        sp.get("status") == "ok"
        and sp["rho"] > 0
        and sp["p_value"] < 0.05
    )
    adoption_elastic = (
        ll_market.get("status") == "ok"
        and ll_market.get("beta", 0) > 0
        and ll_market.get("p_value", 1) < 0.05
    )

    parts = []
    if holders_growing:
        parts.append(
            f"Ex-repo holder count is growing significantly "
            f"(~{t_holder.get('annualized_growth_rate', 0):.1%}/yr annualized)."
        )
    else:
        parts.append("Ex-repo holder growth is not statistically significant.")
    if adoption_tracks_cav:
        parts.append(
            f"Monthly holder growth tracks CAV growth positively "
            f"(ρ={sp.get('rho', 0):.3f}, {_sig_label(sp.get('p_value', 1))})."
        )
    else:
        parts.append("Month-to-month holder growth does not significantly track CAV growth.")
    if n_tested > 0:
        class_names = (
            f" ({', '.join(significant_adoption_classes)})"
            if significant_adoption_classes
            else ""
        )
        parts.append(
            f"{n_broadening}/{n_tested} asset classes show significant positive "
            f"co-movement between monthly holder growth and CAV growth{class_names}."
        )
    if cav_weighted_score is not None:
        parts.append(
            f"CAV-weighted adoption score: {cav_weighted_score:.2f} "
            "(fraction of ex-repo CAV in classes where monthly holder growth "
            "significantly co-moves with CAV growth)."
        )

    return {
        "holder_trend_ex_repo":            t_holder,
        "log_log_cav_holders_ex_repo":     ll_market,
        "spearman_cav_holder_growth":      sp,
        "asset_class_regressions":         asset_class_results,
        "cav_weighted_adoption_score":     cav_weighted_score,
        "repo_exclusion_note": (
            "Repurchase Agreements report zero holders in every observation "
            "and are excluded from all adoption analysis."
        ),
        "holder_proxy_note": (
            "Market-level holder totals sum across asset classes. Wallets holding "
            "multiple classes are counted multiple times — approximate upper bound."
        ),
        "summary": {
            "holders_growing":                holders_growing,
            "adoption_tracks_cav":            adoption_tracks_cav,
            "adoption_elastic":               adoption_elastic,
            "n_broadening_classes":           n_broadening,
            "n_classes_tested":               n_tested,
            "cav_weighted_adoption_score":    cav_weighted_score,
            "significant_adoption_classes":   significant_adoption_classes,
            "interpretation": " ".join(parts),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Liquidity vs Asset Growth
# ─────────────────────────────────────────────────────────────────────────────

def pillar4_liquidity(combined_df: pd.DataFrame) -> dict:
    """
    Pillar 4: Is tokenized value becoming more actively traded, or dormant?

    Repos are excluded entirely — they report zero on-chain volume in every
    observation. Including repo CAV in the market turnover denominator would
    artificially SUPPRESS measured liquidity (large CAV denominator, zero
    volume numerator).

    Volume data may include intra-protocol and custodial transfers, not only
    secondary-market trades.
    """
    mkt = _build_market_aggregates(combined_df)
    df  = combined_df[combined_df["asset_class"] != REPO_CLASS].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Market-level ex-repo turnover (volume / CAV, then 3m rolling)
    mkt["ex_repo_turnover"] = (
        mkt["ex_repo_volume"] / mkt["ex_repo_cav"].where(mkt["ex_repo_cav"] > 0)
    )
    mkt["ex_repo_turnover_3m"] = mkt["ex_repo_turnover"].rolling(3, min_periods=1).mean()

    # ── Turnover trend (3m rolling, ex-repo) ─────────────────────────────────
    t_liq = _ols_trend(mkt["ex_repo_turnover_3m"])
    if t_liq["status"] == "ok":
        direction = "improving" if t_liq["beta"] > 0 else "declining"
        t_liq["liquidity_direction"] = direction
        t_liq["interpretation"] = (
            f"Ex-repo 3m rolling turnover {direction}: β={t_liq['beta']:+.5f}/month "
            f"({_sig_label(t_liq['p_value'])})."
        )

    # ── OLS (growth on growth): monthly turnover growth ~ monthly CAV growth ─
    mkt["ex_repo_turnover_growth"] = (
        mkt["ex_repo_turnover"]
        .replace(0, np.nan)
        .pct_change(fill_method=None)
        .mul(100)
    )
    ll_market = _ols_growth_on_growth(
        mkt["ex_repo_turnover_growth"], mkt["ex_repo_cav_growth"]
    )
    if ll_market["status"] == "ok":
        ll_market["interpretation"] = (
            f"1 pp CAV monthly growth → {ll_market['beta']:.3f} pp turnover monthly growth "
            f"({_sig_label(ll_market['p_value'])})."
        )
        ll_market["method_note"] = (
            "OLS on monthly growth rates (first differences). "
            "Avoids spurious regression from trending levels."
        )

    # ── Spearman: monthly CAV growth vs volume growth ─────────────────────────
    sp = _spearman(mkt["ex_repo_cav_growth"], mkt["ex_repo_volume_growth"])
    if sp["status"] == "ok":
        sp["interpretation"] = (
            f"Monthly ex-repo CAV vs volume growth: ρ={sp['rho']:+.3f} "
            f"({_sig_label(sp['p_value'])})."
        )

    # ── Per-asset-class: monthly turnover growth ~ monthly CAV growth ────────
    # Turnover growth approximated as pct change in turnover_ratio month-over-month.
    # Uses first differences to avoid spurious regression from trending levels.
    cav_weights = df.sort_values("date").groupby("asset_class")["cav"].last()
    total_weight = float(cav_weights.sum())

    asset_class_results: dict[str, dict] = {}
    weighted_positive_sig = 0.0
    significant_liquidity_classes: list[str] = []

    for ac, group in df.groupby("asset_class"):
        g = group.sort_values("date")
        if "turnover_ratio" not in g.columns:
            asset_class_results[ac] = {
                "status": "insufficient_data",
                "reason": "turnover_ratio column missing",
            }
            continue
        turnover_growth = (
            g["turnover_ratio"].replace(0, np.nan).pct_change(fill_method=None).mul(100)
        )
        r = _ols_growth_on_growth(turnover_growth, g["asset_class_cav_growth"])
        if r["status"] == "ok":
            r["interpretation"] = (
                f"1 pp CAV monthly growth → {r['beta']:.3f} pp turnover monthly growth "
                f"({_sig_label(r['p_value'])})."
            )
            if r["beta"] > 0 and r["p_value"] < 0.05:
                weighted_positive_sig += float(cav_weights.get(ac, 0))
                significant_liquidity_classes.append(ac)
        asset_class_results[ac] = r

    n_positive = len(significant_liquidity_classes)
    n_tested = sum(1 for r in asset_class_results.values() if r.get("status") == "ok")
    cav_weighted_score = (
        float(weighted_positive_sig / total_weight) if total_weight > 0 else None
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    liquidity_growing = (
        t_liq.get("status") == "ok"
        and t_liq["beta"] > 0
        and t_liq["p_value"] < 0.05
    )
    liquidity_tracks_cav = (
        sp.get("status") == "ok"
        and sp["rho"] > 0
        and sp["p_value"] < 0.05
    )

    parts = []
    if liquidity_growing:
        parts.append("Ex-repo market turnover (3m rolling) is improving significantly over time.")
    else:
        parts.append("Ex-repo market turnover shows no significant positive trend.")
    if liquidity_tracks_cav:
        parts.append(f"Volume growth tracks CAV growth positively (ρ={sp.get('rho', 0):.3f}).")
    else:
        parts.append("Volume growth does not significantly track CAV growth month-to-month.")
    if n_tested > 0:
        class_names = (
            f" ({', '.join(significant_liquidity_classes)})"
            if significant_liquidity_classes
            else ""
        )
        parts.append(
            f"{n_positive}/{n_tested} asset classes show significant positive "
            f"CAV-to-turnover relationship{class_names}."
        )
    if cav_weighted_score is not None:
        parts.append(f"CAV-weighted liquidity score: {cav_weighted_score:.2f}.")

    return {
        "turnover_trend_ex_repo":          t_liq,
        "log_log_cav_turnover_ex_repo":    ll_market,
        "spearman_cav_volume_growth":      sp,
        "asset_class_regressions":         asset_class_results,
        "cav_weighted_liquidity_score":    cav_weighted_score,
        "repo_exclusion_note": (
            "Repurchase Agreements report zero on-chain volume in every observation "
            "and are excluded from all liquidity analysis. Including repo CAV in the "
            "market turnover denominator would artificially suppress measured liquidity "
            "(adding ~$280B+ CAV with zero volume lowers the ratio mechanically)."
        ),
        "summary": {
            "liquidity_growing":                liquidity_growing,
            "liquidity_tracks_cav":             liquidity_tracks_cav,
            "n_improving_classes":              n_positive,
            "n_classes_tested":                 n_tested,
            "cav_weighted_liquidity_score":     cav_weighted_score,
            "significant_liquidity_classes":    significant_liquidity_classes,
            "interpretation": " ".join(parts),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conclusion
# ─────────────────────────────────────────────────────────────────────────────

def build_conclusion(p1: dict, p2: dict, p3: dict, p4: dict) -> str:
    """
    Plain-English, CAV-weighted conclusion synthesizing all four pillars.

    Does NOT claim causality. Distinguishes statistical from economic
    significance. Flags data limitations, small-base effects, and the repo
    discontinuity without overstating its impact on the thesis.
    """
    s1 = p1["summary"]
    s2 = p2["summary"]
    s3 = p3["summary"]
    s4 = p4["summary"]

    paras: list[str] = []

    # ── Repo context ──────────────────────────────────────────────────────────
    paras.append(
        "Data context: Repurchase Agreements (repos) entered the rwa.xyz dataset in "
        "June 2025, contributing approximately $282 billion in Circulating Asset Value "
        "(CAV) — roughly 92% of the total market at the time. Repos report zero "
        "on-chain holders and zero on-chain volume throughout the observation window. "
        "This is a known data limitation: on-chain repos exist but rwa.xyz does not yet "
        "fully track their holder or transfer data. All four thesis pillars are evaluated "
        "on ex-repo data as the primary measure. This limitation does not change the "
        "conclusions about the non-repo tokenized RWA market."
    )

    # ── Pillar 1 — Growth ─────────────────────────────────────────────────────
    if s1["is_growing"]:
        ann = s1.get("annualized_growth_rate_ex_repo")
        ann_str = f" (~{ann:.1%}/yr annualized)" if ann else ""
        paras.append(
            f"Pillar 1 — Growth: Ex-repo tokenized RWA value has grown significantly{ann_str} "
            "over the January 2023–May 2026 sample window. This reflects genuine accumulation "
            "of tokenized traditional financial assets on-chain, independent of the repo data "
            "entry event."
        )
    else:
        paras.append(
            "Pillar 1 — Growth: Ex-repo CAV growth is not statistically significant "
            "at conventional levels over the full sample window."
        )

    if s1["is_decelerating"]:
        paras.append(
            "Monthly ex-repo growth rates have been decelerating, suggesting the pace of "
            "initial expansion may be moderating — a common pattern as base effects "
            "normalize after early rapid growth."
        )
    elif s1["is_accelerating"]:
        paras.append("Monthly ex-repo growth rates are accelerating, a positive signal for sustained expansion.")

    if s1["ex_repo_break_detected"]:
        paras.append(
            "A significant acceleration in ex-repo CAV growth was detected around June 2025, "
            "suggesting organic market growth also accelerated in this period "
            "independent of the repo data addition."
        )

    # ── Pillar 2 — Composition ────────────────────────────────────────────────
    if s2["is_diversifying_ex_repo"]:
        hhi_snap = p2.get("hhi_snapshots", {})
        e = hhi_snap.get("hhi_ex_repo_earliest", "")
        l = hhi_snap.get("hhi_ex_repo_latest", "")
        hhi_str = f" (HHI: {e:.3f} → {l:.3f})" if e and l else ""
        paras.append(
            f"Pillar 2 — Composition: Among traditional tokenized assets (ex-repo), "
            f"market concentration has declined significantly{hhi_str}. Capital is spreading "
            "across a broader set of asset classes. Full-market HHI is not the primary "
            "lens here — repos holding 85–92% of total CAV mechanically produce "
            "near-monopoly HHI scores despite genuine underlying diversification."
        )
    else:
        paras.append(
            "Pillar 2 — Composition: Ex-repo market concentration has not declined "
            "significantly. A small number of asset classes continue to dominate "
            "tokenized RWA value."
        )

    if s2["active_classes_growing"]:
        paras.append(
            "The number of economically active (CAV > $1M) asset classes has been rising "
            "significantly — new asset types are entering the market at meaningful scale."
        )
    if s2["top5_falling_pre_repo"]:
        paras.append(
            "The top-5 CAV share was declining prior to the repo entry, indicating "
            "diversification momentum that pre-dates the June 2025 data addition."
        )

    # ── Pillar 3 — Adoption ───────────────────────────────────────────────────
    score3 = s3.get("cav_weighted_adoption_score")

    if s3["holders_growing"] and s3["adoption_tracks_cav"]:
        paras.append(
            "Pillar 3 — Adoption: On-chain participation is growing alongside CAV. "
            "Ex-repo holder counts are increasing significantly and co-move positively "
            "with monthly CAV growth. This is consistent with broadening adoption — "
            "new participants appear to be entering alongside capital inflows."
        )
    elif s3["holders_growing"]:
        paras.append(
            "Pillar 3 — Adoption: Holder counts are growing significantly, but "
            "month-to-month holder growth does not closely track CAV growth. "
            "Adoption is increasing on trend but may be driven by structural onboarding "
            "rather than responding directly to capital flow episodes."
        )
    else:
        paras.append(
            "Pillar 3 — Adoption: Ex-repo holder growth is not statistically significant. "
            "Tokenized value is accumulating, but on-chain participation is not keeping pace."
        )

    if score3 is not None:
        adoption_classes = s3.get("significant_adoption_classes", [])
        class_str = f" ({', '.join(adoption_classes)})" if adoption_classes else ""
        if score3 >= 0.5:
            paras.append(
                f"A CAV-weighted majority ({score3:.0%}) of ex-repo asset value sits in "
                f"classes where monthly holder growth co-moves significantly with CAV "
                f"growth{class_str}."
            )
        elif score3 > 0:
            paras.append(
                f"Only {score3:.0%} of ex-repo CAV (by weight) is in asset classes with "
                f"significant positive adoption co-movement — improvement is concentrated "
                f"in specific segments{class_str}."
            )
        else:
            paras.append(
                "No individual asset class shows significant positive co-movement between "
                "monthly holder growth and CAV growth."
            )

    paras.append(
        "Holder counts are on-chain address proxies, not verified unique participants. "
        "Institutional custodians may control multiple wallets, and market-level totals "
        "double-count wallets holding multiple asset classes. These figures are "
        "directional indicators, not precise participation counts."
    )

    # ── Pillar 4 — Liquidity ──────────────────────────────────────────────────
    score4 = s4.get("cav_weighted_liquidity_score")

    if s4["liquidity_growing"] and s4["liquidity_tracks_cav"]:
        paras.append(
            "Pillar 4 — Liquidity: Secondary-market liquidity is developing. "
            "Ex-repo turnover ratios are trending upward and volume growth tracks "
            "CAV growth positively. Tokenized assets are not simply sitting on-chain — "
            "transfer activity is increasing alongside asset value."
        )
    elif s4["liquidity_growing"]:
        paras.append(
            "Pillar 4 — Liquidity: Ex-repo turnover is improving over time, though "
            "volume growth does not consistently co-move with CAV growth month-to-month. "
            "Liquidity is developing but unevenly across the calendar."
        )
    else:
        paras.append(
            "Pillar 4 — Liquidity: Ex-repo market turnover shows no significant positive "
            "trend. Most tokenized RWA activity appears to be buy-and-hold, with "
            "limited secondary-market turnover relative to asset size."
        )

    if score4 is not None:
        liquidity_classes = s4.get("significant_liquidity_classes", [])
        class_str = f" ({', '.join(liquidity_classes)})" if liquidity_classes else ""
        if score4 > 0:
            paras.append(
                f"CAV-weighted liquidity score: {score4:.0%} of ex-repo asset value is in "
                f"classes where monthly turnover growth co-moves significantly with CAV "
                f"growth{class_str}."
            )
        else:
            paras.append(
                "No individual asset class shows significant positive co-movement between "
                "monthly turnover growth and CAV growth."
            )

    # ── Overall verdict ───────────────────────────────────────────────────────
    pro = sum([
        bool(s1["is_growing"]),
        bool(s2["is_diversifying_ex_repo"]),
        bool(s2["active_classes_growing"]),
        bool(s3["holders_growing"]),
        bool(s3["adoption_tracks_cav"]),
        bool(s4["liquidity_growing"]),
    ])

    if pro >= 5:
        verdict = (
            "Overall Assessment: The statistical evidence across all four pillars is "
            "broadly consistent with a structural shift in the tokenized real-world "
            "asset market. Ex-repo CAV is growing significantly, market composition "
            "is diversifying, on-chain participation is increasing, and secondary "
            "liquidity is developing. These findings are associational — they reflect "
            "co-movement in the data, not a causal chain. Repurchase Agreements "
            "dominate headline market metrics but are a separate institutional layer "
            "with different on-chain data characteristics; the cleaner structural "
            "signal consistently emerges from ex-repo analysis."
        )
    elif pro >= 3:
        verdict = (
            "Overall Assessment: The evidence is mixed but directionally positive. "
            "Some structural shift signals are present across multiple pillars — "
            "growth and diversification are occurring in the ex-repo segment — but "
            "signals are not uniform across all asset classes. Adoption and liquidity "
            "improvements vary, and the headline market size is dominated by "
            "Repurchase Agreements whose on-chain data is incomplete. The data is "
            "consistent with early-stage structural development rather than a fully "
            "established shift. These findings are correlational."
        )
    else:
        verdict = (
            "Overall Assessment: The statistical evidence does not support a strong "
            "structural shift claim at this stage. Growth in ex-repo CAV is occurring, "
            "but diversification, adoption, and liquidity signals are limited or "
            "inconsistent across asset classes. The market may be in a pre-structural "
            "phase — tokenization infrastructure is being built and initial use cases "
            "validated, but broad-based secondary adoption has not yet materialized "
            "in the data. No causal conclusions can be drawn from these associations."
        )

    paras.append(verdict)
    return "\n\n".join(paras)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def _make_serializable(obj):
    """Recursively convert numpy types and NaN/inf to JSON-compatible values."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    combined_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
) -> dict:
    """
    Run all four thesis pillars and return a structured, JSON-serializable dict.

    Parameters
    ----------
    combined_df      : per-(date, asset_class) metrics from combined_metrics.csv
    concentration_df : per-date market metrics from concentration_metrics.csv

    Returns
    -------
    dict with keys: pillar1_growth, pillar2_composition, pillar3_adoption,
                    pillar4_liquidity, conclusion, generated_at, data_window,
                    limitations.
    All NaN / inf values are converted to None for JSON compatibility.
    """
    combined_df      = combined_df.copy()
    concentration_df = concentration_df.copy()
    combined_df["date"]      = pd.to_datetime(combined_df["date"])
    concentration_df["date"] = pd.to_datetime(concentration_df["date"])

    p1 = pillar1_growth(combined_df, concentration_df)
    p2 = pillar2_composition(combined_df, concentration_df)
    p3 = pillar3_adoption(combined_df)
    p4 = pillar4_liquidity(combined_df)

    conclusion_text = build_conclusion(p1, p2, p3, p4)

    dates  = combined_df["date"].dropna().sort_values()
    ex_df  = combined_df[combined_df["asset_class"] != REPO_CLASS]

    data_window = {
        "start":                    str(dates.iloc[0].date())    if not dates.empty else None,
        "end":                      str(dates.iloc[-1].date())   if not dates.empty else None,
        "n_months_total":           int(concentration_df["date"].nunique()),
        "n_months_ex_repo":         int(ex_df["date"].nunique()),
        "n_asset_classes_total":    int(combined_df["asset_class"].nunique()),
        "n_asset_classes_ex_repo":  int(ex_df["asset_class"].nunique()),
        "repo_entry_date":          str(REPO_ENTRY_DATE.date()),
    }

    result = {
        "pillar1_growth":      p1,
        "pillar2_composition": p2,
        "pillar3_adoption":    p3,
        "pillar4_liquidity":   p4,
        "conclusion":          conclusion_text,
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "data_window":         data_window,
        "limitations":         LIMITATIONS,
    }

    return _make_serializable(result)
