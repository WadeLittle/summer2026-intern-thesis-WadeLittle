# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Two responsibilities:
  1. Charting — three static thesis pillar charts saved to CHARTS_DIR.
  2. Statistical analysis — OLS, Spearman, and Kendall tests printed to console
     with a plain-English thesis conclusion.

Pipeline: combined DataFrame -> build_* (data_processing) -> chart + stats -> output
"""

import math
import os
from datetime import datetime

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas

from src.config import CHARTS_DIR, ASSET_CLASSES_IN_SCOPE, ANALYSIS_START_DATE
from src.bronze.data_processing import (
    build_composition_shares,
    build_adoption_index,
    build_avg_position_size,
    build_turnover_ratio,
)

COLORS = {
    "US Treasury Debt":       "#2171b5",
    "Stablecoins":            "#74c476",
    "Commodities":            "#d6a520",
    "Real Estate":            "#e05c4b",
    "Stocks":                 "#3aaa5e",
    "Corporate Credit":       "#f7941d",
    "Asset-Backed Credit":    "#4bc4cf",
    "Private Equity":         "#7b2d8b",
    "Venture Capital":        "#2a9d8f",
    "Active Strategies":      "#264653",
    "Diversified Credit":     "#e9c46a",
    "non-US Government Debt": "#457b9d",
    "Cryptocurrencies":       "#f4a261",
    "Specialty Finance":      "#e76f51",
    "Fiat Currency":          "#6d6875",
    "Repurchase Agreements":  "#b5838d",
}

_START_DT = datetime.fromisoformat(ANALYSIS_START_DATE)
_months_of_data = (datetime.now().year - _START_DT.year) * 12 + (datetime.now().month - _START_DT.month)
_DATE_INTERVAL = 12 if _months_of_data > 72 else (6 if _months_of_data > 24 else 3)

DATE_FMT = mdates.DateFormatter("%b %Y")
DATE_LOC = mdates.MonthLocator(interval=_DATE_INTERVAL)
_XLIM_START = pandas.Timestamp(ANALYSIS_START_DATE)


def _subplot_grid(n, ncols=4):
    """Returns (nrows, ncols) for a grid that fits n subplots."""
    return math.ceil(n / ncols), ncols


# ─────────────────────────────────────────────
# Chart utilities
# ─────────────────────────────────────────────

def _save(fig, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[charts] Saved {path}")


def _apply_date_axis(ax, interval=None):
    loc = mdates.MonthLocator(interval=interval or _DATE_INTERVAL)
    ax.xaxis.set_major_formatter(DATE_FMT)
    ax.xaxis.set_major_locator(loc)
    ax.set_xlim(left=_XLIM_START)


# ─────────────────────────────────────────────
# CHART 1: Asset Class Composition
# ─────────────────────────────────────────────

def plot_composition(df):
    """Pillar 1: 100% stacked area chart of CAV share by asset class."""
    data = build_composition_shares(df)
    pivot = (
        data.pivot_table(index="date", columns="asset_class", values="cav_share")
        .reindex(columns=ASSET_CLASSES_IN_SCOPE)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(
        pivot.index,
        [pivot[cls] * 100 for cls in ASSET_CLASSES_IN_SCOPE],
        labels=ASSET_CLASSES_IN_SCOPE,
        colors=[COLORS[cls] for cls in ASSET_CLASSES_IN_SCOPE],
        alpha=0.85,
    )
    ax.set_title("RWA Market Composition by Asset Class", fontsize=14, pad=12)
    ax.set_ylabel("Share of Circulating Asset Value (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    _apply_date_axis(ax)
    ax.margins(x=0)
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    _save(fig, "chart1_composition.png")


# ─────────────────────────────────────────────
# CHART 2: Adoption — Holder Growth vs. Asset Growth
# ─────────────────────────────────────────────

def plot_adoption(df):
    """
    Pillar 2: One subplot per asset class.
    Each subplot: CAV index (dashed) vs. holder index (solid), indexed to each
    class's first available month = 100. Classes that start later appear as NaN
    until their data begins, then rise from 100.
    """
    data = build_adoption_index(df)
    n = len(ASSET_CLASSES_IN_SCOPE)
    nrows, ncols = _subplot_grid(n)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), sharex=True)
    fig.suptitle(
        "Adoption: On-Chain Participation vs. Asset Value Growth\n(Indexed to each class's first available month = 100)",
        fontsize=13,
    )
    axes_flat = axes.flatten()
    bottom_row_start = (nrows - 1) * ncols

    for i, asset_class in enumerate(ASSET_CLASSES_IN_SCOPE):
        ax = axes_flat[i]
        subset = data[data["asset_class"] == asset_class].sort_values("date")
        color = COLORS[asset_class]

        ax.plot(subset["date"], subset["cav_index"],
                color=color, linestyle="--", linewidth=1.8, label="CAV Index")
        ax.plot(subset["date"], subset["holders_index"],
                color=color, linestyle="-", linewidth=1.8, label="Holder Index")
        ax.axhline(100, color="gray", linewidth=0.7, linestyle=":")

        ax.set_title(asset_class, fontsize=9)
        ax.set_ylabel("Index", fontsize=7)
        _apply_date_axis(ax)

        if i < bottom_row_start:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "chart2_adoption.png")


# ─────────────────────────────────────────────
# CHART 2b: Adoption — Average Position Size
# ─────────────────────────────────────────────

def _dollar_fmt(x, _):
    if x >= 1_000_000_000:
        return f"${x/1_000_000_000:.1f}B"
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


def _count_fmt(x, _):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def plot_avg_position_size(df):
    """
    Pillar 2 (supplemental): Average dollar value held per wallet (CAV / holders).
    Declining line = new smaller participants entering = broadening adoption.
    Rising line = existing holders adding capital = concentration.
    """
    data = build_avg_position_size(df)
    n = len(ASSET_CLASSES_IN_SCOPE)
    nrows, ncols = _subplot_grid(n)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), sharex=True)
    fig.suptitle(
        "Adoption: Average Position Size per Wallet by Asset Class",
        fontsize=13,
    )
    axes_flat = axes.flatten()
    bottom_row_start = (nrows - 1) * ncols

    for i, asset_class in enumerate(ASSET_CLASSES_IN_SCOPE):
        ax = axes_flat[i]
        subset = data[data["asset_class"] == asset_class].sort_values("date")
        color = COLORS[asset_class]

        ax.plot(subset["date"], subset["avg_position"],
                color=color, linewidth=2)

        ax.set_title(asset_class, fontsize=9)
        ax.set_ylabel("Avg $ per Wallet", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
        _apply_date_axis(ax)

        if i < bottom_row_start:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "chart2b_avg_position.png")


# ─────────────────────────────────────────────
# CHART 2c: Adoption — Context (Nx3 grid)
# ─────────────────────────────────────────────

def plot_adoption_context(df):
    """
    Pillar 2 (context): Nx3 grid showing the building blocks of the adoption index.
    Rows: one per asset class.
    Columns: raw holder count | raw CAV | index chart (CAV vs holder index).
    """
    data = build_adoption_index(df)
    n = len(ASSET_CLASSES_IN_SCOPE)

    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n), sharex=True)
    fig.suptitle(
        "Adoption: From Raw Data to Index (first available month per class = 100)",
        fontsize=14,
    )

    col_titles = ["Holder Count", "Circulating Asset Value (CAV)", "Index (first month = 100)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i, asset_class in enumerate(ASSET_CLASSES_IN_SCOPE):
        subset = data[data["asset_class"] == asset_class].sort_values("date")
        color = COLORS[asset_class]

        axes[i, 0].set_ylabel(asset_class, fontsize=8, fontweight="bold", labelpad=8)

        axes[i, 0].plot(subset["date"], subset["holders"], color=color, linewidth=1.8)
        axes[i, 0].yaxis.set_major_formatter(mticker.FuncFormatter(_count_fmt))

        axes[i, 1].plot(subset["date"], subset["cav"], color=color, linewidth=1.8)
        axes[i, 1].yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))

        axes[i, 2].plot(subset["date"], subset["cav_index"],
                        color=color, linestyle="--", linewidth=1.8, label="CAV Index")
        axes[i, 2].plot(subset["date"], subset["holders_index"],
                        color=color, linestyle="-", linewidth=1.8, label="Holder Index")
        axes[i, 2].axhline(100, color="gray", linewidth=0.7, linestyle=":")

        for j in range(3):
            _apply_date_axis(axes[i, j])
            if i < n - 1:
                plt.setp(axes[i, j].get_xticklabels(), visible=False)
            else:
                plt.setp(axes[i, j].get_xticklabels(), rotation=30, ha="right", fontsize=7)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=1, fontsize=9,
               bbox_to_anchor=(0.99, 0.01))

    fig.tight_layout()
    _save(fig, "chart2c_adoption_context.png")


# ─────────────────────────────────────────────
# CHART 3: Liquidity — Monthly Turnover Ratio
# ─────────────────────────────────────────────

def plot_liquidity(df):
    """
    Pillar 3: Turnover ratio per asset class.
    Faint raw monthly line + bold 3-month rolling average per asset class.
    """
    data = build_turnover_ratio(df)

    fig, ax = plt.subplots(figsize=(13, 5))
    for asset_class in ASSET_CLASSES_IN_SCOPE:
        subset = data[data["asset_class"] == asset_class].sort_values("date")
        color = COLORS[asset_class]
        ax.plot(subset["date"], subset["turnover_ratio"],
                color=color, linewidth=0.8, alpha=0.3)
        ax.plot(subset["date"], subset["turnover_3m"],
                color=color, linewidth=2, label=asset_class)

    ax.set_title("Liquidity: Monthly Turnover Ratio by Asset Class", fontsize=14, pad=12)
    ax.set_ylabel("Transfer Volume / Avg Monthly Circulating Asset Value")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _apply_date_axis(ax)
    fig.autofmt_xdate()
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "chart3_liquidity.png")


def run_all_charts(df):
    plot_composition(df)
    plot_adoption(df)
    plot_avg_position_size(df)
    plot_adoption_context(df)
    plot_liquidity(df)


# ─────────────────────────────────────────────
# Statistical Analysis
# ─────────────────────────────────────────────

def _fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _sig_label(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "* "
    if p < 0.10:
        return "~ "
    return "  "


def compute_hhi_trend(df):
    """
    Pillar 1: Monthly HHI from CAV shares, then OLS of HHI on time.
    Returns beta, p-value, R², and start/end HHI values.
    """
    data = build_composition_shares(df)
    hhi = (
        data.groupby("date")["cav_share"]
        .apply(lambda s: (s ** 2).sum())
        .reset_index()
        .rename(columns={"cav_share": "hhi"})
        .sort_values("date")
    )
    hhi["t"] = range(len(hhi))

    X = sm.add_constant(hhi["t"])
    model = sm.OLS(hhi["hhi"], X).fit()

    return {
        "beta":      model.params["t"],
        "p_value":   model.pvalues["t"],
        "r_squared": model.rsquared,
        "hhi_start": hhi["hhi"].iloc[0],
        "hhi_end":   hhi["hhi"].iloc[-1],
        "n_months":  len(hhi),
    }


def compute_adoption_stats(df):
    """
    Pillar 2: Spearman correlation (CAV index vs holder index) and
    Kendall tau trend test on the participation ratio, per asset class.
    """
    data = build_adoption_index(df)
    results = {}

    for asset_class, group in data.groupby("asset_class"):
        g = group.sort_values("date").dropna(subset=["cav_index", "holders_index",
                                                      "participation_ratio"])
        if len(g) < 5:
            continue
        rho, rho_p = stats.spearmanr(g["cav_index"], g["holders_index"])

        tau, tau_p = stats.kendalltau(range(len(g)), g["participation_ratio"])

        results[asset_class] = {
            "spearman_rho": rho,
            "spearman_p":   rho_p,
            "kendall_tau":  tau,
            "kendall_p":    tau_p,
            "ratio_start":  g["participation_ratio"].iloc[0],
            "ratio_end":    g["participation_ratio"].iloc[-1],
        }

    return results


def compute_liquidity_trend(df):
    """
    Pillar 3: OLS regression of 3-month rolling turnover ratio on time,
    per asset class. Returns beta, p-value, and R² per class.
    """
    data = build_turnover_ratio(df)
    results = {}

    for asset_class, group in data.groupby("asset_class"):
        g = group.sort_values("date").dropna(subset=["turnover_3m"])
        if len(g) < 5:
            continue
        g = g.copy()
        g["t"] = range(len(g))

        X = sm.add_constant(g["t"])
        model = sm.OLS(g["turnover_3m"], X).fit()

        results[asset_class] = {
            "beta":      model.params["t"],
            "p_value":   model.pvalues["t"],
            "r_squared": model.rsquared,
        }

    return results


def _build_conclusion(hhi, adoption, liquidity):
    """
    Generates a plain-English thesis conclusion from the statistical results.
    Framed as evidence, not causality — suitable for a banking audience.
    """
    n_classes = len(ASSET_CLASSES_IN_SCOPE)
    lines = []

    # --- Pillar 1 verdict ---
    if hhi["p_value"] < 0.05 and hhi["beta"] < 0:
        lines.append(
            f"Market concentration declined at a statistically significant rate "
            f"(β = {hhi['beta']:.4f}/month, p = {_fmt_p(hhi['p_value'])}), with HHI "
            f"falling from {hhi['hhi_start']:.3f} to {hhi['hhi_end']:.3f}. "
            f"This is consistent with gradual diversification across asset classes."
        )
    elif hhi["p_value"] < 0.10 and hhi["beta"] < 0:
        lines.append(
            f"There is marginal evidence of declining concentration "
            f"(β = {hhi['beta']:.4f}/month, p = {_fmt_p(hhi['p_value'])}), "
            f"though the trend is not yet statistically robust at the 5% level."
        )
    else:
        lines.append(
            f"Market concentration showed no statistically significant trend "
            f"(β = {hhi['beta']:.4f}/month, p = {_fmt_p(hhi['p_value'])}). "
            f"The RWA market remained heavily concentrated throughout the period."
        )

    # --- Pillar 2 verdict ---
    n_with_adoption = len(adoption)
    broadening = [ac for ac, r in adoption.items()
                  if r["kendall_tau"] > 0 and r["kendall_p"] < 0.05]
    narrowing  = [ac for ac, r in adoption.items()
                  if r["kendall_tau"] < 0 and r["kendall_p"] < 0.05]

    if len(broadening) >= n_with_adoption * 0.75:
        lines.append(
            f"On-chain participation broadened relative to asset value growth in "
            f"{len(broadening)} of {n_with_adoption} asset classes ({', '.join(broadening)}), "
            f"suggesting wallet adoption is outpacing capital inflows in most categories."
        )
    elif len(broadening) >= 1:
        lines.append(
            f"On-chain participation showed broadening in {len(broadening)} asset class(es) "
            f"({', '.join(broadening)}), but the pattern was not consistent across all classes."
        )
    else:
        lines.append(
            "On-chain participation did not broadly outpace asset value growth. "
            "Capital inflows appear to be concentrated among existing large wallets."
        )

    if narrowing:
        lines.append(
            f"In {', '.join(narrowing)}, value growth outpaced wallet participation, "
            f"which may reflect institutional concentration rather than broad adoption."
        )

    # --- Pillar 3 verdict ---
    n_with_liquidity = len(liquidity)
    improving = [ac for ac, r in liquidity.items()
                 if r["beta"] > 0 and r["p_value"] < 0.05]

    if len(improving) >= n_with_liquidity * 0.75:
        lines.append(
            f"Turnover ratios improved significantly in {len(improving)} of {n_with_liquidity} "
            f"asset classes ({', '.join(improving)}), providing evidence that tokenization is "
            f"delivering measurable secondary market activity, not just on-chain custody."
        )
    elif len(improving) >= 1:
        lines.append(
            f"Liquidity improved significantly in {len(improving)} asset class(es) "
            f"({', '.join(improving)}). Remaining classes showed flat or inconsistent turnover, "
            f"suggesting liquidity development is uneven across the market."
        )
    else:
        lines.append(
            "No asset class showed a statistically significant improvement in turnover ratio. "
            "Circulating value is growing, but transfer activity has not kept pace — "
            "consistent with a buy-and-hold tokenized asset market rather than an active one."
        )

    # --- Overall verdict ---
    pro_signals = (
        (1 if hhi["p_value"] < 0.05 and hhi["beta"] < 0 else 0) +
        len(broadening) +
        len(improving)
    )
    signal_threshold_high = 1 + n_with_adoption // 2 + n_with_liquidity // 2

    if pro_signals >= signal_threshold_high:
        verdict = (
            "Taken together, the data provides meaningful statistical evidence of a structural "
            "shift in RWA tokenization — declining concentration, broadening on-chain "
            "participation, and improving liquidity across multiple asset classes."
        )
    elif pro_signals >= 2:
        verdict = (
            "The data provides mixed but directionally positive evidence. Some structural "
            "shift signals are present, but they are not consistent across all asset classes "
            "or all three pillars. Caution is warranted in characterizing this as a broad "
            "structural shift — it may reflect early-stage diversification rather than a "
            "sustained trend."
        )
    else:
        verdict = (
            "The statistical evidence does not support a strong structural shift claim at this "
            "stage. Growth appears concentrated in value terms, with limited diversification "
            "and uneven liquidity development. The market may be in a pre-structural phase "
            "where tokenization infrastructure is being built but secondary adoption lags."
        )

    lines.append(verdict)
    return lines


def run_analysis(df):
    """
    Runs all three statistical pillars, prints a formatted summary table,
    and prints a plain-English thesis conclusion.
    """
    hhi      = compute_hhi_trend(df)
    adoption = compute_adoption_stats(df)
    liquidity = compute_liquidity_trend(df)

    W = 70
    period_start = df["date"].min().strftime("%b %Y")
    period_end   = df["date"].max().strftime("%b %Y")
    n_classes    = len(ASSET_CLASSES_IN_SCOPE)

    print("\n" + "═" * W)
    print(f"  RWA TOKENIZATION — STRUCTURAL SHIFT ANALYSIS")
    print(f"  Data: rwa.xyz API  |  Period: {period_start}–{period_end}  |  n={hhi['n_months']} months")
    print(f"  Asset classes in scope: {n_classes}")
    print("═" * W)

    # ── Pillar 1 ──────────────────────────────────────────────────────────
    print(f"\nPILLAR 1 — MARKET CONCENTRATION  (HHI)")
    print(f"  Scale: 1.0 = one class dominates  |  {1/n_classes:.2f} = equal weight")
    print(f"  {'Start HHI':<18} {hhi['hhi_start']:.4f}")
    print(f"  {'End HHI':<18} {hhi['hhi_end']:.4f}")
    print(f"  {'OLS β/month':<18} {hhi['beta']:+.5f}   "
          f"p = {_fmt_p(hhi['p_value'])} {_sig_label(hhi['p_value'])}  "
          f"R² = {hhi['r_squared']:.3f}")

    # ── Pillar 2 ──────────────────────────────────────────────────────────
    print(f"\nPILLAR 2 — ON-CHAIN PARTICIPATION")
    print(f"  Spearman ρ: correlation between CAV index and holder index")
    print(f"  Kendall τ:  monotonic trend in participation ratio (holders/CAV), + = broadening")
    print()
    hdr = f"  {'Asset Class':<28} {'Spearman ρ':>10}  {'p':>7}    {'Kendall τ':>10}  {'p':>7}  {'Ratio':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for ac in ASSET_CLASSES_IN_SCOPE:
        if ac not in adoption:
            print(f"  {ac:<28} {'(insufficient data)':>50}")
            continue
        r = adoption[ac]
        ratio_dir = f"{r['ratio_start']:.2f} → {r['ratio_end']:.2f}"
        print(f"  {ac:<28} {r['spearman_rho']:>+10.3f}  {_fmt_p(r['spearman_p']):>7} "
              f"{_sig_label(r['spearman_p'])}  {r['kendall_tau']:>+10.3f}  "
              f"{_fmt_p(r['kendall_p']):>7} {_sig_label(r['kendall_p'])} "
              f"  {ratio_dir:>12}")

    # ── Pillar 3 ──────────────────────────────────────────────────────────
    print(f"\nPILLAR 3 — LIQUIDITY  (3-month rolling turnover OLS)")
    print(f"  β/month: change in monthly turnover ratio per month  |  + = improving liquidity")
    print()
    hdr3 = f"  {'Asset Class':<28} {'β/month':>10}  {'p':>7}    {'R²':>6}"
    print(hdr3)
    print("  " + "─" * (len(hdr3) - 2))
    for ac in ASSET_CLASSES_IN_SCOPE:
        if ac not in liquidity:
            print(f"  {ac:<28} {'(insufficient data)':>30}")
            continue
        r = liquidity[ac]
        print(f"  {ac:<28} {r['beta']:>+10.5f}  {_fmt_p(r['p_value']):>7} "
              f"{_sig_label(r['p_value'])}  {r['r_squared']:>6.3f}")

    print(f"\n  Significance: ** p<0.01   * p<0.05   ~ p<0.10")

    # ── Conclusion ────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  THESIS CONCLUSION")
    print("─" * W)
    conclusion = _build_conclusion(hhi, adoption, liquidity)
    for para in conclusion:
        words = para.split()
        line, out = [], []
        for word in words:
            if sum(len(w) for w in line) + len(line) + len(word) > W - 4:
                out.append("  " + " ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            out.append("  " + " ".join(line))
        print("\n".join(out))
        print()
    print("═" * W + "\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from src.bronze.api_client import (
        get_cav_by_asset_class,
        get_holders_by_asset_class,
        get_transfer_volume_by_asset_class,
        cached_request,
    )
    from src.bronze.data_processing import build_combined_dataset

    cav     = cached_request(get_cav_by_asset_class, "cav_by_asset_class")
    holders = cached_request(get_holders_by_asset_class, "holders_by_asset_class")
    volume  = cached_request(get_transfer_volume_by_asset_class, "volume_by_asset_class")
    df      = build_combined_dataset(cav, holders, volume)

    run_all_charts(df)
    run_analysis(df)
