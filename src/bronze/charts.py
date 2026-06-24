"""
Reusable chart functions for thesis bronze layer.

Each function accepts a metrics DataFrame and an output directory path,
saves PNG (and PDF) files, and returns the output path or None if skipped.
All functions guard against missing columns and empty data.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import ASSET_CLASS_COLORS, OTHER_COLOR, REPO_CLASS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (8, 7)
DPI = 150
FONT_TITLE = {"fontsize": 14, "fontweight": "bold"}
FONT_LABEL = {"fontsize": 11}
FONT_LEGEND = {"fontsize": 9}

# Asset classes to keep individually; the rest collapse to "Other"
TOP_N_CLASSES = 6

# Colors for non-class aggregate series (growth rate lines, dual-axis charts)
_SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    """Save PNG to out_dir/png/ and PDF to out_dir/pdf/; return PNG path."""
    png_dir = out_dir / "png"
    pdf_dir = out_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _require_columns(df: pd.DataFrame, cols: list[str], chart_name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [SKIP] {chart_name}: missing columns {missing}")
        return False
    return True


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _fmt_billions(x: float, _pos=None) -> str:
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x:,.0f}"


def _top_n_classes(df: pd.DataFrame, n: int) -> list[str]:
    """Return top-N asset classes by mean CAV descending."""
    return (
        df.groupby("asset_class")["cav"]
        .mean()
        .nlargest(n)
        .index.tolist()
    )


def _collapse_to_other(df: pd.DataFrame, top_classes: list[str]) -> pd.DataFrame:
    """Replace asset classes outside top_classes with 'Other' and reaggregate."""
    df = df.copy()
    df["asset_class"] = df["asset_class"].where(
        df["asset_class"].isin(top_classes), "Other"
    )
    return (
        df.groupby(["date", "asset_class"], as_index=False)
        .agg({c: "sum" for c in df.select_dtypes("number").columns})
    )


def _class_color(cls: str) -> str:
    """Return the canonical color for an asset class, falling back to OTHER_COLOR."""
    return ASSET_CLASS_COLORS.get(cls, OTHER_COLOR)


# ---------------------------------------------------------------------------
# Chart 1 — Total CAV over time
# ---------------------------------------------------------------------------

def chart1_total_cav(
    conc: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "total_cav"]
    if not _require_columns(conc, required, "chart1_total_cav"):
        return None
    if conc["total_cav"].dropna().empty:
        print("  [SKIP] chart1_total_cav: no data")
        return None

    df = _parse_dates(conc).dropna(subset=["total_cav"])

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(df["date"], df["total_cav"], linewidth=2, color=_SERIES_COLORS[0])
        ax.fill_between(df["date"], df["total_cav"], alpha=0.15, color=_SERIES_COLORS[0])
        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title("Total Crypto Asset Value (CAV) Over Time", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Total CAV", **FONT_LABEL)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart1_total_cav")


# ---------------------------------------------------------------------------
# Chart 2 — CAV by asset class (stacked area)
# ---------------------------------------------------------------------------

def chart2_cav_by_asset_class(
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = TOP_N_CLASSES,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart2_cav_by_asset_class"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav"])
    top_classes = _top_n_classes(df, top_n)
    df = _collapse_to_other(df, top_classes)

    pivot = df.pivot_table(index="date", columns="asset_class", values="cav", aggfunc="sum").fillna(0)
    # Order columns: top classes first, Other last
    ordered = top_classes + (["Other"] if "Other" in pivot.columns else [])
    pivot = pivot[ordered]

    colors = [_class_color(c) for c in ordered]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, colors=colors, alpha=0.85)
        ax.set_xlim(pivot.index.min(), pivot.index.max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title("CAV by Asset Class Over Time", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("CAV", **FONT_LABEL)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", **FONT_LEGEND)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart2_cav_by_asset_class")


# HHI concentration (original chart3) removed — repo dominance makes total-market HHI
# uninformative. See chart14 for the cleaner ex-repo concentration story.


# ---------------------------------------------------------------------------
# Chart 3 — Holder count by asset class over time (raw, not indexed)
# ---------------------------------------------------------------------------

def chart3_holder_growth(
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = TOP_N_CLASSES,
) -> Optional[Path]:
    required = ["date", "asset_class", "holders"]
    if not _require_columns(combined, required, "chart3_holder_growth"):
        return None

    df = _parse_dates(combined).dropna(subset=["holders"])
    # Rank by mean holder count so the most-adopted classes appear
    top_classes = (
        df.groupby("asset_class")["holders"]
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for cls in top_classes:
            sub = df[df["asset_class"] == cls]
            if sub.empty:
                continue
            ax.plot(sub["date"], sub["holders"], label=cls,
                    color=_class_color(cls), linewidth=1.8)

        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_title(f"Holder Count by Asset Class Over Time (Top {top_n} by Holders)", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Holders", **FONT_LABEL)
        ax.legend(loc="upper left", **FONT_LEGEND)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart3_holders_by_class")


# ---------------------------------------------------------------------------
# Chart 4 — Turnover ratio over time
# ---------------------------------------------------------------------------

def chart4_turnover_ratio(
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = TOP_N_CLASSES,
) -> Optional[Path]:
    required = ["date", "asset_class", "turnover_ratio"]
    if not _require_columns(combined, required, "chart4_turnover_ratio"):
        return None

    df = _parse_dates(combined).dropna(subset=["turnover_ratio"])
    # Rank by median turnover so the most actively-traded classes appear,
    # not the largest by CAV (which may have near-zero turnover).
    top_classes = (
        df.groupby("asset_class")["turnover_ratio"]
        .median()
        .nlargest(top_n)
        .index.tolist()
    )

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for cls in top_classes:
            sub = df[df["asset_class"] == cls]
            if sub.empty:
                continue
            ax.plot(sub["date"], sub["turnover_ratio"], label=cls, color=_class_color(cls), linewidth=1.8)

        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_title(f"Monthly Turnover Ratio — Top {top_n} Most Active Asset Classes", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Turnover Ratio", **FONT_LABEL)
        ax.legend(loc="upper right", **FONT_LEGEND)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart4_turnover_ratio")


# chart6 removed — total-market rolling growth is dominated by the repo
# expansion spike; chart15 shows the cleaner ex-repo version.
# chart7 removed — multi-month log scatter replaced by chart11 market map.
# chart8 removed — log CAV vs turnover scatter replaced by chart11 market map.


# ---------------------------------------------------------------------------
# Chart 5 — Thesis scorecard summary
# ---------------------------------------------------------------------------

def chart5_scorecard(
    conc: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    """Single-page scorecard with key thesis metrics at latest date."""
    required_conc = ["date", "total_cav", "hhi", "top_5_share", "monthly_cav_growth"]
    required_comb = ["date", "asset_class", "cav", "holders"]
    if not _require_columns(conc, required_conc, "chart5_scorecard"):
        return None
    if not _require_columns(combined, required_comb, "chart5_scorecard"):
        return None

    conc_df = _parse_dates(conc).dropna(subset=["total_cav"])
    comb_df = _parse_dates(combined).dropna(subset=["cav"])
    if conc_df.empty:
        print("  [SKIP] chart5_scorecard: no concentration data")
        return None

    latest = conc_df.iloc[-1]
    latest_date = latest["date"].strftime("%B %Y")
    total_cav = latest["total_cav"]
    hhi = latest["hhi"]
    top5 = latest["top_5_share"] * 100
    monthly_growth = latest.get("monthly_cav_growth", float("nan"))

    # Total unique holders at latest date
    latest_comb = comb_df[comb_df["date"] == comb_df["date"].max()]
    total_holders = latest_comb["holders"].sum()

    # Dominant asset class
    dominant_class = latest_comb.nlargest(1, "cav")["asset_class"].values[0] if not latest_comb.empty else "N/A"
    dominant_share = (
        latest_comb.nlargest(1, "cav")["cav"].values[0] / total_cav * 100
        if not latest_comb.empty and total_cav > 0 else float("nan")
    )

    metrics = [
        ("Total CAV", _fmt_billions(total_cav)),
        ("HHI", f"{hhi:.4f}"),
        ("Top-5 Share", f"{top5:.1f}%"),
        ("Monthly Growth", f"{monthly_growth:.1f}%" if not np.isnan(monthly_growth) else "N/A"),
        ("Total Holders", f"{total_holders:,.0f}"),
        ("Largest Class", f"{dominant_class}\n({dominant_share:.1f}% of CAV)"),
    ]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")

        ax.text(
            0.5, 0.97,
            f"Thesis Scorecard — {latest_date}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=15, fontweight="bold"
        )

        n_cols = 3
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        box_w = 1.0 / n_cols
        box_h = 0.72 / n_rows

        for idx, (label, value) in enumerate(metrics):
            col = idx % n_cols
            row = idx // n_cols
            x = (col + 0.5) * box_w
            y = 0.88 - row * box_h - box_h * 0.5

            ax.text(x, y + 0.04, label, ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#555555")
            ax.text(x, y - 0.06, value, ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, fontweight="bold", color="#1f77b4")

            rect = plt.Rectangle(
                (col * box_w + 0.02, 0.88 - (row + 1) * box_h + 0.02),
                box_w - 0.04, box_h - 0.04,
                transform=ax.transAxes,
                fill=True, facecolor="#f0f4f8", edgecolor="#c0c8d0",
                linewidth=1, clip_on=False
            )
            ax.add_patch(rect)

        fig.tight_layout()

    return _savefig(fig, out_dir, "chart5_scorecard")


# ---------------------------------------------------------------------------
# Additional helpers for new charts
# ---------------------------------------------------------------------------

def _label_line_end(
    ax: plt.Axes,
    x_vals: pd.Series,
    y_vals: pd.Series,
    label: str,
    color: str,
    fontsize: int = 8,
    pad: int = 4,
) -> None:
    """Annotate the last non-NaN point of a line with its label."""
    mask = pd.notna(y_vals)
    if not mask.any():
        return
    last_x = x_vals[mask].iloc[-1]
    last_y = float(y_vals[mask].iloc[-1])
    ax.annotate(
        label, xy=(last_x, last_y), xytext=(pad, 0),
        textcoords="offset points", va="center", ha="left",
        color=color, fontsize=fontsize, clip_on=False,
        fontweight="semibold",
    )


def _ex_repo_total_cav_series(combined: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed Series of total CAV excluding Repurchase Agreements."""
    df = _parse_dates(combined)
    return (
        df[df["asset_class"] != REPO_CLASS]
        .groupby("date")["cav"]
        .sum()
        .rename("total_cav_ex_repo")
        .reset_index()
    )


def _rolling_growth_from_series(s: pd.Series, window_3m: int = 3, window_6m: int = 6) -> pd.DataFrame:
    """Compute monthly, 3m, and 6m rolling CAV growth rates from a total_cav Series."""
    monthly = s.pct_change() * 100
    r3 = s.pct_change(window_3m) * 100
    r6 = s.pct_change(window_6m) * 100
    return pd.DataFrame({"monthly": monthly, "rolling_3m": r3, "rolling_6m": r6})


def _recompute_hhi(combined: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Recompute HHI and top-5 share by date from combined metrics."""
    df = _parse_dates(combined).copy()
    if exclude:
        df = df[~df["asset_class"].isin(exclude)]
    total = df.groupby("date")["cav"].transform("sum")
    df["share"] = df["cav"] / total
    hhi = df.groupby("date")["share"].apply(lambda s: (s ** 2).sum()).rename("hhi")
    top5 = df.groupby("date").apply(
        lambda g: g.nlargest(5, "share")["share"].sum(), include_groups=False
    ).rename("top_5_share")
    return pd.concat([hhi, top5], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Chart 6 — Total CAV over time, Ex-Repurchase Agreements
# ---------------------------------------------------------------------------

def chart6_ex_repo_total_cav(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart6_ex_repo_total_cav"):
        return None

    df = _ex_repo_total_cav_series(combined)
    if df.empty:
        print("  [SKIP] chart6_ex_repo_total_cav: no data after excluding repo")
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(df["date"], df["total_cav_ex_repo"], linewidth=2, color=_SERIES_COLORS[2])
        ax.fill_between(df["date"], df["total_cav_ex_repo"], alpha=0.15, color=_SERIES_COLORS[2])
        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title(
            f"Total CAV Over Time — Excluding {REPO_CLASS}", **FONT_TITLE
        )
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Total CAV (ex-Repo)", **FONT_LABEL)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart6_ex_repo_total_cav")


# ---------------------------------------------------------------------------
# Chart 7 — CAV by Asset Class, Ex-Repo (stacked area)
# ---------------------------------------------------------------------------

def chart7_ex_repo_by_class(
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = TOP_N_CLASSES,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart7_ex_repo_by_class"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav"])
    df = df[df["asset_class"] != REPO_CLASS]
    if df.empty:
        print("  [SKIP] chart7_ex_repo_by_class: no data after excluding repo")
        return None

    top_classes = _top_n_classes(df, top_n)
    df = _collapse_to_other(df, top_classes)
    pivot = df.pivot_table(index="date", columns="asset_class", values="cav", aggfunc="sum").fillna(0)
    ordered = top_classes + (["Other"] if "Other" in pivot.columns else [])
    pivot = pivot[[c for c in ordered if c in pivot.columns]]
    colors = [_class_color(c) for c in ordered]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, colors=colors, alpha=0.85)
        ax.set_xlim(pivot.index.min(), pivot.index.max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title(f"CAV by Asset Class — Excluding {REPO_CLASS}", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("CAV", **FONT_LABEL)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", **FONT_LEGEND)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart7_ex_repo_by_class")


# ---------------------------------------------------------------------------
# Chart 8 — Latest CAV by Asset Class (horizontal bar, with & without repo)
# ---------------------------------------------------------------------------

def chart8_latest_composition(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart8_latest_composition"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav"])
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    snap = snap[snap["cav"] > 0].sort_values("cav", ascending=True)
    total_all = snap["cav"].sum()

    snap_ex = snap[snap["asset_class"] != REPO_CLASS].copy()
    total_ex = snap_ex["cav"].sum()
    snap_ex = snap_ex.sort_values("cav", ascending=True)

    with plt.style.context(STYLE):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Market Composition — {latest_date.strftime('%B %Y')}",
            **FONT_TITLE, y=1.02
        )

        # Left: including repo
        ax_left.barh(snap["asset_class"], snap["cav"],
                     color=[_class_color(c) for c in snap["asset_class"]])
        ax_left.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax_left.set_title("All Asset Classes", fontsize=11)
        ax_left.set_xlabel("CAV", **FONT_LABEL)
        ax_left.tick_params(axis="y", labelsize=8)
        for bar, (_, row) in zip(ax_left.patches, snap.iterrows()):
            share = row["cav"] / total_all * 100
            ax_left.text(
                bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%", va="center", ha="left", fontsize=7, color="#333333"
            )

        # Right: excluding repo
        ax_right.barh(snap_ex["asset_class"], snap_ex["cav"],
                      color=[_class_color(c) for c in snap_ex["asset_class"]])
        ax_right.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax_right.set_title(f"Excluding {REPO_CLASS}", fontsize=11)
        ax_right.set_xlabel("CAV", **FONT_LABEL)
        ax_right.tick_params(axis="y", labelsize=8)
        for bar, (_, row) in zip(ax_right.patches, snap_ex.iterrows()):
            share = row["cav"] / total_ex * 100
            ax_right.text(
                bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%", va="center", ha="left", fontsize=7, color="#333333"
            )

        fig.tight_layout()

    return _savefig(fig, out_dir, "chart8_latest_composition")


# ---------------------------------------------------------------------------
# Chart 9 — Asset Class Share of Total CAV Over Time
# ---------------------------------------------------------------------------

def chart9_cav_share_over_time(
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = TOP_N_CLASSES,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart9_cav_share_over_time"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav"])
    total_by_date = df.groupby("date")["cav"].sum()
    df = df.copy()
    df["cav_share_pct"] = df["cav"] / df["date"].map(total_by_date) * 100

    df_ex = df[df["asset_class"] != REPO_CLASS].copy()
    total_ex_by_date = df_ex.groupby("date")["cav"].sum()
    df_ex["cav_share_pct"] = df_ex["cav"] / df_ex["date"].map(total_ex_by_date) * 100

    def _make_pivot(data: pd.DataFrame, n: int) -> pd.DataFrame:
        top = _top_n_classes(data, n)
        data = _collapse_to_other(data.copy(), top)
        pivot = data.pivot_table(
            index="date", columns="asset_class", values="cav_share_pct", aggfunc="sum"
        ).fillna(0)
        ordered = top + (["Other"] if "Other" in pivot.columns else [])
        return pivot[[c for c in ordered if c in pivot.columns]]

    pivot_all = _make_pivot(df, top_n)
    pivot_ex = _make_pivot(df_ex, top_n)
    colors_all = [_class_color(c) for c in pivot_all.columns]
    colors_ex = [_class_color(c) for c in pivot_ex.columns]

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
        fig.suptitle("Asset Class Share of Total CAV Over Time", **FONT_TITLE, y=1.02)

        ax1.stackplot(pivot_all.index, pivot_all.T.values, labels=pivot_all.columns,
                      colors=colors_all, alpha=0.85)
        ax1.set_xlim(pivot_all.index.min(), pivot_all.index.max())
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax1.set_title("All Asset Classes", fontsize=11)
        ax1.set_xlabel("Date", **FONT_LABEL)
        ax1.set_ylabel("Share of Total CAV", **FONT_LABEL)
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h[::-1], l[::-1], loc="lower left", **FONT_LEGEND)

        ax2.stackplot(pivot_ex.index, pivot_ex.T.values, labels=pivot_ex.columns,
                      colors=colors_ex, alpha=0.85)
        ax2.set_xlim(pivot_ex.index.min(), pivot_ex.index.max())
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax2.set_title(f"Excluding {REPO_CLASS}", fontsize=11)
        ax2.set_xlabel("Date", **FONT_LABEL)
        ax2.set_ylabel("Share of Ex-Repo CAV", **FONT_LABEL)
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h2[::-1], l2[::-1], loc="lower left", **FONT_LEGEND)

        for ax in (ax1, ax2):
            fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart9_cav_share_over_time")


# ---------------------------------------------------------------------------
# Chart 10 — Before vs After Repo Expansion (stacked bars)
# ---------------------------------------------------------------------------

def chart10_before_after_repo(
    combined: pd.DataFrame,
    out_dir: Path,
    expansion_date: str = "2025-06-01",
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart10_before_after_repo"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav"])
    cutoff = pd.Timestamp(expansion_date)

    # Snapshots: latest month before expansion, and latest available month
    before_date = df[df["date"] < cutoff]["date"].max()
    after_date = df["date"].max()

    if pd.isna(before_date):
        print("  [SKIP] chart10_before_after_repo: no data before expansion date")
        return None

    snap_before = df[df["date"] == before_date][["asset_class", "cav"]].copy()
    snap_after = df[df["date"] == after_date][["asset_class", "cav"]].copy()

    # Express as % of respective totals
    snap_before["share"] = snap_before["cav"] / snap_before["cav"].sum() * 100
    snap_after["share"] = snap_after["cav"] / snap_after["cav"].sum() * 100

    # Keep top classes consistent across both snapshots
    combined_cav = (
        pd.concat([snap_before, snap_after])
        .groupby("asset_class")["cav"].mean()
        .nlargest(TOP_N_CLASSES)
        .index.tolist()
    )
    def _collapse(snap: pd.DataFrame) -> pd.Series:
        s = snap.copy()
        s["asset_class"] = s["asset_class"].where(s["asset_class"].isin(combined_cav), "Other")
        return s.groupby("asset_class")["share"].sum().reindex(combined_cav + ["Other"], fill_value=0)

    before_shares = _collapse(snap_before)
    after_shares = _collapse(snap_after)

    labels = before_shares.index.tolist()
    colors = [_class_color(cls) for cls in labels]
    bar_labels = [before_date.strftime("%b %Y"), after_date.strftime("%b %Y")]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(2)
        bar_width = 0.5
        bottoms = np.zeros(2)

        for i, cls in enumerate(labels):
            heights = np.array([before_shares[cls], after_shares[cls]])
            bars = ax.bar(x, heights, bar_width, bottom=bottoms, color=colors[i], label=cls, alpha=0.88)
            # Label segments > 3%
            for bar, h in zip(bars, heights):
                if h > 3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f"{h:.1f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold"
                    )
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_title("Market Structure: Before vs After Repo Expansion", **FONT_TITLE)
        ax.set_ylabel("Share of Total CAV (%)", **FONT_LABEL)
        ax.set_ylim(0, 105)
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], leg_labels[::-1], loc="upper right",
                  bbox_to_anchor=(1.22, 1), **FONT_LEGEND)
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart10_before_after_repo")


# ---------------------------------------------------------------------------
# Chart 11 — Latest Market Map: CAV vs Holders (bubble, direct labels)
# ---------------------------------------------------------------------------

def chart11_latest_market_map(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav", "holders", "turnover_ratio"]
    if not _require_columns(combined, required, "chart11_latest_market_map"):
        return None

    df = _parse_dates(combined).dropna(subset=["cav", "holders"])
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    snap = snap[(snap["cav"] > 0) & (snap["holders"] > 0)].copy()
    if snap.empty:
        print("  [SKIP] chart11_latest_market_map: no valid data at latest date")
        return None

    snap["turnover_ratio"] = snap["turnover_ratio"].fillna(0).clip(lower=0)
    # Scale bubble size: minimum visible size of 60, max 1200
    max_t = snap["turnover_ratio"].max()
    if max_t > 0:
        snap["bubble_size"] = 60 + (snap["turnover_ratio"] / max_t) * 1140
    else:
        snap["bubble_size"] = 200

    classes = snap["asset_class"].tolist()
    color_map = {cls: _class_color(cls) for cls in classes}

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))
        for _, row in snap.iterrows():
            ax.scatter(
                np.log10(row["holders"]), np.log10(row["cav"]),
                s=row["bubble_size"], color=color_map[row["asset_class"]],
                alpha=0.75, linewidths=0.5, edgecolors="white", zorder=3
            )
            ax.annotate(
                row["asset_class"],
                xy=(np.log10(row["holders"]), np.log10(row["cav"])),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7.5, color="#222222",
                fontweight="semibold",
            )

        # Legend for bubble size
        if max_t > 0:
            for t_val, lbl in [(0.01, "1% turnover"), (0.1, "10% turnover")]:
                sz = 60 + (t_val / max_t) * 1140
                ax.scatter([], [], s=sz, color="#aaaaaa", label=lbl, alpha=0.75)
            ax.legend(title="Bubble = Turnover", loc="lower right", **FONT_LEGEND)

        ax.set_title(
            f"Market Map: CAV vs Holders — {latest_date.strftime('%B %Y')}",
            **FONT_TITLE,
        )
        ax.set_xlabel("log₁₀(Holders)", **FONT_LABEL)
        ax.set_ylabel("log₁₀(CAV in USD)", **FONT_LABEL)
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart11_latest_market_map")


# ---------------------------------------------------------------------------
# Chart 12 — Median Turnover Ratio by Asset Class (bar chart)
# ---------------------------------------------------------------------------

def chart12_median_turnover(
    combined: pd.DataFrame,
    out_dir: Path,
    lookback_months: int = 12,
) -> Optional[Path]:
    required = ["date", "asset_class", "turnover_ratio"]
    if not _require_columns(combined, required, "chart12_median_turnover"):
        return None

    df = _parse_dates(combined).dropna(subset=["turnover_ratio"])
    cutoff = df["date"].max() - pd.DateOffset(months=lookback_months)
    df = df[df["date"] > cutoff]
    if df.empty:
        print("  [SKIP] chart12_median_turnover: no data in lookback window")
        return None

    median_t = (
        df.groupby("asset_class")["turnover_ratio"]
        .median()
        .sort_values(ascending=True)
    )

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(median_t.index, median_t.values,
                       color=[_class_color(c) for c in median_t.index])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=2))
        ax.set_title(
            f"Median Monthly Turnover Ratio by Asset Class\n"
            f"(Last {lookback_months} months)",
            **FONT_TITLE,
        )
        ax.set_xlabel("Median Turnover Ratio", **FONT_LABEL)
        ax.tick_params(axis="y", labelsize=9)

        # Value labels on bars
        for bar, val in zip(bars, median_t.values):
            ax.text(
                bar.get_width() + median_t.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", va="center", fontsize=8
            )

        fig.tight_layout()

    return _savefig(fig, out_dir, "chart12_median_turnover")


# ---------------------------------------------------------------------------
# Chart 13 — Total Holders Over Time
# ---------------------------------------------------------------------------

def chart13_total_holders(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "asset_class", "holders"]
    if not _require_columns(combined, required, "chart13_total_holders"):
        return None

    df = _parse_dates(combined).dropna(subset=["holders"])
    total_holders = df.groupby("date")["holders"].sum().reset_index()
    total_holders.columns = ["date", "total_holders"]

    if total_holders.empty:
        print("  [SKIP] chart13_total_holders: no data")
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(total_holders["date"], total_holders["total_holders"],
                linewidth=2, color=_SERIES_COLORS[4])
        ax.fill_between(total_holders["date"], total_holders["total_holders"],
                        alpha=0.15, color=_SERIES_COLORS[4])
        ax.set_xlim(total_holders["date"].min(), total_holders["date"].max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_title("Total Holders Over Time (All Asset Classes)", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Total Holders", **FONT_LABEL)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart13_total_holders")


# ---------------------------------------------------------------------------
# Chart 14 — Ex-Repo Market Diversification (top-5 share, single line)
# ---------------------------------------------------------------------------

def chart14_ex_repo_hhi(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    """Single-line chart: share of ex-repo CAV held by the top-5 asset classes.
    Falling = market broadening; rising = concentration increasing."""
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart14_ex_repo_hhi"):
        return None

    conc_df = _recompute_hhi(combined, exclude=[REPO_CLASS])
    conc_df = _parse_dates(conc_df).dropna(subset=["top_5_share"])
    if conc_df.empty:
        print("  [SKIP] chart14_ex_repo_hhi: no data after excluding repo")
        return None

    top5_pct = conc_df["top_5_share"] * 100

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.plot(conc_df["date"], top5_pct, color=_SERIES_COLORS[0], linewidth=2)
        ax.fill_between(conc_df["date"], top5_pct, top5_pct.min(),
                        alpha=0.12, color=_SERIES_COLORS[0])
        ax.set_xlim(conc_df["date"].min(), conc_df["date"].max())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_title(
            f"Ex-Repo Market Diversification: Top-5 Share of CAV\n"
            f"(Excluding {REPO_CLASS})",
            **FONT_TITLE,
        )
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Top-5 Asset Class Share of Ex-Repo CAV", **FONT_LABEL)
        ax.annotate(
            "↓ Falling = market broadening",
            xy=(0.02, 0.06), xycoords="axes fraction",
            fontsize=9, color="#555555", style="italic",
        )
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart14_ex_repo_concentration")


# ---------------------------------------------------------------------------
# Chart 15 — Rolling CAV Growth, Ex-Repo
# ---------------------------------------------------------------------------

def chart15_ex_repo_rolling_growth(
    combined: pd.DataFrame,
    out_dir: Path,
) -> Optional[Path]:
    required = ["date", "asset_class", "cav"]
    if not _require_columns(combined, required, "chart15_ex_repo_rolling_growth"):
        return None

    ex_repo = _ex_repo_total_cav_series(combined).set_index("date")["total_cav_ex_repo"]
    if ex_repo.empty:
        print("  [SKIP] chart15_ex_repo_rolling_growth: no ex-repo data")
        return None

    growth = _rolling_growth_from_series(ex_repo)
    df = ex_repo.reset_index().rename(columns={"total_cav_ex_repo": "cav"})
    df = df.join(growth, on="date")

    has_3m = df["rolling_3m"].dropna().shape[0] > 0
    has_6m = df["rolling_6m"].dropna().shape[0] > 0
    if not has_3m and not has_6m:
        print("  [SKIP] chart15_ex_repo_rolling_growth: no rolling growth data")
        return None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        if has_3m:
            ax.plot(df["date"], df["rolling_3m"], label="3-Month Rolling Growth (ex-Repo)",
                    color=_SERIES_COLORS[2], linewidth=2)
        if has_6m:
            ax.plot(df["date"], df["rolling_6m"], label="6-Month Rolling Growth (ex-Repo)",
                    color=_SERIES_COLORS[3], linewidth=2, linestyle="--")

        ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlim(df["date"].min(), df["date"].max())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.set_title(f"Rolling CAV Growth — Excluding {REPO_CLASS}", **FONT_TITLE)
        ax.set_xlabel("Date", **FONT_LABEL)
        ax.set_ylabel("Growth Rate (%)", **FONT_LABEL)
        ax.legend(**FONT_LEGEND)
        fig.autofmt_xdate()
        fig.tight_layout()

    return _savefig(fig, out_dir, "chart15_ex_repo_rolling_growth")
