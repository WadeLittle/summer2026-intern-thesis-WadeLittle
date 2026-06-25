#!/usr/bin/env python3
# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Phase 4 — Bronze: Statistical Analysis runner.

Loads the saved metrics CSVs, runs all four thesis pillars, and produces:
  data/stats_summary.json       Full structured statistical output.
  data/conclusion.txt           Plain-English thesis conclusion.
  data/statistical_results.csv  Flattened table of key statistical outputs.

Usage (from repo root):
    python scripts/run_analysis.py
"""

import csv
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bronze.stats import run_analysis
from src.config import RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_metrics(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_path     = os.path.join(results_dir, "combined_metrics.csv")
    concentration_path = os.path.join(results_dir, "concentration_metrics.csv")

    for path in (combined_path, concentration_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required metrics file not found: {path}\n"
                "Run 'python scripts/build_dataset.py' first."
            )

    combined_df      = pd.read_csv(combined_path)
    concentration_df = pd.read_csv(concentration_path)
    print(f"[run_analysis] Loaded {len(combined_df):,} rows from {combined_path}")
    print(f"[run_analysis] Loaded {len(concentration_df):,} rows from {concentration_path}")
    return combined_df, concentration_df


def _save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size_kb = os.path.getsize(path) / 1024
    print(f"[run_analysis] Saved {path} ({size_kb:.1f} KB)")


def _save_conclusion(conclusion: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(conclusion)
        f.write("\n")
    print(f"[run_analysis] Saved {path}")


def _flatten_ols(result: dict, pillar: str, test_name: str) -> list[dict]:
    """Convert an OLS result dict into one or more flat CSV rows."""
    status = result.get("status", "unknown")
    base = {
        "pillar":    pillar,
        "test":      test_name,
        "status":    status,
        "n":         result.get("n"),
        "beta":      result.get("beta"),
        "p_value":   result.get("p_value"),
        "ci_lower":  result.get("confidence_interval", [None, None])[0],
        "ci_upper":  result.get("confidence_interval", [None, None])[1],
        "r_squared": result.get("r_squared"),
        "significant_05": (
            result.get("p_value") is not None and result.get("p_value") < 0.05
        ) if status == "ok" else None,
        "interpretation": result.get("interpretation", ""),
    }
    return [base]


def _flatten_spearman(result: dict, pillar: str, test_name: str) -> list[dict]:
    status = result.get("status", "unknown")
    return [{
        "pillar":         pillar,
        "test":           test_name,
        "status":         status,
        "n":              result.get("n"),
        "beta":           result.get("rho"),      # rho in the beta column for comparability
        "p_value":        result.get("p_value"),
        "ci_lower":       None,
        "ci_upper":       None,
        "r_squared":      None,
        "significant_05": (
            result.get("p_value") is not None and result.get("p_value") < 0.05
        ) if status == "ok" else None,
        "interpretation": result.get("interpretation", ""),
    }]


def _build_flat_rows(results: dict) -> list[dict]:
    """Extract key statistical outputs into a flat list of row dicts."""
    rows: list[dict] = []

    p1 = results.get("pillar1_growth", {})
    rows += _flatten_ols(p1.get("log_cav_trend_ex_repo", {}),       "pillar1", "log_cav_trend_ex_repo")
    rows += _flatten_ols(p1.get("log_cav_trend_full_market", {}),   "pillar1", "log_cav_trend_full_market")
    rows += _flatten_ols(p1.get("growth_acceleration_ex_repo", {}), "pillar1", "growth_acceleration_ex_repo")
    rows += _flatten_ols(p1.get("rolling_growth_trend_ex_repo", {}), "pillar1", "rolling_growth_trend_ex_repo")
    sb = p1.get("structural_break_at_repo_entry", {})
    if sb.get("status") == "ok":
        rows += _flatten_ols(
            {
                "status":               sb["status"],
                "n":                    sb["n"],
                "beta":                 sb.get("beta_t_D"),
                "p_value":              sb.get("p_t_D"),
                "confidence_interval":  sb.get("ci_t_D", [None, None]),
                "r_squared":            sb.get("r_squared"),
                "interpretation":       sb.get("interpretation", ""),
            },
            "pillar1",
            "structural_break_slope_change",
        )

    p2 = results.get("pillar2_composition", {})
    rows += _flatten_ols(p2.get("hhi_trend_ex_repo", {}),            "pillar2", "hhi_trend_ex_repo")
    rows += _flatten_ols(p2.get("hhi_trend_full_market", {}),        "pillar2", "hhi_trend_full_market")
    rows += _flatten_ols(p2.get("top_5_share_trend_pre_repo", {}),   "pillar2", "top5_share_trend_pre_repo")
    rows += _flatten_ols(p2.get("active_class_count_trend_ex_repo", {}), "pillar2", "active_class_count_trend_ex_repo")

    p3 = results.get("pillar3_adoption", {})
    rows += _flatten_ols(p3.get("holder_trend_ex_repo", {}),         "pillar3", "holder_trend_ex_repo")
    rows += _flatten_ols(p3.get("log_log_cav_holders_ex_repo", {}),  "pillar3", "log_log_cav_holders_ex_repo")
    rows += _flatten_spearman(p3.get("spearman_cav_holder_growth", {}), "pillar3", "spearman_cav_holder_growth")
    for ac, r in p3.get("asset_class_regressions", {}).items():
        rows += _flatten_ols(r, "pillar3", f"per_class_holder_elasticity_{ac}")

    p4 = results.get("pillar4_liquidity", {})
    rows += _flatten_ols(p4.get("turnover_trend_ex_repo", {}),        "pillar4", "turnover_trend_ex_repo")
    rows += _flatten_ols(p4.get("log_log_cav_turnover_ex_repo", {}),  "pillar4", "log_log_cav_turnover_ex_repo")
    rows += _flatten_spearman(p4.get("spearman_cav_volume_growth", {}), "pillar4", "spearman_cav_volume_growth")
    for ac, r in p4.get("asset_class_regressions", {}).items():
        rows += _flatten_ols(r, "pillar4", f"per_class_turnover_elasticity_{ac}")

    return rows


def _save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"[run_analysis] No rows to write → {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_analysis] Saved {len(rows)} rows → {path}")


def _print_summary(results: dict) -> None:
    """Print a concise console summary of key findings."""
    dw = results.get("data_window", {})
    print()
    print("=" * 70)
    print("  RWA TOKENIZATION — PHASE 4 STATISTICAL ANALYSIS")
    print(f"  Data: {dw.get('start')} to {dw.get('end')}")
    print(f"  Months: {dw.get('n_months_ex_repo')} ex-repo | "
          f"{dw.get('n_months_total')} total")
    print(f"  Asset classes: {dw.get('n_asset_classes_ex_repo')} ex-repo | "
          f"{dw.get('n_asset_classes_total')} total")
    print(f"  Repo entry date: {dw.get('repo_entry_date')} "
          f"(repos excluded from primary analysis)")
    print("=" * 70)

    for label, pillar_key in [
        ("PILLAR 1 — GROWTH",      "pillar1_growth"),
        ("PILLAR 2 — COMPOSITION", "pillar2_composition"),
        ("PILLAR 3 — ADOPTION",    "pillar3_adoption"),
        ("PILLAR 4 — LIQUIDITY",   "pillar4_liquidity"),
    ]:
        s = results.get(pillar_key, {}).get("summary", {})
        print(f"\n{label}")
        print(f"  {s.get('interpretation', 'No summary available.')}")

    print()
    print("─" * 70)
    print("  CONCLUSION (excerpt):")
    print("─" * 70)
    conclusion = results.get("conclusion", "")
    last_para = conclusion.split("\n\n")[-1] if conclusion else ""
    for line in last_para.split(". "):
        if line.strip():
            print(f"  {line.strip()}.")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[run_analysis] Phase 4 — Bronze statistical analysis")

    combined_df, concentration_df = _load_metrics(RESULTS_DIR)

    print("[run_analysis] Running all four pillars...")
    results = run_analysis(combined_df, concentration_df)
    print("[run_analysis] Analysis complete.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path        = os.path.join(RESULTS_DIR, "stats_summary.json")
    conclusion_path  = os.path.join(RESULTS_DIR, "conclusion.txt")
    csv_path         = os.path.join(RESULTS_DIR, "statistical_results.csv")

    _save_json(results, json_path)
    _save_conclusion(results.get("conclusion", ""), conclusion_path)

    flat_rows = _build_flat_rows(results)
    _save_csv(flat_rows, csv_path)

    _print_summary(results)


if __name__ == "__main__":
    main()
