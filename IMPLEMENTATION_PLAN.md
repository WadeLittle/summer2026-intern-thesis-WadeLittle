# Implementation Plan — RWA Tokenization Thesis

**Thesis question:** Is real-world asset tokenization a structural shift, or institutional window dressing?

**Questions I want to answer:**

| Question | Metric | Chart | Statistical test |
|---|---|---|---|
| Is tokenization growing? | total CAV, monthly growth | total CAV line chart | log CAV trend (OLS) |
| Is growth speeding up/slowing? | rolling growth rates | rolling growth line chart | growth rate trend (OLS) |
| Is composition changing? | CAV share by asset class | stacked area chart | share trend (OLS) |
| Is the market less concentrated? | HHI, top-5 share | HHI line chart | HHI trend (OLS) |
| Is adoption growing with assets? | holders, holders per $1M CAV | CAV vs holders scatterplot | log holders vs log CAV (OLS) |
| Is liquidity growing with assets? | volume, turnover | CAV vs turnover scatterplot | turnover vs log CAV (OLS + Spearman) |

---

## Three-Tier Architecture

This project is built in three independent, runnable tiers.

```text
Bronze — Data pipeline, metrics, static charts, statistical analysis
         Run via: python scripts/build_dataset.py && python scripts/run_analysis.py

Silver — Streamlit dashboard built on top of Bronze outputs
         Run via: streamlit run silver/app.py

Gold   — Chatbot + optional MCP/live data built on top of Silver
         Run via: streamlit run gold/app.py
```

Each tier must be independently runnable. A tier should not require the next tier to exist.

Bronze produces results. Silver reads from those results. Gold enhances Silver with natural language and optional live data.

---

## Guiding Principles

* Do one phase at a time.
* Do not build later-phase features early.
* Every phase must produce a runnable artifact.
* API calls should be separated from analysis, charts, and Streamlit.
* Streamlit should read from saved local results, not re-fetch API data on page load.
* Chatbot functionality comes after the data, metrics, charts, and statistics are stable.
* MCP/live-data integration comes last.
* Never hardcode API keys.
* Never commit `.env` or cache files.
* Do not claim causality from statistical correlations.

---

# Phase 1 — Bronze: RWA Data Pipeline

## Goal

Create a reliable RWA.xyz data pipeline that fetches or loads cached API data, transforms it into a clean monthly dataset, validates the output, and saves it for downstream metrics, charts, statistics, and the Silver dashboard.

This phase produces data only. No charts. No metrics. No stats. No Streamlit.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Modify | `src/config.py` | Set `ANALYSIS_START_DATE = "2023-01-01"`. Move Stablecoins and Cryptocurrencies out of the main RWA scope (track separately). Add `RESULTS_DIR = "data"`. |
| Move/Modify | `src/bronze/api_client.py` | Move from `src/api_client.py`. Preserve existing API request behavior. Only the smallest import-path changes needed so scripts run from the repo root without `sys.path` manipulation. |
| Move/Modify | `src/bronze/data_processing.py` | Move from `src/data_processing.py`. Add or update logic to build and save a combined monthly dataset. Add input validation for required columns, dates, numeric fields, and negative values. |
| Create | `src/__init__.py` | Empty file to establish `src` as a package. |
| Create | `src/bronze/__init__.py` | Empty file to establish `src/bronze` as a package. |
| Create | `scripts/build_dataset.py` | Standalone script: fetch/load cache → process → validate → save CSV. |
| Create | `data/` | Output folder. Include `.gitkeep`. |

---

## Required Output

`data/combined_monthly.csv`

Required columns:

```text
date
asset_class
cav
holders
volume
```

Rules:

```text
Start no earlier than 2023-01-01.
End at the last complete month.
No row should have negative CAV.
CAV, holders, and volume should be numeric where present.
Missing values should be reported in a data quality summary.
Do not fabricate rows. If an in-scope asset class has no data, log it clearly.
Running twice within the cache TTL must use cached data.
```

---

## Acceptance Criteria

* `python scripts/build_dataset.py` runs from the repo root without error.
* `data/combined_monthly.csv` is produced with: `date, asset_class, cav, holders, volume`.
* Date range starts no earlier than `2023-01-01`.
* No row has negative CAV.
* Missing classes or fields are logged clearly.
* Script prints a data quality summary.
* Second run within cache TTL uses cached data.
* `from src.config import ASSET_CLASSES_IN_SCOPE` works from the repo root.

---

## Commands to Run

```bash
pip install -r requirements.txt
python scripts/build_dataset.py
head -5 data/combined_monthly.csv
python -c "import pandas as pd; df = pd.read_csv('data/combined_monthly.csv'); print(df.dtypes); print(df['date'].min(), df['date'].max()); print(df['asset_class'].unique())"
```

---

# Phase 1b — REMOVED

## Decision

Phase 1b was originally planned as a TradFi benchmark pipeline using yfinance to pull historical market data for comparison against tokenized asset classes. 

This phase was removed due to the struggle to find good data that can be used to compare against the pulled RWA data. The data wasn't strong enough to strengthen the overall analysis
and thesis argument. To avoid unnecessary noise, I have decided to remove this tradfi comparison at this point.

---

# Phase 2 — Bronze: Metrics Layer

## Goal

Create a clean, testable metrics module that computes all derived fields needed for charts, statistical analysis, and thesis interpretation.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/bronze/metrics.py` | Reusable metric functions and `build_all_metrics(df)`. |
| Create | `src/bronze/metrics_config.py` | Constants: rolling window, minimum data points, HHI floor rules. |
| Modify | `src/bronze/data_processing.py` | Keep only raw-to-clean transformation. Move derived metric logic to `src/bronze/metrics.py`. |
| Modify | `scripts/build_dataset.py` | After saving `combined_monthly.csv`, call `build_all_metrics()` and save metric outputs. |
| Create | `tests/test_metrics.py` | Unit tests for core metric functions using synthetic data. No API calls. |

---

## Metrics to Implement

### Pillar 1 — Growth

```text
total_cav                  # aggregate CAV across all in-scope asset classes per month
monthly_cav_growth         # month-over-month % change in total CAV
rolling_3m_cav_growth      # 3-month rolling average of monthly_cav_growth
rolling_6m_cav_growth      # 6-month rolling average of monthly_cav_growth
cav_index                  # index to 100 at each asset class's first non-null CAV month
asset_class_cav_growth     # monthly_cav_growth computed per asset class
```

### Pillar 2 — Composition

```text
cav_share                  # each asset class's share of total CAV per month
hhi                        # sum(share_i^2) across in-scope asset classes per month
top_5_share                # CAV share of the top 5 asset classes per month
asset_class_count          # number of in-scope asset classes with any CAV
active_asset_class_count   # number with CAV > threshold (e.g. > $1M)
```

### Pillar 3 — Adoption

```text
holders_index              # index to 100 at first non-null holders month per asset class
monthly_holder_growth      # month-over-month % change in holders
holders_per_million_cav    # holders / (cav / 1_000_000)
avg_position               # cav / holders (handle zero holders safely)
```

### Pillar 4 — Liquidity

```text
turnover_ratio             # volume / cav (handle zero cav safely)
turnover_3m                # 3-month rolling average of turnover_ratio
monthly_volume_growth      # month-over-month % change in volume
```

---

## Required Outputs

`data/combined_metrics.csv` columns:

```text
date, asset_class, cav, holders, volume,
cav_share, cav_index, asset_class_cav_growth,
holders_index, monthly_holder_growth, holders_per_million_cav, avg_position,
turnover_ratio, turnover_3m, monthly_volume_growth
```

`data/concentration_metrics.csv` columns:

```text
date, total_cav, monthly_cav_growth, rolling_3m_cav_growth, rolling_6m_cav_growth,
hhi, top_5_share, asset_class_count, active_asset_class_count
```

---

## Acceptance Criteria

* `python scripts/build_dataset.py` produces both metric output files.
* `cav_share` sums to approximately 1.0 for every date where CAV data is available.
* `cav_index` equals exactly 100 for each asset class at its first non-null, positive CAV month.
* No division-by-zero errors when `holders == 0` or `cav == 0`.
* Unit tests pass: `python -m pytest tests/test_metrics.py -v`.

---

## Commands to Run

```bash
python scripts/build_dataset.py
python -m pytest tests/test_metrics.py -v
python -c "import pandas as pd; df = pd.read_csv('data/combined_metrics.csv', parse_dates=['date']); print(df.columns.tolist()); print(df.tail())"
python -c "import pandas as pd; hhi = pd.read_csv('data/concentration_metrics.csv'); print(hhi.tail())"
```

---

# Phase 3 — Bronze: Static Charts

## Goal

Create clean, thesis-focused static charts from saved metrics outputs. These charts should be useful for a 10-minute presentation and should not require re-fetching API data. All chart code lives in `src/bronze/`.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/bronze/charts.py` | Reusable chart functions, one function per chart. |
| Create | `scripts/build_charts.py` | Load saved metrics → generate static charts → save PNGs. |
| Create | `charts/` | Static chart output folder. |

---

## Charts to Build

```text
chart1_total_cav.png              — total CAV over time (line chart)
chart2_cav_by_asset_class.png     — stacked area chart by asset class
chart3_hhi_concentration.png      — HHI and top-5 share over time
chart4_holder_growth.png          — holders index by asset class
chart5_turnover_ratio.png         — turnover ratio over time
chart6_rolling_growth.png         — rolling 3m and 6m CAV growth rates
chart7_cav_vs_holders.png         — log CAV vs log holders scatterplot
chart8_cav_vs_turnover.png        — log CAV vs turnover scatterplot
chart9_scorecard.png              — thesis scorecard summary (if data supports it)
```

At minimum, produce charts 1–5.

---

## Chart Requirements

* Charts must read from `data/combined_metrics.csv` and `data/concentration_metrics.csv`.
* No API calls.
* No statistical tests.
* Save PNG files to `charts/`. Save SVG or PDF versions where easy.
* Use presentation-ready titles and axis labels.
* Avoid cluttered legends; group minor asset classes as "Other" if needed.
* Do not let x-axis start before the first available date in the data.
* Any chart that cannot be built due to missing data should be skipped with a clear message.
* Target chart PNGs under 3 MB where practical.

---

## Acceptance Criteria

* `python3 scripts/build_charts.py` runs without error.
* At least 5 core chart PNGs are created in `charts/`.
* No chart x-axis begins before the first date in the data.
* Each chart function can be called independently with a metrics DataFrame.
* Running the script a second time overwrites outputs rather than duplicating them.

---

## Commands to Run

```bash
python3 scripts/build_charts.py
ls -lh charts/
```

---

## Phase 3 Expansion — Chart Improvements and Additions

This section documents the changes made after the initial Phase 3 implementation, including charts that were added, revised, or removed, and the reasoning behind each decision.

### Output Structure

Charts are split into two subfolders for cleaner organization:

```
charts/
  png/   — all PNG files (presentation-ready, DPI 150)
  pdf/   — all PDF files (vector, suitable for LaTeX or print)
```

`build_charts.py` clears both folders before each run so no stale outputs accumulate.

### Data Freshness Enforcement

`build_charts.py` now checks the modification time of `data/combined_metrics.csv` and `data/concentration_metrics.csv` against `CACHE_TTL_SECONDS` (defined in `src/config.py`, default 24 hours) before building any chart. If either file is missing or older than the TTL, `build_dataset.py` is invoked automatically to fetch fresh API data and recompute all metrics. This ensures charts always reflect up-to-date data without requiring the user to run two scripts manually.

### Uniform Asset Class Colors

All charts pull asset class colors from a single dict, `ASSET_CLASS_COLORS`, defined in `src/config.py`. Every chart function calls `_class_color(cls)` which looks up this dict with an `OTHER_COLOR` fallback. This guarantees that a given asset class always renders in the same color across every chart in the deck — critical for a presentation where the audience will build a mental model of which color maps to which class.

Non-class aggregate series (rolling growth lines, HHI lines, total-CAV fills) use a separate `_SERIES_COLORS` list in `charts.py` to keep the class color palette free of collision.

### Repurchase Agreements Dominance Problem

After mid-2025, Repurchase Agreements account for ~85% of total CAV. This single class distorts every aggregate chart: total CAV jumps, HHI spikes, rolling growth is dominated by the repo entry event, and the stacked area chart becomes unreadable for all other classes. The expansion addresses this by providing parallel "ex-repo" versions of the key time-series charts so both stories can be told:

> "Including repos, tokenized RWAs are already very large."
> "Excluding repos, the rest of the market is smaller but more diverse and analytically meaningful."

### Charts Removed

| Chart | Reason |
|---|---|
| `chart3_hhi_concentration` | Total-market HHI is trivially high once repos dominate (~0.75+). The story — "repo dominates" — is already told more clearly by chart2 and chart10. |
| `chart6_rolling_growth` | The repo expansion creates a spike that collapses the y-axis and makes the pre-2025 history invisible. Chart19 covers the same metric on ex-repo data. |
| `chart7_cav_vs_holders` | Plotting all monthly observations for all classes produces overlapping blobs with no clear message. Replaced by chart11 which uses only the latest month with direct labels. |
| `chart8_cav_vs_turnover` | Same issue as chart7. The three-way relationship (CAV, holders, turnover) is shown more clearly in chart11 via bubble sizing. |

### Charts Revised

| Chart | Change | Reason |
|---|---|---|
| `chart3_holder_growth` | Replaced holders index (base-100) with raw holder counts by class. | The indexed format caused extreme y-axis values for fast-growing classes, forcing a log scale and making the chart unreadable. Raw counts are directly interpretable. |
| `chart4_turnover_ratio` | Changed filter from top-N by CAV to top-N by **median turnover**. | A class can be large by CAV but have near-zero turnover (e.g. US Treasury Debt held as long-term collateral). The old filter showed the biggest classes, not the most actively traded ones. |
| `chart8_latest_composition` | Added share-of-total percentage labels to the **left (all classes) panel** as well as the right (ex-repo) panel. | Both panels are equally useful for the presentation; the percentage annotation gives immediate context without requiring the viewer to read the x-axis scale. |
| `chart14_ex_repo_hhi` | Replaced dual-axis HHI + top-5 share chart with a **single-line ex-repo top-5 share** chart. | The dual-axis format was confusing and HHI is too technical for a presentation audience. A single line showing "what fraction of the ex-repo market do the top 5 classes hold?" is immediately readable. A falling line means the market is broadening, annotated directly on the chart. |

### New Charts Added (10–19)

| Chart | Description |
|---|---|
| `chart6_ex_repo_total_cav` | Total CAV over time, excluding Repurchase Agreements. Shows the growth trajectory of the non-repo market independently. |
| `chart7_ex_repo_by_class` | Stacked area chart by asset class, excluding Repurchase Agreements. Shows which classes drive the ex-repo market and how their relative sizes have shifted. |
| `chart8_latest_composition` | Horizontal bar chart of the latest month's CAV by asset class, two panels: all classes and ex-repo. Percentage share annotated on both panels. Best single slide for showing current market structure. |
| `chart9_cav_share_over_time` | Two-panel stacked area chart showing each asset class as a percentage of total CAV over time (left: all, right: ex-repo). Answers whether the market is becoming more or less diversified. |
| `chart10_before_after_repo` | Two stacked bars comparing market composition before the repo expansion (May 2025) and after (latest month). Expansion date is detected programmatically as the first month where repo share exceeds 50%. |
| `chart11_latest_market_map` | Bubble scatter of all asset classes at the latest date only. x = holders, y = CAV, bubble size = turnover ratio. Each point is directly labeled. Tells the full story of size, adoption, and liquidity in one chart. |
| `chart12_median_turnover` | Horizontal bar chart of median monthly turnover ratio by asset class over the last 12 months. Cleaner than the monthly line chart for identifying which classes are most actively traded. |
| `chart13_total_holders` | Total holder count summed across all asset classes over time. Simple adoption signal that is not distorted by repo. |
| `chart14_ex_repo_concentration` | Single-line chart of the top-5 asset class share of ex-repo CAV over time. Falling = market broadening. Annotated with a direction cue for the audience. |
| `chart15_ex_repo_rolling_growth` | Rolling 3-month and 6-month CAV growth rates computed from ex-repo CAV only. Shows whether the non-repo market is accelerating or decelerating without the repo entry distortion. |

---

# Phase 4 — Bronze: Statistical Analysis

## Goal

Create a clean statistical analysis layer that evaluates the thesis across four pillars using the metrics outputs. Produce structured results and a plain-English conclusion.

All statistical code lives in `src/bronze/`.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/bronze/stats.py` | Statistical functions and `run_analysis(df) -> dict`. |
| Create | `scripts/run_analysis.py` | Load saved metrics → run all pillars → save results. |
| Create | `tests/test_stats.py` | Unit tests for statistical functions using synthetic data. |
| Create | `data/stats_summary.json` | Structured statistical output. |
| Create | `data/conclusion.txt` | Plain-English conclusion. |
| Create | `data/statistical_results.csv` | Flattened table of key statistical outputs. |

---

## Statistical Tests by Pillar

### Pillar 1 — Growth / Rate of Tokenization

**Goal:** Determine whether tokenized RWA value is growing, slowing, or accelerating.

```text
OLS trend on log CAV:
  ln(total_cav_t) = alpha + beta*time

Growth acceleration/deceleration:
  monthly_cav_growth_t = alpha + beta*time

Rolling growth trend:
  rolling_3m_cav_growth_t = alpha + beta*time

Optional structural break (if data window supports it):
  ln(total_cav_t) = alpha + beta1*time + beta2*post_event + beta3*time_after_event
```

Output per test: `beta, p_value, confidence_interval, annualized_growth_rate, growth_direction, interpretation`

Interpretation logic:

```text
Structural shift evidence is stronger when:
- total CAV is growing significantly
- growth is persistent across multiple months
- growth is not explained only by one outlier month
- growth is not limited to one asset class

Window-dressing risk is higher when:
- growth is slowing materially
- growth is driven by one short-lived spike
- growth is mostly concentrated in one asset class
```

---

### Pillar 2 — Composition / Diversification

**Goal:** Determine whether growth is broadening across asset classes or becoming more concentrated.

```text
HHI trend:
  HHI_t = alpha + beta*time

Top share trend:
  top_5_share_t = alpha + beta*time

Active asset class count trend:
  active_asset_class_count_t = alpha + beta*time
```

Output per test: `beta, p_value, confidence_interval, interpretation`

Interpretation logic:

```text
Structural shift evidence is stronger when:
- HHI is falling
- top-5 share is falling
- active asset class count is rising
- multiple asset classes gain meaningful CAV share

Window-dressing risk is higher when:
- HHI remains high or rises
- top-5 share remains high or rises
- new asset classes remain economically tiny
- total growth is dominated by one asset class
```

---

### Pillar 3 — Adoption vs Asset Growth

**Goal:** Determine whether growth in tokenized value is accompanied by growth in holders or user participation.

```text
Holder growth trend:
  ln(holders_t) = alpha + beta*time

Relationship between asset growth and holder growth:
  ln(holders_t) = alpha + beta*ln(cav_t)

Spearman correlation:
  monthly_cav_growth vs monthly_holder_growth
```

Asset-class-level tests where sample size allows:

```text
  ln(holders_it) = alpha + beta*ln(cav_it)
```

Output: `beta, p_value, confidence_interval, correlation, interpretation`

Interpretation logic:

```text
Structural shift evidence is stronger when:
- holders grow alongside CAV
- holders_per_million_cav is stable or rising
- adoption broadens across multiple asset classes

Window-dressing risk is higher when:
- CAV grows but holders remain flat
- avg_position rises because of larger balances, not more holders
- holder growth is concentrated in one asset class
```

Limitation note:

```text
Holder count is an on-chain participation proxy, not a perfect measure of unique users.
One institution can control multiple wallets; one wallet can represent multiple users.
```

---

### Pillar 4 — Liquidity vs Asset Growth

**Goal:** Determine whether tokenized assets are becoming more actively used or simply sitting on-chain.

```text
Liquidity trend:
  turnover_3m_t = alpha + beta*time

Relationship between asset growth and liquidity:
  turnover_3m_t = alpha + beta*ln(cav_t)

Spearman correlation:
  monthly_cav_growth vs monthly_volume_growth
```

Asset-class-level tests where sample size allows:

```text
  turnover_3m_it = alpha + beta*ln(cav_it)
```

Output: `beta, p_value, confidence_interval, correlation, interpretation`

Interpretation logic:

```text
Structural shift evidence is stronger when:
- volume grows alongside CAV
- turnover ratio is stable or rising
- liquidity improves across multiple asset classes

Window-dressing risk is higher when:
- CAV grows but volume does not
- turnover ratio falls as the market grows
- large tokenized assets remain mostly inactive
```

---

## Conclusion Logic

The conclusion should be CAV-weighted, not a simple raw count of asset classes.

The final verdict evaluates:

```text
1. Is tokenized RWA value growing?
2. Is growth accelerating, stable, or slowing?
3. Is market composition becoming more diversified?
4. Is adoption growing alongside asset value?
5. Is liquidity growing alongside asset value?
```

Structural shift evidence is stronger when:

```text
- CAV is growing significantly
- growth is persistent, not just one spike
- HHI is falling
- active asset class count is rising
- holder participation is rising
- turnover/liquidity is stable or improving
```

Window-dressing risk is higher when:

```text
- CAV growth is slowing
- CAV growth is concentrated in one asset class
- HHI remains high or rises
- holder growth is weak
- turnover is low or falling
```

The conclusion must distinguish:

```text
statistical significance
economic significance
data limitations
correlation vs causality
small-base effects
```

Do not claim causality. If sample size is too small, mark the result as `insufficient_data` rather than claiming weak/no effect.

---

## Required Outputs

`data/stats_summary.json` structure:

```text
pillar1_growth
pillar2_composition
pillar3_adoption
pillar4_liquidity
conclusion
generated_at
data_window
limitations
```

---

## Acceptance Criteria

* `python scripts/run_analysis.py` runs without error.
* `data/stats_summary.json` is produced with all four pillars.
* `data/conclusion.txt` is produced.
* Confidence intervals are included where OLS is used.
* Results with insufficient sample size are clearly flagged.
* The conclusion is CAV-weighted.
* The conclusion does not claim causality.
* Unit tests pass: `python -m pytest tests/test_stats.py -v`.

---

## Commands to Run

```bash
python scripts/run_analysis.py
python -m pytest tests/test_stats.py -v
python -c "import json; d=json.load(open('data/stats_summary.json')); print(list(d.keys()))"
cat data/conclusion.txt
```

---

## Phase 4 Amendments — Post-Implementation Corrections

This section documents corrections and additions made after the initial Phase 4 implementation.

### Spurious Regression: ln(holders) ~ ln(cav) in Levels

**Problem discovered:** The per-class Pillar 3 regressions originally fit `ln(holders) ~ ln(cav)` using monthly observations over the full 41-month window. Durbin-Watson statistics of 0.26–0.69 were found across all 12 asset classes (healthy threshold ≥1.5), indicating heavily autocorrelated residuals — the hallmark of spurious regression. When two non-stationary series (both trending upward over the same period) are regressed against each other in levels, OLS inflates t-statistics and produces near-zero p-values regardless of any true relationship. The original result — 12/12 classes significant, CAV-weighted adoption score of 1.00 — was an artifact, not a finding.

The same problem applied to the per-class Pillar 4 regressions (`turnover_3m ~ ln(cav)`) and the market-level log-log regressions in both pillars.

**Fix:** Added a new primitive `_ols_growth_on_growth(pct_y, pct_x)` that regresses monthly growth rates (first differences of log levels) on each other. Growth rates are stationary and do not share a common time trend, so OLS inference is valid. The per-class regressions in Pillars 3 and 4 were updated to use this function:

| Pillar | Old test | New test |
|--------|----------|----------|
| P3 per-class | `ln(holders) ~ ln(cav)` | `monthly_holder_growth ~ asset_class_cav_growth` |
| P4 per-class | `turnover_3m ~ ln(cav)` | `pct_change(turnover_ratio) ~ asset_class_cav_growth` |
| P3 market-level | `ln(holders) ~ ln(cav)` | `ex_repo_holders_growth ~ ex_repo_cav_growth` |
| P4 market-level | `turnover ~ ln(cav)` | `ex_repo_turnover_growth ~ ex_repo_cav_growth` |

The `_ols_log_log` function was retained (with `spurious_regression_warning` and `durbin_watson` fields added to its output) for structural or cross-sectional use cases, but is no longer used for time-series co-movement tests.

**Impact on findings:**

| Metric | Before (spurious) | After (corrected) |
|--------|------------------|-------------------|
| P3 classes significant | 12/12 | 2/12 |
| P3 CAV-weighted adoption score | 1.00 | 0.16 |
| P4 classes significant | 10/12 | 2/12 |
| P4 CAV-weighted liquidity score | 0.97 | 0.04 |

The market-level Spearman correlation (ρ=0.505, p<0.01) was not affected — it already operated on growth rates, not levels.

### Named Asset Class Disclosure in Pillar 3 and Pillar 4

**Change:** The per-class loops in Pillars 3 and 4 now collect the names of asset classes that pass the significance threshold into `significant_adoption_classes` and `significant_liquidity_classes` lists respectively. These lists are embedded in three places:

1. The pillar `summary.interpretation` string (e.g. `"2/12 asset classes ... (Commodities, Venture Capital)"`)
2. The pillar `summary` dict as a named field for downstream consumers (Streamlit, chatbot)
3. The `conclusion.txt` output via `build_conclusion()`, which reads `s3["significant_adoption_classes"]` and `s4["significant_liquidity_classes"]` directly into the client-facing prose

This ensures that wherever the adoption and liquidity scores appear — in the JSON, in the console summary, and in the conclusion that clients read — the specific market segments driving the result are named, not just counted.

**Conclusion.txt output (as of current run):**

```
Only 16% of ex-repo CAV (by weight) is in asset classes with significant positive
adoption co-movement — improvement is concentrated in specific segments
(Commodities, Venture Capital).

CAV-weighted liquidity score: 4% of ex-repo asset value is in classes where monthly
turnover growth co-moves significantly with CAV growth (Private Equity, non-US
Government Debt).
```

---

# Phase 5 — Silver: Streamlit Dashboard

## Goal

Create a working Streamlit dashboard in the `silver/` folder that lets users explore thesis findings through saved Bronze outputs. The Silver tier reads from `data/` and does not re-fetch API data.

Silver can be launched independently of Gold.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/silver/app.py` | Main Streamlit entry point for the Silver tier. |
| Create | `src/silver/components/chart_panel.py` | Reusable chart rendering components. |
| Create | `src/silver/components/stats_panel.py` | Displays statistical outputs from `data/stats_summary.json`. |
| Create | `src/silver/components/conclusion_panel.py` | Displays conclusion from `data/conclusion.txt`. |
| Modify | `requirements.txt` | Add `streamlit`. |
| Create | `.streamlit/config.toml` | Basic Streamlit settings. |

---

## Dashboard Sections

### Overview

```text
KPI cards: latest total CAV, active asset class count, total holders, total volume, latest HHI
Charts: total CAV over time, CAV by asset class (stacked area)
```

### Pillar 1 — Growth

```text
Total CAV line chart
Rolling 3m and 6m growth rate chart
Monthly CAV growth chart
```

### Pillar 2 — Composition

```text
HHI over time
CAV share stacked area chart
Top-5 share trend
Active asset class count trend
```

### Pillar 3 — Adoption

```text
Holder growth index
Average position size
Holders per $1M CAV
CAV vs holders scatterplot
```

### Pillar 4 — Liquidity

```text
Volume over time
Turnover ratio chart
CAV vs turnover scatterplot
```

### Statistical Evidence

```text
Display stats_summary.json pillar-by-pillar
Show beta, p_value, confidence interval, interpretation for each test
```

### Conclusion

```text
Display conclusion.txt
Show structural-shift evidence summary
Show window-dressing risk summary
```

---

## Streamlit Requirements

* App reads from: `data/combined_metrics.csv`, `data/concentration_metrics.csv`, `data/stats_summary.json`, `data/conclusion.txt`.
* No API calls during runtime.
* Sidebar: date range filter, asset class multiselect, page selector.
* Missing result files show a clear instruction (e.g. "Run `python scripts/build_dataset.py` first.").
* Use existing matplotlib charts for the MVP. Do not migrate to Plotly in this phase.

---

## Acceptance Criteria

* `streamlit run src/silver/app.py` starts without error.
* App opens at `localhost:8501`.
* Date range and asset class filters work.
* All four pillar sections render.
* Statistical output and conclusion render.
* App does not call the API.
* Missing data is handled gracefully.
* Page loads in under 5 seconds on normal local execution.

---

## Commands to Run

```bash
pip install streamlit
streamlit run src/silver/app.py
```

---

# Phase 6 — Gold: Local Results Chatbot

## Goal

Add a natural-language interface in the `gold/` folder that lets users ask thesis-specific questions and receive answers grounded in the project's saved local outputs. Gold can be launched independently.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/gold/app.py` | Main Streamlit entry point for the Gold tier (Silver + Chat). |
| Create | `src/gold/components/chatbot_panel.py` | Streamlit chat UI. |
| Create | `src/gold/chat/context_builder.py` | Builds a concise context from saved results. |
| Create | `src/gold/chat/llm_client.py` | Thin wrapper around the chosen LLM API. |
| Modify | `requirements.txt` | Add required LLM SDK. |
| Modify | `.env.example` | Add LLM API key placeholder. |

---

## Chatbot Data Sources

```text
data/stats_summary.json
data/conclusion.txt
data/combined_metrics.csv
data/concentration_metrics.csv
```

The chatbot must not re-run analysis, re-fetch API data, invent missing values, or answer outside the thesis scope.

---

## Minimum Supported Questions

```text
What is the HHI trend?
Is the market becoming more diversified?
Which asset classes show stronger adoption?
Which asset classes have weak liquidity?
What evidence supports a structural shift?
What evidence supports window-dressing risk?
What are the main limitations of this analysis?
What data window is being used?
```

---

## Acceptance Criteria

* `streamlit run src/gold/app.py` starts without error.
* Chatbot answers thesis questions using actual saved results.
* Unrelated questions are redirected.
* API keys are loaded from `.env`, never hardcoded.
* Chatbot does not re-run analysis or call the RWA API.

---

# Phase 7 — Gold: Optional MCP / Live Data

## Goal

After the local Gold chatbot works, optionally add MCP or live-data integration so the chatbot can answer questions about the latest API state.

This phase is optional and should not block the thesis presentation.

---

## When To Start

Only start if Phases 1–6 are all complete and presentation-ready.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/gold/mcp/client.py` | MCP client wrapper. |
| Create | `src/gold/mcp/tools.py` | Safe tool functions for scoped data lookup. |
| Create | `src/gold/mcp/context_builder.py` | Builds live-data context if MCP is available. |
| Modify | `src/gold/components/chatbot_panel.py` | Add optional live-data mode if MCP is configured. |

---

## MCP Guardrails

* MCP mode must be optional. The dashboard must work without it.
* Chatbot must clearly distinguish cached thesis analysis from live MCP results.
* Live results must not overwrite saved thesis outputs unless the user explicitly re-runs the pipeline.
* MCP errors must not crash the app.

---

# File Structure After All Phases

```text
/
├── cache/                         # API cache (gitignored)
│
├── charts/                        # Static presentation-ready PNGs
│
├── data/                          # All datasets: pipeline outputs and static reference files
│   ├── .gitkeep
│   ├── combined_monthly.csv       # Phase 1 output
│   ├── combined_metrics.csv       # Phase 2 output
│   ├── concentration_metrics.csv  # Phase 2 output
│   ├── stats_summary.json         # Phase 4 output
│   ├── conclusion.txt             # Phase 4 output
│   └── statistical_results.csv   # Phase 4 output
│
├── scripts/
│   ├── build_dataset.py           # Runs Phase 1 + 2
│   ├── build_charts.py            # Runs Phase 3
│   └── run_analysis.py            # Runs Phase 4
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Shared config for all tiers
│   │
│   ├── bronze/                    # Data pipeline, metrics, charts, stats
│   │   ├── __init__.py
│   │   ├── api_client.py
│   │   ├── data_processing.py
│   │   ├── metrics.py
│   │   ├── metrics_config.py
│   │   ├── charts.py
│   │   └── stats.py
│   │
│   ├── silver/                    # Streamlit dashboard
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── components/
│   │       ├── chart_panel.py
│   │       ├── stats_panel.py
│   │       └── conclusion_panel.py
│   │
│   └── gold/                      # Chatbot + optional MCP
│       ├── __init__.py
│       ├── app.py
│       ├── components/
│       │   └── chatbot_panel.py
│       ├── chat/
│       │   ├── context_builder.py
│       │   └── llm_client.py
│       └── mcp/
│           ├── client.py
│           ├── context_builder.py
│           └── tools.py
│
├── tests/
│   ├── test_metrics.py
│   └── test_stats.py
│
├── .env
├── .env.example
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md
```

---

# How to Run Each Tier Independently

```bash
# Bronze — build all data, metrics, charts, and statistical analysis
python scripts/build_dataset.py
python scripts/build_charts.py
python scripts/run_analysis.py

# Silver — Streamlit dashboard (requires Bronze results to exist)
streamlit run src/silver/app.py

# Gold — Chatbot + MCP (requires Bronze results to exist)
streamlit run src/gold/app.py
```

---

# Git and Reproducibility Rules

## Commit after each successful phase

```bash
git commit -m "Phase 1: RWA data pipeline"
git commit -m "Phase 2: metrics layer"
git commit -m "Phase 3: static charts"
git commit -m "Phase 4: statistical analysis"
git commit -m "Phase 5 (Silver): Streamlit dashboard"
git commit -m "Phase 6 (Gold): local thesis chatbot"
git commit -m "Phase 7 (Gold): optional MCP integration"
```

## Gitignore

```text
.env
cache/
__pycache__/
*.pyc
.venv/
```

---

# Final Build Order

```text
Phase 1:  Bronze — RWA Data Pipeline
Phase 2:  Bronze — Metrics Layer
Phase 3:  Bronze — Static Charts
Phase 4:  Bronze — Statistical Analysis (4 pillars)
Phase 5:  Silver — Streamlit Dashboard
Phase 6:  Gold   — Local Results Chatbot
Phase 7:  Gold   — Optional MCP / Live Data
```

Do not move to the next phase until the current phase's acceptance criteria are met.
