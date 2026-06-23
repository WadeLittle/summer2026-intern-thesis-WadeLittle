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
| Is growth meaningful vs TradFi? | tokenized share of TradFi | benchmark share chart | relative growth comparison |

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
| Modify | `src/config.py` | Set `ANALYSIS_START_DATE = "2023-01-01"`. Move Stablecoins and Cryptocurrencies out of the main RWA scope (track separately). Add `RESULTS_DIR = "results"`. |
| Move/Modify | `src/bronze/api_client.py` | Move from `src/api_client.py`. Preserve existing API request behavior. Only the smallest import-path changes needed so scripts run from the repo root without `sys.path` manipulation. |
| Move/Modify | `src/bronze/data_processing.py` | Move from `src/data_processing.py`. Add or update logic to build and save a combined monthly dataset. Add input validation for required columns, dates, numeric fields, and negative values. |
| Create | `src/__init__.py` | Empty file to establish `src` as a package. |
| Create | `src/bronze/__init__.py` | Empty file to establish `src/bronze` as a package. |
| Create | `scripts/build_dataset.py` | Standalone script: fetch/load cache → process → validate → save CSV. |
| Create | `results/` | Output folder. Include `.gitkeep`. |

---

## Required Output

`results/combined_monthly.csv`

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
* `results/combined_monthly.csv` is produced with: `date, asset_class, cav, holders, volume`.
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
head -5 results/combined_monthly.csv
python -c "import pandas as pd; df = pd.read_csv('results/combined_monthly.csv'); print(df.dtypes); print(df['date'].min(), df['date'].max()); print(df['asset_class'].unique())"
```

---

# Phase 1b — Bronze: TradFi Benchmark Pipeline

## Goal

Pull and cache TradFi benchmark data from a free API (Yahoo Finance via `yfinance`) to enable the Pillar 5 comparison in Phase 4. This runs alongside or immediately after Phase 1.

This phase produces benchmark data only.

---

## Why yfinance

`yfinance` is free, requires no API key, covers the required asset classes, and supports the needed date range. Responses should be cached locally with the same TTL approach used for RWA data.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/bronze/tradfi_client.py` | Fetches benchmark data from yfinance with local caching. |
| Create | `data/benchmarks/tradfi_benchmarks.csv` | Manually curated seed file for benchmarks not available in yfinance (optional fallback). |
| Modify | `scripts/build_dataset.py` | After RWA data, also fetch and save TradFi benchmark data. |
| Modify | `requirements.txt` | Add `yfinance`. |

---

## Benchmark Targets

| Benchmark | Ticker / Proxy | Comparison |
|---|---|---|
| U.S. Treasuries / Money Markets | `^IRX`, `SHY`, or manual AUM seed | Tokenized Treasuries |
| Private Credit | Manual seed or `^GSPC` proxy | Tokenized private credit |
| Gold | `GLD` or `GC=F` | Tokenized commodities/gold |
| Broad ETF/Funds | `SPY` AUM proxy | Tokenized funds |
| Real Estate | `VNQ` | Tokenized real estate |

---

## Required Output

`results/tradfi_benchmarks.csv`

Required columns:

```text
date
benchmark_name
asset_class
tradfi_value
source
notes
```

Rules:

```text
Date range should match or exceed the RWA analysis window (2023-01-01 to latest month).
Values should represent AUM or market size, not price.
Cached locally to avoid redundant API calls.
If a benchmark cannot be fetched, log it clearly and skip rather than crashing.
```

---

## Acceptance Criteria

* `python scripts/build_dataset.py` produces `results/tradfi_benchmarks.csv`.
* At least 3 benchmark series are present.
* Dates align with the RWA data range.
* Cached data is used on second run within TTL.
* Missing benchmarks are logged, not silently dropped.

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

### Pillar 5 — TradFi Benchmark

```text
tokenized_share_of_tradfi  # tokenized_cav / tradfi_value per benchmark pair
relative_growth_rate       # monthly_cav_growth - tradfi_growth per benchmark pair
```

---

## Required Outputs

`results/combined_metrics.csv` columns:

```text
date, asset_class, cav, holders, volume,
cav_share, cav_index, asset_class_cav_growth,
holders_index, monthly_holder_growth, holders_per_million_cav, avg_position,
turnover_ratio, turnover_3m, monthly_volume_growth
```

`results/concentration_metrics.csv` columns:

```text
date, total_cav, monthly_cav_growth, rolling_3m_cav_growth, rolling_6m_cav_growth,
hhi, top_5_share, asset_class_count, active_asset_class_count
```

`results/tradfi_comparison.csv` columns:

```text
date, benchmark_name, asset_class, tokenized_cav, tradfi_value,
tokenized_share_of_tradfi, relative_growth_rate
```

---

## Acceptance Criteria

* `python scripts/build_dataset.py` produces all three metric output files.
* `cav_share` sums to approximately 1.0 for every date where CAV data is available.
* `cav_index` equals exactly 100 for each asset class at its first non-null, positive CAV month.
* No division-by-zero errors when `holders == 0` or `cav == 0`.
* Unit tests pass: `python -m pytest tests/test_metrics.py -v`.

---

## Commands to Run

```bash
python scripts/build_dataset.py
python -m pytest tests/test_metrics.py -v
python -c "import pandas as pd; df = pd.read_csv('results/combined_metrics.csv', parse_dates=['date']); print(df.columns.tolist()); print(df.tail())"
python -c "import pandas as pd; hhi = pd.read_csv('results/concentration_metrics.csv'); print(hhi.tail())"
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
chart9_tradfi_comparison.png      — tokenized share of TradFi benchmarks
chart10_scorecard.png             — thesis scorecard summary (if data supports it)
```

At minimum, produce charts 1–5.

---

## Chart Requirements

* Charts must read from `results/combined_metrics.csv`, `results/concentration_metrics.csv`, and `results/tradfi_comparison.csv`.
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

* `python scripts/build_charts.py` runs without error.
* At least 5 core chart PNGs are created in `charts/`.
* No chart x-axis begins before the first date in the data.
* Each chart function can be called independently with a metrics DataFrame.
* Running the script a second time overwrites outputs rather than duplicating them.

---

## Commands to Run

```bash
python scripts/build_charts.py
ls -lh charts/
```

---

# Phase 4 — Bronze: Statistical Analysis

## Goal

Create a clean statistical analysis layer that evaluates the thesis across five pillars using the metrics outputs. Produce structured results and a plain-English conclusion.

All statistical code lives in `src/bronze/`.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/bronze/stats.py` | Statistical functions and `run_analysis(df) -> dict`. |
| Create | `scripts/run_analysis.py` | Load saved metrics → run all pillars → save results. |
| Create | `tests/test_stats.py` | Unit tests for statistical functions using synthetic data. |
| Create | `results/stats_summary.json` | Structured statistical output. |
| Create | `results/conclusion.txt` | Plain-English conclusion. |
| Create | `results/statistical_results.csv` | Flattened table of key statistical outputs. |

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

### Pillar 5 — TradFi Benchmark Comparison

**Goal:** Put tokenized RWA growth in context against comparable traditional finance markets.

```text
Tokenized share of comparable market:
  tokenized_share_of_tradfi_t = tokenized_cav_t / tradfi_value_t

Relative growth rate:
  relative_growth_rate_t = monthly_cav_growth_t - tradfi_growth_t

OLS trend on tokenized share:
  tokenized_share_t = alpha + beta*time
```

Benchmarks:

```text
Tokenized Treasuries     vs U.S. Treasury or money market fund AUM (SHY, ^IRX)
Tokenized private credit vs private credit AUM (manual seed)
Tokenized commodities    vs gold ETF AUM or gold market proxy (GLD, GC=F)
Tokenized funds          vs ETF/mutual fund AUM (SPY proxy)
Tokenized real estate    vs REIT or commercial real estate proxy (VNQ)
```

Output: `beta, p_value, confidence_interval, interpretation`

Interpretation logic:

```text
Structural shift evidence is stronger when:
- tokenized assets grow faster than comparable TradFi benchmarks
- tokenized share of the comparable TradFi market rises over time
- growth persists even after accounting for small-base effects

Window-dressing risk is higher when:
- tokenized growth is large in % terms but economically tiny relative to TradFi
- tokenized share of the comparable market remains flat
- no adoption or liquidity improvement despite headline growth
```

Limitation note:

```text
TradFi comparisons are benchmarks, not perfect equivalents. Differences in market structure,
investor base, liquidity, regulation, and product design should be explained clearly.
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
6. Is tokenized growth meaningful relative to comparable TradFi markets?
```

Structural shift evidence is stronger when:

```text
- CAV is growing significantly
- growth is persistent, not just one spike
- HHI is falling
- active asset class count is rising
- holder participation is rising
- turnover/liquidity is stable or improving
- tokenized share of comparable TradFi markets is rising
```

Window-dressing risk is higher when:

```text
- CAV growth is slowing
- CAV growth is concentrated in one asset class
- HHI remains high or rises
- holder growth is weak
- turnover is low or falling
- tokenized market size remains economically tiny relative to TradFi benchmarks
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

`results/stats_summary.json` structure:

```text
pillar1_growth
pillar2_composition
pillar3_adoption
pillar4_liquidity
pillar5_tradfi
conclusion
generated_at
data_window
limitations
```

---

## Acceptance Criteria

* `python scripts/run_analysis.py` runs without error.
* `results/stats_summary.json` is produced with all five pillars.
* `results/conclusion.txt` is produced.
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
python -c "import json; d=json.load(open('results/stats_summary.json')); print(list(d.keys()))"
cat results/conclusion.txt
```

---

# Phase 5 — Silver: Streamlit Dashboard

## Goal

Create a working Streamlit dashboard in the `silver/` folder that lets users explore thesis findings through saved Bronze outputs. The Silver tier reads from `results/` and does not re-fetch API data.

Silver can be launched independently of Gold.

---

## Files to Create or Modify

| Action | File | What changes |
|---|---|---|
| Create | `src/silver/app.py` | Main Streamlit entry point for the Silver tier. |
| Create | `src/silver/components/chart_panel.py` | Reusable chart rendering components. |
| Create | `src/silver/components/stats_panel.py` | Displays statistical outputs from `results/stats_summary.json`. |
| Create | `src/silver/components/conclusion_panel.py` | Displays conclusion from `results/conclusion.txt`. |
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

### Pillar 5 — TradFi Benchmark

```text
Tokenized share of TradFi benchmarks
Relative growth rate vs benchmarks
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

* App reads from: `results/combined_metrics.csv`, `results/concentration_metrics.csv`, `results/tradfi_comparison.csv`, `results/stats_summary.json`, `results/conclusion.txt`.
* No API calls during runtime.
* Sidebar: date range filter, asset class multiselect, page selector.
* Missing result files show a clear instruction (e.g. "Run `python scripts/build_dataset.py` first.").
* Use existing matplotlib charts for the MVP. Do not migrate to Plotly in this phase.

---

## Acceptance Criteria

* `streamlit run src/silver/app.py` starts without error.
* App opens at `localhost:8501`.
* Date range and asset class filters work.
* All five pillar sections render.
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
results/stats_summary.json
results/conclusion.txt
results/combined_metrics.csv
results/concentration_metrics.csv
results/tradfi_comparison.csv
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
How does tokenized growth compare to traditional finance benchmarks?
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
├── data/
│   └── benchmarks/
│       └── tradfi_benchmarks.csv  # Manual seed for benchmarks not in yfinance
│
├── results/
│   ├── combined_monthly.csv
│   ├── combined_metrics.csv
│   ├── concentration_metrics.csv
│   ├── tradfi_benchmarks.csv
│   ├── tradfi_comparison.csv
│   ├── stats_summary.json
│   ├── conclusion.txt
│   └── statistical_results.csv
│
├── scripts/
│   ├── build_dataset.py           # Runs Phase 1 + 1b + 2
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
│   │   ├── tradfi_client.py
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
git commit -m "Phase 1b: TradFi benchmark pipeline"
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
Phase 1b: Bronze — TradFi Benchmark Pipeline
Phase 2:  Bronze — Metrics Layer
Phase 3:  Bronze — Static Charts
Phase 4:  Bronze — Statistical Analysis (5 pillars)
Phase 5:  Silver — Streamlit Dashboard
Phase 6:  Gold   — Local Results Chatbot
Phase 7:  Gold   — Optional MCP / Live Data
```

Do not move to the next phase until the current phase's acceptance criteria are met.
