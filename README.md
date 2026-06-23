# Thesis — Is RWA Tokenization a Structural Shift or Institutional Window Dressing?
2026 Summer Internship Program · Market Intelligence Research Case Study

## Thesis Question

Are financial institutions structurally migrating real-world assets to blockchain-based infrastructure — treating on-chain settlement and custody as a superior alternative to legacy financial rails — or is this a temporary positioning strategy with limited staying power?

The thesis is tested across four statistical pillars using live data from the [rwa.xyz](https://rwa.xyz) API:

| Pillar | Question | Method |
|--------|----------|--------|
| 1 — Growth | Is tokenized RWA value growing and at what rate? | OLS on log CAV, rolling growth trends |
| 2 — Composition | Is growth diversifying beyond US Treasury Debt? | HHI trend, top-5 share, active class count (OLS) |
| 3 — Adoption | Are more wallets participating relative to capital inflows? | OLS on log holders, Spearman ρ |
| 4 — Liquidity | Is transfer activity growing with circulating value? | Turnover ratio OLS, Spearman ρ |

---

## Three-Tier Architecture

This project is built in three independently runnable tiers. Each tier depends on the outputs of the tier below it, but does not require the tier above it to exist.

```
Bronze — Data pipeline, metrics, static charts, statistical analysis
         Run via: python3 scripts/build_dataset.py
                  python3 scripts/build_charts.py
                  python3 scripts/run_analysis.py

Silver — Streamlit dashboard built on top of Bronze outputs
         Run via: streamlit run src/silver/app.py

Gold   — Chatbot + optional live MCP data built on top of Silver
         Run via: streamlit run src/gold/app.py
```

---

## Build Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bronze: RWA Data Pipeline | ✅ Complete |
| 2 | Bronze: Metrics Layer | Planned |
| 3 | Bronze: Static Charts | Planned |
| 4 | Bronze: Statistical Analysis (4 pillars) | Planned |
| 5 | Silver: Streamlit Dashboard | Planned |
| 6 | Gold: Local Results Chatbot | Planned |
| 7 | Gold: Optional MCP / Live Data | Planned |

---

## Project Structure

```
summer2026-intern-thesis-WadeLittle/
│
├── scripts/
│   ├── build_dataset.py         ← Phase 1+2: fetch, process, validate, save CSV
│   ├── build_charts.py          ← Phase 3: load metrics → generate static PNGs
│   └── run_analysis.py          ← Phase 4: load metrics → run stats → save results
│
├── src/
│   ├── __init__.py
│   ├── config.py                ← Shared config: API credentials, asset scope, paths
│   │
│   └── bronze/                  ← All data pipeline, metrics, chart, and stats code
│       ├── __init__.py
│       ├── api_client.py        ← rwa.xyz API calls + 24-hour file cache
│       ├── data_processing.py   ← Raw JSON → clean monthly DataFrame + metric builders
│       ├── metrics.py           ← Derived metric functions (Phase 2)
│       ├── metrics_config.py    ← Metric constants: windows, thresholds (Phase 2)
│       ├── charts.py            ← Static chart functions (Phase 3)
│       └── stats.py             ← Statistical test functions (Phase 4)
│
├── data/                        ← All pipeline outputs (gitignored except .gitkeep)
│   ├── combined_monthly.csv     ← Phase 1 output: base dataset
│   ├── combined_metrics.csv     ← Phase 2 output
│   ├── concentration_metrics.csv
│   ├── stats_summary.json       ← Phase 4 output
│   ├── conclusion.txt
│   └── statistical_results.csv
│
├── tests/
│   ├── test_data_processing.py  ← Phase 1 tests (24 tests, all passing)
│   ├── test_metrics.py          ← Phase 2 tests (planned)
│   └── test_stats.py            ← Phase 4 tests (planned)
│
├── charts/                      ← Static chart PNGs (Phase 3 output)
├── cache/                       ← 24-hour API response cache (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Set up your environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# open .env and set: RWA_API_KEY=your_key_here
```

### 3. Build the dataset (Phase 1)

```bash
python3 scripts/build_dataset.py
```

This fetches CAV, holder counts, and transfer volume from rwa.xyz (cached for 24 hours), validates the output, prints a data quality summary, and saves the base dataset to `data/combined_monthly.csv`.

### 4. Verify the output

```bash
head -5 data/combined_monthly.csv

python3 -c "
import pandas as pd
df = pd.read_csv('data/combined_monthly.csv')
print(df.dtypes)
print(df['date'].min(), df['date'].max())
print(df['asset_class'].unique())
"
```

### 5. Run the test suite

```bash
python3 -m pytest tests/ -v
```

---

## Asset Classes in Scope

Thirteen traditional RWA asset classes are included in the primary analysis. Stablecoins, Cryptocurrencies, and Fiat Currency are tracked separately — they behave as digital-native or currency-proxy instruments rather than tokenized real-world assets and would distort concentration and adoption metrics if included.

| Asset Class | Category |
|-------------|----------|
| US Treasury Debt | Fixed Income |
| Corporate Credit | Fixed Income |
| Asset-Backed Credit | Fixed Income |
| Diversified Credit | Fixed Income |
| non-US Government Debt | Fixed Income |
| Repurchase Agreements | Fixed Income |
| Real Estate | Alternative |
| Private Equity | Alternative |
| Venture Capital | Alternative |
| Active Strategies | Alternative |
| Specialty Finance | Alternative |
| Stocks | Equity |
| Commodities | Real Assets |

**Supplemental (tracked separately):** Stablecoins · Cryptocurrencies · Fiat Currency

---

## Phase 1 Output

`data/combined_monthly.csv` — 493 rows, 13 asset classes, 2023-01 to 2026-05.

| Column | Type | Description |
|--------|------|-------------|
| `date` | string (YYYY-MM-DD) | Month-start date |
| `asset_class` | string | One of the 13 in-scope classes |
| `cav` | float | Average monthly circulating asset value (USD) |
| `holders` | float | End-of-month holder count |
| `volume` | float | Total monthly transfer volume (USD) |

Data quality guarantees enforced at build time:
- Date range starts no earlier than 2023-01-01
- Partial current month excluded
- No negative CAV rows
- Missing holders/volume filled with 0 (zero activity, not missing data)
- CAV gaps left as NaN (absence of data ≠ zero value)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas / numpy | Data manipulation and computation |
| matplotlib | Static chart generation (Phase 3) |
| scipy / statsmodels | Statistical tests: OLS, Spearman, Kendall (Phase 4) |
| streamlit | Interactive dashboard (Phase 5+) |
| python-dotenv | API key management via `.env` |
| pytest | Test suite |
| rwa.xyz API | Live RWA market data |

---

## Statistical Methods (Phase 4)

### OLS Regression (Pillars 1, 2, 3, 4)

Fits a trend line through the metric over time. The slope (β) quantifies direction and speed; the p-value indicates whether that slope is distinguishable from zero. Confidence intervals are included in all OLS outputs.

**Limitations:** Assumes linearity; sensitive to outliers; short time series reduces statistical power.

---

### Spearman Correlation (Pillars 3, 4)

Measures whether two metrics move in the same direction over time. Rank-based, so it does not assume a linear relationship or equal scales.

**Limitations:** Captures co-movement direction, not magnitude or causation. Both metrics trending upward in a growing market can produce high ρ from shared time trend rather than a meaningful relationship.

---

### Kendall Tau (Pillar 3)

Tests whether the participation ratio (holders index / CAV index) shows consistent monotonic movement. Positive τ = holders outpacing CAV (broadening adoption); negative τ = CAV outpacing holders (concentration).

**Limitations:** Captures directional consistency only, not magnitude. Short time series limits power.

---

### Relative Growth Index (Pillar 3)

Each asset class is indexed to its own first valid monthly observation = 100. This shows relative growth *within* each class, not absolute adoption size. Baseline quality flags (`low_cav_baseline_flag`, `low_holders_baseline_flag`, `late_start_flag`) are attached to every row so that small or late baselines are visible when interpreting results.

---

## Limitations

- Correlation ≠ causation; structural-shift signals may reflect macro conditions (rate environment, regulatory clarity) rather than tokenization-specific dynamics
- Single data source (rwa.xyz); coverage gaps in smaller asset classes may affect completeness
- Holder count is an on-chain wallet proxy, not a perfect measure of unique users — one institution can control multiple wallets
- Short overall time series (starting 2023-01) limits statistical power for trend detection
- Asset classes analyzed in isolation, not in portfolio context
- Results with insufficient sample size are flagged `insufficient_data` in the statistical output rather than reported as weak/no effect
