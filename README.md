# Thesis — Is RWA Tokenization Undergoing a Structural Shift?
2026 Summer Internship Program · Market Intelligence Research Case Study

## Project Overview

This project investigates whether financial institutions are structurally migrating real-world assets to blockchain-based infrastructure — treating on-chain settlement and custody as a superior alternative to legacy financial rails. The analysis tests this through three statistics backed pillars: whether asset-class composition is diversifying beyond tokenized Treasuries, whether wallet participation is broadening relative to capital inflows, and whether secondary market activity is keeping pace with circulating value.

The analysis is anchored to January 2024 (the BlackRock BUIDL launch), a widely recognized inflection point for institutional RWA adoption. Data is pulled live from the [rwa.xyz](https://rwa.xyz) API.

The thesis is tested across three statistical pillars:

| Pillar | Question | Method |
|--------|----------|--------|
| 1 — Market Concentration | Is the market diversifying beyond US Treasury Debt? | HHI trend via OLS |
| 2 — On-Chain Participation | Are more wallets participating relative to capital inflows? | Spearman ρ + Kendall τ |
| 3 — Liquidity | Is transfer activity growing relative to circulating value? | Turnover ratio via OLS |

**Current tier: Lower/Bronze** — static statistical analysis + 3 charts + console report.

---

## Project Structure

```
summer2026-intern-thesis-WadeLittle/
│
├── src/
│   ├── config.py            ← API key, measure slugs, asset class scope, date window
│   ├── api_client.py        ← rwa.xyz API calls + 24-hour file cache
│   ├── data_processing.py   ← Raw JSON → cleaned monthly DataFrame + pillar builders
│   │
│   └── bronze/
│       └── analysis.py      ← Run: python src/bronze/analysis.py
│
├── charts/                  ← Auto-generated (created on first run)
├── cache/                   ← Auto-generated API cache (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Set up your environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and add your rwa.xyz API key:

```bash
cp .env.example .env
# then open .env and set: RWA_API_KEY=your_key_here
```

### 3. Run the Bronze tier analysis

```bash
python src/bronze/analysis.py
```

**What it does:**

- Fetches CAV, holder counts, and transfer volume from the rwa.xyz API (cached for 24 hours)
- Generates 5 charts saved to `./charts/`:
  - `chart1_composition.png` — 100% stacked area: share of circulating asset value by class
  - `chart2_adoption.png` — 2×2 subplots: CAV index vs. holder index per asset class
  - `chart2b_avg_position.png` — 2×2 subplots: average dollar value held per wallet per asset class
  - `chart2c_adoption_context.png` — 4×3 grid: raw holders | raw CAV | index, one row per asset class
  - `chart3_liquidity.png` — monthly turnover ratio with 3-month rolling average
- Prints a full statistical summary to the console (HHI OLS, Spearman/Kendall per asset class, liquidity OLS)
- Prints a plain-English thesis conclusion with an overall verdict

---

## Asset Classes in Scope

| Asset Class | Description |
|-------------|-------------|
| US Treasury Debt | Tokenized T-bills and government bonds |
| Commodities | Tokenized gold and other commodities |
| Real Estate | Tokenized real estate funds and properties |
| Stocks | Tokenized equities |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas / numpy | Data manipulation and computation |
| matplotlib | Chart generation |
| scipy / statsmodels | Statistical tests (OLS, Spearman, Kendall) |
| python-dotenv | API key management via `.env` |
| rwa.xyz API | Live RWA market data |

---

## Statistical Methods

### OLS Regression (Pillars 1 and 3)

Fits a straight line through the metric over time. The slope (β) quantifies direction and speed of the trend; the p-value tells you whether that slope is distinguishable from zero with statistical confidence.

**Why chosen:** Both HHI (concentration) and turnover ratio are single continuous metrics where the core question is simply whether they are trending up or down over time. OLS is the most direct tool for that.

**Limitations:**
- Assumes the relationship is linear — concentration could drop sharply then plateau, which a straight line misrepresents
- Sensitive to outliers; one anomalous month can shift the slope meaningfully
- Small sample (~18 months) makes it harder to reach significance even when a real trend exists

---

### Spearman Correlation (Pillar 2)

Measures whether the CAV index and holder index move in the same direction over time. Unlike standard Pearson correlation, Spearman works on ranks rather than raw values, so it does not assume the relationship is linear.

**Why chosen:** CAV and holder counts can grow at very different rates and scales. A rank-based method is more appropriate than Pearson because it captures whether the two metrics move in sync without being distorted by magnitude differences.

**Limitations:**
- Tells you *whether* they move together but not *by how much* or *who is leading*
- Both metrics trend upward in a growing market almost by definition, so a high ρ may partly reflect a shared time trend rather than a meaningful relationship between the two specifically

---

### Kendall Tau (Pillar 2)

Tests whether the participation ratio (holders index / CAV index) shows a consistent monotonic trend — meaning it moves mostly in one direction rather than bouncing randomly. A positive τ means holders are generally outpacing CAV (broadening adoption); negative means CAV is outpacing holders (concentration).

**Why chosen:** The participation ratio is the direct measure of whether adoption is broadening. Kendall tau tests whether the *directional consistency* of that ratio holds across the full time period, which is more meaningful than just comparing start and end values. It is also more robust than OLS for ratio data that can be volatile month to month.

**Limitations:**
- Only captures directional consistency, not magnitude — a ratio that ticked up by 0.001 every month gets the same positive τ as one that surged
- Short time series limits statistical power here as well

---


## Limitations

- Correlation ≠ causation; structural-shift signals may reflect macro conditions (rate environment, regulatory clarity) rather than tokenization-specific dynamics
- Analysis window starts Jan 2024 — a short time series limits statistical power
- Single API source (rwa.xyz); coverage gaps in smaller asset classes may affect results
- Asset classes analyzed in isolation, not in portfolio context
