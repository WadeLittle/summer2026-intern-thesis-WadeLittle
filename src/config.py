# Copyright (c) 2026 Wade Little. All rights reserved.
# Central configuration: constants used across the project.

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RWA_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "RWA_API_KEY not found. Make sure .env exists and contains RWA_API_KEY=your_key"
    )

BASE_URL = "https://api.rwa.xyz"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

CACHE_DIR = "cache"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours, matches rwa.xyz's daily update cadence

CHARTS_DIR = "charts"
RESULTS_DIR = "data"

# API filter date — set early to capture all available history.
ANALYSIS_START_DATE = "2023-01-01"

# Measure slugs (confirmed against live API responses)
MEASURE_CAV = "circulating_asset_value_dollar"
MEASURE_HOLDERS = "holding_addresses_count"
MEASURE_VOLUME = "daily_transfer_volume_dollar"

# Traditional RWA asset classes — the primary scope of this thesis.
# Stablecoins, Cryptocurrencies, and Fiat Currency are excluded here because
# they behave as digital-native or currency instruments rather than tokenized
# real-world assets; including them distorts concentration and adoption metrics.
ASSET_CLASSES_IN_SCOPE = [
    "US Treasury Debt",
    "Commodities",
    "Real Estate",
    "Stocks",
    "Corporate Credit",
    "Asset-Backed Credit",
    "Private Equity",
    "Venture Capital",
    "Active Strategies",
    "Diversified Credit",
    "non-US Government Debt",
    "Specialty Finance",
    "Repurchase Agreements",
]

# Tracked separately — digital-native and currency-proxy assets that rwa.xyz
# reports alongside traditional RWAs but are outside the thesis scope.
ASSET_CLASSES_SUPPLEMENTAL = [
    "Stablecoins",
    "Cryptocurrencies",
    "Fiat Currency",
]

# ---------------------------------------------------------------------------
# Visualization — asset class color identity
# ---------------------------------------------------------------------------

# The class excluded from all "ex-repo" chart variants.
REPO_CLASS = "Repurchase Agreements"

# Fallback color for collapsed "Other" buckets and any unknown class name.
OTHER_COLOR = "#c7c7c7"

# One fixed color per in-scope asset class.
# All charts import from here so every class renders in the same color
# regardless of which chart it appears in.
ASSET_CLASS_COLORS: dict[str, str] = {
    "Active Strategies":      "#1f77b4",  # tab blue
    "Asset-Backed Credit":    "#ff7f0e",  # tab orange
    "Commodities":            "#2ca02c",  # tab green
    "Corporate Credit":       "#d62728",  # tab red
    "Diversified Credit":     "#9467bd",  # tab purple
    "Private Equity":         "#8c564b",  # tab brown
    "Real Estate":            "#e377c2",  # tab pink
    "Repurchase Agreements":  "#17becf",  # tab cyan
    "Specialty Finance":      "#bcbd22",  # tab yellow-green
    "Stocks":                 "#aec7e8",  # tab light blue
    "US Treasury Debt":       "#ffbb78",  # tab light orange
    "Venture Capital":        "#98df8a",  # tab light green
    "non-US Government Debt": "#ff9896",  # tab light red
    "Other":                  "#c7c7c7",  # light grey (same as OTHER_COLOR)
}