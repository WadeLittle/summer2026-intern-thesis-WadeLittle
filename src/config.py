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

# Jan 2024 = BUIDL launch, a recognized inflection point for institutional
# RWA adoption. 
ANALYSIS_START_DATE = "2024-01-01"

# Measure slugs (confirmed against live API responses)
MEASURE_CAV = "circulating_asset_value_dollar"
MEASURE_HOLDERS = "holding_addresses_count"
MEASURE_VOLUME = "daily_transfer_volume_dollar"

# Asset classes in scope.
ASSET_CLASSES_IN_SCOPE = [
    "US Treasury Debt",
    "Commodities",
    "Real Estate",
    "Stocks",
]