# Copyright (c) 2026 Wade Little. All rights reserved.

"""
Everything related to talking to the rwa.xyz API:
authenticated requests, the timeseries query builder, and a
24-hour file cache so repeated runs don't re-hit the API.
"""

import os
import json
import time
from urllib.parse import quote
import requests

from src.config import (
    BASE_URL, HEADERS, CACHE_DIR, CACHE_TTL_SECONDS,
    MEASURE_CAV, MEASURE_HOLDERS, MEASURE_VOLUME, ANALYSIS_START_DATE,
)


def cached_request(fetch_fn, cache_key):
    """
    Wraps any fetch function with a 24-hour file cache.
    Fresh cache -> use it. Stale/missing -> refetch.
    Refetch fails + stale cache exists -> fall back to stale with a warning.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    cache_exists = os.path.exists(path)
    if cache_exists:
        age_seconds = time.time() - os.path.getmtime(path)
        if age_seconds < CACHE_TTL_SECONDS:
            print(f"[cache] Using cached data for '{cache_key}' "
                  f"(age: {age_seconds / 3600:.1f}h, TTL: {CACHE_TTL_SECONDS / 3600:.0f}h)")
            with open(path) as f:
                return json.load(f)

    try:
        print(f"[cache] Fetching fresh data for '{cache_key}'...")
        data = fetch_fn()
    except Exception as e:
        if cache_exists:
            print(f"[cache] Refresh failed for '{cache_key}' ({e}). Using stale cache.")
            with open(path) as f:
                return json.load(f)
        raise

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[cache] Saved '{cache_key}' to {path}")
    return data


def _timeseries_query(measure_slug, endpoint, group_by="asset_class",
                      aggregate_function="sum", interval="day", mode="stock"):
    """
    Shared query builder for the /aggregates/timeseries endpoints.
    endpoint options: 'assets' or 'tokens'
    """
    query = {
        "filter": {
            "operator": "and",
            "filters": [
                {
                    "operator": "equals",
                    "field": "measure_slug",
                    "value": measure_slug
                },
                {
                    "operator": "onOrAfter",
                    "field": "date",
                    "value": ANALYSIS_START_DATE
                }
            ]
        },
        "aggregate": {
            "groupBy": group_by,
            "aggregateFunction": aggregate_function,
            "interval": interval,
            "mode": mode
        }
    }
    query_param = quote(json.dumps(query))
    url = f"{BASE_URL}/v4/{endpoint}/aggregates/timeseries?query={query_param}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_cav_by_asset_class():
    """CAV by asset class, daily, stock mode. Resampled to monthly in data_processing."""
    return _timeseries_query(MEASURE_CAV, endpoint="tokens", mode="stock")


def get_holders_by_asset_class():
    """Holder counts by asset class, daily, stock mode. Resampled to monthly in data_processing."""
    return _timeseries_query(MEASURE_HOLDERS, endpoint="tokens", mode="stock")


def get_transfer_volume_by_asset_class():
    """Transfer volume by asset class, daily, flow mode. Summed to monthly in data_processing."""
    return _timeseries_query(MEASURE_VOLUME, endpoint="tokens", mode="flow")
