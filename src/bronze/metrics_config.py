# Copyright (c) 2026 Wade Little. All rights reserved.
"""
Configuration constants for the Bronze metrics layer.

Centralising these here means tuning a window or threshold is a one-line
change that propagates everywhere automatically.
"""

# Rolling-window sizes (months)
ROLLING_3M = 3
ROLLING_6M = 6

# Minimum non-null observations required before a rolling window produces
# a result.  1 means the first available month still gets a value (no
# leading NaNs from the window), which is appropriate for visual charts
# that should not have a dead zone at the start of the series.
MIN_PERIODS_ROLLING = 1

# A month where CAV > ACTIVE_CAV_THRESHOLD counts toward
# active_asset_class_count.  $1 M keeps micro/test deployments out of
# the denominator while capturing any commercially meaningful tranche.
ACTIVE_CAV_THRESHOLD = 1_000_000

# Baseline-quality thresholds — used to flag, not suppress, index values.
# Baselines below these levels can amplify percentage growth optically.
MIN_BASELINE_CAV = 1_000_000        # $1 M minimum credible CAV baseline
MIN_BASELINE_HOLDERS = 10           # 10 holders minimum credible baseline

# Minimum asset classes needed for HHI to be meaningful.
# A single-class market trivially yields HHI = 1.0.
HHI_MIN_CLASSES = 2
