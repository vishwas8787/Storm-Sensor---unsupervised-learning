"""Feature engineering for extreme weather event grouping.

Transforms raw NOAA Storm Events into interpretable, non-negative,
log-scaled impact features and cyclical seasonal features suitable
for distance-based clustering (Hierarchical and DBSCAN).
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "LOG_TOTAL_INJURIES",
    "LOG_TOTAL_DEATHS",
    "LOG_PROPERTY_DAMAGE_USD",
    "LOG_CROP_DAMAGE_USD",
    "LOG_DURATION_HOURS",
    "MONTH_SIN",
    "MONTH_COS",
]

RAW_FEATURE_COLUMNS = [
    "TOTAL_INJURIES",
    "TOTAL_DEATHS",
    "PROPERTY_DAMAGE_USD",
    "CROP_DAMAGE_USD",
    "DURATION_HOURS",
]


def engineer_features(df):
    """Compute impact, duration, and cyclical seasonality features.

    Handling Choices:
    1. Casualty features: Direct and indirect injuries/deaths are summed.
       Missing values are treated as 0 (no casualties reported).
    2. Damage features: In NOAA/NWS Storm Events reports, missing property
       and crop damage indicates no confirmed or quantifiable economic
       damage occurred for that event. They are imputed as $0.00.
    3. Duration: Computed in hours from BEGIN_DATE_TIME to END_DATE_TIME.
       Invalid or missing durations are imputed with the median event duration.
    4. Log1p transformation: Applied to highly skewed continuous impact variables
       (injuries, deaths, property damage, crop damage, duration) to stabilize
       variance and prevent extreme outliers from crushing distance metrics.
    5. Seasonality: Sine and cosine transforms of the event begin month
       preserve the cyclical nature of the calendar year (December is adjacent to January).
    """
    result = df.copy()

    # Casualties: sum direct + indirect
    for col in ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).clip(lower=0)

    result["TOTAL_INJURIES"] = result["INJURIES_DIRECT"] + result["INJURIES_INDIRECT"]
    result["TOTAL_DEATHS"] = result["DEATHS_DIRECT"] + result["DEATHS_INDIRECT"]

    # Dates and duration
    result["BEGIN_DATE_TIME"] = pd.to_datetime(result["BEGIN_DATE_TIME"], errors="coerce")
    result["END_DATE_TIME"] = pd.to_datetime(result["END_DATE_TIME"], errors="coerce")

    duration_hours = (result["END_DATE_TIME"] - result["BEGIN_DATE_TIME"]).dt.total_seconds() / 3600.0
    duration_hours = duration_hours.mask(duration_hours < 0)
    median_duration = duration_hours.median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 0.5  # fallback default 30 mins
    result["DURATION_HOURS"] = duration_hours.fillna(median_duration)

    # Damage: missing damage in NOAA reports denotes 0 confirmed damage
    for source, target in [("PROPERTY_DAMAGE_USD", "PROPERTY_DAMAGE_USD"), ("CROP_DAMAGE_USD", "CROP_DAMAGE_USD")]:
        if source in result.columns:
            val = pd.to_numeric(result[source], errors="coerce").fillna(0.0).clip(lower=0)
            result[target] = val
        else:
            result[target] = 0.0

    # Log1p transformations
    result["LOG_TOTAL_INJURIES"] = np.log1p(result["TOTAL_INJURIES"])
    result["LOG_TOTAL_DEATHS"] = np.log1p(result["TOTAL_DEATHS"])
    result["LOG_PROPERTY_DAMAGE_USD"] = np.log1p(result["PROPERTY_DAMAGE_USD"])
    result["LOG_CROP_DAMAGE_USD"] = np.log1p(result["CROP_DAMAGE_USD"])
    result["LOG_DURATION_HOURS"] = np.log1p(result["DURATION_HOURS"])

    # Cyclical seasonality
    month = result["BEGIN_DATE_TIME"].dt.month.fillna(6).astype(int)
    result["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    result["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)

    # Ensure latitude and longitude are numeric if present
    for coord in ["BEGIN_LAT", "BEGIN_LON"]:
        if coord in result.columns:
            result[coord] = pd.to_numeric(result[coord], errors="coerce")

    return result
