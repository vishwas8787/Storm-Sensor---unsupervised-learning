"""Create interpretable event-level features for clustering."""

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


def engineer_features(df):
    """Return a copy with compact, nonnegative and seasonal features.

    Missing damage is preserved as missing here. Imputation is fitted later
    on the clustering sample so the decision is explicit and reproducible.
    """
    result = df.copy()
    for column in ["INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["TOTAL_INJURIES"] = result[["INJURIES_DIRECT", "INJURIES_INDIRECT"]].sum(axis=1, min_count=1)
    result["TOTAL_DEATHS"] = result[["DEATHS_DIRECT", "DEATHS_INDIRECT"]].sum(axis=1, min_count=1)

    result["DURATION_HOURS"] = (
        (result["END_DATE_TIME"] - result["BEGIN_DATE_TIME"]).dt.total_seconds() / 3600
    )
    result.loc[result["DURATION_HOURS"] < 0, "DURATION_HOURS"] = np.nan

    for source, target in [
        ("TOTAL_INJURIES", "LOG_TOTAL_INJURIES"),
        ("TOTAL_DEATHS", "LOG_TOTAL_DEATHS"),
        ("PROPERTY_DAMAGE_USD", "LOG_PROPERTY_DAMAGE_USD"),
        ("CROP_DAMAGE_USD", "LOG_CROP_DAMAGE_USD"),
        ("DURATION_HOURS", "LOG_DURATION_HOURS"),
    ]:
        values = pd.to_numeric(result[source], errors="coerce")
        values = values.mask(values < 0)
        result[target] = np.log1p(values)

    month = result["BEGIN_DATE_TIME"].dt.month
    result["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    result["MONTH_COS"] = np.cos(2 * np.pi * month / 12)
    return result
