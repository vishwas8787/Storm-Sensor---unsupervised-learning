
import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR / "data" / "raw"
    / "StormEvents_details_all.csv"
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "metrics"

CLEANED_FILE = PROCESSED_DIR / "cleaned_storm_data.csv"


# ==========================================
# DAMAGE CONVERSION
# ==========================================

def convert_damage_column(series):

    values = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )

    # Extract numeric amount and optional suffix
    extracted = values.str.extract(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([KMB]?)$"
    )

    amount = pd.to_numeric(
        extracted[0],
        errors="coerce"
    )

    multiplier = extracted[1].map({
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000
    })

    converted = amount * multiplier

    # Negative damage values are invalid
    converted = converted.mask(converted < 0)

    # Identify non-empty values that could not be converted
    unparsed = values.notna() & values.ne("") & amount.isna()

    return converted, unparsed


# ==========================================
# DATA VALIDATION
# ==========================================

def validate_data(df):

    required_columns = [
        "EVENT_ID",
        "YEAR",
        "EVENT_TYPE",
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ==========================================
# MAIN PREPROCESSING FUNCTION
# ==========================================

def preprocess_data(df):

    print("\n========== DATA PREPROCESSING ==========")

    df = df.copy()

    validate_data(df)

    initial_rows = len(df)

    # --------------------------------------
    # 1. Remove duplicate events
    # --------------------------------------

    duplicate_count = df["EVENT_ID"].duplicated().sum()

    print("\nDuplicate events:", duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates(
            subset="EVENT_ID",
            keep="first"
        )

    # --------------------------------------
    # 2. Convert date columns
    # --------------------------------------

    print("\nConverting date columns...")

    date_columns = [
        "BEGIN_DATE_TIME",
        "END_DATE_TIME"
    ]

    for col in date_columns:

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    invalid_dates = df[date_columns].isna().sum()

    print("Invalid or missing dates:")
    print(invalid_dates)

    # --------------------------------------
    # 3. Convert property damage
    # --------------------------------------

    print("\nConverting property damage...")

    (
        df["PROPERTY_DAMAGE_USD"],
        property_unparsed
    ) = convert_damage_column(
        df["DAMAGE_PROPERTY"]
    )

    # --------------------------------------
    # 4. Convert crop damage
    # --------------------------------------

    print("Converting crop damage...")

    (
        df["CROP_DAMAGE_USD"],
        crop_unparsed
    ) = convert_damage_column(
        df["DAMAGE_CROPS"]
    )

    print(
        "Unparsed property damage values:",
        property_unparsed.sum()
    )

    print(
        "Unparsed crop damage values:",
        crop_unparsed.sum()
    )

    # --------------------------------------
    # 5. Validate injury and death counts
    # --------------------------------------

    impact_columns = [
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT"
    ]

    print("\nValidating injury and death counts...")

    for col in impact_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        invalid_count = (
            (df[col] < 0).sum()
        )

        print(f"{col} negative values:", invalid_count)

        df.loc[df[col] < 0, col] = np.nan

    # --------------------------------------
    # 6. Missing value report
    # --------------------------------------

    print("\nGenerating missing value report...")

    missing_report = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percentage": (
            df.isna().mean() * 100
        )
    })

    missing_report = missing_report.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print(missing_report.head(20))

    # --------------------------------------
    # 7. Save processed dataset
    # --------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEANED_FILE,
        index=False
    )

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values_report.csv"
    )

    # --------------------------------------
    # 8. Final summary
    # --------------------------------------

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Original rows:", initial_rows)
    print("Final rows:", len(df))
    print("Removed rows:", initial_rows - len(df))

    print("Original columns: 51")
    print("Final columns:", len(df.columns))

    print("Cleaned dataset saved to:")
    print(CLEANED_FILE)

    return df


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    print("Loading NOAA dataset...")

    df = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    print("Original shape:", df.shape)

    cleaned_df = preprocess_data(df)

    print("\nFinal shape:", cleaned_df.shape)

    print("\nPreprocessing completed successfully.")
