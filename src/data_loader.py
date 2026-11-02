
import pandas as pd
from pathlib import Path


# Dataset directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DETAILS_FILE = DATA_DIR / "StormEvents_details_all.csv"


def load_details_data():

    if not DETAILS_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DETAILS_FILE}"
        )

    print("Loading NOAA Storm Events Details...")

    df = pd.read_csv(
        DETAILS_FILE,
        low_memory=False
    )

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


def inspect_details_data(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))

    print("\n========== YEAR COVERAGE ==========")

    print(df["YEAR"].describe())

    print("\n========== EVENT TYPE DISTRIBUTION ==========")

    print(df["EVENT_TYPE"].value_counts().head(15))

    print("\n========== DUPLICATE CHECK ==========")

    print("Duplicate EVENT_IDs:", df["EVENT_ID"].duplicated().sum())

    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print(missing.head(20))

    print("\n========== NUMERICAL SUMMARY ==========")

    print(
        df[[
            "MAGNITUDE",
            "INJURIES_DIRECT",
            "INJURIES_INDIRECT",
            "DEATHS_DIRECT",
            "DEATHS_INDIRECT"
        ]].describe()
    )


if __name__ == "__main__":

    df = load_details_data()

    inspect_details_data(df)
