
import pandas as pd
from pathlib import Path

# Project data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def inspect_datasets():

    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    datasets = {}

    print("=" * 60)
    print("NOAA STORM EVENTS DATASET INSPECTION")
    print("=" * 60)

    for file in csv_files:

        print(f"\nDataset: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        datasets[file.stem] = df

        print("-" * 40)
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nFirst 5 records:")
        print(df.head())

        print("\nMissing values:")
        print(df.isnull().sum().sum())

    return datasets


if __name__ == "__main__":
    inspect_datasets()
