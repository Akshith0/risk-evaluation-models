import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned")

os.makedirs(PROCESSED_PATH, exist_ok=True)

# --------------------------------------------------
# Cleaning Function
# --------------------------------------------------

def load_and_clean(file_path):

    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # -------------------------
    # Detect date column
    # -------------------------

    possible_date_cols = [
        "date",
        "datetime",
        "timestamp"
    ]

    date_col = next(
        (c for c in possible_date_cols if c in df.columns),
        None
    )

    if date_col is None:
        raise ValueError(f"No date column found in {file_path}")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    if date_col != "date":
        df.drop(columns=[date_col], inplace=True)

    # Remove invalid dates
    df.dropna(subset=["date"], inplace=True)

    # Sort chronologically
    df.sort_values("date", inplace=True)

    # Remove duplicate trading days
    df.drop_duplicates(subset="date", inplace=True)

    # -------------------------
    # Check required columns
    # -------------------------

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns {missing} in {file_path}"
        )

    # Keep only required columns
    df = df[["date"] + required]

    # -------------------------
    # Convert numerics
    # -------------------------

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill missing values
    df = df.ffill().dropna()

    # Remove impossible values
    for col in ["open", "high", "low", "close"]:
        df = df[df[col] > 0]

    df = df[df["volume"] >= 0]

    # Final sort
    df = df.sort_values("date").reset_index(drop=True)

    # Final sanity check
    assert df["date"].is_unique

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    files = [
        f for f in os.listdir(RAW_PATH)
        if f.endswith(".csv")
    ]

    dataframes = {}

    print("\n========== CLEANING DATA ==========\n")

    for file in files:

        path = os.path.join(RAW_PATH, file)
        name = file.replace(".csv", "")

        df = load_and_clean(path)

        print(
            f"{name}: {len(df)} rows "
            f"| {df['date'].min().date()} -> {df['date'].max().date()}"
        )

        dataframes[name] = df

    # ---------------------------------
    # Find overlapping date range
    # ---------------------------------

    start_date = max(
        df["date"].min()
        for df in dataframes.values()
    )

    end_date = min(
        df["date"].max()
        for df in dataframes.values()
    )

    print("\nCommon Date Range:")
    print(start_date.date(), "->", end_date.date())

    # ---------------------------------
    # Save cleaned datasets
    # ---------------------------------

    print("\nSaving cleaned files...\n")

    for name, df in dataframes.items():
        name = name.replace("_data", "")
        df = df[
            (df["date"] >= start_date)
            & (df["date"] <= end_date)
        ].reset_index(drop=True)

        output = os.path.join(
            PROCESSED_PATH,
            f"{name}_clean.csv"
        )

        df.to_csv(output, index=False)

        print(
            f"Saved {name}_clean.csv "
            f"({len(df)} rows)"
        )

    print("\nCleaning complete.\n")


if __name__ == "__main__":
    main()