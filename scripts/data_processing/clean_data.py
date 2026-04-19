import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_and_clean(file_path):
    df = pd.read_csv(file_path)

    # standardize columns
    df.columns = [col.strip().lower() for col in df.columns]

    # find date column
    possible_date_cols = ["date", "datetime", "timestamp"]
    date_col = next((col for col in possible_date_cols if col in df.columns), None)

    if date_col is None:
        raise ValueError(f"No date column in {file_path}")

    # convert to datetime
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    if date_col != "date":
        df = df.drop(columns=[date_col])

    # drop invalid dates
    df = df.dropna(subset=["date"])

    # sort and reset index
    df = df.sort_values("date").reset_index(drop=True)

    # remove duplicates
    df = df.drop_duplicates(subset="date")

    # check required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")

    df = df[["date"] + required_cols]

    return df


def main():
    files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]

    dataframes = {}

    for file in files:
        path = os.path.join(RAW_PATH, file)
        name = file.replace(".csv", "")

        df = load_and_clean(path)

        print(f"{name}: {len(df)} rows")

        dataframes[name] = df

    # find common range
    min_date = max(df["date"].min() for df in dataframes.values())
    max_date = min(df["date"].max() for df in dataframes.values())

    # trim and save
    for name, df in dataframes.items():
        df = df[(df["date"] >= min_date) & (df["date"] <= max_date)].reset_index(drop=True)

        output_path = os.path.join(PROCESSED_PATH, f"{name}_clean.csv")
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()