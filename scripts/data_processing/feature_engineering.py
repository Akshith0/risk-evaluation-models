import pandas as pd
import os

input_folder = "../../data/processed/"
output_folder = "../../data/features/"

os.makedirs(output_folder, exist_ok=True)


def create_features(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith("_processed.csv"):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # ensure sorted
            df = df.sort_values("date").reset_index(drop=True)

            if "Log_Return" not in df.columns:
                print(f"Log_Return not found in {filename}, skipping")
                continue

            # convert types
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["Log_Return"] = pd.to_numeric(df["Log_Return"], errors="coerce")

            # rolling features
            df["rolling_mean_5"] = df["Log_Return"].rolling(5).mean().shift(1)
            df["rolling_vol_5"] = df["Log_Return"].rolling(5).std().shift(1)

            df["rolling_mean_10"] = df["Log_Return"].rolling(10).mean().shift(1)
            df["rolling_vol_10"] = df["Log_Return"].rolling(10).std().shift(1)

            df["ma_5"] = df["close"].rolling(5).mean().shift(1)
            df["ma_10"] = df["close"].rolling(10).mean().shift(1)

            # target
            df["target"] = df["Log_Return"].shift(-1)

            # drop NaNs
            df = df.dropna()

            # save
            name = filename.replace("_clean_processed.csv", "")
            output_path = os.path.join(output_folder, f"{name}_features.csv")
            df.to_csv(output_path, index=False)

            print(f"{name}: done")


if __name__ == "__main__":
    create_features(input_folder, output_folder)