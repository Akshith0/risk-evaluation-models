import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "returns")

os.makedirs(OUTPUT_PATH, exist_ok=True)

def calculate_log_returns(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith("_clean.csv"):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # ensure numeric
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

            # calculate log returns
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

            # drop NaNs created by shift
            df = df.dropna().reset_index(drop=True)

            # save
            new_filename = filename.replace("_clean.csv", "_returns.csv")
            output_path = os.path.join(OUTPUT_PATH, new_filename)

            df.to_csv(output_path, index=False)
            print(f"Saved {new_filename}")

if __name__ == "__main__":
    calculate_log_returns(INPUT_PATH)