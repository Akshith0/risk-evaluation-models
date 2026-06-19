import pandas as pd
import os

from volatility_metrics import add_volatility_metrics
from market_regimes import create_market_regimes
from garch_features import add_garch_features
from target_variable import add_target_variable


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "returns")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "final", "final_dataset.csv")

dfs = []

for file in os.listdir(INPUT_PATH):
    if file.endswith("_returns.csv"):

        df = pd.read_csv(os.path.join(INPUT_PATH, file))

        # Add stock name
        ticker = file.replace("_returns.csv", "")
        df["ticker"] = ticker

        # Add features
        df = add_volatility_metrics(df)
        df = create_market_regimes(df)
        df = add_garch_features(df)
        df = add_target_variable(df)

        dfs.append(df)


if not dfs:
    raise ValueError(f"No *_returns.csv files found in {INPUT_PATH}")
# Combine all stocks
final_df = pd.concat(dfs, ignore_index=True)

# Remove rows with NaNs from rolling windows
final_df = final_df.dropna().reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Final dataset saved!")

