import os
import pandas as pd

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

        # Read stock data
        df = pd.read_csv(os.path.join(INPUT_PATH, file))

        # Add ticker column
        ticker = file.replace("_returns.csv", "")
        df["ticker"] = ticker

        # Add all engineered features
        df = add_volatility_metrics(df)
        df = create_market_regimes(df)
        df = add_garch_features(df)
        df = add_target_variable(df)

        dfs.append(df)

# Make sure files were found
if not dfs:
    raise ValueError(f"No *_returns.csv files found in {INPUT_PATH}")

# Combine all stocks
final_df = pd.concat(dfs, ignore_index=True)

# Sort by stock then date
final_df = final_df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

# Remove rows with missing values
final_df = final_df.dropna().reset_index(drop=True)

# Move ticker and date to the front
cols = ["ticker", "date"] + [
    col for col in final_df.columns
    if col not in ["ticker", "date"]
]

final_df = final_df[cols]

# Save final dataset
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Final dataset saved to: {OUTPUT_PATH}")