import os
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_pinball_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "predictions"
)

results = []

for file in os.listdir(DATA_PATH):
    if file.endswith("_predictions.csv"):

        df = pd.read_csv(os.path.join(DATA_PATH, file))

        mae = mean_absolute_error(df["actual"], df["predicted"])
        rmse = root_mean_squared_error(df["actual"], df["predicted"])

        row = {
            "Model": file.replace("_predictions.csv", ""),
            "MAE": mae,
            "RMSE": rmse,
            "Pinball Loss": None
        }

        if file in [
            "quantile_regression_predictions.csv",
            "quantile_regression_full_predictions.csv"
        ]:
            row["Pinball Loss"] = mean_pinball_loss(
                df["actual"],
                df["predicted"],
                alpha=0.5
            )

        results.append(row)

comparison = pd.DataFrame(results)

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print(comparison)
print("Model comparison saved!")