import os
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "final",
    "final_dataset.csv"
)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

train = df[(df["date"] >= "2022-06-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"]

features = [
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "MA50",
    "MA200",
    "BullBear",
    "TrendStrength",
    "garch_forecast"
]

X_train = train[features]
y_train = train["target_volatility"]

X_test = test[features]
y_true = test["target_volatility"]

results = test[["ticker", "date"]].copy()
results["actual"] = y_true.values

quantiles = [0.1, 0.5, 0.9]

for q in quantiles:

    model = QuantileRegressor(
        quantile=q,
        alpha=0.001,
        solver="highs"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Save predictions
    results[f"q{int(q * 100)}"] = preds

y_pred = results["q50"]

mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)

print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# ---------- Save Results ----------
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "quantile_regression_full_predictions.csv"
    ),
    index=False
)

print("Full multi-quantile regression predictions saved!")