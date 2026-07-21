import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

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
    "volatility_60"
]

X_train = train[features]
y_train = train["target_volatility"]

X_test = test[features]
y_true = test["target_volatility"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²  : {r2:.6f}")

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.6f}")

print(f"Intercept: {model.intercept_:.6f}")

results = test[["ticker", "date"]].copy()
results["actual"] = y_true
results["predicted"] = y_pred

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results.to_csv(
    os.path.join(OUTPUT_DIR, "linear_regression_predictions.csv"),
    index=False
)

print("Linear regression predictions saved!")