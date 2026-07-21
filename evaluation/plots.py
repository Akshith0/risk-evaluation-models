import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tables",
    "model_comparison.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

comparison = pd.read_csv(TABLE_PATH)

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["MAE"])
plt.ylabel("MAE")
plt.title("Mean Absolute Error by Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mae_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["RMSE"])
plt.ylabel("RMSE")
plt.title("Root Mean Squared Error by Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_comparison.png"))
plt.close()

best_model = comparison.loc[comparison["MAE"].idxmin(), "Model"]

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "predictions",
    f"{best_model}_predictions.csv"
)

pred = pd.read_csv(PREDICTION_PATH)
pred["date"] = pd.to_datetime(pred["date"])

plt.figure(figsize=(6, 6))
plt.scatter(pred["actual"], pred["predicted"], alpha=0.5)
plt.xlabel("Actual Volatility")
plt.ylabel("Predicted Volatility")
plt.title(f"Actual vs Predicted Volatility ({best_model})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.close()

stocks = ["NVDA", "PLTR", "WMT", "SPX"]

for ticker in stocks:
    stock_df = pred[pred["ticker"] == ticker]

    if stock_df.empty:
        continue

    stock_df = stock_df.sort_values("date")

    plt.figure(figsize=(12, 5))
    plt.plot(stock_df["date"], stock_df["actual"], label="Actual")
    plt.plot(stock_df["date"], stock_df["predicted"], label="Predicted")
    plt.title(f"{ticker}: Actual vs Predicted Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_time_series.png"))
    plt.close()

print("Figures saved!")