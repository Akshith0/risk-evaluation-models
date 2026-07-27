import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Style ----------
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ---------- Paths ----------
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


# ---------- MAE Bar Chart ----------
fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    comparison["Model"],
    comparison["MAE"],
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.8
)

ax.set_xlabel("Model")
ax.set_ylabel("Mean Absolute Error")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "mae_comparison.png"))
plt.savefig(os.path.join(OUTPUT_DIR, "mae_comparison.pdf"))
plt.close()


# ---------- RMSE Bar Chart ----------
fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(
    comparison["Model"],
    comparison["RMSE"],
    color="#55A868",
    edgecolor="black",
    linewidth=0.8
)

ax.set_xlabel("Model")
ax.set_ylabel("Root Mean Squared Error")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "rmse_comparison.png"))
plt.savefig(os.path.join(OUTPUT_DIR, "rmse_comparison.pdf"))
plt.close()


# ---------- Best Model ----------
best_model = comparison.loc[
    comparison["MAE"].idxmin(),
    "Model"
]

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "predictions",
    f"{best_model}_predictions.csv"
)

pred = pd.read_csv(PREDICTION_PATH)
pred["date"] = pd.to_datetime(pred["date"])


# ---------- Scatter Plot ----------
fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax.scatter(
    pred["actual"],
    pred["predicted"],
    color="#4C72B0",
    alpha=0.6,
    s=18
)

lims = [
    min(pred["actual"].min(), pred["predicted"].min()),
    max(pred["actual"].max(), pred["predicted"].max())
]

ax.plot(lims, lims, "r--", linewidth=1.5)

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("Actual Annualized Volatility")
ax.set_ylabel("Predicted Annualized Volatility")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.pdf"))
plt.close()


# ---------- Time Series ----------
stocks = ["NVDA", "PLTR", "WMT", "SPX"]

for ticker in stocks:

    stock_df = pred[pred["ticker"] == ticker]

    if stock_df.empty:
        continue

    stock_df = stock_df.sort_values("date")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        stock_df["date"],
        stock_df["actual"],
        color="black",
        linewidth=2,
        label="Actual"
    )

    ax.plot(
        stock_df["date"],
        stock_df["predicted"],
        color="#4C72B0",
        linestyle="--",
        linewidth=2,
        label="Predicted"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")

    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker.lower()}_time_series.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker.lower()}_time_series.pdf"))

    plt.close()


print("Figures saved!")