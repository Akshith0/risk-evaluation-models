import os
import pandas as pd
import matplotlib.pyplot as plt
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

        if file in [
            "quantile_regression_predictions.csv",
            "quantile_regression_full_predictions.csv"
        ]:

            df["predicted"] = df["q50"]

            mae = mean_absolute_error(
                df["actual"],
                df["predicted"]
            )

            rmse = root_mean_squared_error(
                df["actual"],
                df["predicted"]
            )

            pinball_q10 = mean_pinball_loss(
                df["actual"],
                df["q10"],
                alpha=0.1
            )

            pinball_q50 = mean_pinball_loss(
                df["actual"],
                df["q50"],
                alpha=0.5
            )

            pinball_q90 = mean_pinball_loss(
                df["actual"],
                df["q90"],
                alpha=0.9
            )

            avg_pinball = (
                pinball_q10 +
                pinball_q50 +
                pinball_q90
            ) / 3

            row = {
                "Model": file.replace("_predictions.csv", ""),
                "MAE": mae,
                "RMSE": rmse,
                "Pinball Loss": avg_pinball
            }

        else:

            mae = mean_absolute_error(
                df["actual"],
                df["predicted"]
            )

            rmse = root_mean_squared_error(
                df["actual"],
                df["predicted"]
            )

            row = {
                "Model": file.replace("_predictions.csv", ""),
                "MAE": mae,
                "RMSE": rmse,
                "Pinball Loss": None
            }

        results.append(row)

comparison = pd.DataFrame(results)

comparison_pdf = comparison.copy()

comparison_pdf["Model"] = comparison_pdf["Model"].replace({
    "historical_volatility": "Historical Volatility",
    "linear_regression": "Linear Regression",
    "garch": "GARCH(1,1)",
    "quantile_regression": "Quantile Regression",
    "quantile_regression_full": "Quantile Regression (Full)"
})

comparison_pdf["MAE"] = comparison_pdf["MAE"].map("{:.3f}".format)
comparison_pdf["RMSE"] = comparison_pdf["RMSE"].map("{:.3f}".format)
comparison_pdf["Pinball Loss"] = comparison_pdf["Pinball Loss"].map(
    lambda x: "-" if pd.isna(x) else f"{x:.3f}"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "tables"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(8, 2.6))
ax.axis("off")

table = ax.table(
    cellText=comparison_pdf.values,
    colLabels=comparison_pdf.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row == 0:
        cell.set_facecolor("#D9EAF7")
        cell.set_text_props(weight="bold")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.pdf"),
    bbox_inches="tight"
)

plt.close()

print(comparison_pdf)
print("Model comparison saved!")