import os
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

input_folder = "../../data/features/"
output_folder = "../../outputs/quantiles/"

os.makedirs(output_folder, exist_ok=True)

q10_model_alpha = 0.1
q50_model_alpha = 0.5
q90_model_alpha = 0.9

def load_data():
    datasets = {}

    for filename in os.listdir(input_folder):
        if filename.endswith("_data_features.csv"):
            path = os.path.join(input_folder, filename)
            df = pd.read_csv(path)

            df = df.sort_values("date").reset_index(drop=True)

            train = df[df["date"] < "2025-01-01"]
            test = df[df["date"] >= "2025-01-01"]

            feature_cols = [
                "rolling_mean_5",
                "rolling_vol_5",
                "rolling_mean_10",
                "rolling_vol_10",
                "ma_5",
                "ma_10"
            ]

            datasets[filename.replace("_data_features.csv", "")] = {
                "X_train": train[feature_cols],
                "y_train": train["target"],
                "X_test": test[feature_cols],
                "y_test": test["target"],
                "dates_test": test["date"],
                "rolling": test["rolling_mean_5"]
            }

    return datasets

def main():
    data = load_data()

    for name, d in data.items():
        X_train = d["X_train"]
        y_train = d["y_train"]
        X_test = d["X_test"]
        y_test = d["y_test"]
        dates = d["dates_test"]

        q10_model = GradientBoostingRegressor(loss="quantile", alpha=q10_model_alpha)
        q50_model = GradientBoostingRegressor(loss="quantile", alpha=q50_model_alpha)
        q90_model = GradientBoostingRegressor(loss="quantile", alpha=q90_model_alpha)

        q10_model.fit(X_train, y_train)
        q50_model.fit(X_train, y_train)
        q90_model.fit(X_train, y_train)

        q10_pred = q10_model.predict(X_test)
        q50_pred = q50_model.predict(X_test)
        q90_pred = q90_model.predict(X_test)

        output = pd.DataFrame({
            "date": dates,
            "actual": y_test,
            "q10": q10_pred,
            "q50": q50_pred,
            "q90": q90_pred
        })
        
        output_path = os.path.join(output_folder, f"{name}_quantiles.csv")
        output.to_csv(output_path, index=False)

        print(f"{name}: done")
if __name__ == "__main__":
    main()