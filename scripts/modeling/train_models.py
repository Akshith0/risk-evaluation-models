import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

input_folder = "../../data/features/"
output_folder = "../../outputs/predictions/"

os.makedirs(output_folder, exist_ok=True)


def load_data():
    datasets = {}

    for filename in os.listdir(input_folder):
        if filename.endswith("_features.csv"):
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

            datasets[filename.replace("_features.csv", "")] = {
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

        # baselines
        zero_pred = [0] * len(y_test)
        rolling_pred = d["rolling"]

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # Ridge Regression
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # save results
        output = pd.DataFrame({
            "date": dates,
            "actual": y_test,
            "zero": zero_pred,
            "rolling": rolling_pred,
            "linear": lr_pred,
            "ridge": ridge_pred,
            "rf": rf_pred
        })

        output_path = os.path.join(output_folder, f"{name}_predictions.csv")
        output.to_csv(output_path, index=False)

        print(f"{name}: done")


if __name__ == "__main__":
    main()