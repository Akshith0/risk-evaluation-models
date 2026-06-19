import os
import pandas as pd

input_folder = "../../data/features/"
output_folder = "../../outputs/predictions/"

os.makedirs(output_folder, exist_ok=True)


def run_baselines():
    for filename in os.listdir(input_folder):
        if filename.endswith("_features.csv"):
            path = os.path.join(input_folder, filename)
            df = pd.read_csv(path)

            # sort
            df = df.sort_values("date").reset_index(drop=True)

            # split
            train = df[df["date"] < "2025-01-01"]
            test = df[df["date"] >= "2025-01-01"]

            # actual values
            actual = test["target"]

            # baseline 1: zero
            zero_pred = [0] * len(test)

            # baseline 2: rolling mean
            rolling_pred = test["rolling_mean_5"]

            # create output df
            output = pd.DataFrame({
                "date": test["date"],
                "actual": actual,
                "zero": zero_pred,
                "rolling": rolling_pred
            })

            # save
            name = filename.replace("_features.csv", "")
            output_path = os.path.join(output_folder, f"{name}_baseline.csv")
            output.to_csv(output_path, index=False)

            print(f"{name}: done")


if __name__ == "__main__":
    run_baselines()