import numpy as np

def add_target_variable(df, horizon=10):
    df = df.copy()

    targets = [np.nan] * len(df)

    for i in range(len(df) - horizon):
        future_returns = df["log_return"].iloc[i+1:i+horizon+1]
        targets[i] = future_returns.std() * np.sqrt(252)

    df["target_volatility"] = targets

    return df