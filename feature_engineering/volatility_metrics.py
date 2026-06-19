import pandas as pd
import numpy as np

WINDOWS = [5, 10, 20,60]
def add_volatility_metrics(df):
    df=df.copy()
    annualization = np.sqrt(252)
    for window in WINDOWS:
        df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std() * annualization
    return df