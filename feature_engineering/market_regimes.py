import numpy as np
import pandas as pd

def create_market_regimes(df):
    # Calculate moving averages
    df["MA50"] = df['close'].rolling(50).mean()
    df["MA200"] = df['close'].rolling(200).mean()

    # Create Bull/Bear regime based on moving averages
    df["BullBear"] = (df["MA50"] > df["MA200"]).astype(int)

    # Calculate volatility (using rolling standard deviation)
    df["Volatility"] = df['close'].rolling(20).std()

    # Create High/Low Volatility regime based on volatility threshold
    volatility_threshold = df["Volatility"].median()
    df["HighLowVolatility"] = (df["Volatility"] > volatility_threshold).astype(int)

     # Trend strength
    df["TrendStrength"] = (
        (df["close"] - df["MA50"]) / df["MA50"]
    )

    return df



