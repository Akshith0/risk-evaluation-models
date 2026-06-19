import yfinance as yf
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw")
ticker = "ticker_name"
data = yf.download(ticker, start="2022-05-30", end="2026-01-01")
data.head()
new_filename = f"{ticker}_data.csv"
output_path = os.path.join(OUTPUT_PATH, new_filename)
data.to_csv(output_path, index=True)