import yfinance as yf
ticker = "ADBE"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")
data.head()
data.to_csv(f"{ticker}_data.csv")