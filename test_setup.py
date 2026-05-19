import pandas as pd
import yfinance as yf

ticker = yf.Ticker("AAPL")
data = ticker.history(period="5d")

print("AAPL - Last 5 Days")
print("-" * 50)
print(data)
