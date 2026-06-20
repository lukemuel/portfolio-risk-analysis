"""
Stage 1: Data Ingestion + Returns
-----------------------------------
Goal: pull historical prices for a small set of stocks,
compute daily returns and cumulative returns.
"""

import yfinance as yf
import pandas as pd

# -----------------------------
# 1. Define the portfolio universe
# -----------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM"]

start_date = "2022-01-01"
end_date = "2024-12-31"

# -----------------------------
# 2. Download price data
# -----------------------------
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
prices = raw_data["Close"]

# -----------------------------
# 3. Basic validation
# -----------------------------
missing_counts = prices.isna().sum()
print("Missing values per ticker:")
print(missing_counts)

prices = prices.ffill()
prices = prices.dropna()

print(f"\nFinal date range: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Total trading days: {len(prices)}")

# -----------------------------
# 4. Calculate daily returns
# -----------------------------
daily_returns = prices.pct_change().dropna()

print("\nDaily returns sample:")
print(daily_returns.head())

# -----------------------------
# 5. Calculate cumulative returns
# -----------------------------
cumulative_returns = (1 + daily_returns).cumprod() - 1

print("\nCumulative returns (final values):")
print(cumulative_returns.iloc[-1])

# -----------------------------
# 6. Sanity check output
# -----------------------------
print("\nDaily return summary stats:")
print(daily_returns.describe())
