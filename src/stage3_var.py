"""
Stage 3: Value-at-Risk (Historical Simulation)
-------------------------------------------------
Goal: estimate the maximum expected loss at 95% and 99%
confidence levels using historical portfolio returns.
"""

import yfinance as yf
import numpy as np
import pandas as pd

# -----------------------------
# 1. Define portfolio + weights
# -----------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM"]
weights = np.array([0.25, 0.25, 0.25, 0.25])

start_date = "2022-01-01"
end_date = "2024-12-31"

# -----------------------------
# 2. Download data + portfolio returns
# -----------------------------
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=False)
prices = raw_data["Close"].ffill().dropna()
daily_returns = prices.pct_change().dropna()
portfolio_returns = daily_returns.dot(weights)

# -----------------------------
# 3. Historical VaR calculation
# -----------------------------
# Historical simulation: sort actual past returns, find the percentile cutoff.
# 95% VaR = the 5th percentile of the return distribution.
# 99% VaR = the 1st percentile of the return distribution.
# We take the negative because VaR is reported as a positive "loss" number.
var_95 = -np.percentile(portfolio_returns, 5)
var_99 = -np.percentile(portfolio_returns, 1)

print(f"1-Day 95% VaR: {var_95:.4f} ({var_95:.2%})")
print(f"1-Day 99% VaR: {var_99:.4f} ({var_99:.2%})")

# -----------------------------
# 4. Translate to dollar terms
# -----------------------------
portfolio_value = 100_000  # assume a $100k portfolio
var_95_dollars = portfolio_value * var_95
var_99_dollars = portfolio_value * var_99

print(f"\nOn a ${portfolio_value:,} portfolio:")
print(f"95% VaR: ${var_95_dollars:,.2f} (expect to lose more than this only 5% of days)")
print(f"99% VaR: ${var_99_dollars:,.2f} (expect to lose more than this only 1% of days)")

# -----------------------------
# 5. Backtest: how many days actually exceeded VaR?
# -----------------------------
# A well-calibrated 95% VaR should be breached ~5% of the time historically.
breaches_95 = (portfolio_returns < -var_95).sum()
total_days = len(portfolio_returns)
breach_rate_95 = breaches_95 / total_days

print(f"\nBacktest: 95% VaR breached on {breaches_95} of {total_days} days "
      f"({breach_rate_95:.2%} — should be close to 5%)")

# -----------------------------
# 6. Worst single day (for context / drawdown intuition)
# -----------------------------
worst_day = portfolio_returns.min()
worst_date = portfolio_returns.idxmin()
print(f"\nWorst single day: {worst_date.date()} ({worst_day:.2%})")

# -----------------------------
# 7. Drawdown (bonus — peak-to-trough decline)
# -----------------------------
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
max_drawdown_date = drawdown.idxmin()

print(f"\nMax drawdown: {max_drawdown:.2%} (on {max_drawdown_date.date()})")
