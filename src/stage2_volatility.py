"""
Stage 2: Portfolio Construction + Volatility
-----------------------------------------------
Goal: combine individual stocks into a weighted portfolio,
then measure portfolio-level risk (volatility).
"""

import yfinance as yf
import numpy as np
import pandas as pd

# -----------------------------
# 1. Define portfolio + weights
# -----------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM"]
weights = np.array([0.25, 0.25, 0.25, 0.25])  # equal-weighted for now

start_date = "2022-01-01"
end_date = "2024-12-31"

# -----------------------------
# 2. Download data + calculate daily returns
# -----------------------------
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=False)
prices = raw_data["Close"].ffill().dropna()
daily_returns = prices.pct_change().dropna()

# -----------------------------
# 3. Calculate portfolio daily returns
# -----------------------------
# Weighted sum of each stock's daily return, row by row.
# This gives us ONE return series representing the whole portfolio.
portfolio_returns = daily_returns.dot(weights)

print("Portfolio daily returns sample:")
print(portfolio_returns.head())

# -----------------------------
# 4. Individual stock volatility (annualized)
# -----------------------------
individual_vol = daily_returns.std() * np.sqrt(252)
print("\nIndividual stock annualized volatility:")
print(individual_vol)

# Weighted average volatility — what you'd get if there were NO diversification benefit
weighted_avg_vol = (individual_vol * weights).sum()
print(f"\nWeighted average volatility (no diversification benefit): {weighted_avg_vol:.4f}")

# -----------------------------
# 5. Actual portfolio volatility (annualized)
# -----------------------------
# This accounts for correlation between assets via the covariance matrix.
# Formula: portfolio variance = w^T * Cov * w
cov_matrix = daily_returns.cov() * 252  # annualize covariance
portfolio_variance = weights.T @ cov_matrix.values @ weights
portfolio_volatility = np.sqrt(portfolio_variance)

print(f"Actual portfolio volatility (with diversification): {portfolio_volatility:.4f}")

# -----------------------------
# 6. Show the diversification benefit
# -----------------------------
diversification_benefit = weighted_avg_vol - portfolio_volatility
print(f"\nDiversification benefit: {diversification_benefit:.4f} "
      f"({diversification_benefit / weighted_avg_vol:.1%} reduction in risk)")

# -----------------------------
# 7. Correlation matrix (helps explain WHY diversification works)
# -----------------------------
print("\nCorrelation matrix between assets:")
print(daily_returns.corr())
