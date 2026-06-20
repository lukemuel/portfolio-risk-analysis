"""
Stage 4: CAPM Beta Estimation vs. S&P 500
---------------------------------------------
Goal: estimate portfolio beta via linear regression against
the S&P 500, and compute CAPM expected return.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------
# 1. Define portfolio + weights
# -----------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM"]
weights = np.array([0.25, 0.25, 0.25, 0.25])

start_date = "2022-01-01"
end_date = "2024-12-31"

# -----------------------------
# 2. Download portfolio data + market data (S&P 500)
# -----------------------------
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=False)
prices = raw_data["Close"].ffill().dropna()
daily_returns = prices.pct_change().dropna()
portfolio_returns = daily_returns.dot(weights)

market_raw = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True, threads=False)
market_prices = market_raw["Close"].squeeze()
market_returns = market_prices.pct_change().dropna()

# -----------------------------
# 3. Align dates between portfolio and market
# -----------------------------
# Both series need the exact same trading days to compare apples-to-apples.
aligned = pd.concat([portfolio_returns, market_returns], axis=1, join="inner")
aligned.columns = ["portfolio", "market"]

# -----------------------------
# 4. Run linear regression: portfolio returns vs. market returns
# -----------------------------
# Beta = slope of this regression line.
# Alpha = intercept (excess return not explained by the market).
slope, intercept, r_value, p_value, std_err = stats.linregress(
    aligned["market"], aligned["portfolio"]
)

beta = slope
alpha = intercept
r_squared = r_value ** 2

print(f"Beta: {beta:.4f}")
print(f"Alpha (daily): {alpha:.6f}")
print(f"R-squared: {r_squared:.4f} (% of portfolio movement explained by the market)")
print(f"P-value: {p_value:.6f} (statistical significance of beta)")

# -----------------------------
# 5. Interpret beta in plain English
# -----------------------------
if beta > 1:
    interpretation = "more volatile than the market"
elif beta < 1:
    interpretation = "less volatile than the market"
else:
    interpretation = "moves in line with the market"

print(f"\nInterpretation: This portfolio is {interpretation} "
      f"(beta of {beta:.2f} means a 1% market move tends to produce "
      f"a {beta:.2f}% portfolio move).")

# -----------------------------
# 6. CAPM expected return
# -----------------------------
# CAPM formula: E(R) = Rf + Beta * (Market Return - Rf)
risk_free_rate = 0.045  # approx current 3-month T-bill rate, annualized
market_annual_return = market_returns.mean() * 252  # annualize daily market return

capm_expected_return = risk_free_rate + beta * (market_annual_return - risk_free_rate)

print(f"\nMarket annualized return (period): {market_annual_return:.2%}")
print(f"Risk-free rate assumption: {risk_free_rate:.2%}")
print(f"CAPM expected annual return: {capm_expected_return:.2%}")
