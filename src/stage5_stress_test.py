"""
Stage 5: Stress Testing + Visualization + Reporting
-------------------------------------------------------
Goal: apply historical crisis scenarios to the current portfolio,
and generate a visual risk report.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# -----------------------------
# 1. Define portfolio + weights
# -----------------------------
tickers = ["AAPL", "MSFT", "JPM", "XOM"]
weights = np.array([0.25, 0.25, 0.25, 0.25])

start_date = "2022-01-01"
end_date = "2024-12-31"

# -----------------------------
# 2. Download data + portfolio returns (recent period, for normal-time charts)
# -----------------------------
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=False)
prices = raw_data["Close"].ffill().dropna()
daily_returns = prices.pct_change().dropna()
portfolio_returns = daily_returns.dot(weights)

# -----------------------------
# 3. Stress test: 2008 Financial Crisis
# -----------------------------
# Pull actual 2008 returns for the SAME stocks (where available), apply CURRENT weights.
# Note: this assumes current tickers existed in 2008 (true for AAPL, MSFT, JPM, XOM).
crisis_2008_start = "2008-09-01"
crisis_2008_end = "2009-03-31"

raw_2008 = yf.download(tickers, start=crisis_2008_start, end=crisis_2008_end, auto_adjust=True, threads=False)
prices_2008 = raw_2008["Close"].ffill().dropna()
returns_2008 = prices_2008.pct_change().dropna()
portfolio_returns_2008 = returns_2008.dot(weights)

cumulative_2008 = (1 + portfolio_returns_2008).cumprod() - 1
total_loss_2008 = cumulative_2008.iloc[-1]

print("=== STRESS TEST: 2008 Financial Crisis (Sep 2008 - Mar 2009) ===")
print(f"Total portfolio return: {total_loss_2008:.2%}")
print(f"Worst single day: {portfolio_returns_2008.min():.2%}")

# -----------------------------
# 4. Stress test: 2020 COVID Crash
# -----------------------------
crisis_2020_start = "2020-02-01"
crisis_2020_end = "2020-04-30"

raw_2020 = yf.download(tickers, start=crisis_2020_start, end=crisis_2020_end, auto_adjust=True, threads=False)
prices_2020 = raw_2020["Close"].ffill().dropna()
returns_2020 = prices_2020.pct_change().dropna()
portfolio_returns_2020 = returns_2020.dot(weights)

cumulative_2020 = (1 + portfolio_returns_2020).cumprod() - 1
total_loss_2020 = cumulative_2020.iloc[-1]

print("\n=== STRESS TEST: 2020 COVID Crash (Feb 2020 - Apr 2020) ===")
print(f"Total portfolio return: {total_loss_2020:.2%}")
print(f"Worst single day: {portfolio_returns_2020.min():.2%}")

# -----------------------------
# 5. Dollar impact on a $100k portfolio
# -----------------------------
portfolio_value = 100_000
print(f"\n=== DOLLAR IMPACT ON ${portfolio_value:,} PORTFOLIO ===")
print(f"2008 Crisis scenario: ${portfolio_value * total_loss_2008:,.2f}")
print(f"2020 COVID scenario: ${portfolio_value * total_loss_2020:,.2f}")

# -----------------------------
# 6. VISUALIZATION — build a 4-panel report
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Portfolio Risk Report", fontsize=16, fontweight="bold")

# Panel 1: Cumulative returns (2022-2024 normal period)
cumulative_normal = (1 + portfolio_returns).cumprod() - 1
axes[0, 0].plot(cumulative_normal.index, cumulative_normal.values, color="steelblue")
axes[0, 0].set_title("Cumulative Portfolio Returns (2022-2024)")
axes[0, 0].set_ylabel("Cumulative Return")
axes[0, 0].axhline(0, color="black", linewidth=0.8)

# Panel 2: Drawdown over time (normal period)
cumulative_value = (1 + portfolio_returns).cumprod()
running_max = cumulative_value.cummax()
drawdown = (cumulative_value - running_max) / running_max
axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.6)
axes[0, 1].set_title("Portfolio Drawdown (2022-2024)")
axes[0, 1].set_ylabel("Drawdown")

# Panel 3: Return distribution with VaR marked
var_95 = -np.percentile(portfolio_returns, 5)
axes[1, 0].hist(portfolio_returns, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
axes[1, 0].axvline(-var_95, color="red", linestyle="--", linewidth=2, label=f"95% VaR ({-var_95:.2%})")
axes[1, 0].set_title("Daily Return Distribution")
axes[1, 0].set_xlabel("Daily Return")
axes[1, 0].legend()

# Panel 4: Stress test comparison bar chart
scenarios = ["2008 Crisis", "2020 COVID Crash"]
scenario_returns = [total_loss_2008, total_loss_2020]
colors = ["darkred" if r < 0 else "darkgreen" for r in scenario_returns]
axes[1, 1].bar(scenarios, scenario_returns, color=colors)
axes[1, 1].set_title("Historical Stress Test: Total Return")
axes[1, 1].set_ylabel("Total Return")
axes[1, 1].axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("portfolio_risk_report.png", dpi=150)
print("\nChart saved as portfolio_risk_report.png")
