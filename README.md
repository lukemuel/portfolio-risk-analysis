# Portfolio Risk Analysis

A staged build of a quantitative portfolio risk management system in Python, demonstrating return modeling, volatility, Value-at-Risk (VaR), CAPM beta estimation, and historical stress testing.

## Project Goal

This project is built incrementally to demonstrate financial risk analysis fundamentals using real market data:

- **Stage 1 (complete):** Data ingestion + return calculations
- **Stage 2:** Portfolio construction + volatility
- **Stage 3:** Value-at-Risk (historical simulation)
- **Stage 4:** CAPM beta estimation vs. S&P 500
- **Stage 5:** Historical stress testing + visualization

## Stage 1: Data Ingestion + Returns

Pulls historical adjusted close prices for a small portfolio of stocks (AAPL, MSFT, JPM, XOM) using `yfinance`, validates the data for gaps, and calculates daily and cumulative returns.

### What it does
- Downloads 3 years of daily price data (2022–2024)
- Validates and handles missing data
- Calculates daily percentage returns
- Calculates cumulative returns over the period
- Outputs summary statistics for sanity-checking the data

### Sample output
## Tech Stack

- **pandas / numpy** — data wrangling and numerical calculations
- **yfinance** — historical market data
- **scipy** — statistical functions (used in later stages for VaR and regression)
- **matplotlib / seaborn** — visualization (used in later stages)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/stage1_returns.py
```

## Status

🚧 In progress — Stage 1 complete, Stage 2 (portfolio volatility) in development.
