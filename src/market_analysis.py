import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKERS = ["AAPL", "^GSPC"]
START_DATE = "2021-01-01"

print("Downloading historical market data...")

raw = yf.download(
    TICKERS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
)

if raw.empty:
    raise RuntimeError(
        "No market data downloaded. Check your internet connection and try again."
    )

prices = raw["Close"].copy()
prices = prices.dropna(how="all").ffill().dropna()

if not all(ticker in prices.columns for ticker in TICKERS):
    raise RuntimeError("One or more requested tickers are missing.")

prices = prices[TICKERS]
returns = prices.pct_change().dropna()

if returns.empty:
    raise RuntimeError("Not enough data to calculate returns.")

trading_days = 252

annual_returns = (1 + returns).prod() ** (
    trading_days / len(returns)
) - 1

annual_volatility = returns.std() * np.sqrt(trading_days)

wealth = (1 + returns).cumprod()
drawdown = wealth / wealth.cummax() - 1
max_drawdown = drawdown.min()

summary = pd.DataFrame({
    "Annualised Return": annual_returns,
    "Annualised Volatility": annual_volatility,
    "Maximum Drawdown": max_drawdown,
})

print("\nQUANTITATIVE RESEARCH REPORT")
print("=" * 60)
print(f"Data start: {prices.index.min().date()}")
print(f"Data end:   {prices.index.max().date()}")
print(f"Daily observations: {len(returns)}")
print("\nPerformance and risk metrics:")
print(summary.to_string(float_format=lambda x: f"{x:.2%}"))

prices.to_csv("results/market_prices.csv")
returns.to_csv("results/daily_returns.csv")
summary.to_csv("results/risk_summary.csv")

fig, axes = plt.subplots(3, 1, figsize=(11, 12))

normalised = prices / prices.iloc[0] * 100
normalised.plot(ax=axes[0], title="Growth of 100: AAPL vs S&P 500")
axes[0].set_ylabel("Indexed value")

annual_vol = returns.rolling(20).std() * np.sqrt(trading_days)
annual_vol.plot(ax=axes[1], title="20-Day Rolling Annualised Volatility")
axes[1].set_ylabel("Volatility")

drawdown.plot(ax=axes[2], title="Drawdown")
axes[2].set_ylabel("Drawdown")

fig.tight_layout()
fig.savefig("results/market_analysis.png", dpi=150)
plt.show()

print("\nSaved analysis files in the results/ directory.")