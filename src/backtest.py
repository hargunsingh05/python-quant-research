import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "AAPL"
START_DATE = "2021-01-01"
SHORT_WINDOW = 20
LONG_WINDOW = 50
TRANSACTION_COST = 0.001
TRADING_DAYS = 252

print(f"Downloading {TICKER} data...")

raw = yf.download(
    TICKER,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
)

if raw.empty:
    raise RuntimeError("No market data downloaded.")

prices = raw["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]

prices = prices.dropna().rename("Price")

data = pd.DataFrame({"Price": prices})
data["Return"] = data["Price"].pct_change()

data["SMA_Short"] = data["Price"].rolling(SHORT_WINDOW).mean()
data["SMA_Long"] = data["Price"].rolling(LONG_WINDOW).mean()

# Signal uses today's closing price.
# Position is shifted so today's signal only affects tomorrow's return.
data["Signal"] = (
    data["SMA_Short"] > data["SMA_Long"]
).astype(float)

data["Position"] = data["Signal"].shift(1).fillna(0.0)

# Remove warm-up period before evaluating either approach.
data = data.dropna(subset=["SMA_Long", "Return"]).copy()

data["Turnover"] = data["Position"].diff().abs()
data["Turnover"] = data["Turnover"].fillna(data["Position"].abs())

data["Strategy_Return"] = (
    data["Position"] * data["Return"]
    - TRANSACTION_COST * data["Turnover"]
)

data["Buy_Hold_Return"] = data["Return"]

data["Strategy_Equity"] = (1 + data["Strategy_Return"]).cumprod()
data["Buy_Hold_Equity"] = (1 + data["Buy_Hold_Return"]).cumprod()

def calculate_metrics(returns, equity):
    returns = returns.dropna()

    annual_return = equity.iloc[-1] ** (
        TRADING_DAYS / len(returns)
    ) - 1

    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    daily_risk_free_rate = 0.0
    excess_returns = returns - daily_risk_free_rate

    sharpe = (
        excess_returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "Cumulative Return": equity.iloc[-1] - 1,
        "Annualised Return": annual_return,
        "Annualised Volatility": annual_volatility,
        "Sharpe Ratio (rf=0)": sharpe,
        "Maximum Drawdown": drawdown.min(),
    }

strategy_metrics = calculate_metrics(
    data["Strategy_Return"], data["Strategy_Equity"]
)

buy_hold_metrics = calculate_metrics(
    data["Buy_Hold_Return"], data["Buy_Hold_Equity"]
)

results = pd.DataFrame({
    "Moving Average Strategy": strategy_metrics,
    "Buy and Hold": buy_hold_metrics,
})

# Count transitions between invested and uninvested states.
trade_entries = (
    (data["Position"] == 1) &
    (data["Position"].shift(1).fillna(0) == 0)
).sum()

time_in_market = data["Position"].mean()

print("\nBACKTEST REPORT")
print("=" * 60)
print(f"Ticker: {TICKER}")
print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
print(f"Observations: {len(data)}")
print(f"Moving averages: {SHORT_WINDOW}/{LONG_WINDOW} days")
print(f"Transaction cost: {TRANSACTION_COST:.2%} per unit of turnover")
print(f"Strategy entries: {trade_entries}")
print(f"Time in market: {time_in_market:.1%}")
print("\nPerformance metrics:")
print(results.to_string(float_format=lambda x: f"{x:.3f}"))

data.to_csv("results/backtest_results.csv")
results.to_csv("results/backtest_metrics.csv")

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

data[["Strategy_Equity", "Buy_Hold_Equity"]].plot(ax=axes[0])
axes[0].set_title("Strategy vs Buy-and-Hold")
axes[0].set_ylabel("Growth of £1 (before taxes)")

strategy_drawdown = (
    data["Strategy_Equity"] / data["Strategy_Equity"].cummax() - 1
)

buy_hold_drawdown = (
    data["Buy_Hold_Equity"] / data["Buy_Hold_Equity"].cummax() - 1
)

pd.DataFrame({
    "Strategy": strategy_drawdown,
    "Buy and Hold": buy_hold_drawdown,
}).plot(ax=axes[1])

axes[1].set_title("Drawdown Comparison")
axes[1].set_ylabel("Drawdown")

fig.tight_layout()
fig.savefig("results/backtest.png", dpi=150)
plt.show()

print("\nSaved updated backtest results in results/.")