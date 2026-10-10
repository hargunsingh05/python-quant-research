import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER = "AAPL"
START_DATE = "2021-01-01"
SPLIT_DATE = "2025-01-01"
SHORT_WINDOW = 20
LONG_WINDOW = 50
TRANSACTION_COST = 0.001
TRADING_DAYS = 252


def calculate_metrics(returns):
    returns = returns.dropna()

    equity = (1 + returns).cumprod()
    cumulative_return = equity.iloc[-1] - 1
    annual_return = equity.iloc[-1] ** (
        TRADING_DAYS / len(returns)
    ) - 1
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    daily_std = returns.std(ddof=1)
    sharpe = (
        returns.mean() / daily_std * np.sqrt(TRADING_DAYS)
        if daily_std > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "Cumulative Return": cumulative_return,
        "Annualised Return": annual_return,
        "Annualised Volatility": annual_volatility,
        "Sharpe Ratio (rf=0)": sharpe,
        "Maximum Drawdown": drawdown.min(),
    }


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

# Indicators use historical prices only.
data["SMA_Short"] = data["Price"].rolling(SHORT_WINDOW).mean()
data["SMA_Long"] = data["Price"].rolling(LONG_WINDOW).mean()

data["Signal"] = (
    data["SMA_Short"] > data["SMA_Long"]
).astype(float)

# Today's closing signal determines tomorrow's position.
data["Position"] = data["Signal"].shift(1).fillna(0.0)

data["Turnover"] = data["Position"].diff().abs()
data["Turnover"] = data["Turnover"].fillna(data["Position"].abs())

data["Strategy_Return"] = (
    data["Position"] * data["Return"]
    - TRANSACTION_COST * data["Turnover"]
)

data["Buy_Hold_Return"] = data["Return"]

# Start evaluation once both indicators and returns are available.
data = data.dropna(subset=["SMA_Long", "Return"]).copy()

# Development and out-of-sample periods use the same fixed strategy.
development = data[data.index < SPLIT_DATE].copy()
out_of_sample = data[data.index >= SPLIT_DATE].copy()

if len(development) < 2 or len(out_of_sample) < 2:
    raise RuntimeError("Not enough observations in one of the test periods.")

# Calculate each period's growth independently, starting at £1.
for period in (development, out_of_sample):
    period["Strategy_Equity"] = (
        1 + period["Strategy_Return"]
    ).cumprod()
    period["Buy_Hold_Equity"] = (
        1 + period["Buy_Hold_Return"]
    ).cumprod()

development_metrics = pd.DataFrame({
    "Moving Average Strategy": calculate_metrics(
        development["Strategy_Return"]
    ),
    "Buy and Hold": calculate_metrics(
        development["Buy_Hold_Return"]
    ),
})

oos_metrics = pd.DataFrame({
    "Moving Average Strategy": calculate_metrics(
        out_of_sample["Strategy_Return"]
    ),
    "Buy and Hold": calculate_metrics(
        out_of_sample["Buy_Hold_Return"]
    ),
})

print("\nDEVELOPMENT PERIOD")
print("=" * 60)
print(
    f"{development.index.min().date()} to "
    f"{development.index.max().date()}"
)
print(development_metrics.to_string(float_format=lambda x: f"{x:.3f}"))

print("\nOUT-OF-SAMPLE PERIOD")
print("=" * 60)
print(
    f"{out_of_sample.index.min().date()} to "
    f"{out_of_sample.index.max().date()}"
)
print(oos_metrics.to_string(float_format=lambda x: f"{x:.3f}"))

# Save period-specific results.
development_metrics.to_csv("results/development_metrics.csv")
oos_metrics.to_csv("results/out_of_sample_metrics.csv")
development.to_csv("results/development_results.csv")
out_of_sample.to_csv("results/out_of_sample_results.csv")

# Plot out-of-sample equity and drawdown only.
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

out_of_sample[
    ["Strategy_Equity", "Buy_Hold_Equity"]
].plot(ax=axes[0])
axes[0].set_title("Out-of-Sample: Strategy vs Buy-and-Hold")
axes[0].set_ylabel("Growth of £1")

strategy_drawdown = (
    out_of_sample["Strategy_Equity"]
    / out_of_sample["Strategy_Equity"].cummax() - 1
)
buy_hold_drawdown = (
    out_of_sample["Buy_Hold_Equity"]
    / out_of_sample["Buy_Hold_Equity"].cummax() - 1
)

pd.DataFrame({
    "Strategy": strategy_drawdown,
    "Buy and Hold": buy_hold_drawdown,
}).plot(ax=axes[1])

axes[1].set_title("Out-of-Sample Drawdown")
axes[1].set_ylabel("Drawdown")

fig.tight_layout()
fig.savefig("results/out_of_sample_backtest.png", dpi=150)
plt.show()

print("\nSaved development and out-of-sample reports in results/.")