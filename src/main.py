import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducible example data
np.random.seed(42)

dates = pd.bdate_range(start="2025-01-01", periods=252)
daily_returns = np.random.normal(
    loc=0.0004,
    scale=0.015,
    size=len(dates),
)

prices = 100 * (1 + pd.Series(daily_returns, index=dates)).cumprod()

data = pd.DataFrame({"Price": prices})
data["Daily Return"] = data["Price"].pct_change()
data["Volatility (20d)"] = (
    data["Daily Return"].rolling(window=20).std() * np.sqrt(252)
)

total_return = data["Price"].iloc[-1] / data["Price"].iloc[0] - 1
annualised_volatility = data["Daily Return"].std() * np.sqrt(252)

print("QUANTITATIVE RESEARCH REPORT")
print("-" * 32)
print(f"Observations: {len(data)}")
print(f"Total return: {total_return:.2%}")
print(f"Annualised volatility: {annualised_volatility:.2%}")
print("\nLatest observations:")
print(data.tail())

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

data["Price"].plot(ax=axes[0], title="Simulated Asset Price")
axes[0].set_ylabel("Price")

data["Volatility (20d)"].plot(
    ax=axes[1],
    title="20-Day Rolling Annualised Volatility",
)
axes[1].set_ylabel("Volatility")
axes[1].set_xlabel("Date")

fig.tight_layout()
fig.savefig("results/initial_analysis.png", dpi=150)
plt.show()
