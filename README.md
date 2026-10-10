Python Quantitative Research

A quantitative research project using Python, NumPy, pandas, Matplotlib and Yahoo Finance data to analyse asset prices, measure risk and evaluate a moving-average trading strategy.

Project Features

* Historical market-data analysis
* Daily returns and annualised risk metrics
* Rolling volatility and drawdown analysis
* 20-day and 50-day moving-average strategy
* Transaction costs included in strategy returns
* Buy-and-hold benchmark
* Chronological development and out-of-sample evaluation
* CSV reports and performance charts

Strategy

The strategy holds the asset when its 20-day simple moving average is above its 50-day simple moving average. Otherwise, it holds cash.

Signals are shifted by one trading day so a closing-price signal does not earn the return from that same day’s price movement. A transaction cost of 0.10% per unit of turnover is included.

Data and Evaluation

* Asset: Apple Inc. (AAPL)
* Data source: Yahoo Finance via yfinance
* Development period: March 2021 – December 2024
* Out-of-sample period: January 2025 – October 2026
* Benchmark: Buy-and-hold AAPL
* Risk-free rate for Sharpe ratio: 0%

The moving-average parameters are fixed at 20 and 50 days. The out-of-sample period is used to evaluate the strategy, not to choose its parameters.

Results

Development period

Metric	Strategy	Buy and hold
Cumulative return	27.6%	106.2%
Annualised return	6.6%	21.0%
Annualised volatility	18.0%	26.1%
Sharpe ratio	0.445	0.861
Maximum drawdown	-29.8%	-30.9%

Out-of-sample period

Metric	Strategy	Buy and hold
Cumulative return	7.7%	35.4%
Annualised return	4.3%	18.8%
Annualised volatility	19.9%	30.0%
Sharpe ratio	0.311	0.723
Maximum drawdown	-20.7%	-30.2%

Interpretation

The strategy had lower realised volatility and a smaller maximum drawdown than buy-and-hold in both periods, but it also produced lower returns and lower Sharpe ratios. In these tests, the simpler buy-and-hold benchmark performed better.

This is a useful negative result: a trading rule should be evaluated against a relevant benchmark rather than assumed to be profitable.

Project Structure

* src/main.py — simulated price-series analysis
* src/market_analysis.py — historical market-data and risk analysis
* src/backtest.py — moving-average backtest
* results/ — generated charts, metrics and CSV reports

How to Run

Install the dependencies:

pip install -r requirements.txt

Run the analyses from the project root:

python src/main.py
python src/market_analysis.py
python src/backtest.py

The market-data scripts require an internet connection.

Limitations

* Historical results do not predict future performance.
* The analysis covers one equity ticker.
* The strategy uses daily data and a simple moving-average rule.
* Transaction costs are simplified; slippage, market impact, taxes and execution constraints are not fully modelled.
* The test does not establish that the strategy is profitable or suitable for live trading.
* Yahoo Finance data may be revised or have data-quality limitations.
* The strategy has not been compared across a broad universe of assets or market regimes.

Future Improvements

* Test across multiple assets and market regimes.
* Add more detailed execution-cost assumptions.
* Add automated tests for performance metrics.
* Compare alternative strategies using pre-defined evaluation rules.
* Improve robustness analysis without tuning to the out-of-sample period.