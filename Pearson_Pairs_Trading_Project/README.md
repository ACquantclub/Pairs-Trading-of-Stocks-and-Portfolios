# Strategy # 

# Pairs Trading with K-Means Clustering and LSTM

## Overview
This project implements a machine learning-based **pairs trading** strategy using a combination of **K-Means clustering** for pair selection and **Long Short-Term Memory (LSTM)** networks for trade execution. The **Pearson Correlation Coefficient** is used to ensure that selected pairs have a high historical correlation, making them suitable candidates for pairs trading. This approach leverages both unsupervised learning (clustering) and supervised learning (LSTM) to identify trading opportunities and make decisions based on future price predictions.

## Workflow

1. **Data Collection:**
   - Historical price data for multiple securities is collected and preprocessed. This includes adjusting for stock splits, handling missing values, and computing daily returns.

2. **Pair Selection Using K-Means and Pearson Correlation:**
   - **K-Means Clustering** is used to group stocks based on their historical price movements. Stocks in the same cluster are expected to have similar price patterns.
   - After clustering, the **Pearson Correlation Coefficient** is applied to the securities within each cluster to filter out pairs with high correlation (typically > 0.8). This ensures that only highly correlated pairs are selected for trading.

3. **Spread Calculation:**
   - The spread between the selected pair (i.e., the difference or ratio between their prices) is calculated to track the relative price movements.
   - A historical mean and standard deviation of the spread are calculated to identify deviations (mean reversion signals).

4. **Prediction with LSTM:**
   - An **LSTM model** is trained to predict the future values of the spread based on its historical data, along with other relevant features such as volatility, volume, and technical indicators.
   - The model provides predictions on whether the spread will widen or narrow, which serves as the basis for the trading signals.

5. **Trading Strategy:**
   - **Entry Signals:** When the spread deviates significantly from the historical mean (e.g., 2 standard deviations), the strategy opens positions. If the spread widens, the strategy buys the underperforming security and sells the overperforming one.
   - **Exit Signals:** Positions are closed when the spread reverts to the mean, or if the LSTM model predicts a reversal.
   
6. **Backtesting and Evaluation:**
   - The strategy is backtested on historical data to evaluate its performance. Metrics like cumulative returns, Sharpe ratio, and maximum drawdown are used to assess profitability and risk.

## Key Features

- **Unsupervised Learning for Pair Selection:** K-Means clustering groups stocks into clusters, and Pearson correlation is used to filter out the best pairs within each cluster.
- **LSTM for Predictive Modeling:** An LSTM model is used to predict future spread movements, enabling dynamic buy/sell decisions.
- **Mean Reversion Strategy:** Trades are based on the assumption that the spread between a pair of correlated securities will revert to its historical mean over time.
- **Robust Backtesting:** The strategy is tested on historical data with risk-adjusted performance metrics.

