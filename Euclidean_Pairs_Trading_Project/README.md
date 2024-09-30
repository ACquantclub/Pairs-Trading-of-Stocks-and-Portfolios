# Pairs Trading with Euclidean Distance and ML

## Overview
This project explores pairs trading using the **Euclidean distance** approach for pair selection. The strategy involves calculating the Euclidean distance between securities to find the most similar pairs based on historical price movements. Simple machine learning algorithms such as **K-Nearest Neighbors (KNN)** are used for pair selection, while a **Random Forest Regressor** is employed to track future spread movements and execute trades.

## Workflow

1. **Data Collection:**
   - Historical price data for multiple securities is collected and preprocessed, including price adjustments and handling missing values. Daily returns are computed to represent price movements over time.

2. **Pair Selection Using Euclidean Distance and KNN:**
   - **Euclidean Distance** is used to calculate the similarity between securities based on their historical price movements. The smaller the distance, the more similar the securities.
   - **K-Nearest Neighbors (KNN)** is applied to identify pairs by finding the nearest securities (smallest Euclidean distance) in the dataset. This ensures that pairs with the most similar price behaviors are selected.

3. **Spread Calculation:**
   - The spread between selected pairs is calculated using the price difference or price ratio to monitor the relative movement of the pair.
   - Historical mean and standard deviation of the spread are computed to detect deviations for potential trading opportunities.

4. **Prediction with Random Forest Regressor:**
   - A **Random Forest Regressor** model is trained to predict future values of the spread using features such as past spread values, price ratios, volatility, and technical indicators.
   - The model provides forecasts on whether the spread will increase or decrease, signaling when to execute buy or sell trades.

5. **Trading Strategy:**
   - **Entry Signals:** When the spread deviates significantly from its historical mean (e.g., 2 standard deviations), the strategy opens positions (buy the underperforming security, sell the overperforming one).
   - **Exit Signals:** Positions are closed when the spread returns to its mean or the Random Forest model predicts a reversal.
   
6. **Backtesting and Evaluation:**
   - The strategy is backtested on historical data to evaluate performance, with metrics such as cumulative returns, Sharpe ratio, and maximum drawdown used to assess profitability and risk.

## Key Features

- **Euclidean Distance for Pair Selection:** Euclidean distance calculates the similarity between securities based on their historical price movements, ensuring that selected pairs are highly similar.
- **K-Nearest Neighbors (KNN):** KNN is used to find the closest pairs in terms of Euclidean distance, simplifying the pair selection process.
- **Random Forest for Spread Prediction:** A Random Forest Regressor is employed to predict future spread movements, providing a robust and simple method for generating trading signals.
- **Mean Reversion Trading Strategy:** The trading strategy assumes that the spread between correlated pairs will revert to its historical mean over time, enabling profitable trades based on this assumption.


