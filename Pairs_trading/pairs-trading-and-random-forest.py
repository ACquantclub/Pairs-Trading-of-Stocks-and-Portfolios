import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats

def pairs_trading_random_forest(pair_data, lookback=30, prediction_window=5, z_score_threshold=2):
    """
    Track stocks and identify buy/sell signals using Random Forest Regression.
    
    :param pair_data: DataFrame containing 'price1' and 'price2' columns for the pair
    :param lookback: Number of past days to use for feature creation
    :param prediction_window: Number of days to predict into the future
    :param z_score_threshold: Z-score threshold for generating trxading signals
    :return: DataFrame with buy/sell signals
    """
    
    # Calculate the spread
    pair_data['spread'] = pair_data['price1'] - pair_data['price2']
    
    # Create features
    for i in range(1, lookback + 1):
        pair_data[f'spread_lag_{i}'] = pair_data['spread'].shift(i)
        pair_data[f'return1_lag_{i}'] = pair_data['price1'].pct_change(i)
        pair_data[f'return2_lag_{i}'] = pair_data['price2'].pct_change(i)
    
    # Calculate rolling mean and std of the spread
    pair_data['spread_ma'] = pair_data['spread'].rolling(window=lookback).mean()
    pair_data['spread_std'] = pair_data['spread'].rolling(window=lookback).std()
    
    # Calculate z-score
    pair_data['z_score'] = (pair_data['spread'] - pair_data['spread_ma']) / pair_data['spread_std']
    
    # Prepare the feature matrix X and target variable y
    features = [f'spread_lag_{i}' for i in range(1, lookback + 1)] + \
               [f'return1_lag_{i}' for i in range(1, lookback + 1)] + \
               [f'return2_lag_{i}' for i in range(1, lookback + 1)] + \
               ['spread_ma', 'spread_std', 'z_score']
    
    X = pair_data.dropna()[features]
    y = pair_data['spread'].shift(-prediction_window).dropna()
    
    # Align X and y
    X = X.iloc[:-prediction_window]
    y = y.iloc[lookback:]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    pair_data['spread_prediction'] = np.nan
    pair_data.loc[X.index, 'spread_prediction'] = rf_model.predict(X)
    
    # Generate trading signals
    pair_data['signal'] = 0
    long_entry = (pair_data['z_score'] < -z_score_threshold) & (pair_data['spread_prediction'] > pair_data['spread'])
    short_entry = (pair_data['z_score'] > z_score_threshold) & (pair_data['spread_prediction'] < pair_data['spread'])
    pair_data.loc[long_entry, 'signal'] = 1
    pair_data.loc[short_entry, 'signal'] = -1
    
    # Close positions when z-score crosses zero or prediction reverses
    long_exit = (pair_data['z_score'] > 0) | (pair_data['spread_prediction'] < pair_data['spread'])
    short_exit = (pair_data['z_score'] < 0) | (pair_data['spread_prediction'] > pair_data['spread'])
    pair_data.loc[long_exit & (pair_data['signal'].shift(1) == 1), 'signal'] = 0
    pair_data.loc[short_exit & (pair_data['signal'].shift(1) == -1), 'signal'] = 0
    
    return pair_data[['price1', 'price2', 'spread', 'spread_prediction', 'z_score', 'signal']]

# Now let's use this function with our data
result = pairs_trading_random_forest(pair_data)

# Print the first few rows of the result
print(result.head())

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(result.index, result['spread'], label='Spread')
plt.plot(result.index, result['spread_prediction'], label='Predicted Spread')
plt.scatter(result.index[result['signal'] == 1], result['spread'][result['signal'] == 1], color='g', marker='^', label='Buy Signal')
plt.scatter(result.index[result['signal'] == -1], result['spread'][result['signal'] == -1], color='r', marker='v', label='Sell Signal')
plt.title('Pairs Trading: Visa (V) vs Mastercard (MA)')
plt.legend()
plt.show()

# Calculate strategy returns
result['returns'] = result['signal'].shift(1) * result['spread'].pct_change()
cumulative_returns = (1 + result['returns']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns.index, cumulative_returns)
plt.title('Cumulative Returns of Pairs Trading Strategy')
plt.ylabel('Cumulative Returns')
plt.show()

# Print some performance metrics
total_return = cumulative_returns.iloc[-1] - 1
annualized_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
sharpe_ratio = np.sqrt(252) * result['returns'].mean() / result['returns'].std()

print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
