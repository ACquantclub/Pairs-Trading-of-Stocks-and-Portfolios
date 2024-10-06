import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a date range
dates = pd.date_range(start='2020-01-01', end='2024-10-01', freq='B')  # Business days

# Generate synthetic spread data with a stronger mean reversion component
spread = np.random.normal(loc=0.0, scale=0.5, size=len(dates))  # Reduced scale for less variation
spread = pd.Series(spread).cumsum()  # Cumulative sum to introduce trends
spread = spread - spread.rolling(window=20).mean()  # Tighter mean reversion with a smaller window


# uniform, chi squared, t, f instead of just normal
# choose a random one every day

# Create a DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Spread': spread
})

# Drop NaN values resulted from rolling mean
data.dropna(inplace=True)

# Save to CSV
data.to_csv('data/random_ahh_data.csv', index=False)
