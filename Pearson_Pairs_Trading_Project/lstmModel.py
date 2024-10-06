import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and Keras for the LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# For data preprocessing
from sklearn.preprocessing import MinMaxScaler

# Load the spread data
data = pd.read_csv('Pearson_Pairs_Trading_Project/data/random_ahh_data.csv', parse_dates=['Date'], index_col='Date')

# Sort the data by date (just in case)
data.sort_index(inplace=True)

# Extract the spread values
spread = data['Spread'].values.reshape(-1, 1)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
spread_scaled = scaler.fit_transform(spread)


# Define the sequence length
sequence_length = 20  # days

# Create the sequences
X = []
y = []

for i in range(sequence_length, len(spread_scaled)):
    X.append(spread_scaled[i - sequence_length:i, 0])
    y.append(spread_scaled[i, 0])

# Convert to numpy arrays and reshape
X = np.array(X)
y = np.array(y)

# Reshape X for LSTM input (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the split index
train_size = int(len(X) * 0.8)

# Split the data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()

# Add LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))

# Add output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(data.index[train_size + sequence_length:], actual, color='blue', label='Actual Spread')
plt.plot(data.index[train_size + sequence_length:], predictions, color='red', label='Predicted Spread')
plt.title('Spread Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.show()


# Save the model (optional)
#model.save('lstm_spread_model.h5')
