import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch Stock Data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

# Prepare Data for LSTM
def prepare_data(data, time_steps=50):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#  Main Execution
ticker = 'AAPL'  # Change this to any stock symbol
data = get_stock_data(ticker, '2020-01-01', '2024-01-01')
X, y, scaler = prepare_data(data.values)

#  Reshape Data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

#  Train-Test Split (Important for better evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = build_lstm_model((X.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

#  Predict Future Prices
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))  # Convert back to original scale

#  Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
sns.lineplot(x=range(len(y_test)), y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), label='Actual Price')
sns.lineplot(x=range(len(predicted_prices)), y=predicted_prices.flatten(), label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
