# main.py
from data_fetching.data_fetcher import fetch_stock_data
from models.lstm_model import build_lstm_model
from utils.portfolio_monitor import monitor_portfolio
from visualization.plotter import plot_graphs
import logging
import pandas as pd
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def main():
    # 1. User input for tickers and threshold levels
    tickers = input("Enter the tickers of the funds you want to monitor (comma separated): ").strip().split(',')
    tickers = [ticker.strip().upper() for ticker in tickers]
    thresholds = {}
    stock_data = {}
    predictions = {}

    for ticker in tickers:
        threshold = float(input(f"Enter the threshold price for {ticker}: "))
        thresholds[ticker] = threshold

    for ticker in tickers:
        # 2. Get stock data for each ticker
        file_path = fetch_stock_data(ticker, period="5y")
        if not file_path or not os.path.exists(file_path):
            logging.error(f"No data found for {ticker}. Skipping.")
            continue

        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        stock_data[ticker] = df

        # 3. Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])

        # 4. Create sequences
        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)

        # 5. Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # 6. Build and train the model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

        # 7. Make predictions
        predictions[ticker] = []
        last_sequence = scaled_data[-seq_length:]
        for _ in range(10):
            input_seq = last_sequence.reshape((1, seq_length, 1))
            pred_scaled = model.predict(input_seq)[0][0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions[ticker].append(pred)
            last_sequence = np.append(last_sequence[1:], [[pred_scaled]], axis=0)

    # 8. Monitor portfolio for threshold breaches
    alerts = monitor_portfolio(predictions, thresholds)

    # 9. Plot graphs for each stock
    plot_graphs(tickers, predictions, thresholds, stock_data, alerts)

if __name__ == "__main__":
    main()
