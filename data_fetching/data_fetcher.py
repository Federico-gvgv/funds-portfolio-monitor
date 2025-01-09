# /data_fetching/data_fetcher.py
import yfinance as yf
import os
import logging
import pandas as pd

def fetch_stock_data(ticker, save_path="data/", period="5y"):
    """
    Fetch historical stock data and save it to a CSV file.

    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :param save_path: Directory to save the data.
    :param period: Time period for the data (default: '5y').
    :return: Path to the saved CSV file.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logging.warning(f"No data found for ticker: {ticker}")
            return None

        data = data[['Close']]
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data.set_index('Date', inplace=True)

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{ticker}.csv")
        data.to_csv(file_path)
        logging.info(f"Data saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error fetching data for ticker {ticker}: {e}")
        return None
