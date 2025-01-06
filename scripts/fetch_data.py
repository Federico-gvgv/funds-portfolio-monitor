import yfinance as yf
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def fetch_stock_data(ticker, save_path="data/", period="6mo"):
    """
    Fetch historical stock data and save it to a CSV file.

    :param ticker: Stock ticker symbol (e.g., 'AAPL').
    :param save_path: Directory to save the data.
    :param period: Time period for the data (default: '6mo').
    :return: Path to the saved CSV file.
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        # Ensure data is not empty
        if data.empty:
            logging.warning(f"No data found for ticker: {ticker}")
            return None

        # Keep only the 'Close' prices
        data = data[['Close']]

        # Format the Date column to remove time and timezone
        data.reset_index(inplace=True)  # Convert index to a column
        data['Date'] = pd.to_datetime(data['Date']).dt.date  # Extract only the date part
        data.set_index('Date', inplace=True)  # Set the Date column back as index

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save data to a CSV file
        file_path = os.path.join(save_path, f"{ticker}.csv")
        data.to_csv(file_path)
        logging.info(f"Data saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error fetching data for ticker {ticker}: {e}")
        return None
