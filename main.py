from scripts.fetch_data import fetch_stock_data

def main():
    """
    Main function to fetch stock data for a given ticker.
    """
    ticker = "AAPL"  # Apple stock ticker
    file_path = fetch_stock_data(ticker)

    if file_path:
        print(f"Data successfully saved to: {file_path}")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
