# Funds Portfolio Monitor

This project provides a tool to monitor stock portfolios, predict future stock prices using LSTM (Long Short-Term Memory) models, and send alerts if the predicted prices breach a user-defined threshold. The system fetches historical stock data, trains an LSTM model, and visualizes the results with stock price predictions, thresholds, and alert notifications.

## Features
- Fetches historical stock data from Yahoo Finance using `yfinance`.
- Trains an LSTM model to predict future stock prices.
- Allows users to set thresholds for price predictions.
- Provides alert notifications for when the threshold is breached.
- Visualizes historical data, predictions, and alerts with interactive plots.

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine.

```bash
git clone https://github.com/your-username/funds-portfolio-monitor.git
cd funds-portfolio-monitor
```
### 2. Install Dependencies
Install the required dependencies using `pip`.

```bash
pip install -r requirements.txt
```
The `requirements.txt` file includes all necessary packages, such as:
- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `tensorflow`

## Usage

### 1. Run the Program
Run the main script to fetch stock data, train the LSTM model, and visualize the results.

```bash
python main.py
```
### 2. Input Stock Tickers and Thresholds
The program will prompt you to input the stock tickers you want to monitor (comma-separated). It will also ask you to enter the threshold price for each stock.

### 3. View Results
Once the program runs, it will display:
- **Stock graphs:** Historical prices (default period is 5 years), predicted prices (next 10 days), and threshold lines.
- **Alerts:** If the predicted price breaches the threshold, an alert message will be displayed below the graph.