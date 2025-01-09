# /visualization/plotter.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_graphs(tickers, predictions, thresholds, stock_data, alerts):
    """
    Plots all graphs for the stocks in the portfolio, highlighting threshold breaches.

    :param tickers: List of stock tickers.
    :param predictions: Dictionary of predicted prices for each stock.
    :param thresholds: Dictionary of threshold prices for each stock.
    :param stock_data: Dictionary of historical data for each stock.
    :param alerts: Dictionary with alerts for threshold breaches.
    """
    num_stocks = len(tickers)
    fig, axes = plt.subplots(num_stocks, 1, figsize=(14, 7 * num_stocks))

    if num_stocks == 1:
        axes = [axes]

    for idx, ticker in enumerate(tickers):
        ax = axes[idx]

        df = stock_data[ticker]
        ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        ax.plot(future_dates, predictions[ticker], label='Predicted Prices', linestyle='--', marker='o', color='red')

        ax.axhline(y=thresholds[ticker], color='green', linestyle='-', label=f'Threshold: ${thresholds[ticker]}')

        if ticker in alerts and alerts[ticker]:
            first_breach_date = future_dates[alerts[ticker][0] - 1].strftime('%d %B, %Y')
            alert_text = f"Threshold first breached on: {first_breach_date}"
            ax.text(0.5, -0.1, alert_text, ha='center', va='top', transform=ax.transAxes, fontsize=12, color='red')


        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{ticker} Price Prediction and Threshold')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(pad=5.0)
    plt.show()
