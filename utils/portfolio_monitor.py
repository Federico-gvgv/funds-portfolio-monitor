# /utils/portfolio_monitor.py
def monitor_portfolio(predictions, thresholds):
    """
    Monitors the portfolio and triggers alerts if predictions fall below the set threshold.

    :param predictions: Dictionary with dates as keys and predicted prices as values.
    :param thresholds: Dictionary with tickers as keys and threshold prices as values.
    """
    alerts = {}
    for ticker, pred_list in predictions.items():
        for i, price in enumerate(pred_list):
            if price < thresholds[ticker]:
                if ticker not in alerts:
                    alerts[ticker] = []
                alerts[ticker].append(i + 1)  # Store the day number (i + 1)
    return alerts
