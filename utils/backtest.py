# utils/backtest.py
import numpy as np

def build_return_series(close, seq_length, train_size, y_pred_price):
    """
    Compute true & predicted daily returns for the test set
    
    :param close: pandas Series or 1D array of close prices (sorted by date)
    :param seq_length: sequence length used in create_sequences
    :param train_size: number of sequence samples used for training
    :param y_pred_price: model_predictions for X_test, already in price space
    """

    prices = np.asarray(close, dtype=float)

    # how many (X, y) samples there are in total
    num_samples = len(prices) - seq_length
    num_test = num_samples - train_size

    # index (in the original price series) of the first test target
    first_target_idx = seq_length + train_size
    target_indices = np.arange(first_target_idx, first_target_idx + num_test)
    prev_indices = target_indices - 1

    assert len(y_pred_price) == num_test

    true_prices = prices[target_indices]
    prev_prices = prices[prev_indices]

    r_true = true_prices / prev_prices - 1.0
    r_pred = np.asarray(y_pred_price, dtype=float) / prev_prices - 1.0
    return r_true, r_pred

def directional_strategy_returns(r_true, r_pred, theta=0.0):
    """Long the asset when predicted return > theta, else stay in cash"""
    r_true = np.asarray(r_true, dtype=float)
    r_pred = np.asarray(r_pred, dtype=float)

    long_mask = r_pred > theta
    strat_returns = np.where(long_mask, r_true, 0.0)

    hit_rate = (np.sign(r_pred) == np.sign(r_true)).mean()
    return strat_returns, hit_rate

def sharpe_ratio(returns, periods_per_year=252):
    returns = np.asarray(returns, dtype=float)
    if returns.std(ddof=1) == 0:
        return np.nan
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=1)
    return (daily_mean / daily_std) * np.sqrt(periods_per_year)

def max_drawdown(returns):
    returns = np.asarray(returns, dtype=float)
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)    
    dd = cum / peak - 1.0
    return dd.min()

def summary_stats(returns):
    returns = np.asarray(returns, dtype=float)
    cum_return = np.prod(1.0 + returns) - 1.0
    ann_vol = returns.std(ddof=1) * np.sqrt(252)
    sr = sharpe_ratio(returns)
    mdd = max_drawdown(returns)
    return {
        "cum_return": cum_return,
        "ann_vol" : ann_vol,
        "sharpe": sr,
        "max_dd": mdd,
    }


