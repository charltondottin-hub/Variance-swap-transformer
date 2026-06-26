import numpy as np
import pandas as pd


TRADING_DAYS = 252

def daily_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices/prices.shift(1)).dropna()

def realized_variance_daily(prices:pd.Series) -> pd.Series:
    r = daily_returns(prices)
    return (r ** 2) * TRADING_DAYS

def realized_variance_window(prices: pd.Series, window: int) -> pd.Series:
    r = daily_returns(prices)
    rv = r.rolling(window).apply(lambda x: np.sum(x**2)) * (TRADING_DAYS/window)
    return rv.dropna()

def parkinson_estimator(high: pd.Series, low: pd.Series) -> pd.Series:
    log_hl = np.log(high/low) 
    daily_var = (log_hl ** 2) / (4 * np.log(2))
    return daily_var * TRADING_DAYS
    
def garman_klass_estimator(open_: pd.Series, high: pd.Series, low: pd.Series, close:pd.Series) -> pd.Series:
    log_hl = np.log(high/low)
    log_co = np.log(close/open_)
    daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return daily_var * TRADING_DAYS

def forward_realized_variance(prices: pd.Series, horizon: int = 21) -> pd.Series:
    r = daily_returns(prices)
    forward_sum = (r ** 2).rolling(horizon).sum().shift(-horizon)
    return forward_sum * (TRADING_DAYS/horizon) 

