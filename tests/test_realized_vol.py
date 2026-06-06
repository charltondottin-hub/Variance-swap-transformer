import numpy as np
import pandas as pd 
from features.realized_vol import (daily_returns, realized_variance_window, forward_realized_variance,)

def test_daily_returns():
    prices = pd.Series([100, 105, 100] , index = pd.date_range("2020-01-01", periods=3))
    r = daily_returns(prices)
    assert len(r) == 2
    assert np.isclose(r.iloc[0], np.log(105/100))

def test_constant_returns_give_constant_rv():
    np.random.seed(0)
    n = 100
    r = pd.Series([0.01] * n, index = pd.date_range('2020-01-01', periods=n))
    prices = pd.Series(np.exp(np.cumsum(r)), index = r.index) * 100
    rv = realized_variance_window(prices, window=22)
    expected = 0.01**2 * 252
    assert np.allclose(rv.dropna().values, expected, atol=1e-6)

def test_forward_rv_lookahead_check():

    n = 50
    np.random.seed(1)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)), index = pd.date_range('2020-01-01', periods=n))
    fwd = forward_realized_variance(prices, horizon=5)
    t_idx = 10
    r = np.log(prices / prices.shift(1)).dropna()
    manual = (r.iloc[t_idx+1:t_idx+6] ** 2).sum() * 252/5
    assert np.isclose(fwd.iloc[t_idx], manual, atol=1e-10)
