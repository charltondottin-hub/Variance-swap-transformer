'''Loss functions and forecast comparison tests.'''
import numpy as np
import pandas as pd 
from scipy import stats

def qlike(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.values
    predicted = predicted.values
    ratio = actual / predicted
    return float(np.mean(ratio-np.log(ratio)-1))

def mse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean((actual.values - predicted.values) ** 2))

def diebold_mariano(actual:pd.Series, 
                    pred_a:pd.Series,
                    pred_b:pd.Series,
                    loss: str = "qlike",) -> dict:
    common = actual.index.intersection(pred_a.index).intersection(pred_b.index)
    a = actual.loc[common].values
    pa = pred_a.loc[common].values
    pb = pred_b.loc[common].values

    if loss == "qlike":
        la = a/pa - np.log(a/pa) - 1
        lb = a/pb - np.log(a/pb) - 1

    elif loss == "mse":
        la = (a - pa) ** 2
        lb = (a - pb) ** 2
    else: 
        raise ValueError(f"Unknown loss: {loss}")
    
    d = la - lb
    n = len(d)

    h = 21
    gamma_0 = np.var(d)
    weights = [1-k/h for k in range(1,h)]
    auto_cov = np.array([np.cov(d[k:], d[:-k])[0,1] for k in range(1,h)])
    long_run_var = gamma_0 + 2 * np.sum(np.array(weights) * auto_cov)

    dm_stat = np.mean(d) / np.sqrt(long_run_var/n)

    
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n": n}