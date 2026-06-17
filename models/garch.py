from dataclasses import dataclass
import numpy as np
import pandas as pd
from arch import arch_model

@dataclass
class GARCHFit:
    model_result: object
    last_obs_date : pd.Timestamp

def fit_garch(returns: pd.Series, last_date: pd.Timestamp = None) -> GARCHFit:
    if last_date is not None:
        returns = returns.loc[:last_date]

    am = arch_model(returns*100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = am.fit(disp='off')
    return GARCHFit(model_result=res, last_obs_date=returns.index[-1])

def predict_garch_horizon(fit: GARCHFit, horizon: int =21) -> float:
    forecasts = fit.model_result.forecast(horizon=horizon, reindex=False)
    daily_vars_pct = forecasts.variance.values[-1]
    daily_vars = daily_vars_pct / (100**2)
    horizon_var = daily_vars.sum() * (252/horizon)
    return float(horizon_var)

