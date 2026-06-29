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

def predict_garch_daily(params, returns: pd.Series, as_of: pd.Timestamp, horizon: int = 21) -> float:
    """21-day-ahead annualized variance anchored at `as_of`, using fixed params.

    Reuses parameters estimated at the segment's refit date and filters the GARCH
    conditional-variance recursion forward through `as_of` (no re-estimation, no
    look-ahead). This lets us emit a daily forecast while only refitting monthly.
    """
    r = returns.loc[:as_of] * 100
    am = arch_model(r, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = am.fix(params)                      # apply fixed params + filter, no fit
    forecasts = res.forecast(horizon=horizon, reindex=False)
    daily_vars = forecasts.variance.values[-1] / (100**2)
    return float(daily_vars.sum() * (252 / horizon))

