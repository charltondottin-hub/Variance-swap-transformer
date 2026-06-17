from dataclasses import dataclass
from typing import Callable, Any
import pandas as pd
'''Walk-forward backtesting framework for all models.'''
@dataclass
class WalkForwardConfig:
    train_start: str          # e.g. "2005-01-01"
    train_end: str            # initial training cutoff, e.g. "2010-01-01"
    test_end: str             # final evaluation date, e.g. "2024-12-31"
    refit_every: int = 21     # refit every N business days
    expanding: bool = True    # if False, uses a rolling window of fixed size

def walk_forward_backtest(
    features: pd.DataFrame,
    target: pd.Series,
    fit_fn: Callable[[pd.DataFrame, pd.Series], Any],
    predict_fn: Callable[[Any, pd.DataFrame], pd.Series],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Generic walk-forward backtest. Returns a DataFrame with columns ['actual', 'predicted'] indexed by date."""

    common = features.index.intersection(target.index)
    features = features.loc[common]
    target = target.loc[common]  

    test_start = pd.Timestamp(config.train_end)
    test_end = pd.Timestamp(config.test_end)
    train_start = pd.Timestamp(config.train_start)

    rebalance_dates = pd.bdate_range(test_start, test_end, freq=f"{config.refit_every}B")
    
    window_size = len(pd.bdate_range(train_start, test_start))
    predictions = []
    actuals = []

    for i in range(len(rebalance_dates) - 1):
        fit_cutoff = rebalance_dates[i]
        next_cutoff = rebalance_dates[i + 1]

        if config.expanding:
            train_X = features.loc[train_start:fit_cutoff]
            train_y = target.loc[:fit_cutoff]
        else:
            
            rolling_start = fit_cutoff - pd.tseries.offsets.BDay(window_size)
            train_X = features.loc[rolling_start:fit_cutoff]
            train_y = target.loc[rolling_start:fit_cutoff]

        fit = fit_fn(train_X, train_y)

        forecast_X = features.loc[fit_cutoff:next_cutoff]
        if forecast_X.empty:
            continue
        preds = predict_fn(fit, forecast_X)
        actual = target.loc[forecast_X.index]

        predictions.append(preds)
        actuals.append(actual)

    pred_series =pd.concat(predictions)
    actual_series = pd.concat(actuals)
    return pd.DataFrame({"actual": actual_series, "predicted": pred_series}).dropna()