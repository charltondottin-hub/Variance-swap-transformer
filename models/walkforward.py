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

    # Drop the last `horizon` rows of training data so no target depends on
    # returns from after fit_cutoff (target[t] = sum r^2 over t+1..t+horizon).
    horizon = 21

    n_segments = len(rebalance_dates) - 1
    for i in range(n_segments):
        fit_cutoff = rebalance_dates[i]
        next_cutoff = rebalance_dates[i + 1]
        is_last = i == n_segments - 1

        if config.expanding:
            train_X = features.loc[train_start:fit_cutoff].iloc[:-horizon]
            train_y = target.loc[:fit_cutoff].iloc[:-horizon]
        else:
            rolling_start = fit_cutoff - pd.tseries.offsets.BDay(window_size)
            train_X = features.loc[rolling_start:fit_cutoff].iloc[:-horizon]
            train_y = target.loc[rolling_start:fit_cutoff].iloc[:-horizon]

        fit = fit_fn(train_X, train_y)

        # Each segment owns [fit_cutoff, next_cutoff); the final segment closes
        # the interval to include test_end. Prevents boundary duplication.
        if is_last:
            mask = (features.index >= fit_cutoff) & (features.index <= next_cutoff)
        else:
            mask = (features.index >= fit_cutoff) & (features.index < next_cutoff)
        forecast_X = features.loc[mask]
        if forecast_X.empty:
            continue
        preds = predict_fn(fit, forecast_X)
        actual = target.loc[forecast_X.index]

        predictions.append(preds)
        actuals.append(actual)

    pred_series =pd.concat(predictions)
    actual_series = pd.concat(actuals)
    return pd.DataFrame({"actual": actual_series, "predicted": pred_series}).dropna()