"""Walk-forward training and prediction for the transformer."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.dataset import RVDataset, WindowConfig, FeatureScaler
from models.transformer import RVTransformer
from models.train import train_model, TrainConfig


def walk_forward_transformer(
    features: pd.DataFrame,
    target: pd.Series,
    test_start: str,
    test_end: str,
    refit_freq: str = "YS",        # annually (year-start)
    val_frac: float = 0.15,
    window_config: WindowConfig = None,
    train_config: TrainConfig = None,
    model_kwargs: dict = None,
) -> pd.DataFrame:
    window_config = window_config or WindowConfig()
    train_config = train_config or TrainConfig()
    model_kwargs = model_kwargs or {"d_model": 64, "n_heads": 4, "n_layers": 2}
    seq_len = window_config.seq_len

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    refit_dates = pd.date_range(test_start_ts, test_end_ts, freq=refit_freq)
    if len(refit_dates) == 0 or refit_dates[0] > test_start_ts:
        refit_dates = pd.DatetimeIndex([test_start_ts]).append(refit_dates)
    if refit_dates[-1] < test_end_ts:
        refit_dates = refit_dates.append(pd.DatetimeIndex([test_end_ts]))

    all_preds = []
    all_actuals = []

    n_segments = len(refit_dates) - 1
    for i in range(n_segments):
        fit_cutoff = refit_dates[i]
        next_cutoff = refit_dates[i + 1]
        is_last = i == n_segments - 1
        print(f"\n=== Refit at {fit_cutoff.date()}, predicting through {next_cutoff.date()}")

        # Drop the last `horizon` rows so no training target uses returns from
        # after fit_cutoff (target[t] = sum r^2 from t+1..t+horizon, so target[t]
        # is only safe when t+horizon <= fit_cutoff).
        trim = window_config.horizon
        train_features = features.loc[:fit_cutoff].iloc[:-trim]
        train_target = target.loc[:fit_cutoff].iloc[:-trim]
        n = len(train_features)
        val_n = max(seq_len * 2, int(n * val_frac))
        train_n = n - val_n

        scaler = FeatureScaler().fit(train_features.iloc[:train_n])
        scaled = scaler.transform(features)

        # Val slice extends back by seq_len-1 rows so the first val day gets a
        # full lookback window (those extra rows are training-period features
        # used only as context, not as targets — no leakage).
        val_start = max(0, train_n - (seq_len - 1))
        train_ds = RVDataset(scaled.iloc[:train_n], train_target, window_config)
        val_ds = RVDataset(scaled.iloc[val_start:n], train_target, window_config)

        model = RVTransformer(n_features=features.shape[1], **model_kwargs)
        model, _ = train_model(model, train_ds, val_ds, train_config)

        # Test slice: include seq_len-1 rows of pre-context so the first day
        # of the segment gets a prediction (otherwise we silently lose ~seq_len
        # predictions at each refit boundary).
        fit_pos = scaled.index.searchsorted(fit_cutoff)
        next_pos = scaled.index.searchsorted(next_cutoff, side="right")
        test_start_pos = max(0, fit_pos - (seq_len - 1))
        test_features = scaled.iloc[test_start_pos:next_pos]

        test_ds = RVDataset(test_features, target, window_config)
        loader = DataLoader(test_ds, batch_size=train_config.batch_size, shuffle=False)
        model.eval()
        device = next(model.parameters()).device
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                preds.append(model(x).cpu().numpy().ravel())
        if len(preds) == 0:
            print("  (no valid test windows in segment)")
            continue
        rv_preds = np.exp(np.concatenate(preds))
        dates_idx = pd.DatetimeIndex(test_ds.dates())

        # Each segment owns [fit_cutoff, next_cutoff); the final segment closes
        # the interval to include test_end. Guarantees no boundary duplication.
        if is_last:
            mask = (dates_idx >= fit_cutoff) & (dates_idx <= next_cutoff)
        else:
            mask = (dates_idx >= fit_cutoff) & (dates_idx < next_cutoff)
        if not mask.any():
            continue
        sel_dates = dates_idx[mask]
        sel_preds = rv_preds[mask]
        actual = target.loc[sel_dates].values

        all_preds.append(pd.Series(sel_preds, index=sel_dates))
        all_actuals.append(pd.Series(actual, index=sel_dates))

    if not all_preds:
        return pd.DataFrame(columns=["actual", "predicted"])

    pred_series = pd.concat(all_preds).sort_index()
    actual_series = pd.concat(all_actuals).sort_index()
    return pd.DataFrame({"actual": actual_series, "predicted": pred_series}).dropna()
