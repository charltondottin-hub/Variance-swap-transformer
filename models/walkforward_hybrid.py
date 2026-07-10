"""Walk-forward training/prediction for the CNN+transformer pairing.

A two-input port of models/walkforward_transformer.py. Everything the
production driver does - horizon trim, val-slice back-extension, hybrid
scratch/fine-tune schedule, per-segment scaler fits, test pre-context,
segment ownership of [fit_cutoff, next_cutoff) - is inherited unchanged;
the only substantive additions are the surface frame, its own scaler,
and the SurfaceRVTransformer.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.dataset import WindowConfig, FeatureScaler
from models.hybrid_dataset import SurfaceRVDataset, SurfaceScaler
from models.surface_cnn import SurfaceRVTransformer
from models.train import TrainConfig
from models.train_hybrid import train_model, _to


def walk_forward_surface_cnn(
    features: pd.DataFrame,          # baseline + surf_level (n_base cols)
    surface: pd.DataFrame,           # 228 shape columns, same index
    target: pd.Series,
    test_start: str,
    test_end: str,
    refit_freq: str = "YS",
    val_frac: float = 0.15,
    window_config: WindowConfig = None,
    train_config: TrainConfig = None,
    model_kwargs: dict = None,
    channels=(16, 32, 64),
    d_embed: int = 16,
    warm_start: bool = False,
    scratch_at_year_start: bool = False,
    finetune_config: TrainConfig = None,
) -> pd.DataFrame:
    window_config = window_config or WindowConfig()
    train_config = train_config or TrainConfig()
    model_kwargs = model_kwargs or {"d_model": 64, "n_heads": 4, "n_layers": 2}
    seq_len = window_config.seq_len

    if warm_start and finetune_config is None:
        finetune_config = replace(train_config, epochs=5,
                                  lr=train_config.lr / 5, warmup_steps=0)

    test_start_ts, test_end_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)
    refit_dates = pd.date_range(test_start_ts, test_end_ts, freq=refit_freq)
    if len(refit_dates) == 0 or refit_dates[0] > test_start_ts:
        refit_dates = pd.DatetimeIndex([test_start_ts]).append(refit_dates)
    if refit_dates[-1] < test_end_ts:
        refit_dates = refit_dates.append(pd.DatetimeIndex([test_end_ts]))

    all_preds, all_actuals, prev_state = [], [], None
    n_segments = len(refit_dates) - 1
    for i in range(n_segments):
        fit_cutoff, next_cutoff = refit_dates[i], refit_dates[i + 1]
        is_last = i == n_segments - 1
        use_warm = (warm_start and prev_state is not None
                    and not (scratch_at_year_start and fit_cutoff.month == 1))
        mode = "fine-tune" if use_warm else "scratch"
        print(f"\n=== Refit at {fit_cutoff.date()} ({mode}), "
              f"predicting through {next_cutoff.date()}")

        trim = window_config.horizon        # no target uses post-cutoff days
        train_features = features.loc[:fit_cutoff].iloc[:-trim]
        train_target = target.loc[:fit_cutoff].iloc[:-trim]

        n = len(train_features)
        val_n = max(seq_len * 2, int(n * val_frac))
        train_n = n - val_n

        f_scaler = FeatureScaler().fit(train_features.iloc[:train_n])
        s_scaler = SurfaceScaler().fit(
            surface.loc[train_features.index[:train_n]])
        scaled = f_scaler.transform(features)
        s_scaled = s_scaler.transform(surface)

        val_start = max(0, train_n - (seq_len - 1))
        train_ds = SurfaceRVDataset(scaled.iloc[:train_n],
                                    s_scaled.iloc[:train_n],
                                    train_target, window_config)
        val_ds = SurfaceRVDataset(scaled.iloc[val_start:n],
                                  s_scaled.iloc[val_start:n],
                                  train_target, window_config)

        model = SurfaceRVTransformer(n_base=features.shape[1],
                                     channels=channels, d_embed=d_embed,
                                     **model_kwargs)
        if use_warm:
            model.load_state_dict(prev_state)
            model, _ = train_model(model, train_ds, val_ds, finetune_config,
                                   init_as_baseline=True)
        else:
            model, _ = train_model(model, train_ds, val_ds, train_config)
        if warm_start:
            prev_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        fit_pos = scaled.index.searchsorted(fit_cutoff)
        next_pos = scaled.index.searchsorted(next_cutoff, side="right")
        test_start_pos = max(0, fit_pos - (seq_len - 1))
        test_ds = SurfaceRVDataset(scaled.iloc[test_start_pos:next_pos],
                                   s_scaled.iloc[test_start_pos:next_pos],
                                   target, window_config)
        loader = DataLoader(test_ds, batch_size=train_config.batch_size,
                            shuffle=False)
        model.eval()
        device = next(model.parameters()).device
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                preds.append(model(_to(x, device)).cpu().numpy().ravel())
        if len(preds) == 0:
            print("  (no valid test windows in segment)")
            continue
        rv_preds = np.exp(np.concatenate(preds))
        dates_idx = pd.DatetimeIndex(test_ds.dates())
        if is_last:
            mask = (dates_idx >= fit_cutoff) & (dates_idx <= next_cutoff)
        else:
            mask = (dates_idx >= fit_cutoff) & (dates_idx < next_cutoff)
        if not mask.any():
            continue
        all_preds.append(pd.Series(rv_preds[mask], index=dates_idx[mask]))
        all_actuals.append(pd.Series(target.loc[dates_idx[mask]].values,
                                     index=dates_idx[mask]))

    if not all_preds:
        return pd.DataFrame(columns=["actual", "predicted"])
    pred_series = pd.concat(all_preds).sort_index()
    actual_series = pd.concat(all_actuals).sort_index()
    return pd.DataFrame({"actual": actual_series,
                         "predicted": pred_series}).dropna()
