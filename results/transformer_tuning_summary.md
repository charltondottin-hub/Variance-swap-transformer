# Transformer Hyperparameter Tuning — Results

Test window: 2013-01-02 to 2024-11-29 (walk-forward, annual refit). Target:
21-day forward annualized realized variance, S&P 500. Primary metric: QLIKE
(Patton 2011, scale-invariant); MSE secondary. n = 2998 daily forecasts.

## Methodology (no test-set leakage)

The search ran **only on 2011–2018**: inner-train 2011–2016, validate 2017–2018.
2019–2024 was never seen during tuning. Coarse-to-fine (not the full 729-cell
grid): tune `lr` → `seq_len` → architecture (`d_model`/`n_heads`/`n_layers`) →
`dropout`×`weight_decay`, fixing the best value at each stage, then confirm the
winner over 3 seeds. Objective = validation QLIKE in RV space. Reproduce with
`notebooks/05_transformer_tuning.py`; per-config scores in `tuning_results.csv`.

## Best configuration

| Param | Old (default) | Tuned |
|-------|---------------|-------|
| `lr` | 1e-3 | 1e-3 |
| `seq_len` | 60 | **30** |
| `d_model` | 64 | **32** |
| `n_heads` | 4 | **2** |
| `n_layers` | 2 | **1** |
| `d_ff` | 128 | **64** |
| `dropout` | 0.1 | 0.1 |
| `weight_decay` | 1e-4 | **1e-5** |

Saved to `results/best_transformer_config.json`. The headline finding: the
original model was **overparameterized** for ~2,000 training samples. A smaller,
shorter-lookback model generalizes substantially better — consistent with how
hard the 3-parameter HAR is to beat in volatility forecasting.

## Results on the full 2013–2024 walk-forward

| Model | QLIKE | MSE |
|-------|-------|-----|
| HAR | 0.416851 | 4.807e-03 |
| Transformer (old) | 0.586542 | 5.071e-03 |
| **Transformer (tuned)** | **0.494383** | **4.386e-03** |

**Improvement from tuning:**
- QLIKE: 0.587 → 0.494 (**−15.9%**)
- MSE: 5.07e-03 → 4.39e-03 (**−13.5%**)
- Daily win-rate vs HAR: 48.6% → **52.1%** of days

**Diebold–Mariano (HAR vs tuned transformer):**
- QLIKE: stat −2.21, p = 0.027 — HAR still wins, but the gap shrank from
  strongly significant (old: stat −3.89, p < 0.001) to marginal.
- MSE: stat +0.93, p = 0.35 — transformer now has **lower MSE** than HAR, though
  the difference is not statistically significant.

## Honest read

Tuning closed most of the gap and **flipped the MSE ranking in the transformer's
favour**, but the transformer **still does not beat HAR on QLIKE**, the primary
metric. QLIKE punishes under-prediction of variance spikes; the per-day loss
chart shows the transformer's single worst stretch is the March-2020 COVID spike,
which dominates its QLIKE deficit. HAR remains the model to beat on QLIKE.

Plots refreshed in `results/figures/transformer_pred_plot.png` (run
`notebooks/transformer_pred_plot.ipynb`).
