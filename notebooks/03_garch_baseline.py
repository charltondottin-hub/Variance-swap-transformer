import sys, pathlib
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import pandas as pd
from data.fetch import fetch_all
from features.realized_vol import daily_returns, forward_realized_variance
from models.garch import fit_garch, predict_garch_horizon
from models.metrics import qlike, mse

data = fetch_all()
spx = data['spx']['Close']
returns = daily_returns(spx)
target = forward_realized_variance(spx, horizon=21)

preds = {}

rebalance_dates = pd.bdate_range('2010-01-01', '2024-12-31', freq='21B')

for date in rebalance_dates:
    if date not in returns.index:
        continue
    fit = fit_garch(returns, last_date=date)
    pred = predict_garch_horizon(fit, horizon=21)
    preds[date] = pred

pred_series = pd.Series(preds)
results = pd.DataFrame({'actual': target.reindex(pred_series.index), 'predicted': pred_series}).dropna()
print(f"GARCH QLIKE: {qlike(results['actual'], results['predicted']) :.6f}")
print(f"GARCH MSE: {mse(results['actual'], results['predicted']) :.6e}")
results.to_parquet('results/garch_predictions.parquet')