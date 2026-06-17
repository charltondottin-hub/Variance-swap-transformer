import sys, pathlib
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import pandas as pd
from data.fetch import fetch_all
from features.realized_vol import daily_returns, realized_variance_daily, forward_realized_variance
from models.har import build_har_features, fit_har
from models.metrics import qlike, mse, diebold_mariano

har = pd.read_parquet("results/har_predictions.parquet")
gar = pd.read_parquet("results/garch_predictions.parquet")

print(f"HAR   QLIKE: {qlike(har['actual'], har['predicted']):.6f}   MSE: {mse(har['actual'], har['predicted']):.6e}")
print(f"GARCH QLIKE: {qlike(gar['actual'], gar['predicted']):.6f}   MSE: {mse(gar['actual'], gar['predicted']):.6e}")

har = har[~har.index.duplicated(keep="last")].sort_index()
gar = gar[~gar.index.duplicated(keep="last")].sort_index()

common = har.index.intersection(gar.index)
actual = har.loc[common, "actual"]
pred_h = har.loc[common, "predicted"]
pred_g = gar.loc[common, "predicted"]

dm_q = diebold_mariano(actual, pred_h, pred_g, loss="qlike")
dm_m = diebold_mariano(actual, pred_h, pred_g, loss="mse")
print("DM QLIKE (HAR vs GARCH):", dm_q)
print("DM MSE   (HAR vs GARCH):", dm_m)

# HAR full-sample R^2
spx = fetch_all()["spx"]["Close"]
rv = realized_variance_daily(spx)
target = forward_realized_variance(spx, horizon=21)
fit = fit_har(build_har_features(rv), target)
print(f"HAR full-sample R^2: {fit.r_squared:.4f}")