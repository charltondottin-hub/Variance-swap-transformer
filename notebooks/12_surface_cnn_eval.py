"""Evaluate the surface-CNN arm: GARCH benchmark, PCA-arm ablation,
production diagnostic, per-year table, tail-responsiveness check.

The ensemble is recomputed from the seed directory when present, so the
stage-2 (K=40) extension is picked up without re-running stage 1.
GARCH benchmark reuses results/garch_surface_period.parquet if the PCA
arm already built it; otherwise builds it here with the same protocol.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

from data.fetch import fetch_all
from features.realized_vol import daily_returns, forward_realized_variance
from models.garch import fit_garch, predict_garch_daily
from models.metrics import qlike, mse, diebold_mariano

GARCH_PATH = root / "results" / "garch_surface_period.parquet"
SEED_DIR = root / "results" / "surfcnn_ms_preds"
CNN_PATH = root / "results" / "transformer_predictions_surfcnn.parquet"
PCA_PATH = root / "results" / "transformer_predictions_surfpca.parquet"
PROD_PATH = root / "results" / "transformer_predictions.parquet"
TEST_START, TEST_END = "2013-01-01", "2023-12-31"


def build_garch_benchmark() -> pd.DataFrame:
    if GARCH_PATH.exists():
        return pd.read_parquet(GARCH_PATH)
    data = fetch_all()
    spx = data["spx"]["Close"]
    returns = daily_returns(spx)
    target = forward_realized_variance(spx, horizon=21)
    refits = pd.date_range(TEST_START, TEST_END, freq="MS")
    preds = {}
    for i, refit in enumerate(refits):
        seg_end = (refits[i + 1] if i + 1 < len(refits)
                   else pd.Timestamp(TEST_END) + pd.Timedelta(days=1))
        fit = fit_garch(returns, last_date=refit)
        params = fit.model_result.params
        days = returns.loc[refit:seg_end].index
        days = days[days < seg_end]
        for d in days:
            preds[d] = predict_garch_daily(params, returns, d)
        print(f"  {refit.date()}: {len(days)} daily forecasts")
    pred = pd.Series(preds).sort_index()
    out = pd.DataFrame({"actual": target.reindex(pred.index),
                        "predicted": pred}).dropna()
    out.to_parquet(GARCH_PATH)
    return out


def load_cnn() -> pd.DataFrame:
    seeds = sorted(SEED_DIR.glob("seed*.parquet")) if SEED_DIR.exists() else []
    if seeds:
        frames = [pd.read_parquet(p) for p in seeds]
        pred = pd.concat([f["predicted"] for f in frames], axis=1).mean(axis=1)
        out = pd.DataFrame({"actual": frames[0]["actual"],
                            "predicted": pred}).dropna()
        print(f"ensemble recomputed from {len(seeds)} seed files")
        return out
    return pd.read_parquet(CNN_PATH)


def tail_ratio(a: pd.Series, p: pd.Series, q: float = 0.9) -> float:
    """Median forecast/actual on the worst-decile days - the crash
    responsiveness diagnostic from 2026-07-08 (production ~0.46)."""
    hot = a >= a.quantile(q)
    return float((p[hot] / a[hot]).median())


cnn = load_cnn()
bench = build_garch_benchmark()

common = cnn.index.intersection(bench.index)
a = cnn.loc[common, "actual"]
p_cnn = cnn.loc[common, "predicted"]
p_garch = bench.loc[common, "predicted"]
print(f"\ncommon sample: {len(common)} days "
      f"({common[0].date()} .. {common[-1].date()})\n")

print(f"{'model':30s} {'QLIKE':>8s} {'MSE':>12s} {'tail ratio':>10s}")
print(f"{'surface-CNN transformer':30s} {qlike(a, p_cnn):8.4f} "
      f"{mse(a, p_cnn):12.3e} {tail_ratio(a, p_cnn):10.2f}")
print(f"{'GARCH benchmark':30s} {qlike(a, p_garch):8.4f} "
      f"{mse(a, p_garch):12.3e} {tail_ratio(a, p_garch):10.2f}")

for loss in ("qlike", "mse"):
    dm = diebold_mariano(a, p_cnn, p_garch, loss=loss)
    who = "surfcnn better" if dm["dm_stat"] < 0 else "GARCH better"
    print(f"DM {loss:5s} vs GARCH: stat {dm['dm_stat']:+.2f}  "
          f"p={dm['p_value']:.4f}  ({who})")

# ---- ablation gate: the PCA arm, if its predictions exist on disk
if PCA_PATH.exists():
    pca = pd.read_parquet(PCA_PATH)
    c2 = common.intersection(pca.index)
    print(f"\n{'surface-PCA transformer':30s} "
          f"{qlike(pca.loc[c2, 'actual'], pca.loc[c2, 'predicted']):8.4f}"
          f"  (on {len(c2)} common days)")
    dm = diebold_mariano(cnn.loc[c2, "actual"], cnn.loc[c2, "predicted"],
                         pca.loc[c2, "predicted"], loss="qlike")
    who = "CNN better" if dm["dm_stat"] < 0 else "PCA better"
    print(f"DM qlike vs PCA arm: stat {dm['dm_stat']:+.2f}  "
          f"p={dm['p_value']:.4f}  ({who})")

print("\nper-year QLIKE (surfcnn vs GARCH):")
for y in sorted(set(common.year)):
    m = common.year == y
    print(f"  {y}: {qlike(a[m], p_cnn[m]):7.4f}  "
          f"{qlike(a[m], p_garch[m]):7.4f}")

# ---- optional diagnostic: existing production preds, overlap only.
if PROD_PATH.exists():
    prod = pd.read_parquet(PROD_PATH)
    c3 = common.intersection(prod.index)
    dm = diebold_mariano(cnn.loc[c3, "actual"], cnn.loc[c3, "predicted"],
                         prod.loc[c3, "predicted"], loss="qlike")
    print(f"\n[diagnostic] vs production baseline on {len(c3)} days: "
          f"DM {dm['dm_stat']:+.2f}  p={dm['p_value']:.4f}")
