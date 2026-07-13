"""Single-segment ablation: does a clean handoff change the CNN arm's story?

Protocol (mirrors the 2026-07-13 diagnostic ablations): one scratch fit at
2018-01-01 on all prior data, predict calendar 2018, seed 0. Two models under
IDENTICAL split/loop/config — the only difference is the surface pathway:

  A. base-only: production RVTransformer on the 11 baseline features
  B. gated:     GatedSurfaceRVTransformer (GroupNorm encoder (8,16),
                d_embed 4, LayerNorm + zero-init gate)

Both use the purged validation split (embargo 21 = horizon, removing
forward-RV overlap between train and val targets) and the capped warmup.
Headline readouts: QLIKE A vs B (DM test) and the learned gate value -
if the gate stays ~0 the model itself judged the surface pathway useless
under a clean handoff; if it opens AND B <= A, first real edge evidence.

Usage:
    .venv/Scripts/python notebooks/13_gated_ablation_2018.py
    .venv/Scripts/python notebooks/13_gated_ablation_2018.py --smoke  # 2 epochs
"""
import sys, pathlib, argparse, json
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.fetch import fetch_all
from features.surface_tensor import build_cnn_inputs
from features.realized_vol import forward_realized_variance
from models.dataset import RVDataset, WindowConfig, FeatureScaler
from models.hybrid_dataset import SurfaceRVDataset, SurfaceScaler
from models.surface_cnn import GatedSurfaceRVTransformer
from models.transformer import RVTransformer
from models.train import TrainConfig
from models.train_hybrid import train_model, _to
from models.metrics import qlike, mse, diebold_mariano

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="2 epochs, wiring check")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--embargo", type=int, default=21)
args = parser.parse_args()

FIT_CUTOFF = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2018-12-31")

tuned = json.loads((root / "results" / "best_transformer_config.json").read_text())
wc = WindowConfig(seq_len=tuned["seq_len"])
tc = TrainConfig(epochs=2 if args.smoke else 30, lr=tuned["lr"],
                 weight_decay=tuned["weight_decay"])
mk = {"d_model": tuned["d_model"], "n_heads": tuned["n_heads"],
      "n_layers": tuned["n_layers"], "d_ff": 2 * tuned["d_model"],
      "dropout": tuned["dropout"]}
seq_len = wc.seq_len

data = fetch_all()
base, shape = build_cnn_inputs(data)
target = forward_realized_variance(data["spx"]["Close"], horizon=21)

# ---- one walk-forward segment, purged split (same arithmetic as the driver)
trim = wc.horizon
tf = base.loc[:FIT_CUTOFF].iloc[:-trim]
tt = target.loc[:FIT_CUTOFF].iloc[:-trim]
n = len(tf)
val_n = max(seq_len * 2, int(n * 0.15))
train_n = n - val_n - args.embargo
val_start = max(0, train_n + args.embargo - (seq_len - 1))
print(f"segment: {tf.index[0].date()}..{tf.index[-1].date()}  "
      f"train {train_n}, embargo {args.embargo}, val {val_n}")

f_scaler = FeatureScaler().fit(tf.iloc[:train_n])
s_scaler = SurfaceScaler().fit(shape.loc[tf.index[:train_n]])
scaled = f_scaler.transform(base)
s_scaled = s_scaler.transform(shape)

fit_pos = scaled.index.searchsorted(FIT_CUTOFF)
next_pos = scaled.index.searchsorted(TEST_END, side="right")
test_pos = max(0, fit_pos - (seq_len - 1))


def predict(model, test_ds):
    loader = DataLoader(test_ds, batch_size=tc.batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(_to(x, torch.device("cpu"))).numpy().ravel())
    rv = np.exp(np.concatenate(preds))
    idx = pd.DatetimeIndex(test_ds.dates())
    m = (idx >= FIT_CUTOFF) & (idx <= TEST_END)
    return pd.Series(rv[m], index=idx[m])


def run_base(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    train_ds = RVDataset(scaled.iloc[:train_n], tt, wc)
    val_ds = RVDataset(scaled.iloc[val_start:n], tt, wc)
    model = RVTransformer(n_features=base.shape[1], **mk)
    model, _ = train_model(model, train_ds, val_ds, tc)
    test_ds = RVDataset(scaled.iloc[test_pos:next_pos], target, wc)
    return predict(model, test_ds), model


def run_gated(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    train_ds = SurfaceRVDataset(scaled.iloc[:train_n], s_scaled.iloc[:train_n],
                                tt, wc)
    val_ds = SurfaceRVDataset(scaled.iloc[val_start:n], s_scaled.iloc[val_start:n],
                              tt, wc)
    model = GatedSurfaceRVTransformer(n_base=base.shape[1], channels=(8, 16),
                                      d_embed=4, **mk)
    model, _ = train_model(model, train_ds, val_ds, tc)
    test_ds = SurfaceRVDataset(scaled.iloc[test_pos:next_pos],
                               s_scaled.iloc[test_pos:next_pos], target, wc)
    return predict(model, test_ds), model


print("\n--- A: base-only transformer")
p_base, _ = run_base(args.seed)
print("\n--- B: gated surface hybrid")
p_gated, m_gated = run_gated(args.seed)

common = p_base.index.intersection(p_gated.index)
a = target.loc[common]
gate = float(m_gated.gate.detach())
print(f"\n2018 test days: {len(common)}")
print(f"{'model':12s} {'QLIKE':>8s} {'MSE':>12s} {'sd(logpred)':>12s}")
print(f"{'base-only':12s} {qlike(a, p_base.loc[common]):8.4f} "
      f"{mse(a, p_base.loc[common]):12.3e} "
      f"{np.log(p_base.loc[common]).std():12.3f}")
print(f"{'gated':12s} {qlike(a, p_gated.loc[common]):8.4f} "
      f"{mse(a, p_gated.loc[common]):12.3e} "
      f"{np.log(p_gated.loc[common]).std():12.3f}")
print(f"sd(log actual): {np.log(a).std():.3f}")
dm = diebold_mariano(a, p_gated.loc[common], p_base.loc[common], loss="qlike")
who = "gated better" if dm["dm_stat"] < 0 else "base better"
print(f"DM qlike gated vs base: {dm['dm_stat']:+.2f} p={dm['p_value']:.4f} ({who})")
print(f"\nlearned gate value: {gate:+.4f}  "
      f"(0 = surface pathway fully closed)")
