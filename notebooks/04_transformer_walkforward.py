"""Walk-forward transformer driver.

Usage:
    python notebooks/04_transformer_walkforward.py            # full span 2013-01-01..2024-12-31
    python notebooks/04_transformer_walkforward.py --smoke    # one-refit 2023 window, reduced epochs
"""
import sys, pathlib, argparse, json
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import torch
import numpy as np
import pandas as pd
import contextlib, os
from data.fetch import fetch_all
from features.engineer import build_features
from features.realized_vol import forward_realized_variance
from models.walkforward_transformer import walk_forward_transformer
from models.train import TrainConfig
from models.dataset import WindowConfig
from models.metrics import qlike, mse

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true",
                    help="Short window, few epochs — for pipeline sanity check")
parser.add_argument("--config", default=str(root / "results" / "best_transformer_config.json"),
                    help="JSON of tuned hyperparameters (from 05_transformer_tuning.py)")
parser.add_argument("--n-seeds", type=int, default=10,
                    help="Number of random seeds to train and average in level space "
                         "(in-memory deep ensemble). Smoke runs always use 1.")
args = parser.parse_args()

# Load tuned hyperparameters if present, else fall back to library defaults.
cfg_path = pathlib.Path(args.config)
if cfg_path.exists():
    tuned = json.loads(cfg_path.read_text())
    print(f"Loaded tuned config from {cfg_path.name}: {tuned}")
else:
    tuned = {}
    print("No tuned config found — using library defaults.")

window_cfg = WindowConfig(seq_len=tuned.get("seq_len", 60))
model_kwargs = {
    "d_model": tuned.get("d_model", 64),
    "n_heads": tuned.get("n_heads", 4),
    "n_layers": tuned.get("n_layers", 2),
    "d_ff": 2 * tuned.get("d_model", 64),   # tied to d_model, as in tuning
    "dropout": tuned.get("dropout", 0.1),
}

data = fetch_all()
features = build_features(data)
target = forward_realized_variance(data["spx"]["Close"], horizon=21)

if args.smoke:
    test_start, test_end = "2023-01-01", "2024-01-01"
    train_cfg = TrainConfig(epochs=5,
                            lr=tuned.get("lr", 1e-3),
                            weight_decay=tuned.get("weight_decay", 1e-4))
    out_path = root / "results" / "transformer_predictions_smoke.parquet"
else:
    test_start, test_end = "2013-01-01", "2024-12-31"
    train_cfg = TrainConfig(lr=tuned.get("lr", 1e-3),
                            weight_decay=tuned.get("weight_decay", 1e-4))
    out_path = root / "results" / "transformer_predictions.parquet"

print(f"Features: {features.shape}, range {features.index[0].date()}..{features.index[-1].date()}")
print(f"Test window: {test_start} .. {test_end}  (smoke={args.smoke})")
print(f"Model kwargs: {model_kwargs}  seq_len={window_cfg.seq_len}")

# Smoke runs use a single seed for speed; production ensembles n_seeds.
n_seeds = 1 if args.smoke else args.n_seeds
print(f"Ensemble: {n_seeds} seed(s)")

# Train the walk-forward once per seed, keep only each run's prediction series in
# memory (Option B — no per-seed files written), then average across seeds.
seed_preds = []
actual = None
for seed in range(n_seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)   # global RNG, set right before each fresh model is built
    print(f"\n### Seed {seed + 1}/{n_seeds} (seed={seed}) ...")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        res = walk_forward_transformer(
            features, target,
            test_start=test_start,
            test_end=test_end,
            window_config=window_cfg,
            train_config=train_cfg,
            model_kwargs=model_kwargs,
        )
    seed_preds.append(res["predicted"].rename(f"seed{seed}"))
    if actual is None:
        actual = res["actual"]
    print(f"    seed {seed}: QLIKE {qlike(res['actual'], res['predicted']):.6f}")

# Deep ensemble: average the seeds' predictions per date in level space (the space
# QLIKE/MSE score), then score the averaged series once.
ensemble_pred = pd.concat(seed_preds, axis=1).mean(axis=1)
results = pd.DataFrame({"actual": actual, "predicted": ensemble_pred}).dropna()

print(f"\nEnsembled {n_seeds} seed(s): {len(results)} rows, "
      f"{results.index[0].date()}..{results.index[-1].date()}")
print(f"QLIKE: {qlike(results['actual'], results['predicted']):.6f}")
print(f"MSE:   {mse(results['actual'], results['predicted']):.6e}")

out_path.parent.mkdir(parents=True, exist_ok=True)
results.to_parquet(out_path)
print(f"Wrote {out_path}")
