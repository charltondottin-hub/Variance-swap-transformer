"""Walk-forward CNN+transformer with surface images (exp/surface-cnn).

Test window: 2013-01-01 .. 2023-12-31 (the OptionsDX surface ends 2023).

Usage:
    python notebooks/11_surface_cnn_walkforward.py --smoke
    python notebooks/11_surface_cnn_walkforward.py --n-seeds 10 \
        --refit-freq MS --warm-start --annual-scratch \
        --save-seeds-dir results/surfcnn_ms_preds
    # stage 2: extend the same ensemble to K=40 without redoing 0-9
    python notebooks/11_surface_cnn_walkforward.py --seed-start 10 \
        --n-seeds 30 --refit-freq MS --warm-start --annual-scratch \
        --save-seeds-dir results/surfcnn_ms_preds
"""
import sys, pathlib, argparse, json
root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import contextlib, os
import numpy as np
import pandas as pd
import torch

from data.fetch import fetch_all
from features.surface_tensor import build_cnn_inputs
from features.realized_vol import forward_realized_variance
from models.walkforward_hybrid import walk_forward_surface_cnn
from models.train import TrainConfig
from models.dataset import WindowConfig
from models.metrics import qlike, mse

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--config",
                    default=str(root / "results" / "best_transformer_config.json"))
parser.add_argument("--n-seeds", type=int, default=10)
parser.add_argument("--seed-start", type=int, default=0,
                    help="first seed (stage-2 ensemble extension)")
parser.add_argument("--refit-freq", default="YS")
parser.add_argument("--warm-start", action="store_true")
parser.add_argument("--annual-scratch", action="store_true")
parser.add_argument("--channels", default="16,32,64",
                    help="encoder filter ladder, e.g. 16,32 or 16,32,64,128")
parser.add_argument("--d-embed", type=int, default=16)
parser.add_argument("--save-seeds-dir", default=None)
parser.add_argument("--out",
                    default=str(root / "results"
                                / "transformer_predictions_surfcnn.parquet"))
args = parser.parse_args()

channels = tuple(int(c) for c in args.channels.split(","))
cfg_path = pathlib.Path(args.config)
tuned = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
window_cfg = WindowConfig(seq_len=tuned.get("seq_len", 60))
model_kwargs = {
    "d_model": tuned.get("d_model", 64),
    "n_heads": tuned.get("n_heads", 4),
    "n_layers": tuned.get("n_layers", 2),
    "d_ff": 2 * tuned.get("d_model", 64),
    "dropout": tuned.get("dropout", 0.1),
}

data = fetch_all()
base, shape = build_cnn_inputs(data)
target = forward_realized_variance(data["spx"]["Close"], horizon=21)

if args.smoke:
    test_start, test_end = "2023-01-01", "2023-12-31"
    train_cfg = TrainConfig(epochs=5, lr=tuned.get("lr", 1e-3),
                            weight_decay=tuned.get("weight_decay", 1e-4))
    out_path = root / "results" / "transformer_predictions_surfcnn_smoke.parquet"
else:
    test_start, test_end = "2013-01-01", "2023-12-31"
    train_cfg = TrainConfig(lr=tuned.get("lr", 1e-3),
                            weight_decay=tuned.get("weight_decay", 1e-4))
    out_path = pathlib.Path(args.out)

print(f"Base features: {base.shape}, surface: {shape.shape}, "
      f"{base.index[0].date()}..{base.index[-1].date()}")
print(f"Encoder: channels={channels}, d_embed={args.d_embed}")
print(f"Test window: {test_start} .. {test_end}  (smoke={args.smoke})")

n_seeds = 1 if args.smoke else args.n_seeds
seed_preds, actual = [], None
for seed in range(args.seed_start, args.seed_start + n_seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n### Seed {seed} ({seed - args.seed_start + 1}/{n_seeds}) ...")
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
        res = walk_forward_surface_cnn(
            base, shape, target,
            test_start=test_start, test_end=test_end,
            refit_freq=args.refit_freq,
            window_config=window_cfg, train_config=train_cfg,
            model_kwargs=model_kwargs,
            channels=channels, d_embed=args.d_embed,
            warm_start=args.warm_start,
            scratch_at_year_start=args.annual_scratch,
        )
    seed_preds.append(res["predicted"].rename(f"seed{seed}"))
    if actual is None:
        actual = res["actual"]
    print(f"    seed {seed}: QLIKE "
          f"{qlike(res['actual'], res['predicted']):.6f}")
    if args.save_seeds_dir:
        seed_dir = pathlib.Path(args.save_seeds_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        res.to_parquet(seed_dir / f"seed{seed}.parquet")

ensemble_pred = pd.concat(seed_preds, axis=1).mean(axis=1)  # level space
results = pd.DataFrame({"actual": actual,
                        "predicted": ensemble_pred}).dropna()
print(f"\nEnsembled {n_seeds} seed(s) [{args.seed_start}.."
      f"{args.seed_start + n_seeds - 1}]: {len(results)} rows")
print(f"QLIKE: {qlike(results['actual'], results['predicted']):.6f}")
print(f"MSE:   {mse(results['actual'], results['predicted']):.6e}")
results.to_parquet(out_path)
print(f"Wrote {out_path}")
