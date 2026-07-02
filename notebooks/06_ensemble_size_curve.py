"""QLIKE vs ensemble size (number of seeds averaged).

For each ensemble size k = 1..10, averages predictions in level space over
every C(10, k) subset of the ten per-seed runs in results/multiseed_preds/
(base feature set — the production configuration) and computes QLIKE for each
subset. The curve of the mean QLIKE across subsets, with a min-max band, shows
whether adding seeds beyond 10 is likely to help.

Usage:
    python notebooks/06_ensemble_size_curve.py
"""
import sys, pathlib
from itertools import combinations

root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.metrics import qlike

RESULTS = root / "results"
FIGURES = RESULTS / "figures"
PREDS_DIR = RESULTS / "multiseed_preds"
N_SEEDS = 10


def load_seed_predictions():
    """Per-seed prediction series aligned on their common dates."""
    frames = [
        pd.read_parquet(PREDS_DIR / f"seed{s}_base.parquet") for s in range(N_SEEDS)
    ]
    actual = frames[0]["actual"]
    preds = pd.concat(
        [f["predicted"].rename(f"seed{s}") for s, f in enumerate(frames)], axis=1
    ).dropna()
    return actual.loc[preds.index], preds


def qlike_by_ensemble_size(actual, preds):
    """Mean/min/max QLIKE over all seed subsets of each size."""
    rows = []
    for k in range(1, N_SEEDS + 1):
        scores = [
            qlike(actual, preds[list(combo)].mean(axis=1))
            for combo in combinations(preds.columns, k)
        ]
        rows.append(
            {"k": k, "mean": np.mean(scores), "min": np.min(scores),
             "max": np.max(scores), "n_subsets": len(scores)}
        )
    return pd.DataFrame(rows).set_index("k")


def plot_curve(curve, out_name="ensemble_size_qlike.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(curve.index, curve["min"], curve["max"],
                    color="tab:blue", alpha=0.15, label="Min-max across subsets")
    ax.plot(curve.index, curve["mean"], color="tab:blue", marker="o",
            label="Mean QLIKE across subsets")
    ax.set_xlabel("Ensemble size (number of seeds averaged)")
    ax.set_ylabel("QLIKE")
    ax.set_title("Transformer QLIKE vs Ensemble Size (all subsets of 10 seeds)")
    ax.set_xticks(curve.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / out_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    actual, preds = load_seed_predictions()
    curve = qlike_by_ensemble_size(actual, preds)
    print(curve.round(6).to_string())
    marginal = curve["mean"].diff()
    print("\nMarginal QLIKE change per added seed:")
    print(marginal.round(6).to_string())
    plot_curve(curve)
    print(f"\nFigure written to {FIGURES / 'ensemble_size_qlike.png'}")
