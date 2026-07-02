"""Generate figures for the writeup.

Each figure is a standalone function; run the whole script to regenerate all
figures into results/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)


def plot_actual_vs_model_recent_5y(pred_file, model_label, color, out_name):
    """Actual vs model predictions, most recent 5 years, one time series.

    The target at date t is variance realized over the *next* 21 trading days,
    so for visual alignment the actual series is shifted forward 21 trading
    days (plotted at the end of its window) to match when the model reacts.
    """
    df = pd.read_parquet(RESULTS / pred_file)
    df["actual"] = df["actual"].shift(21)  # timestamp at window end
    start = df.index.max() - pd.DateOffset(years=5)
    df = df.loc[df.index >= start]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["actual"], label="Actual (realized)", color="black", linewidth=1.0)
    ax.plot(
        df.index,
        df["predicted"],
        label=model_label,
        color=color,
        linewidth=1.0,
        alpha=0.85,
    )
    ax.set_ylabel("21-day forward realized variance")
    ax.set_title(f"Actual vs {model_label} Predictions (Last 5 Years)")
    ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(FIGURES / out_name, dpi=150)
    plt.close(fig)


def plot_transformer_vs_garch_recent_5y():
    """Transformer vs GARCH predictions overlaid, most recent 5 years."""
    tf = pd.read_parquet(RESULTS / "transformer_predictions.parquet")
    gr = pd.read_parquet(RESULTS / "garch_predictions.parquet")
    start = tf.index.max() - pd.DateOffset(years=5)
    tf = tf.loc[tf.index >= start]
    gr = gr.loc[gr.index >= start]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gr.index, gr["predicted"], label="GARCH", color="tab:orange", linewidth=1.0)
    ax.plot(
        tf.index,
        tf["predicted"],
        label="Transformer",
        color="tab:blue",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.set_ylabel("21-day forward realized variance")
    ax.set_title("Transformer vs GARCH Predictions (Last 5 Years)")
    ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "transformer_vs_garch_5y.png", dpi=150)
    plt.close(fig)


def _qlike_by_year(df):
    ratio = df["actual"] / df["predicted"]
    return (ratio - np.log(ratio) - 1).groupby(df.index.year).mean()


def transformer_best_years_qlike(n_years=4):
    """Calendar years where the transformer has the lowest QLIKE.

    QLIKE is scale-free, so it ranks years by tracking quality rather than by
    volatility level (low-vol years get no free pass, unlike MSE).
    """
    df = pd.read_parquet(RESULTS / "transformer_predictions.parquet")
    return _qlike_by_year(df).nsmallest(n_years).index.sort_values()


def plot_actual_vs_model_years_panel(pred_file, model_label, color, years,
                                     out_name, suptitle):
    """Actual vs model predictions, one subplot per calendar year.

    Each panel is titled with the model's own QLIKE for that year, computed on
    the aligned (unshifted) series; the actual series is then shifted 21
    trading days for visual alignment, as in the full-sample plots.
    """
    df = pd.read_parquet(RESULTS / pred_file)
    qlike_by_year = _qlike_by_year(df)

    df["actual"] = df["actual"].shift(21)  # timestamp at window end

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, year in zip(axes.flat, years):
        sub = df.loc[df.index.year == year]
        ax.plot(sub.index, sub["actual"], label="Actual (realized)",
                color="black", linewidth=1.0)
        ax.plot(sub.index, sub["predicted"], label=model_label,
                color=color, linewidth=1.0, alpha=0.85)
        ax.set_title(f"{year}  (QLIKE = {qlike_by_year[year]:.3f})")
        ax.margins(x=0.01)
    axes.flat[0].legend()
    fig.supylabel("21-day forward realized variance")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(FIGURES / out_name, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_actual_vs_model_recent_5y(
        "transformer_predictions.parquet", "Transformer", "tab:blue",
        "actual_vs_transformer_5y.png",
    )
    plot_actual_vs_model_recent_5y(
        "garch_predictions.parquet", "GARCH", "tab:orange",
        "actual_vs_garch_5y.png",
    )
    plot_transformer_vs_garch_recent_5y()
    best_years = transformer_best_years_qlike()
    plot_actual_vs_model_years_panel(
        "transformer_predictions.parquet", "Transformer", "tab:blue",
        best_years, "transformer_best_years_qlike.png",
        "Actual vs Transformer: Four Best Years by QLIKE",
    )
    plot_actual_vs_model_years_panel(
        "garch_predictions.parquet", "GARCH", "tab:orange",
        best_years, "garch_same_years_qlike.png",
        "Actual vs GARCH: Transformer's Four Best Years",
    )
    print(f"Figures written to {FIGURES}")
