"""Aligned inputs for the CNN arm: baseline features + level scalar, plus
the ATM-normalized shape grid as per-day images.

Consumes results/vol_surface_grid.parquet (built on main). Mirrors
features/surface_pca.py's normalization exactly, so the two arms differ
only in how the identical shape object is compressed. Kept self-contained
(no import from surface_pca) so this branch stands even if exp/surface-pca
is parked unmerged.
"""
import pathlib

import numpy as np
import pandas as pd

from features.engineer import build_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "results" / "vol_surface_grid.parquet"
ATM_COL = "iv_t030_d50"        # the 30-day at-the-money node


def build_cnn_inputs(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(baseline features + surf_level, shape grid), on common dates.

    Column order of the shape frame is the GRID_COLS order written by
    fit_surface.py (maturity-major), which is what lets the dataset
    reshape a row straight into a (12, 19) image.
    """
    grid = pd.read_parquet(GRID_PATH).dropna()   # only fully fitted days
    level = grid[ATM_COL]
    shape = grid.div(level, axis=0)              # each day / its own ATM
    base = build_features(data).join(
        np.log(level).rename("surf_level"), how="inner").dropna()
    common = base.index.intersection(shape.index)
    return base.loc[common], shape.loc[common]
