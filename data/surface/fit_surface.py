"""Fit one day's cleaned quotes into the constant (maturity x delta) grid.

Per quote date: merge implied forwards, keep OTM quotes in the 5-95
delta band, aggregate into (expiry, delta-bucket) median cells, fit a
thin-plate spline in (delta, log tau), evaluate on the fixed grid.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from data.surface.black76 import implied_vol, call_delta

GRID_DAYS = np.array([7, 14, 21, 30, 45, 60, 91, 122, 152, 182, 273, 365])
GRID_DELTAS = np.round(np.arange(0.05, 0.951, 0.05), 2)
GRID_COLS = [f"iv_t{d:03d}_d{int(round(dl * 100)):02d}"
             for d in GRID_DAYS for dl in GRID_DELTAS]

DTE_MIN, DTE_MAX = 4, 550      # fitting band (brackets the 7-365d grid)
MIN_PREMIUM = 0.05             # drop sub-nickel lottery tickets
DELTA_BAND = (0.05, 0.95)      # Shailesh's tail exclusion
MIN_BUCKETS = 60               # refuse to fit a sparse day
SMOOTHING = 1e-3               # thin-plate smoothing strength
SHORT_END_MAX_DTE = 14         # nearest expiry must be within this many days;
                               # pre-2017 (Friday-only weeklies) the front
                               # usable expiry is often 8-11d, so the 7d grid
                               # row may be a short log-tau extrapolation


def prepare_day(day: pd.DataFrame, fwds: pd.DataFrame) -> pd.DataFrame:
    """OTM quotes with forward-based IV and delta, filtered to the band."""
    day = day.merge(fwds[["expire_date", "F", "DF"]], on="expire_date",
                    how="left").dropna(subset=["F"])
    day = day[(day["dte"] >= DTE_MIN) & (day["dte"] <= DTE_MAX)].copy()
    day["tau"] = day["dte"] / 365.0
    day["is_call"] = day["strike"] >= day["F"]     # OTM side selection
    day["mid"] = np.where(day["is_call"],
                          (day["c_bid"] + day["c_ask"]) / 2,
                          (day["p_bid"] + day["p_ask"]) / 2)
    day["quote_bid"] = np.where(day["is_call"], day["c_bid"], day["p_bid"])
    day = day[(day["quote_bid"] > 0) & (day["mid"] >= MIN_PREMIUM)].copy()
    day["iv"] = implied_vol(day["mid"].to_numpy() / day["DF"].to_numpy(),
                            day["F"].to_numpy(), day["strike"].to_numpy(),
                            day["tau"].to_numpy(),
                            day["is_call"].to_numpy())
    day = day.dropna(subset=["iv"])
    day = day[(day["iv"] > 0.01) & (day["iv"] < 3.0)].copy()
    day["delta"] = call_delta(day["F"].to_numpy(), day["strike"].to_numpy(),
                              day["tau"].to_numpy(), day["iv"].to_numpy())
    lo, hi = DELTA_BAND
    return day[(day["delta"] >= lo) & (day["delta"] <= hi)]


def parity_gap(band: pd.DataFrame) -> float:
    """Acceptance test: median |call IV - put IV| where both sides trade."""
    b = band[band["delta"].between(0.35, 0.65)
             & (band["c_bid"] > 0) & (band["p_bid"] > 0)]
    if len(b) < 10:
        return np.nan
    args = (b["F"].to_numpy(), b["strike"].to_numpy(), b["tau"].to_numpy())
    ivc = implied_vol(((b["c_bid"] + b["c_ask"]) / 2
                       / b["DF"]).to_numpy(), *args,
                      np.ones(len(b), dtype=bool))
    ivp = implied_vol(((b["p_bid"] + b["p_ask"]) / 2
                       / b["DF"]).to_numpy(), *args,
                      np.zeros(len(b), dtype=bool))
    return float(np.nanmedian(np.abs(ivc - ivp)))


def fit_day(day: pd.DataFrame, fwds: pd.DataFrame) -> tuple[dict, dict]:
    """One quote date -> (grid-value dict, QC dict). All-NaN if unfittable."""
    band = prepare_day(day, fwds)
    qc = {"n_quotes": len(band), "parity_gap": parity_gap(band),
          "n_buckets": 0, "fit_rmse": np.nan}
    empty = dict.fromkeys(GRID_COLS, np.nan)
    if band.empty:
        return empty, qc
    # median-aggregate into (expiry, delta-bucket) cells: robust to any
    # single junk quote, and keeps the spline solve small and fast
    band = band.copy()
    band["dbucket"] = (band["delta"] / 0.025).round() * 0.025
    agg = band.groupby(["expire_date", "dbucket"]).agg(
        iv=("iv", "median"), delta=("delta", "median"),
        tau=("tau", "median")).reset_index()
    qc["n_buckets"] = len(agg)
    # straddle guard: long end interpolation-only; short end allows a small
    # extrapolation down to the 7d row (see SHORT_END_MAX_DTE)
    if (len(agg) < MIN_BUCKETS
            or agg["tau"].min() > SHORT_END_MAX_DTE / 365.0
            or agg["tau"].max() < GRID_DAYS[-1] / 365.0):
        return empty, qc
    pts = np.column_stack([agg["delta"].to_numpy(),
                           np.log(agg["tau"].to_numpy())])
    rbf = RBFInterpolator(pts, agg["iv"].to_numpy(),
                          kernel="thin_plate_spline", smoothing=SMOOTHING)
    qc["fit_rmse"] = float(np.sqrt(np.mean((rbf(pts)
                                            - agg["iv"].to_numpy())**2)))
    gd, gt = np.meshgrid(GRID_DELTAS, np.log(GRID_DAYS / 365.0))
    vals = rbf(np.column_stack([gd.ravel(), gt.ravel()]))
    return dict(zip(GRID_COLS, vals)), qc
