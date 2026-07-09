"""Implied forward and discount factor per expiration via put-call parity.

For European options, C - P = DF * (F - K) at every strike K. Regressing
(call mid - put mid) on K across strikes therefore identifies both the
discount factor (-slope) and the forward (intercept / DF) with no outside
rate or dividend data.
"""
import numpy as np
import pandas as pd

MIN_PAIRS = 6            # refuse to fit a line through fewer points
STRIKE_WINDOW = 0.20     # use strikes within +/-20% of spot
DF_BOUNDS = (0.5, 1.05)  # plausible discount factors; else reject


def implied_forward(g: pd.DataFrame) -> tuple[float, float, int]:
    """One (quote_date, expire_date) group -> (F, DF, n_pairs)."""
    ok = ((g["c_bid"] > 0) & (g["c_ask"] > 0)
          & (g["p_bid"] > 0) & (g["p_ask"] > 0))
    g = g[ok]
    if g.empty:
        return np.nan, np.nan, 0
    spot = g["underlying_last"].iloc[0]
    g = g[(g["strike"] > (1 - STRIKE_WINDOW) * spot)
          & (g["strike"] < (1 + STRIKE_WINDOW) * spot)]
    if len(g) < MIN_PAIRS:
        return np.nan, np.nan, len(g)
    K = g["strike"].to_numpy()
    y = ((g["c_bid"] + g["c_ask"]) / 2
         - (g["p_bid"] + g["p_ask"]) / 2).to_numpy()
    slope, intercept = np.polyfit(K, y, 1)
    df_ = -slope
    if not (DF_BOUNDS[0] < df_ < DF_BOUNDS[1]):
        return np.nan, np.nan, len(g)
    return intercept / df_, df_, len(g)


def forwards_for_day(day: pd.DataFrame) -> pd.DataFrame:
    """All expirations of one quote date -> table of (F, DF, n_pairs)."""
    rows = []
    for exp, g in day.groupby("expire_date"):
        F, df_, n = implied_forward(g)
        rows.append({"expire_date": exp, "F": F, "DF": df_,
                     "n_pairs": n, "dte": g["dte"].iloc[0]})
    return pd.DataFrame(rows)
