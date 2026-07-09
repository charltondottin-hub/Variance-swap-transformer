"""Black-76 pricing, vectorized implied-vol bisection, and forward delta.

All prices here are UNDISCOUNTED (divide observed mids by DF before
calling). sigma is annualized, tau in years.
"""
import numpy as np
from scipy.special import erf

SQRT2 = np.sqrt(2.0)


def ncdf(x):
    """Standard normal CDF, vectorized (erf is numpy-aware)."""
    return 0.5 * (1.0 + erf(x / SQRT2))


def black76_price(F, K, tau, sigma, is_call):
    st = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * st**2) / st
    d2 = d1 - st
    call = F * ncdf(d1) - K * ncdf(d2)
    put = K * ncdf(-d2) - F * ncdf(-d1)
    return np.where(is_call, call, put)


def implied_vol(price, F, K, tau, is_call, lo=1e-3, hi=5.0, iters=60):
    """Invert Black-76 by bisection, all quotes at once.

    Quotes outside no-arbitrage bounds (below intrinsic, above F or K)
    return NaN rather than a fake vol.
    """
    price, F = np.asarray(price, float), np.asarray(F, float)
    K, tau = np.asarray(K, float), np.asarray(tau, float)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0),
                         np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    valid = (price > intrinsic) & (price < upper) & (tau > 0) & (F > 0)
    lo_a = np.full(price.shape, lo)
    hi_a = np.full(price.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        too_low = black76_price(F, K, tau, mid, is_call) < price
        lo_a = np.where(too_low, mid, lo_a)   # answer is above mid
        hi_a = np.where(too_low, hi_a, mid)   # answer is at or below mid
    out = 0.5 * (lo_a + hi_a)
    return np.where(valid, out, np.nan)


def call_delta(F, K, tau, sigma):
    """N(d1): the call-equivalent delta used as the surface's strike axis."""
    st = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * st**2) / st
    return ncdf(d1)
