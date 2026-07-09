"""Acceptance tests + diagnostic figures for the fitted surface.

Checks: (1) coverage by year, (2) Shailesh's parity acceptance test,
(3) fitted ATM level tracks the VIX, (4) calm-vs-crash surface snapshots.
"""
import sys, pathlib

root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.fetch import fetch_all
from data.surface.fit_surface import GRID_DAYS, GRID_DELTAS

FIG = root / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

grid = pd.read_parquet(root / "results" / "vol_surface_grid.parquet")
qc = pd.read_parquet(root / "results" / "vol_surface_qc.parquet")

# ---- 1. coverage: how many days produced a full 228-node grid?
ok = grid.notna().all(axis=1)
print(f"days: {len(grid)}   fully fitted: {ok.sum()} ({ok.mean():.1%})")
print(ok.groupby(grid.index.year).mean().round(3).to_string())

# ---- 2. parity acceptance test, every day
print(f"parity IV gap: median {qc['parity_gap'].median():.4f}   "
      f"95th pct {qc['parity_gap'].quantile(0.95):.4f}")
assert qc["parity_gap"].median() < 0.005, "parity gap too wide - check forwards"

# ---- 3. the fitted level must track the VIX
vix = fetch_all()["vix"]["Close"] / 100
atm = grid["iv_t030_d50"].dropna()
common = atm.index.intersection(vix.index)
corr = np.corrcoef(atm.loc[common], vix.loc[common])[0, 1]
print(f"corr(fitted ATM 30d, VIX) = {corr:.3f} on {len(common)} days")
assert corr > 0.90, "fitted level should track the VIX closely"

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(common, vix.loc[common], lw=0.7, label="VIX / 100")
ax.plot(common, atm.loc[common], lw=0.7, label="fitted ATM 30d IV")
ax.legend()
ax.set_title("Surface level vs VIX")
fig.tight_layout()
fig.savefig(FIG / "surface_atm_vs_vix.png", dpi=150)

# ---- 4. calm day vs crash day snapshots
for label, d in [("calm", "2017-07-14"), ("crash", "2020-03-16")]:
    if pd.Timestamp(d) not in grid.index:
        continue
    z = grid.loc[d].to_numpy().reshape(len(GRID_DAYS), len(GRID_DELTAS))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(range(0, 19, 3), [f"{x:.2f}" for x in GRID_DELTAS[::3]])
    ax.set_yticks(range(len(GRID_DAYS)), GRID_DAYS)
    ax.set_xlabel("call delta (right edge = OTM puts)")
    ax.set_ylabel("maturity (days)")
    ax.set_title(f"Fitted surface {d} ({label})")
    fig.colorbar(im, label="implied vol")
    fig.tight_layout()
    fig.savefig(FIG / f"surface_{label}.png", dpi=150)

print(f"figures written to {FIG}")
