"""Build the daily fitted vol-surface grid from cached OptionsDX chains.

Usage:
    python notebooks/07_build_surface.py                 # all cached years
    python notebooks/07_build_surface.py --years 2013    # just one year
    python notebooks/07_build_surface.py --force         # rebuild

Writes results/surface/grid_YYYY.parquet per year (checkpoints), then
combines into results/vol_surface_grid.parquet + vol_surface_qc.parquet.
"""
import sys, pathlib, argparse, time

root = pathlib.Path(__file__).resolve().parent
while root != root.parent and not (root / ".git").exists():
    root = root.parent
sys.path.insert(0, str(root))

import pandas as pd

from data.surface.load_chains import build_year, CACHE_DIR
from data.surface.forwards import forwards_for_day
from data.surface.fit_surface import fit_day, GRID_COLS

OUT_DIR = root / "results" / "surface"
OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, nargs="*", default=None)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

years = args.years or sorted(
    int(p.stem.split("_")[1]) for p in CACHE_DIR.glob("chains_*.parquet"))

for year in years:
    out = OUT_DIR / f"grid_{year}.parquet"
    if out.exists() and not args.force:
        print(f"{year}: already built, skipping")
        continue
    chains = build_year(year)
    grid_rows, qc_rows, dates = [], [], []
    t0 = time.time()
    for i, (qd, day) in enumerate(chains.groupby("quote_date")):
        fwds = forwards_for_day(day)
        grid, qc = fit_day(day, fwds)
        grid_rows.append(grid)
        qc_rows.append(qc)
        dates.append(qd)
        if (i + 1) % 50 == 0:
            print(f"  {year}: {i + 1} days, {time.time() - t0:.0f}s")
    idx = pd.DatetimeIndex(dates, name="date")
    gdf = pd.DataFrame(grid_rows, index=idx)
    qdf = pd.DataFrame(qc_rows, index=idx)
    pd.concat([gdf, qdf], axis=1).to_parquet(out)
    n_ok = gdf.notna().all(axis=1).sum()
    print(f"{year}: {n_ok}/{len(gdf)} days fully fitted -> {out.name}")

# ---- combine all years into the two files experiments consume
parts = [pd.read_parquet(p) for p in sorted(OUT_DIR.glob("grid_*.parquet"))]
full = pd.concat(parts).sort_index()
full[GRID_COLS].to_parquet(root / "results" / "vol_surface_grid.parquet")
qc_cols = [c for c in full.columns if c not in GRID_COLS]
full[qc_cols].to_parquet(root / "results" / "vol_surface_qc.parquet")
print(f"vol_surface_grid.parquet: {len(full)} days x {len(GRID_COLS)} nodes")
