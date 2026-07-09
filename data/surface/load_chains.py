"""Extract OptionsDX 7z archives and cache cleaned per-year chain parquets.

Raw archives live in data/raw/optionsdx/*.7z (gitignored). Each holds
monthly files like spx_eod_201301.txt. Output: data/cache/optionsdx/
chains_YYYY.parquet with typed, de-duplicated quotes.
"""
from pathlib import Path

import pandas as pd
import py7zr

RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / "optionsdx"
EXTRACT_DIR = RAW_DIR / "extracted"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "optionsdx"

KEEP = ["quote_date", "underlying_last", "expire_date", "dte", "strike",
        "c_bid", "c_ask", "p_bid", "p_ask", "c_iv", "p_iv",
        "c_volume", "p_volume"]


def extract_all() -> None:
    """Unpack every archive; skips months already extracted (idempotent)."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for archive in sorted(RAW_DIR.glob("*.7z")):
        with py7zr.SevenZipFile(archive) as z:
            todo = [n for n in z.getnames() if not (EXTRACT_DIR / n).exists()]
            if todo:
                print(f"{archive.name}: extracting {len(todo)} file(s)")
                z.extract(path=EXTRACT_DIR, targets=todo)


def parse_month(path: Path) -> pd.DataFrame:
    """One monthly txt -> typed DataFrame with clean column names."""
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [c.strip().strip("[]").lower() for c in df.columns]
    df = df[[c for c in KEEP if c in df.columns]].copy()
    for c in df.columns:
        if c not in ("quote_date", "expire_date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expire_date"] = pd.to_datetime(df["expire_date"])
    return df


def build_year(year: int, force: bool = False) -> pd.DataFrame:
    """Concatenate the year's months into one cached parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"chains_{year}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    files = sorted(EXTRACT_DIR.glob(f"spx_eod_{year}*.txt"))
    if not files:
        raise FileNotFoundError(f"no extracted files for {year}; "
                                "run extract_all() first")
    df = pd.concat([parse_month(f) for f in files], ignore_index=True)
    # 2021 ships as a half-year archive plus Q3/Q4 quarterlies; any
    # overlapping months would double-count, so de-duplicate hard.
    df = df.drop_duplicates(subset=["quote_date", "expire_date", "strike"])
    df = df.sort_values(["quote_date", "expire_date", "strike"])
    df.to_parquet(out)
    print(f"chains_{year}.parquet: {len(df):,} rows, "
          f"{df['quote_date'].nunique()} trading days")
    return df


if __name__ == "__main__":
    extract_all()
    for year in range(2010, 2024):
        build_year(year)
