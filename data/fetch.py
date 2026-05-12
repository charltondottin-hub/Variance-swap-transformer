"Fetches S&P 500, VIX, VIX term structure, and 3m T-Bill rate." 
"Caches everything as parquet for fast reload" 
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path    
import yfinance as yf
import pandas as pd 
from fredapi import Fred

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START = "2005-01-01"
END = "2025-01-01"

YF_TICKERS = {
    "spx": "^GSPC", 
    "vix": "^VIX",
    "vix9d": "^VIX9D",
    "vix3m": "^VIX3M",
    "vix6m": "^VIX6M",
}

def fetch_yfinance(name: str, ticker: str, force: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    
    print(f"Downlading {ticker} ...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df

def fetch_dtb3(force: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "dtb3.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    
    print("Downloading 3m T-Bill rate from FRED ...")
    fred = Fred()
    df = fred.get_series("DTB3", observation_start=START, observation_end=END)
    df = df.to_frame(name="DTB3")
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df

def fetch_all(force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for name, ticker in YF_TICKERS.items():
        out[name] = fetch_yfinance(name, ticker, force)
    out["dtb3"] = fetch_dtb3(force)
    return out

if __name__ == "__main__":
    data = fetch_all(force=False)
    for name, df in data.items():
        print(f"{name:8s} | {df.index[0].date()} to {df.index[-1].date()} | {len(df)} rows")
   
    