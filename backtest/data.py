# backtest/data.py — cached daily price loader for the backtest engine.
#
# Design rule (the FMP-paywall insurance): fetch from yfinance ONCE, cache to
# disk as CSV, then read locally forever after. No provider change, rate-limit,
# or outage can take data we've already pulled. `refresh=True` forces a re-pull.

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
FETCH_START = "1990-01-01"  # pull max history once; date filtering happens on read


def get_prices(ticker, start=None, end=None, refresh=False):
    """Daily prices for `ticker`, adjusted for splits and dividends.

    The cache always holds the asset's FULL history; `start`/`end` only slice the
    returned view, so one cache file serves any sub-window (walk-forward, etc.).
    First call downloads from yfinance and caches to backtest/cache/<ticker>.csv;
    later calls read the cache. Returns a DataFrame indexed by date with
    Open/High/Low/Close/Volume columns. Close is dividend+split adjusted, so a
    buy-and-hold equity curve on it is total return (dividends reinvested).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=FETCH_START, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {ticker!r}")

        # Single-ticker download still comes back with a (field, ticker) column
        # MultiIndex in recent yfinance — flatten to plain field names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"

        df.to_csv(path)

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    return df


if __name__ == "__main__":
    spy = get_prices("SPY")
    print(spy.tail())
    print(f"\n{len(spy)} rows | {spy.index.min().date()} -> {spy.index.max().date()}")
    print("cached at", os.path.join(CACHE_DIR, "SPY.csv"))
