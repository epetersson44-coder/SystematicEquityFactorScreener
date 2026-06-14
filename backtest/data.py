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

OHLC = ["Open", "High", "Low", "Close"]


def _validate(df, ticker):
    """Chan ch.3 data hygiene — fail LOUD on broken data, not silently.

    A backtest is only as trustworthy as its prices. Garbage here (a duplicated
    date, a zero close, a high below its low) silently corrupts every downstream
    number, so we check structure on every load and raise rather than limp on.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates not sorted ascending")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()][:3].tolist()
        raise ValueError(f"{ticker}: duplicate dates, e.g. {dupes}")
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    if df[OHLC].isnull().any().any():
        raise ValueError(f"{ticker}: NaNs in OHLC ({int(df[OHLC].isnull().sum().sum())} cells)")
    if (df[OHLC] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive OHLC prices present")
    bad = df["High"] < df["Low"]
    if bad.any():
        raise ValueError(f"{ticker}: High < Low on {int(bad.sum())} bar(s), e.g. {df.index[bad][0].date()}")
    return df


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

    _validate(df, ticker)                 # trust nothing unvalidated downstream

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
