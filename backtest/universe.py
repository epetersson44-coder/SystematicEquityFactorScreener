# backtest/universe.py — multi-ticker price panels for the cross-sectional engine.
#
# Phase 3 needs N tickers aligned by date, not one OHLC frame. get_universe()
# batch-downloads via yfinance (ONE call — far faster and gentler on rate limits
# than N calls), then hands back aligned (date x ticker) matrices, one per field
# (Close, Open). Cached to disk so we pull once, like the single-asset loader.
#
# SURVIVORSHIP-BIAS WARNING (read this): the ticker list is TODAY's S&P 500. It
# excludes every company that was dropped, delisted, acquired, or went bankrupt
# along the way. A backtest on it only ever holds survivors, so it OVERSTATES
# returns — the losers that would have dragged you down aren't in the data. This is
# acceptable for proving the momentum machinery; it is NOT a clean result. The
# honest fix (point-in-time index membership) is a later, harder job. Documented,
# not hidden.

import os
import pandas as pd
import yfinance as yf

from backtest.data import CACHE_DIR

UNIVERSE_START = "2005-01-01"   # enough history for momentum + walk-forward; keeps the pull sane


def load_tickers(name="sp500"):
    """Read a saved constituent list (one ticker per line)."""
    path = os.path.join(os.path.dirname(__file__), f"{name}_tickers.txt")
    with open(path) as f:
        return [t.strip() for t in f if t.strip()]


def download_panel(tickers, fields=("Close", "Open"), start=UNIVERSE_START):
    """Batch-download `tickers`; return {field: (date x ticker) DataFrame}. No cache."""
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("yfinance returned no data for the universe")
    out = {}
    for f in fields:
        panel = raw[f].copy()                      # (date x ticker) for this field
        panel.index.name = "Date"
        out[f] = panel.sort_index()
    return out


def _validate_panel(panel, field):
    """Panel hygiene — NaNs are allowed (pre-IPO / delisted), but structure isn't.
    Dead (all-NaN) tickers must already be dropped before this runs."""
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{field} panel: dates not sorted")
    if panel.index.has_duplicates:
        raise ValueError(f"{field} panel: duplicate dates")
    if (panel.notna().sum() == 0).any():
        raise ValueError(f"{field} panel: all-NaN ticker slipped through the drop")
    if (panel <= 0).any().any():
        raise ValueError(f"{field} panel: non-positive prices present")


def _drop_dead(panels):
    """Drop tickers with NO data in ANY field (failed downloads). Keeps fields aligned
    on the same surviving ticker set. Returns (clean_panels, dropped_list)."""
    dead = set()
    for panel in panels.values():
        dead |= set(panel.columns[panel.notna().sum() == 0])
    clean = {f: p.drop(columns=[c for c in dead if c in p.columns]) for f, p in panels.items()}
    return clean, sorted(dead)


def get_universe(name="sp500", fields=("Close", "Open"), start=None, end=None, refresh=False):
    """Aligned price panels for a named universe: {field: (date x ticker) DataFrame}.

    Batch-downloads + caches on first call (cache/universe_<name>_<field>.csv), then
    reads the cache. `start`/`end` slice the returned view; the cache holds full
    history. NaNs in a column mean that stock wasn't trading then — expected.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {f: os.path.join(CACHE_DIR, f"universe_{name}_{f}.csv") for f in fields}

    if not refresh and all(os.path.exists(p) for p in paths.values()):
        panels = {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()}
    else:
        panels = download_panel(load_tickers(name), fields=fields)
        for f, panel in panels.items():
            panel.to_csv(paths[f])

    panels, dropped = _drop_dead(panels)        # tolerate failed-download tickers
    if dropped:
        print(f"[universe] dropped {len(dropped)} dead ticker(s): {dropped}")

    for f, panel in panels.items():
        _validate_panel(panel, f)
        if start is not None:
            panel = panel[panel.index >= pd.to_datetime(start)]
        if end is not None:
            panel = panel[panel.index <= pd.to_datetime(end)]
        panels[f] = panel
    return panels


if __name__ == "__main__":
    panels = get_universe("sp500")
    close = panels["Close"]
    print(f"S&P 500 panel: {close.shape[0]} days x {close.shape[1]} tickers")
    print(f"range {close.index.min().date()} -> {close.index.max().date()}")
    coverage = close.notna().mean(axis=1)
    print(f"avg ticker coverage: {coverage.iloc[0]*100:.0f}% at start -> {coverage.iloc[-1]*100:.0f}% at end")
    print(f"(coverage rises as more of today's names became public — survivorship caveat applies)")
