# backtest/tests/_helpers.py — shared synthetic-data fixtures.
#
# Both suites (test_backtest = correctness, stress_test = adversarial) build their
# inputs from these. Deterministic and offline so a test never touches the network
# or depends on cache state. One definition each — no copy-paste drift.

import numpy as np
import pandas as pd

from backtest.strategy import Strategy


def make_df(closes, opens=None, start="2010-01-04"):
    """OHLC frame on a business-day index from a close series.

    Open defaults to Close; High/Low bracket the two so the bars always pass
    data._validate. Volume is a constant placeholder (unused by the engine).
    """
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes), name="Date")
    return pd.DataFrame(
        {"Open": opens, "High": np.maximum(opens, closes),
         "Low": np.minimum(opens, closes), "Close": closes,
         "Volume": np.ones(len(closes))}, index=idx)


def rising(n, daily=0.001, p0=100.0):
    """Deterministic smooth geometric uptrend — no randomness."""
    return p0 * (1 + daily) ** np.arange(n)


def random_walk(n, p0=100.0, vol=0.01, seed=42):
    """Deterministic geometric random walk, strictly positive (re-seeded each call)."""
    steps = np.random.default_rng(seed).normal(0, vol, n)
    return p0 * np.exp(np.cumsum(steps))


class ConstantWeight(Strategy):
    """Test double: always returns a fixed target weight."""

    def __init__(self, w):
        self.w = w

    def target_weight(self, history):
        return self.w


# ---------------------------------------------------------- multi-asset (Phase 3)
def make_panel(closes, tickers=None, opens=None, start="2010-01-04"):
    """Build {'Close','Open'} (date x ticker) panels from a 2D array of closes.

    closes: array shape (days, n_tickers). opens defaults to closes. Used by the
    cross-sectional engine tests."""
    closes = np.asarray(closes, dtype=float)
    if closes.ndim == 1:
        closes = closes.reshape(-1, 1)
    n, k = closes.shape
    tickers = tickers if tickers is not None else [f"T{j}" for j in range(k)]
    idx = pd.bdate_range(start, periods=n, name="Date")
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return {
        "Close": pd.DataFrame(closes, index=idx, columns=tickers),
        "Open": pd.DataFrame(opens, index=idx, columns=tickers),
    }


def rising_panel(n, k, daily=0.0005, p0=100.0):
    """k tickers all rising at the same geometric rate (a known-answer fixture)."""
    col = p0 * (1 + daily) ** np.arange(n)
    return make_panel(np.repeat(col.reshape(-1, 1), k, axis=1))


def random_panel(n, k, vol=0.012, p0=100.0, seed=0):
    """k independent geometric random walks, strictly positive."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, vol, (n, k))
    return make_panel(p0 * np.exp(np.cumsum(steps, axis=0)))
