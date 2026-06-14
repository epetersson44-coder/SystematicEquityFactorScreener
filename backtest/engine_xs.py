# backtest/engine_xs.py — cross-sectional (multi-asset) backtest engine.
#
# The Phase-3 sibling of engine.py. Same honest discipline — next-open fill, mark to
# market each bar, look-ahead provable by the "corrupt the future" test — but the
# portfolio now holds a BASKET: cash + shares across N tickers, and the strategy
# returns a weight per ticker instead of a scalar. engine.py (single asset) stays as
# the simple, hardened reference; this is its own module because the data shape (a
# date x ticker panel) and the weight interface genuinely differ.
#
# Long-only, no leverage: weights are >= 0 and sum to <= 1 (the rest is cash). That
# guard relaxes deliberately when shorting arrives (pairs trading, later in Phase 3).

import numpy as np
import pandas as pd

from backtest.constants import INITIAL_CAPITAL


class MultiPortfolio:
    """Cash + a basket of single-name positions ({ticker: shares}, only non-zero kept)."""

    def __init__(self, cash):
        self.cash = float(cash)
        self.shares = {}

    def equity(self, prices):
        """Mark to market: cash + sum(shares * price) over held names. `prices` is a
        Series (ticker -> price) for the current bar; held names always have a price."""
        v = self.cash
        for t, sh in self.shares.items():
            v += sh * prices[t]
        return v

    def rebalance(self, target_weights, prices, cost=None):
        """Trade so each name becomes its target fraction of current equity.

        target_weights: Series {ticker: weight}, weights >= 0 summing to <= 1.
        prices: Series of fill prices for this bar. Names with a NaN/<=0 fill price
        are skipped (can't trade them) rather than crashing. Returns total fee.
        """
        if (target_weights < 0).any():
            raise ValueError("negative target weight (long-only — no shorting yet)")
        total = float(target_weights.sum())
        if total > 1.0 + 1e-9:
            raise ValueError(f"target weights sum to {total:.4f} > 1 (no leverage)")

        eq = self.equity(prices)
        target_shares = {}
        for t, w in target_weights.items():
            if w <= 0:
                continue
            p = prices.get(t, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue                                   # untradeable this bar -> stays cash
            target_shares[t] = w * eq / p

        fee = 0.0
        for t in set(self.shares) | set(target_shares):
            delta = target_shares.get(t, 0.0) - self.shares.get(t, 0.0)
            if abs(delta) < 1e-12:
                continue
            p = prices[t]
            f = cost(delta, p) if cost else 0.0
            self.cash -= delta * p + f
            fee += f
            if t in target_shares:
                self.shares[t] = target_shares[t]
            else:
                self.shares.pop(t, None)
        return fee


def run_xs(panels, strategy, initial_capital=INITIAL_CAPITAL, cost=None, fill="next_open"):
    """Walk a price panel bar by bar applying a cross-sectional strategy; return the
    equity curve.

    panels: {"Close": (date x ticker) DataFrame, "Open": same} from backtest.universe.
    strategy: CrossSectionalStrategy — target_weights(closes, i) -> Series or None (hold).
    fill: "next_open" (honest, default) or "close" (validation/optimistic).
    """
    closes, opens = panels["Close"], panels["Open"]
    dates = closes.index
    n = len(dates)
    pf = MultiPortfolio(initial_capital)
    equity = np.empty(n)
    pending = None

    for i in range(n):
        if fill == "next_open":
            if pending is not None:                        # yesterday's decision...
                pf.rebalance(pending, opens.iloc[i], cost) # ...fills at today's open
                pending = None
            w = strategy.target_weights(closes, i)
            if w is not None:
                pending = w                                # schedule for next open
        elif fill == "close":
            w = strategy.target_weights(closes, i)
            if w is not None:
                pf.rebalance(w, closes.iloc[i], cost)
        else:
            raise ValueError(f"unknown fill mode {fill!r}")
        equity[i] = pf.equity(closes.iloc[i])              # mark at today's close

    return pd.Series(equity, index=dates, name="equity")
