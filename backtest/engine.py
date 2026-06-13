# backtest/engine.py — event-driven backtest engine.
#
# The loop walks bars in time order and shows the strategy ONLY the history up to
# and including the current bar, so look-ahead is structurally impossible — not a
# rule to remember, a thing the code physically can't do. A `Portfolio` holds
# cash + shares and marks to market each bar to build the equity curve.
#
# Fill timing is the one real correctness knob:
#   "next_open" (default) — decide on today's close, fill at tomorrow's open.
#                           Honest: never trade at a price you couldn't have known.
#   "close"               — decide and fill at the same bar's close. Optimistic
#                           (mild look-ahead); used to validate accounting against
#                           the analytic buy-and-hold, and as a best-case bound.

import numpy as np
import pandas as pd


class Portfolio:
    """Cash + a single asset position. (Multi-asset arrives with the Phase 3 universe.)"""

    def __init__(self, cash):
        self.cash = float(cash)
        self.shares = 0.0

    def equity(self, price):
        """Total mark-to-market value at `price`."""
        return self.cash + self.shares * price

    def rebalance(self, target_weight, price, cost=None):
        """Trade so the asset becomes `target_weight` of current equity (0..1).

        `cost` is an optional callable (delta_shares, price) -> dollar fee.
        Returns (delta_shares, fee).
        """
        eq = self.equity(price)
        target_shares = target_weight * eq / price
        delta = target_shares - self.shares
        fee = cost(delta, price) if cost else 0.0
        self.cash -= delta * price + fee
        self.shares = target_shares
        return delta, fee


def run(prices, strategy, initial_capital=10_000, cost=None, fill="next_open"):
    """Walk `prices` bar by bar applying `strategy`; return the equity curve.

    prices: DataFrame with Open/Close indexed by date (from backtest.data).
    strategy: object with target_weight(history) -> weight in [0, 1].
    cost: optional callable (delta_shares, price) -> dollar fee (None = free).
    fill: "next_open" (honest, default) or "close" (optimistic, for validation).
    """
    pf = Portfolio(initial_capital)
    opens = prices["Open"].to_numpy()
    closes = prices["Close"].to_numpy()
    n = len(prices)
    equity = np.empty(n)
    pending = None

    for i in range(n):
        if fill == "next_open":
            if pending is not None:                        # yesterday's decision...
                pf.rebalance(pending, opens[i], cost)      # ...fills at today's open
            pending = strategy.target_weight(prices.iloc[:i + 1])
        elif fill == "close":
            w = strategy.target_weight(prices.iloc[:i + 1])
            pf.rebalance(w, closes[i], cost)
        else:
            raise ValueError(f"unknown fill mode {fill!r}")
        equity[i] = pf.equity(closes[i])                   # mark at today's close

    return pd.Series(equity, index=prices.index, name="equity")


if __name__ == "__main__":
    from backtest.data import get_prices
    from backtest.strategy import BuyAndHold
    from backtest import baseline

    spy = get_prices("SPY", start="2000-01-01")

    analytic = baseline.buy_and_hold(spy)                    # closed-form benchmark
    engine_close = run(spy, BuyAndHold(), fill="close")      # must match it exactly
    engine_open = run(spy, BuyAndHold(), fill="next_open")   # honest entry

    diff = np.max(np.abs(analytic.to_numpy() - engine_close.to_numpy()))
    print(f"engine(fill=close) vs analytic baseline -> match={np.allclose(analytic.to_numpy(), engine_close.to_numpy())}, max abs diff={diff:.2e}\n")

    baseline.print_summary("engine buy&hold (fill=close)", engine_close)
    baseline.print_summary("engine buy&hold (fill=next_open)", engine_open)
