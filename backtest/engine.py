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
#
# Idle cash can earn a risk-free rate (`cash_rate`). Default 0 keeps the engine's
# behaviour — and the analytic-baseline validation — unchanged. Set it (e.g. 0.04)
# for an honest comparison: a strategy that sits in cash through bear markets
# (like the SMA) really would collect T-bill interest while it waits; scoring that
# cash at 0% understates it. Pair cash_rate with the same rf in metrics.sharpe().

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
        if not 0.0 <= target_weight <= 1.0:
            raise ValueError(
                f"target_weight must be in [0, 1] (long-only, no leverage/short); got {target_weight}"
            )
        eq = self.equity(price)
        target_shares = target_weight * eq / price
        delta = target_shares - self.shares
        fee = cost(delta, price) if cost else 0.0
        self.cash -= delta * price + fee
        self.shares = target_shares
        return delta, fee


def run(prices, strategy, initial_capital=10_000, cost=None, fill="next_open",
        cash_rate=0.0, periods_per_year=252):
    """Walk `prices` bar by bar applying `strategy`; return the equity curve.

    prices: DataFrame with Open/Close indexed by date (from backtest.data).
    strategy: object with target_weight(history) -> weight in [0, 1].
    cost: optional callable (delta_shares, price) -> dollar fee (None = free).
    fill: "next_open" (honest, default) or "close" (optimistic, for validation).
    cash_rate: annual risk-free rate earned on idle cash (default 0 = no interest,
        which leaves the analytic-baseline validation exact). Accrued daily.
    """
    pf = Portfolio(initial_capital)
    opens = prices["Open"].to_numpy()
    closes = prices["Close"].to_numpy()
    n = len(prices)
    equity = np.empty(n)
    pending = None
    daily_cash_factor = 1.0 + cash_rate / periods_per_year

    for i in range(n):
        if cash_rate:                                      # one day passes: idle
            pf.cash *= daily_cash_factor                   # cash earns the rf rate
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
