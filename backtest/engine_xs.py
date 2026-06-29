# backtest/engine_xs.py — cross-sectional (multi-asset) backtest engine.
#
# The Phase-3 sibling of engine.py. Same honest discipline — next-open fill, mark to
# market each bar, look-ahead provable by the "corrupt the future" test — but the
# portfolio now holds a BASKET: cash + shares across N tickers, and the strategy
# returns a weight per ticker instead of a scalar. engine.py (single asset) stays as
# the simple, hardened reference; this is its own module because the data shape (a
# date x ticker panel) and the weight interface genuinely differ.
#
# LONG-ONLY by default (weights >= 0, sum <= 1 — the rest is cash); a stray negative
# weight from a book that's supposed to be long-only still raises, which catches bugs.
# Pass allow_short=True to opt into LONG-SHORT: negative weights become short
# positions, and the [0,1] guard is replaced by a GROSS-exposure cap (sum|w| <=
# gross_max, default 2.0 = 100% long / 100% short, i.e. dollar-neutral). The signed
# accounting is the SAME formula — a short just has negative shares: it adds cash when
# opened and gains when the price falls. Shorts also accrue a borrow cost each bar
# (borrow_bps), because borrowing shares isn't free — especially for the small-caps
# this lab screens, where names can be hard or impossible to borrow.

import numpy as np
import pandas as pd

from backtest.constants import INITIAL_CAPITAL, TRADING_DAYS


class MultiPortfolio:
    """Cash + a basket of single-name positions ({ticker: shares}, only non-zero kept).
    Shares may be negative (a short) when allow_short=True."""

    def __init__(self, cash, allow_short=False, gross_max=None):
        self.cash = float(cash)
        self.shares = {}
        self.allow_short = allow_short
        # gross cap only bites in short mode; default 2.0 there (dollar-neutral), 1.0 long-only
        self.gross_max = gross_max if gross_max is not None else (2.0 if allow_short else 1.0)
        self._last_px = {}                                 # last finite price seen per name
        self._entry_px = {}                                # fill price each position was (re)set at

    def equity(self, prices):
        """Mark to market: cash + sum(shares * price) over held names (signed). `prices`
        is a Series (ticker -> price) for the current bar. If a held name has no price
        this bar (delisted/halted mid-hold), it's marked at its last finite price rather
        than silently poisoning equity with NaN — a held name was bought at a real price,
        so a carried mark always exists."""
        v = self.cash
        for t, sh in self.shares.items():
            px = prices.get(t, np.nan)
            if np.isfinite(px):
                self._last_px[t] = px
            else:
                px = self._last_px.get(t, np.nan)          # carry forward
            v += sh * px
        return v

    def accrue_borrow(self, prices, daily_rate):
        """Charge the daily borrow fee on SHORT notional and deduct it from cash. Longs
        are free; only sh < 0 pays. Returns the dollar charge."""
        if daily_rate <= 0:
            return 0.0
        charge = 0.0
        for t, sh in self.shares.items():
            if sh < 0:
                px = prices.get(t, np.nan)
                if not np.isfinite(px):
                    px = self._last_px.get(t, np.nan)
                if np.isfinite(px):
                    charge += abs(sh * px) * daily_rate
        self.cash -= charge
        return charge

    def rebalance(self, target_weights, prices, cost=None):
        """Trade so each name becomes its target fraction of current equity.

        target_weights: Series {ticker: weight}. Long-only (default): weights >= 0
        summing to <= 1. Short mode (allow_short): any sign, with sum|weight| <=
        gross_max. prices: Series of fill prices for this bar; names with a NaN/<=0 fill
        price are skipped (can't trade them) rather than crashing. Returns total fee.
        """
        if not self.allow_short:
            if (target_weights < 0).any():
                raise ValueError("negative target weight (long-only book — pass allow_short=True to short)")
            total = float(target_weights.sum())
            if total > self.gross_max + 1e-9:
                raise ValueError(f"target weights sum to {total:.4f} > {self.gross_max} (over-leveraged)")
        else:
            gross = float(target_weights.abs().sum())
            if gross > self.gross_max + 1e-9:
                raise ValueError(f"gross exposure {gross:.4f} > gross_max {self.gross_max} (over-leveraged)")

        eq = self.equity(prices)
        target_shares = {}
        for t, w in target_weights.items():
            if w == 0:
                continue                                   # exact-zero weight -> close (handled below)
            p = prices.get(t, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue                                   # untradeable this bar -> position unchanged
            target_shares[t] = w * eq / p                  # w < 0 -> negative shares (short)

        fee = 0.0
        for t in set(self.shares) | set(target_shares):
            delta = target_shares.get(t, 0.0) - self.shares.get(t, 0.0)
            if abs(delta) < 1e-12:
                continue
            p = prices.get(t, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue                                   # can't trade an unpriceable (delisted)
                                                           # name -> leave the position frozen
            f = cost(delta, p) if cost else 0.0
            self.cash -= delta * p + f                     # signed: buying spends, shorting adds cash
            fee += f
            if t in target_shares:
                self.shares[t] = target_shares[t]
                self._last_px[t] = p                       # seed carry-forward from the fill price
                self._entry_px[t] = p                      # stop-loss reference (this fill)
            else:
                self.shares.pop(t, None)
                self._entry_px.pop(t, None)
        return fee

    def apply_stops(self, prices, stop_loss, cost=None):
        """Daily risk overlay: sell any LONG that has fallen >= `stop_loss` (fraction, e.g.
        0.20) below its entry/fill price — to CASH, at the current price. Once stopped, the
        name stays out until the strategy re-buys it at the next rebalance. Long-only (a value
        book); shorts are left alone. Returns total fee. Models gap-through loosely: the fill
        is the current price, which on a trigger bar is already at/below the stop level."""
        if not stop_loss:
            return 0.0
        fee = 0.0
        for t, sh in list(self.shares.items()):
            if sh <= 0:
                continue
            p = prices.get(t, np.nan)
            entry = self._entry_px.get(t, np.nan)
            if not (np.isfinite(p) and p > 0 and np.isfinite(entry) and entry > 0):
                continue
            if p <= entry * (1.0 - stop_loss):
                f = cost(-sh, p) if cost else 0.0
                self.cash += sh * p - f                    # liquidate the position to cash
                fee += f
                self.shares.pop(t, None)
                self._entry_px.pop(t, None)
        return fee


def run_xs(panels, strategy, initial_capital=INITIAL_CAPITAL, cost=None, fill="next_open",
           allow_short=False, gross_max=None, borrow_bps=0.0, stop_loss=None,
           leverage=1.0, financing_bps=0.0):
    """Walk a price panel bar by bar applying a cross-sectional strategy; return the
    equity curve.

    panels: {"Close": (date x ticker) DataFrame, "Open": same} from backtest.universe.
    strategy: CrossSectionalStrategy — target_weights(closes, i) -> Series or None (hold).
    fill: "next_open" (honest, default) or "close" (validation/optimistic).
    allow_short / gross_max: enable long-short and cap gross exposure (see MultiPortfolio).
    borrow_bps: annual borrow cost (basis points) charged daily on short notional.
    stop_loss: optional fraction (e.g. 0.20). When set, each bar any LONG down >= stop_loss
        from its entry is sold to cash (checked at the close), held out until the next rebalance.
    leverage: long-only gross cap (>1 borrows to buy more than equity). The strategy must emit
        weights summing to <= leverage. financing_bps: annual interest (bps) on borrowed cash
        (charged daily on negative cash) — leverage isn't free.
    """
    closes, opens = panels["Close"], panels["Open"]
    dates = closes.index
    n = len(dates)
    eff_gross = gross_max if gross_max is not None else (leverage if not allow_short else None)
    pf = MultiPortfolio(initial_capital, allow_short=allow_short, gross_max=eff_gross)
    daily_borrow = (borrow_bps / 10_000.0) / TRADING_DAYS if borrow_bps else 0.0
    daily_fin = (financing_bps / 10_000.0) / TRADING_DAYS if financing_bps else 0.0
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
        if stop_loss:
            pf.apply_stops(closes.iloc[i], stop_loss, cost) # daily stop check at the close
        if daily_borrow:
            pf.accrue_borrow(closes.iloc[i], daily_borrow)  # holding cost on shorts, on today's book
        if daily_fin and pf.cash < 0:
            pf.cash += pf.cash * daily_fin                  # interest on borrowed cash (cash < 0)
        equity[i] = pf.equity(closes.iloc[i])              # mark at today's close

    return pd.Series(equity, index=dates, name="equity")
