# backtest/strategy.py — trading strategies.
#
# Two families:
#  - single-asset (Strategy): target_weight(history) -> scalar 0..1, run by engine.run
#  - cross-sectional (CrossSectionalStrategy): target_weights(closes, i) -> Series of
#    per-ticker weights, run by engine_xs.run_xs over a price panel (Phase 3+).
#
# A target weight is the fraction of equity to hold; weight is the general form —
# it handles sizing and rebalancing, and a +1/0 signal is just the binary case.

import pandas as pd


class Strategy:
    """Base class. Override target_weight(history) -> weight in [0, 1].

    `history` is the price frame up to AND including the current bar. The engine
    never hands you future bars, so look-ahead isn't possible from in here.
    """

    def target_weight(self, history):
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Fully invested, always. The benchmark expressed as a strategy."""

    def target_weight(self, history):
        return 1.0


class SMACrossover(Strategy):
    """Trend following via moving-average crossover (the "golden/death cross").

    Fully invested when the fast SMA is above the slow SMA, flat (in cash) when
    below. Windows are 50/200 by DESIGN and left untuned — optimizing them, and
    watching that optimization fall apart out-of-sample, is the Phase 2 lesson.

    Until there are `slow` bars of history the slow SMA is undefined, so the
    strategy stays flat through the warmup rather than guessing.
    """

    def __init__(self, fast=50, slow=200):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow")
        self.fast = fast
        self.slow = slow

    def target_weight(self, history):
        close = history["Close"]
        if len(close) < self.slow:
            return 0.0
        fast_ma = close.iloc[-self.fast:].mean()
        slow_ma = close.iloc[-self.slow:].mean()
        return 1.0 if fast_ma > slow_ma else 0.0


class WalkForwardSMA(Strategy):
    """Execute a walk-forward schedule: in each out-of-sample test window, trade the
    SMA pair that was optimized on the train window *before* it.

    `schedule` is a time-ordered list of (test_start, test_end, fast, slow). The
    params for a window are chosen only from data preceding it (built by
    optimize.walk_forward_schedule), so this never trades on a parameter it couldn't
    have known — the honest, out-of-sample version of an optimized strategy. Flat on
    any date outside every window (e.g. the initial train period).
    """

    def __init__(self, schedule):
        self.schedule = schedule

    def target_weight(self, history):
        date = history.index[-1]
        for start, end, fast, slow in self.schedule:
            if start <= date <= end:
                return SMACrossover(fast, slow).target_weight(history)
        return 0.0


# ----------------------------------------------------------- cross-sectional (Phase 3)
class CrossSectionalStrategy:
    """Base for strategies that rank a UNIVERSE and hold a basket.

    target_weights(closes, i) sees the full close panel and the current bar index i,
    and MUST use only rows [:i+1] (the look-ahead corruption test in the xs stress
    suite proves it doesn't peek). Return a Series {ticker: weight} (weights >= 0,
    sum <= 1) to rebalance to, or None to hold the current basket untouched — so a
    monthly strategy trades ~12x/yr, not every bar.
    """

    def target_weights(self, closes, i):
        raise NotImplementedError


class CrossSectionalMomentum(CrossSectionalStrategy):
    """Classic 12-1 cross-sectional momentum, rebalanced monthly.

    On the first trading bar of each month, rank every name by its return over the
    `lookback` window ending `skip` days ago (the skip drops the most recent month to
    sidestep short-term reversal — the standard "12 minus 1" construction), and hold
    the top `top` fraction, equal-weighted and fully invested. Flat until warmed up.
    """

    def __init__(self, lookback=252, skip=21, top=0.1):
        self.lookback = lookback        # ~12 months
        self.skip = skip                # ~1 month skipped (reversal guard)
        self.top = top                  # top decile by default

    def target_weights(self, closes, i):
        dates = closes.index
        if i == 0 or dates[i].month == dates[i - 1].month:
            return None                                  # not a new month -> hold
        return self.rank(closes, i)

    def rank(self, closes, i):
        """The momentum ranking for bar i, ignoring the monthly gate — returns the
        equal-weight top-`top` basket (or None if not enough history). Used by
        target_weights on month-starts, and by the live tracker to pick on demand."""
        if i < self.lookback + self.skip:
            return None                                  # not enough history yet
        p_end = closes.iloc[i - self.skip]               # price 1 month ago
        p_start = closes.iloc[i - self.skip - self.lookback]   # price 13 months ago
        momentum = (p_end / p_start - 1).dropna()        # pre-IPO names drop out
        tradable = closes.iloc[i].dropna().index         # must have a price to buy NOW
        momentum = momentum[momentum.index.isin(tradable)]
        if len(momentum) < 10:
            return None
        n_top = max(1, int(len(momentum) * self.top))
        winners = momentum.nlargest(n_top).index
        return pd.Series(1.0 / n_top, index=winners)     # equal weight, fully invested
