# backtest/strategy.py — trading strategies.
#
# Each strategy returns a TARGET WEIGHT: the fraction of equity to hold in the
# asset, 0 (flat) to 1 (fully invested). Weight is the general form — it handles
# sizing and rebalancing, and a +1/0 signal is just the binary special case.


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
