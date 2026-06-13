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
