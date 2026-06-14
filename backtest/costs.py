# backtest/costs.py — transaction cost models.
#
# A cost model is a callable (delta_shares, price) -> dollar fee, matching the
# engine's `cost=` hook. v1 folds commission + half-spread + slippage into ONE
# knob: a fixed number of basis points charged on the traded notional,
# |delta_shares| * price. Cheap for liquid names (SPY ~2 bps), much higher for
# small-caps (crank it up). Modeling slippage as a separate price impact can come
# later if the realism is needed; as a P&L drag, a bps fee is equivalent for now.

DEFAULT_BPS = 2.0  # SPY: ~zero commission + a penny-wide spread


def proportional(bps=DEFAULT_BPS):
    """Return a cost function charging `bps` basis points of traded notional.

    bps=2 -> 0.02% of |delta_shares * price| per trade. A no-op trade (delta=0)
    costs nothing, so holding a position incurs no ongoing fee.
    """
    rate = bps / 10_000.0

    def cost(delta_shares, price):
        return abs(delta_shares) * price * rate

    return cost
