# backtest/momentum_ls.py — momentum long-short, harvested harder, with crash protection.
#
# The factor analysis (factor_analysis.py) found momentum is the real edge (IC t=3.91, L/S
# Sharpe ~1.0) while value/quality is dead. This harvests momentum properly — and, because
# momentum's signature risk is the rare VIOLENT CRASH (2009: the short side, prior losers,
# rips +100% in weeks), it ships with the crash guard built in, not bolted on later.
#
# Failsafe 1 — VOLATILITY MANAGEMENT (Barroso & Santa-Clara 2015): scale exposure to a
#   target vol using the strategy's OWN trailing realized vol (lagged, so look-ahead-free).
#   Momentum crashes are preceded by vol spikes, so de-levering on high vol dodges them.
# Failsafe 2 — MARKET TREND FILTER: hold only when the market is above its 200-day average;
#   flat otherwise. Momentum crashes cluster in the rebound after a market decline.
#
# Tested on the S&P panel back to 2005 — which CONTAINS the real 2009 crash — so the
# failsafe is demonstrated, not assumed. SURVIVORSHIP CAVEAT: that panel is today's
# constituents, so the short side misses delisted losers; absolute numbers are inflated,
# but the crash mechanics + failsafe behaviour are real. (The survivorship-free SimFin
# data only spans 2020-2025 — no crash to test against.)
#
# Run:  python -m backtest.momentum_ls

import numpy as np
import pandas as pd

from backtest.universe import get_universe
from backtest.data import get_prices
from backtest.engine_xs import run_xs
from backtest.strategy import CrossSectionalStrategy
from backtest import metrics, costs
from backtest.constants import TRADING_DAYS, INITIAL_CAPITAL


class MomentumLS(CrossSectionalStrategy):
    """Dollar-neutral cross-sectional momentum: long the top `decile`, short the bottom
    `decile` by 12-1 momentum (trailing 12m return, skipping the last month), rebalanced
    monthly. Gross 1.0 (±0.5 per side) before any vol overlay."""

    def __init__(self, lookback=252, skip=21, decile=0.1):
        self.lookback, self.skip, self.decile = lookback, skip, decile

    def target_weights(self, closes, i):
        dates = closes.index
        if i == 0 or dates[i].month == dates[i - 1].month:
            return None                                      # rebalance on month-starts only
        if i < self.lookback + self.skip:
            return None
        mom = (closes.iloc[i - self.skip] / closes.iloc[i - self.skip - self.lookback] - 1).dropna()
        mom = mom[mom.index.isin(closes.iloc[i].dropna().index)]   # must be tradeable today
        if len(mom) < 20:
            return None
        n = max(1, int(len(mom) * self.decile))
        win, los = mom.nlargest(n).index, mom.nsmallest(n).index
        w = pd.Series(0.0, index=win.union(los))
        w[win] = 0.5 / n                                     # long top decile
        w[los] = -0.5 / n                                    # short bottom decile
        return w


def run_raw(panel, cost_bps=10, borrow_bps=50):
    """Raw momentum L/S equity — next-open fill, transaction cost + short borrow modelled."""
    return run_xs(panel, MomentumLS(), cost=costs.proportional(cost_bps),
                  allow_short=True, gross_max=1.0, borrow_bps=borrow_bps, fill="next_open")


def vol_managed(equity, target_vol=0.10, window=126, max_leverage=2.0):
    """FAILSAFE 1 (Barroso & Santa-Clara). Scale each day's exposure toward `target_vol`
    using the strategy's OWN trailing realized vol, LAGGED one day (look-ahead-free). High
    vol -> de-lever -> dodge the crash; calm -> lever up to `max_leverage`. NOTE: applied at
    the return level, so it doesn't re-model the extra cost of the leverage — conservative
    for the crash claim (the failsafe REDUCES exposure exactly when it matters)."""
    ret = metrics._daily_returns(equity)
    realized = ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    lev = (target_vol / realized).shift(1).clip(upper=max_leverage).fillna(0.0)
    return INITIAL_CAPITAL * (1 + ret * lev).cumprod()


def trend_filtered(equity, market, ma=200):
    """FAILSAFE 2: hold the book only when `market` is above its `ma`-day average, else flat.
    Lagged to avoid look-ahead. Momentum crashes cluster in the post-decline rebound; staying
    flat while the market is below its average sidesteps that window."""
    ret = metrics._daily_returns(equity)
    sma = market.rolling(ma).mean()
    # 1.0 when market > its MA, lagged a day; on float (not bool) to avoid downcast warnings
    risk_on = (market > sma).astype(float).shift(1).reindex(ret.index).ffill().fillna(0.0) > 0.5
    return INITIAL_CAPITAL * (1 + ret.where(risk_on, 0.0)).cumprod()


def _row(name, eq):
    return {"strategy": name,
            "CAGR_%": round(metrics.cagr(eq) * 100, 1),
            "vol_%": round(metrics.annualized_volatility(eq) * 100, 1),
            "Sharpe": round(metrics.sharpe(eq), 2),
            "maxDD_%": round(metrics.max_drawdown(eq) * 100, 0),
            "DD_days": int(metrics.max_drawdown_duration(eq)),
            "final_$": round(float(eq.iloc[-1]))}


def run_comparison(start="2005-01-01"):
    """Backtest raw vs vol-managed vs trend-filtered vs both, over 2005-2026 (contains the
    2009 momentum crash). Returns (rows, curves dict)."""
    panel = get_universe("sp500", start=start)
    market = get_prices("SPY")["Close"]
    raw = run_raw(panel)
    vm = vol_managed(raw)
    tf = trend_filtered(raw, market)
    both = trend_filtered(vm, market)
    curves = {"raw momentum L/S": raw, "+ vol-managed (failsafe 1)": vm,
              "+ trend filter (failsafe 2)": tf, "+ both failsafes": both}
    rows = [_row(n, e) for n, e in curves.items()]
    return rows, curves


def _crash_return(eq, lo="2009-02-01", hi="2009-06-30"):
    seg = eq[(eq.index >= lo) & (eq.index <= hi)]
    return (seg.iloc[-1] / seg.iloc[0] - 1) * 100 if len(seg) > 1 else float("nan")


if __name__ == "__main__":
    rows, curves = run_comparison()
    print("\n=== Momentum long-short, harvested harder (S&P, 2005-2026, cost+borrow) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== The 2009 momentum crash (Feb-Jun 2009) — does the failsafe work? ===")
    for n, e in curves.items():
        print(f"  {n:<32} {_crash_return(e):+6.1f}%")
    print("\n  SURVIVORSHIP-INFLATED (today's S&P constituents); the crash mechanics + failsafe")
    print("  behaviour are real, the absolute returns are not. Vol-managed leverage cost not re-modelled.")
