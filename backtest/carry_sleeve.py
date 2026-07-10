# backtest/carry_sleeve.py — cross-asset CARRY sleeve (the one remaining frontier-pusher).
#
# Carry = the return you earn for simply HOLDING an asset if its price doesn't move — its
# yield/roll (Koijen-Moskowitz-Pedersen-Vrugt, "Carry", 2018). It's a distinct premium from
# trend and value, and crucially it tends to be NEGATIVELY related to trend in crises: carry
# is a short-volatility, "reach-for-yield" bet that crashes exactly when trend (long safe
# havens) pays off. So carry + trend is the classic diversifying pair — each covers the other's
# weakness. The point of this sleeve is to add a genuinely new RETURN stream and test whether
# it pushes the SPY+trend frontier OUTWARD (not just slides along it).
#
# Honest data limit: with free ETF data the cleanest point-in-time carry proxy is each ETF's
# TRAILING-12-MONTH DISTRIBUTION YIELD (dividends already paid / price). That captures BOND and
# EQUITY carry well; it MISSES commodity roll-yield (backwardation/contango) and FX rate-
# differential carry, which need futures-curve / short-rate data we don't have free. So this is
# a fixed-income+equity carry sleeve, not full cross-asset carry — a defensible start.
#
# Construction (standard): each month weight the positive-carry ETFs by carry-per-unit-vol
# (risk-adjusted carry), vol-target the sleeve, cap leverage. Trailing dividends are known as
# of the date -> point-in-time clean, no look-ahead.

import os

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.data import CACHE_DIR
from backtest.trend_sleeve import etf_panel, ETFS
from backtest.engine_xs import run_xs
from backtest.strategy import CrossSectionalStrategy
from backtest import costs, metrics


def trailing_yield_panel(tickers=ETFS, refresh=False):
    """Trailing-12-month distribution yield per ETF (the carry proxy), aligned to the price
    panel's dates. Point-in-time: only dividends already paid by each date are counted."""
    # cache keyed by universe — one unkeyed file silently served the default universe
    # to any caller with different tickers (the stale-universe class etf_panel guards
    # against; ninth review, F10 — carry is closed [POLICY], fixed for if it reopens)
    path = os.path.join(CACHE_DIR, f"carry_yield_{'_'.join(sorted(tickers))}.csv")
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    close = etf_panel(tickers)["Close"]
    yld = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for t in close.columns:                                 # only ETFs actually in the panel
        try:
            div = yf.Ticker(t).dividends
        except Exception:
            div = pd.Series(dtype=float)
        if len(div):
            div.index = pd.to_datetime(div.index).tz_localize(None)
            daily = pd.Series(0.0, index=close.index)
            for dt, amt in div.items():
                fwd = close.index[close.index >= dt]            # post to first trading day on/after
                if len(fwd):
                    daily.loc[fwd[0]] += float(amt)
            ttm = daily.rolling(252).sum()                      # ~12 months of trading days
            yld[t] = ttm / close[t]
    os.makedirs(CACHE_DIR, exist_ok=True)
    yld.to_csv(path)
    return yld


class Carry(CrossSectionalStrategy):
    """Cross-asset carry: monthly, long the positive-carry ETFs weighted by carry-per-unit-vol
    (risk-adjusted carry), normalized and scaled to a target portfolio vol (capped). Long-only
    (retail-implementable). `yld` is the trailing-yield panel (carry signal)."""
    def __init__(self, yld, vol_lb=63, target_vol=0.10, every=21, max_gross=2.0):
        self.yld, self.vol_lb, self.target_vol, self.every, self.max_gross = (
            yld, vol_lb, target_vol, every, max_gross)

    def target_weights(self, closes, i):
        if i < 252 or i % self.every != 0:
            return None
        d = closes.index[i]
        carry = self.yld.loc[d] if d in self.yld.index else pd.Series(dtype=float)
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        on, raw = [], {}
        for t in closes.columns:
            c, v = carry.get(t, np.nan), (rets[t].std() if t in rets else np.nan)
            if np.isfinite(c) and c > 0 and np.isfinite(v) and v > 0:
                raw[t] = c / (v * np.sqrt(252))                 # carry per unit of risk
                on.append(t)
        if not on:
            return pd.Series(dtype=float)
        w = pd.Series(raw)
        w = w / w.sum()
        cov = rets[on].cov() * 252
        pvol = float(np.sqrt(w.values @ cov.values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


def run_carry(cost_bps=5, target_vol=0.10, max_gross=2.0, financing_bps=400):
    """Equity curve of the carry sleeve (may lever to target vol -> financing on borrow)."""
    panels = etf_panel()
    yld = trailing_yield_panel()
    strat = Carry(yld, target_vol=target_vol, max_gross=max_gross)
    return run_xs(panels, strat, cost=costs.proportional(cost_bps), fill="next_open",
                  leverage=max_gross, financing_bps=financing_bps)
