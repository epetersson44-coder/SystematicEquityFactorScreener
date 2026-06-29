# backtest/trend_sleeve.py — cross-asset time-series-momentum (trend) sleeve + SPY blend.
#
# Research-plan Step 4: stop forcing momentum into the small-cap equity bucket; test trend as a
# SEPARATE, genuinely uncorrelated diversifier. This is the one path the evidence (and our own
# blend test) says can lift a PORTFOLIO's risk-adjusted return — not by picking better stocks,
# but by holding something that rises when equities crash (crisis alpha; Moskowitz-Ooi-Pedersen
# "Time Series Momentum", Hurst-Ooi-Pedersen "A Century of Evidence on Trend-Following").
#
# Retail-accessible build: liquid ETFs across 4 asset classes (US + intl equity, long + mid
# Treasuries, gold, broad commodities), long-FLAT 12-month time-series momentum, monthly. Each
# instrument is held at 1/N when its trailing 12-month return is positive, else that slice sits
# in CASH. The crisis-alpha mechanism: in an equity crash the safe-haven legs (bonds, gold)
# trend UP, so the sleeve rotates into them while equities go to cash -> positive return exactly
# when SPY is most negative -> low/negative correlation.
#
# Tested over the FULL ETF history (~2007+, so it INCLUDES the 2008 GFC, 2020 COVID, 2022 bear)
# — testing trend only over the 2011-2026 SPY bull would understate it (SPY's Sharpe was
# abnormally high and trend had a weak decade). Window choice is itself part of the lesson.

import os

import numpy as np
import pandas as pd

from backtest.data import CACHE_DIR
from backtest.universe import download_panel
from backtest.engine_xs import run_xs
from backtest.strategy import CrossSectionalStrategy
from backtest import costs, metrics

ETFS = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]    # US eq, intl eq, long UST, mid UST, gold, commodities
TREND_START = "2006-07-01"                            # earliest all 6 ETFs have data (DBC ~Feb 2006)


def etf_panel(tickers=ETFS, refresh=False):
    """{'Close','Open'} (date x ETF) total-return-adjusted, from yfinance, cached."""
    paths = {f: os.path.join(CACHE_DIR, f"trend_{f}.csv") for f in ("Close", "Open")}
    if not refresh and all(os.path.exists(p) for p in paths.values()):
        return {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    panels = download_panel(tickers, fields=("Close", "Open"), start=TREND_START)
    for f, p in panels.items():
        p.to_csv(paths[f])
    return panels


class TSMOM(CrossSectionalStrategy):
    """Long-flat 12-month time-series momentum, rebalanced monthly. Hold each instrument at
    1/n_universe when its trailing `look`-day return is positive, else 0 (cash). Fully invested
    when all instruments trend up; rotates to cash as they roll over (the defensive profile)."""
    def __init__(self, look=252, n_universe=len(ETFS), every=21):
        self.look, self.n, self.every = look, n_universe, every

    def target_weights(self, closes, i):
        if i < self.look or i % self.every != 0:
            return None
        row, past = closes.iloc[i], closes.iloc[i - self.look]
        w = {}
        for t in closes.columns:
            p0, pm = row.get(t), past.get(t)
            if p0 and pm and np.isfinite(p0) and np.isfinite(pm) and p0 / pm - 1 > 0:
                w[t] = 1.0 / self.n
        return pd.Series(w, dtype=float)             # empty Series -> all cash (sells everything)


def run_trend(cost_bps=5, panels=None):
    """Equity curve of the cross-asset trend sleeve (cheap ETF costs)."""
    panels = panels or etf_panel()
    return run_xs(panels, TSMOM(), cost=costs.proportional(cost_bps), fill="next_open")


def _ann_stats(eq):
    return {"cagr": metrics.cagr(eq), "sharpe": metrics.sharpe(eq), "maxdd": metrics.max_drawdown(eq)}


def analyze(cost_bps=5, start=None):
    """Trend sleeve vs SPY buy-hold, plus pre-committed SPY+trend blends (constant-mix on daily
    returns). Returns (stats_dict, curves_dict). `start` slices the window (None = full ~2007+)."""
    panels = etf_panel()
    trend_eq = run_trend(cost_bps, panels)
    spy = panels["Close"]["SPY"].dropna()
    spy_eq = 10_000 * spy / spy.iloc[0]
    df = pd.DataFrame({"SPY": spy_eq, "trend": trend_eq}).dropna()
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    df = 10_000 * df / df.iloc[0]
    rets = df.pct_change().dropna()
    corr = float(rets["SPY"].corr(rets["trend"]))

    def blend(ws):
        br = sum(w * rets[c] for c, w in ws.items())
        eq = (1 + br).cumprod()
        return _ann_stats(eq), eq
    iv_s, iv_t = 1 / rets["SPY"].std(), 1 / rets["trend"].std()
    blends = {
        "SPY 100%": {"SPY": 1.0},
        "trend 100%": {"trend": 1.0},
        "60/40 SPY+trend": {"SPY": 0.6, "trend": 0.4},
        "50/50 SPY+trend": {"SPY": 0.5, "trend": 0.5},
        "risk-parity SPY+trend": {"SPY": iv_s / (iv_s + iv_t), "trend": iv_t / (iv_s + iv_t)},
    }
    stats, curves = {}, {}
    for name, ws in blends.items():
        st, eq = blend(ws)
        stats[name] = {**st, "w": {k: round(v, 2) for k, v in ws.items()}}
        curves[name] = 10_000 * eq / eq.iloc[0]
    stats["_corr_trend_spy"] = round(corr, 2)
    stats["_window"] = (str(df.index[0].date()), str(df.index[-1].date()))
    return stats, curves
