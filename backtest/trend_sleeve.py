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

# Clean cross-asset universe: US + dev-intl equity, long + mid Treasuries, gold, commodities.
# NOTE (tested 2026-06-28): expanding to 10 (adding EEM/LQD/HYG/UUP) actually HURT the trend
# sleeve (Sharpe 0.71 -> 0.59) — EM equity, credit, and the dollar trend less cleanly and add
# noise, not breadth. More instruments is not better here; these 6 are the keepers.
ETFS = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]
TREND_START = "2006-07-01"


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


class VolTargetTSMOM(CrossSectionalStrategy):
    """Vol-targeted cross-asset trend (the standard managed-futures construction). Monthly: hold
    instruments with positive 12-month momentum, weight them INVERSE to recent volatility (equal
    risk per bet = risk parity among the 'on' set), then scale the whole sleeve to a TARGET
    annualized portfolio vol (estimated from the recent covariance), capped at `max_gross`. So
    the sleeve runs hot when many uncorrelated trends are calm and dials down when vol spikes or
    trends roll over — the mechanism behind trend's smooth risk profile."""
    def __init__(self, look=252, vol_lb=63, target_vol=0.10, every=21, max_gross=2.0,
                 long_short=False):
        self.look, self.vol_lb, self.target_vol, self.every, self.max_gross = (
            look, vol_lb, target_vol, every, max_gross)
        self.long_short = long_short                     # True: SHORT down-trending assets too

    def target_weights(self, closes, i):
        if i < self.look or i % self.every != 0:
            return None
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        sign = {}
        for t in closes.columns:
            p0, pm = closes.iloc[i].get(t), closes.iloc[i - self.look].get(t)
            v = rets[t].std() if t in rets else np.nan
            if not (p0 and pm and np.isfinite(p0) and np.isfinite(pm) and np.isfinite(v) and v > 0):
                continue
            mom = p0 / pm - 1
            if mom > 0:
                sign[t] = 1.0
            elif self.long_short and mom < 0:
                sign[t] = -1.0                           # short the downtrend (managed-futures style)
        if not sign:
            return pd.Series(dtype=float)                # all cash
        on = list(sign)
        invvol = 1.0 / (rets[on].std() * np.sqrt(252))   # inverse-vol risk weights, signed
        w = pd.Series({t: sign[t] * invvol[t] for t in on})
        w = w / w.abs().sum()                            # normalize GROSS to 1
        cov = rets[on].cov() * 252
        pvol = float(np.sqrt(w.values @ cov.values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


def run_trend(cost_bps=5, panels=None, vol_target=True, target_vol=0.10, max_gross=2.0,
              financing_bps=400, long_short=False, borrow_bps=50):
    """Equity curve of the cross-asset trend sleeve. vol_target=True uses the vol-targeted
    managed-futures construction (may lever to hit target_vol → financing on borrowed cash).
    long_short=True shorts down-trending assets (real managed-futures profile → stronger
    crisis alpha), charging borrow on the short legs."""
    panels = panels or etf_panel()
    if vol_target:
        strat = VolTargetTSMOM(target_vol=target_vol, max_gross=max_gross, long_short=long_short)
        return run_xs(panels, strat, cost=costs.proportional(cost_bps), fill="next_open",
                      allow_short=long_short, gross_max=max_gross,
                      leverage=(1.0 if long_short else max_gross),
                      financing_bps=(0 if long_short else financing_bps),
                      borrow_bps=(borrow_bps if long_short else 0.0))
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

    # #1: lever the best-Sharpe blend (risk-parity) up to SPY's volatility — the apples-to-apples
    # "same risk, whose return wins?" test. Financing (~4%/yr) charged on the borrowed cash.
    rp = blends["risk-parity SPY+trend"]
    rp_ret = sum(w * rets[c] for c, w in rp.items())
    spy_vol, rp_vol = rets["SPY"].std() * np.sqrt(252), rp_ret.std() * np.sqrt(252)
    L = spy_vol / rp_vol if rp_vol > 0 else 1.0
    lev_ret = L * rp_ret - (L - 1) * (400 / 10_000) / 252
    lev_eq = (1 + lev_ret).cumprod()
    label = f"risk-parity LEVERED {L:.1f}x (=SPY vol)"
    stats[label] = {**_ann_stats(lev_eq), "w": {"blend": round(L, 2)}}
    curves[label] = 10_000 * lev_eq / lev_eq.iloc[0]

    stats["_corr_trend_spy"] = round(corr, 2)
    stats["_window"] = (str(df.index[0].date()), str(df.index[-1].date()))
    stats["_levered_label"] = label
    return stats, curves
