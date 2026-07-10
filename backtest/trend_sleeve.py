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

# The ADOPTED signal construction (2026-07-01 A/B, all-offset-swept so the comparison is
# timing-luck-controlled): the Moskowitz-Ooi-Pedersen 1/3/12-month ensemble DOMINATES the
# single 252d lookback — its WORST rebalance offset (blend Sharpe 0.875) beats the single
# look's MEDIAN (0.854); luck-free all-21 average 0.943 vs 0.869; maxDD -16% vs -18%;
# better in every crisis window (GFC +6.9% vs +5.3%, COVID -0.2% vs -1.8%, 2022 -6.4% vs
# -11.0%) and lower SPY corr (0.24 vs 0.29). Faster lookbacks exit fast crashes sooner.
# Adopted on distribution dominance, not a cherry-picked draw. The BIL cash-hurdle gate
# (hurdle=True) tested a WASH on this window (rf ~0 for half of it) — available, not default.
ENSEMBLE_LOOKS = (21, 63, 252)


def etf_panel(tickers=ETFS, refresh=False):
    """{'Close','Open'} (date x ETF) total-return-adjusted, from yfinance, cached.
    The default 6-ETF universe keeps its historical cache name; any other set gets its
    own keyed files so alternate universes can't silently overwrite the main cache."""
    key = "" if set(tickers) == set(ETFS) else "_" + "_".join(sorted(tickers))
    paths = {f: os.path.join(CACHE_DIR, f"trend{key}_{f}.csv") for f in ("Close", "Open")}
    if not refresh and all(os.path.exists(p) for p in paths.values()):
        cached = {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()}
        if set(cached["Close"].columns) == set(tickers):     # cache must match the requested universe
            return cached                                     # (else fall through & re-fetch — guards
        #                                                       against the silent-stale-universe bug)
    os.makedirs(CACHE_DIR, exist_ok=True)
    panels = download_panel(tickers, fields=("Close", "Open"), start=TREND_START)
    for f, p in panels.items():
        p.to_csv(paths[f])
    return panels


class TSMOM(CrossSectionalStrategy):
    """Long-flat 12-month time-series momentum, rebalanced monthly. Hold each instrument at
    1/n_universe when its trailing `look`-day return is positive, else 0 (cash). Fully invested
    when all instruments trend up; rotates to cash as they roll over (the defensive profile)."""
    def __init__(self, look=252, n_universe=len(ETFS), every=21, offset=0):
        self.look, self.n, self.every = look, n_universe, every
        self.offset = offset % every        # which bar of the cycle to trade on (timing-luck knob)

    def target_weights(self, closes, i):
        if i < self.look or i % self.every != self.offset:
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
    trends roll over — the mechanism behind trend's smooth risk profile.

    looks: optional tuple of lookbacks, e.g. (21, 63, 252) — the Moskowitz-Ooi-Pedersen
    ensemble. Each instrument's signal becomes the AVERAGE of the per-lookback signs, so
    positions scale in thirds as trends at different speeds agree, instead of cliffing on
    one 252-day number. Default None = single `look` (the original construction).
    hurdle_col: optional column (e.g. "BIL") treated as the CASH leg — a trend must beat
    the T-bill return over the same window to count as up (MOP measure momentum on EXCESS
    returns; with cash at 4-5%, an asset up +3%/yr is a DOWN trend). The hurdle column is
    a reference only, never traded; bars where it has no data fall back to a 0 hurdle."""
    def __init__(self, look=252, vol_lb=63, target_vol=0.10, every=21, max_gross=1.0,
                 long_short=False, offset=0, looks=None, hurdle_col=None, vol_df=None):
        self.look, self.vol_lb, self.target_vol, self.every, self.max_gross = (
            look, vol_lb, target_vol, every, max_gross)
        self.long_short = long_short                     # True: SHORT down-trending assets too
        self.offset = offset % every        # which bar of the cycle to trade on (timing-luck knob)
        self.looks = tuple(looks) if looks else (look,)
        self.hurdle_col = hurdle_col
        # optional externally-computed ANNUALIZED vol panel (date x ticker), e.g. the
        # Yang-Zhang range estimator (volatility.yang_zhang) — smoother than the default
        # close-to-close std. (Baltas-Kosowski report ~17% turnover reduction at THEIR
        # cadence; our A/B measured it does NOT transfer to monthly — -0.8%, a wash, see
        # volatility.py header. Banked option.) None = close-to-close, bit-identical.
        self.vol_df = vol_df

    def _hurdle(self, closes, i, lk):
        """Cash return over the same window (the excess-return gate), 0 if unavailable."""
        if self.hurdle_col is None:
            return 0.0
        b0 = closes.iloc[i].get(self.hurdle_col)
        bm = closes.iloc[i - lk].get(self.hurdle_col)
        if b0 and bm and np.isfinite(b0) and np.isfinite(bm):
            return b0 / bm - 1
        return 0.0

    def target_weights(self, closes, i):
        if i < max(self.looks) or i % self.every != self.offset:
            return None
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        if self.vol_df is not None and not self.vol_df.index.equals(closes.index):
            # positional lookup below — a same-length but date-shifted panel would
            # silently size positions off the WRONG days' vols (red-team attack #3)
            raise ValueError("vol_df index must exactly match the closes panel index")
        vol_row = self.vol_df.iloc[i] if self.vol_df is not None else None
        strength = {}                                    # signed signal in [-1, 1] per name
        for t in closes.columns:
            if t == self.hurdle_col:
                continue                                 # reference leg, never traded
            p0 = closes.iloc[i].get(t)
            if vol_row is not None:                      # external (annualized) estimator
                v = float(vol_row.get(t, np.nan)) / np.sqrt(252)   # to daily units
            else:
                v = rets[t].std() if t in rets else np.nan
            if not (p0 and np.isfinite(p0) and np.isfinite(v) and v > 0):
                continue
            sigs = []
            for lk in self.looks:
                pm = closes.iloc[i - lk].get(t)
                if not (pm and np.isfinite(pm)):
                    sigs = None
                    break                                # no full history -> skip the name
                mom = p0 / pm - 1 - self._hurdle(closes, i, lk)
                sigs.append(1.0 if mom > 0 else (-1.0 if self.long_short else 0.0))
            if sigs is None:
                continue
            s = float(np.mean(sigs))
            if s != 0.0:
                strength[t] = s
        if not strength:
            return pd.Series(dtype=float)                # all cash
        on = list(strength)
        if vol_row is not None:
            vols = vol_row[on].astype(float)             # annualized, external estimator
            invvol = 1.0 / vols
            corr = rets[on].corr()                       # correlation still from cc returns
            cov = corr * np.outer(vols, vols)            # annualized cov, estimator-scaled
        else:
            invvol = 1.0 / (rets[on].std() * np.sqrt(252))   # inverse-vol risk weights
            cov = rets[on].cov() * 252
        w = pd.Series({t: strength[t] * invvol[t] for t in on})
        w = w / w.abs().sum()                            # normalize GROSS to 1
        pvol = float(np.sqrt(w.values @ cov.values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


def run_trend(cost_bps=5, panels=None, vol_target=True, target_vol=0.10, max_gross=1.0,
              financing_bps=400, long_short=False, borrow_bps=50, cash_rate=0.0, offset=0,
              looks=ENSEMBLE_LOOKS, hurdle=False):
    """Equity curve of the cross-asset trend sleeve. UNLEVERAGED by default (max_gross=1.0: the
    sleeve scales DOWN toward its vol target and parks the rest in cash, but never borrows) —
    removing the old 2x cap actually IMPROVED the blend (Sharpe 0.85→0.90, maxDD −21%→−18%):
    the leverage's financing cost + amplified drawdowns outweighed it. Pass max_gross>1 to opt
    back into the levered (managed-futures) construction; financing is charged on borrowed cash.
    long_short=True shorts down-trending assets (real managed-futures profile → stronger
    crisis alpha), charging borrow on the short legs. cash_rate credits idle cash with the rf
    rate (the sleeve parks in cash when assets aren't trending — that cash should earn T-bills).
    looks: signal lookback ensemble — default the adopted MOP 1/3/12-month blend (see
    ENSEMBLE_LOOKS note); pass looks=(252,) to reproduce the original single-look results."""
    hurdle_col = "BIL" if hurdle else None
    if panels is None:
        panels = etf_panel(ETFS + ["BIL"]) if hurdle else etf_panel()
    if hurdle and "BIL" not in panels["Close"].columns:
        raise ValueError("hurdle=True needs BIL in the panel (use etf_panel(ETFS + ['BIL']))")
    if vol_target:
        strat = VolTargetTSMOM(target_vol=target_vol, max_gross=max_gross, long_short=long_short,
                               offset=offset, looks=looks, hurdle_col=hurdle_col)
        return run_xs(panels, strat, cost=costs.proportional(cost_bps), fill="next_open",
                      allow_short=long_short, gross_max=max_gross,
                      leverage=(1.0 if long_short else max_gross),
                      financing_bps=(0 if long_short else financing_bps),
                      borrow_bps=(borrow_bps if long_short else 0.0), cash_rate=cash_rate)
    return run_xs(panels, TSMOM(offset=offset), cost=costs.proportional(cost_bps), fill="next_open",
                  cash_rate=cash_rate)


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
