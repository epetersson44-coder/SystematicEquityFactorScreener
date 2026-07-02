# backtest/volatility.py — range-based volatility estimation (Yang-Zhang 2000).
#
# The sleeve's vol targeting originally used the rolling close-to-close standard
# deviation: one datapoint per day, so a 63-day window that lags regime shifts and
# jitters position sizes. Yang-Zhang combines three components measured from the full
# OHLC bar — overnight variance (close->open), open-to-close variance, and the
# Rogers-Satchell range term — into a drift-independent estimator ~7x more efficient
# per observation. Baltas & Kosowski ("Demystifying Time-Series Momentum Strategies")
# size TSMOM positions with it and report ~17% lower turnover at no performance cost:
# a smoother vol estimate means the vol target resizes less erratically. Erik surfaced
# this from the reading list (2026-07-01) — the intended reading->lock pipeline working.
#
# A/B RESULT (2026-07-01, all 21 offsets, turnover measured at the decision level):
# blend Sharpe 0.929 vs 0.929, turnover -0.8% — the 17% does NOT transfer to our
# setting. Why: B-K resize positions at high frequency, where day-to-day estimator
# noise IS the turnover; our sleeve rebalances MONTHLY, and a monthly cadence already
# low-pass-filters vol noise — our turnover comes from trend-signal flips, not
# estimator jitter. Kept as an option (VolTargetTSMOM(vol_df=yang_zhang(...))), NOT
# the default: identical results, extra OHLC dependency. A context-dependence lesson,
# not a failed estimator.

import numpy as np
import pandas as pd


def yang_zhang(open_, high, low, close, window=63, periods_per_year=252):
    """Annualized Yang-Zhang volatility, rolling `window`, per column.

    All four inputs are (date x ticker) DataFrames on the same index. Returns a
    DataFrame of annualized vols; NaN where the window isn't full (or OHLC missing).

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_open-to-close + (1-k) * sigma^2_RS
    with k = 0.34 / (1.34 + (n+1)/(n-1))  (Yang & Zhang 2000, drift-independent).
    """
    o = np.log(open_ / close.shift(1))                  # overnight (close -> open)
    c = np.log(close / open_)                           # open -> close
    rs = (np.log(high / close) * np.log(high / open_)   # Rogers-Satchell range term
          + np.log(low / close) * np.log(low / open_))
    sig_o2 = o.rolling(window).var()
    sig_c2 = c.rolling(window).var()
    sig_rs2 = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sig_o2 + k * sig_c2 + (1 - k) * sig_rs2
    return np.sqrt(var.clip(lower=0.0) * periods_per_year)


def close_to_close(close, window=63, periods_per_year=252):
    """The original estimator, for A/B reference: rolling std of daily returns."""
    return close.pct_change().rolling(window).std() * np.sqrt(periods_per_year)
