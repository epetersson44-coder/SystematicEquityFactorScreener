# backtest/mechanism.py — the CRISIS-MECHANISM instrument.
#
#   .venv/bin/python -m backtest.mechanism
#
# The drawdown protocol's -30% step says: VERIFY the system is doing what its backtest
# did in comparable tape — "the sleeve should be rotating defensive" — and never ask
# "do I feel like owning this?". This module is that verification, PRE-BUILT while calm
# so it exists when it's needed: for every immutable lock it computes the book's
# posture on the aggressive<->defensive axis, and calibrates the reading against what
# the SAME construction did inside 2008 / 2020 / 2022 and in calm bulls.
#
# No stored state: the locks are the facts; the gauge is recomputed from locks + panel
# on demand (the lab's standard pattern). Three numbers per lock:
#   deployed%  — sum of sleeve weights (the vol target's risk appetite; cash is defense)
#   defensive% — share of DEPLOYED sleeve in duration+gold (IEF/TLT/GLD)
#                vs risk assets (SPY/EFA/DBC)
#   book beta  — realized 63d equity beta of the LOCKED book to SPY (UPRO counted as
#                3x SPY — financing drag is irrelevant at beta precision)
# ssoB decomposes exactly (UPRO = the equity leg, rest = 2/3 x sleeve, SGOV = cash).
# COARSE INSTRUMENT, by design: data-vintage re-basing moves these a few percent —
# meaningless for a gauge whose crisis signal is "beta 1.7 vs beta 0.6".
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

DEFENSIVE = ("IEF", "TLT", "GLD")
RISK = ("SPY", "EFA", "DBC")
BETA_WINDOW = 63

# Calibration dates: what the construction did in the named regimes (backtest, via the
# clean-room replica's independently-verified weight formula).
REFERENCE_DATES = [
    ("2008-10-01", "GFC depth"),
    ("2009-03-02", "GFC bottom"),
    ("2017-06-01", "calm bull"),
    ("2020-03-20", "COVID crash"),
    ("2021-12-01", "pre-bear top"),
    ("2022-09-01", "inflation bear"),
]


def book_beta(weights, closes, i, window=BETA_WINDOW):
    """Realized beta of a weight dict to SPY over the trailing `window` days ending at
    bar i. UPRO counts as 3x SPY; SGOV counts as 0 (cash-like)."""
    rets = closes.pct_change().iloc[max(1, i - window + 1):i + 1]
    spy = rets["SPY"]
    book = pd.Series(0.0, index=rets.index)
    for t, w in weights.items():
        if t == "SGOV":
            continue
        if t == "UPRO":
            book = book + 3.0 * w * spy
        elif t in rets.columns:
            book = book + w * rets[t]
    var = float(spy.var())
    return float(book.cov(spy) / var) if var > 0 else float("nan")


def sleeve_posture(sleeve_w):
    """(deployed, defensive_share, risk_share) from sleeve weights (fractions of the
    sleeve slice; deployed = their sum, shares are of the deployed part)."""
    dep = sum(sleeve_w.values())
    if dep <= 0:
        return 0.0, 0.0, 0.0
    d = sum(w for t, w in sleeve_w.items() if t in DEFENSIVE) / dep
    r = sum(w for t, w in sleeve_w.items() if t in RISK) / dep
    return dep, d, r


def lock_rows(closes):
    rows = []
    for book in ("sso_stack", "blend"):
        for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                               "picks", book, "*.json"))):
            rec = json.load(open(f))
            ts = pd.Timestamp(rec["data_asof"])
            if ts not in closes.index:
                idx = closes.index[closes.index <= ts]
                if not len(idx):
                    continue
                ts = idx[-1]
            i = closes.index.get_loc(ts)
            picks = rec["picks"]
            if book == "sso_stack":                        # exact decomposition
                sleeve_w = {t: w / (2.0 / 3.0) for t, w in picks.items()
                            if t not in ("UPRO", "SGOV")}
            else:                                          # blend: coarse (SPY leg mixed in)
                sleeve_w = {t: w for t, w in picks.items() if t not in ("SPY", "SGOV")}
            dep, d, r = sleeve_posture(sleeve_w)
            beta = book_beta(picks, closes, i)
            rows.append({"book": book, "lock": rec["data_asof"],
                         "sleeve deployed": f"{dep:5.0%}", "defensive": f"{d:5.0%}",
                         "risk": f"{r:5.0%}", "book beta": f"{beta:5.2f}"})
    return rows


def reference_rows(closes):
    from backtest.replica import replica_weights
    rows = []
    for date, label in REFERENCE_DATES:
        idx = closes.index[closes.index <= pd.Timestamp(date)]
        if not len(idx) or closes.index.get_loc(idx[-1]) < 260:
            continue
        i = closes.index.get_loc(idx[-1])
        tw = replica_weights(closes, i) or {}
        dep, d, r = sleeve_posture(tw)
        ssob = {"UPRO": 1.0 / 3.0}
        for t, w in tw.items():
            ssob[t] = ssob.get(t, 0.0) + (2.0 / 3.0) * w
        beta = book_beta(ssob, closes, i)
        rows.append({"regime": label, "date": idx[-1].date().isoformat(),
                     "sleeve deployed": f"{dep:5.0%}", "defensive": f"{d:5.0%}",
                     "risk": f"{r:5.0%}", "ssoB beta": f"{beta:5.2f}"})
    return rows


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from backtest.trend_sleeve import etf_panel
    closes = etf_panel()["Close"]

    print("LOCKED books — posture at each immutable lock:")
    print(pd.DataFrame(lock_rows(closes)).to_string(index=False))

    print("\nCALIBRATION — the same construction inside known regimes (backtest,"
          " replica-verified formula):")
    print(pd.DataFrame(reference_rows(closes)).to_string(index=False))

    print("\nHow to read this in a drawdown (the -30% protocol step): the question is"
          "\nNEVER the P&L — it is whether the current lock's row has MOVED toward the"
          "\ncrisis rows (deployment down, defensive share up, beta down) as the trend"
          "\nsignals digest the tape. Rotation lags by design (monthly locks, 1/3/12mo"
          "\nsignals): expect the move to show 1-3 locks into a sustained decline, as it"
          "\ndid in the calibration rows — NOT on day one of a gap-down. If 3+ locks"
            " into a bear the row still looks like 'calm bull', THAT is mechanism"
            " failure — the tripwire conversation, not the hold-through conversation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
