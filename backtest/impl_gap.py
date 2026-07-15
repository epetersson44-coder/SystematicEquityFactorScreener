# backtest/impl_gap.py — the IMPLEMENTATION-GAP monitor: real account vs paper twin.
#
#   .venv/bin/python -m backtest.impl_gap            # record today's point + report
#   .venv/bin/python -m backtest.impl_gap report     # report only, no new row
#
# The best replication check the lab gets for free: every month, the REAL Chase
# account (portfolio.py holdings marked live, with its junk-quote guard) is compared
# to a PAPER TWIN — the frictionless sso_stack book reconstructed from the immutable
# locks at lock prices (dashboard._daily_book), scaled to the actual dollars deployed.
# The gap series IS the all-in implementation cost: spreads, fill slippage, the
# SPLG/IAU/PDBC substitutions' tracking difference, execution buffering, and (once
# realized) taxes. A slowly drifting negative gap of a few bps/month is the expected
# price of reality; a JUMP is a fill error, a data error, or a broker problem — and a
# growing gap that nobody can attribute is the replica check failing in production.
#
# CONVENTIONS: the twin is FRICTIONLESS (no fees/taxes), so the expected gap is <= 0
# and slowly widening. Every REAL cash flow into the strategy must be registered in
# FLOWS at its lock date — the twin compounds each flow from its own anchor, so
# deposits don't masquerade as performance. The log is append-only and git-committed
# (backtest/impl_gap_log.csv): one row per recording, typically one per lock day plus
# any ad-hoc checks.
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

# Every real dollar that entered the strategy, anchored at its deployment date.
# APPEND a row here for each future deposit (deploys at locks only, per the covenant).
FLOWS = [
    ("2026-07-13", 12_259.51),                 # go-live: full liquidation deployed
]

LOG_PATH = os.path.join(os.path.dirname(__file__), "impl_gap_log.csv")


def _twin_from(curve, flows, asof=None):
    """Pure core: scale each flow by the twin curve from its anchor to `asof`.
    curve: daily equity Series (any base). flows: [(date_str, amount)]."""
    if asof is None:
        asof = curve.index[-1]
    asof = pd.Timestamp(asof)
    end_idx = curve.index[curve.index <= asof]
    if not len(end_idx):
        raise ValueError(f"twin curve starts {curve.index[0].date()}, after asof {asof.date()}")
    end_val = float(curve.loc[end_idx[-1]])
    total = 0.0
    for date, amount in flows:
        anchor_idx = curve.index[curve.index >= pd.Timestamp(date)]
        if not len(anchor_idx):
            continue                                       # flow anchored after curve end
        total += float(amount) * end_val / float(curve.loc[anchor_idx[0]])
    return total


def twin_value(asof=None):
    """(value, twin_asof_date). twin_asof is the panel's last FULLY-REAL bar — taken
    from the raw panel BEFORE forward-filling, because _daily_book ffills feed gaps
    and an ffilled row would report a fresh date carrying stale prices. When twin_asof
    lags today (feed outage, weekend), the gap reading mixes dates and must be read as
    PROVISIONAL: the real side marks live while the twin is frozen at twin_asof."""
    import json
    import glob
    from backtest import dashboard
    from backtest.universe import download_panel
    curve = dashboard._daily_book("sso_stack")
    if curve is None or curve.empty:
        raise RuntimeError("no sso_stack daily curve — are the locks present?")
    locks = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                          "picks", "sso_stack", "*.json")))
    names = sorted({t for f in locks for t in json.load(open(f))["picks"]})
    raw = download_panel(names)["Close"].dropna(how="any")
    true_asof = min(raw.index[-1], curve.index[-1]) if len(raw) else curve.index[-1]
    return _twin_from(curve, FLOWS, asof=asof or true_asof), true_asof.date()


def real_value():
    """Mark portfolio.py's holdings to live prices (junk-quote guard included)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import portfolio
    prices = portfolio.fetch_prices(list(portfolio.HOLDINGS))
    missing = [t for t, p in prices.items() if p is None]
    if missing:
        raise RuntimeError(f"real_value: no price for {missing}")
    val = sum(sh * prices[t] for t, (sh, _) in portfolio.HOLDINGS.items())
    return val + portfolio.CASH


def record(note=""):
    real = real_value()
    twin, twin_asof = twin_value()
    gap_bps = (real / twin - 1.0) * 1e4
    today = dt.date.today()
    stale = int(np.busday_count(twin_asof, today))
    if stale > 1:
        note = (note + " " if note else "") + f"[PROVISIONAL: twin frozen at {twin_asof}, " \
                                              f"real marks live — dates mixed]"
    row = {"date": today.isoformat(), "twin_asof": twin_asof.isoformat(),
           "real_$": round(real, 2), "twin_$": round(twin, 2),
           "gap_bps": round(gap_bps, 1), "note": note}
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)
    print(f"recorded: real ${real:,.2f} vs twin ${twin:,.2f} ({twin_asof}) "
          f"-> gap {gap_bps:+.1f}bps" + (f"  ({note})" if note else ""))
    return row


def report():
    if not os.path.exists(LOG_PATH):
        print("no gap log yet — run: python -m backtest.impl_gap")
        return None
    log = pd.read_csv(LOG_PATH)
    print(log.to_string(index=False))
    if len(log) >= 2:
        drift = log["gap_bps"].iloc[-1] - log["gap_bps"].iloc[0]
        days = (pd.Timestamp(log["date"].iloc[-1]) - pd.Timestamp(log["date"].iloc[0])).days
        if days > 0:
            print(f"\ngap drift: {drift:+.1f}bps over {days}d "
                  f"(~{drift / days * 365:+.0f}bps/yr all-in implementation cost)")
    return log


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        record(" ".join(sys.argv[1:]))
        print()
        report()
