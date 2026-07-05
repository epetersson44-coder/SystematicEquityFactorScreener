# backtest/preflight.py — pre-lock health checks for the live-money pipeline.
#
# Run BEFORE /picks locks anything (and especially before real-money orders):
#     python -m backtest.preflight
#
# Checks, each PASS/WARN/FAIL, non-zero exit on any FAIL:
#   1. ETF panel freshness — last bar within a few business days of today (a silently
#      stale cache would lock last month's signals with today's date).
#   2. Complete quote row — every sleeve ETF priced on the last bar (the red-team #1
#      guard, run proactively instead of exploding mid-lock).
#   3. Live tickers priceable — UPRO / SGOV (the real-account legs) return a quote.
#   4. T-bill series health — ^IRX cache reaches near-present (shadow-book financing).
#   5. Independent-vendor cross-check — SPY's latest close vs Stooq (free, separate
#      infrastructure from Yahoo). Catches a corrupted/adjusted-weird vendor feed —
#      the one Data-grade improvement available without paying for data: we can't buy
#      breadth, but we can verify what we get. Network-tolerant (WARN if unreachable).

import sys
import datetime as dt

import numpy as np
import pandas as pd

STALE_BDAYS = 5


def _age_bdays(last_date, today=None):
    today = today or pd.Timestamp(dt.date.today())
    return int(np.busday_count(last_date.date(), today.date()))


def check_panel(refresh=True):
    """(ok, message) — panel reaches near-present and the last row is complete."""
    from backtest.trend_sleeve import etf_panel, ETFS
    closes = etf_panel(refresh=refresh)["Close"]
    age = _age_bdays(closes.index[-1])
    if age > STALE_BDAYS:
        return False, f"panel stale: last bar {closes.index[-1].date()} ({age} bdays old)"
    last = closes.iloc[-1]
    missing = [t for t in ETFS if not (pd.notna(last.get(t)) and last.get(t) > 0)]
    if missing:
        return False, f"partial last row — missing {missing}"
    return True, f"panel fresh ({closes.index[-1].date()}), all {len(ETFS)} ETFs priced"


def check_live_tickers(tickers=("UPRO", "SGOV")):
    from backtest.data import get_prices
    bad = []
    for t in tickers:
        try:
            px = float(get_prices(t, refresh=True)["Close"].iloc[-1])
            if not (np.isfinite(px) and px > 0):
                bad.append(t)
        except Exception:                                  # noqa: BLE001
            bad.append(t)
    return (not bad), ("live tickers priced: " + ", ".join(tickers) if not bad
                       else f"unpriceable: {bad}")


def check_tbills():
    from backtest.leverage_study import tbill_series
    idx = pd.bdate_range(end=dt.date.today(), periods=10)
    rf = tbill_series(idx, refresh=True)
    last = float(rf.iloc[-1])
    if not (np.isfinite(last) and 0.0 <= last < 0.15):
        return False, f"rf series unhealthy (last={last})"
    return True, f"T-bill series healthy (rf={last:.2%})"


def stooq_close(symbol="spy.us"):
    """Latest daily close from Stooq (independent vendor). Raises on failure."""
    import io
    import requests
    r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    return float(df["Close"].iloc[-1]), df.index[-1]


def check_cross_vendor(tolerance=0.01):
    """EVERY live signal input (6 sleeve ETFs + UPRO) vs Stooq within 1% — a bad print
    on any single vendor row would otherwise flow straight into a lock (sixth review,
    F7d; previously SPY-only). NOTE: Yahoo's auto_adjust rescales history but the LATEST
    close matches spot, so a same-date comparison is fair. WARN (not FAIL) if Stooq is
    unreachable — it rate-limits; SPY is probed first and failure skips the rest."""
    from backtest.trend_sleeve import etf_panel, ETFS
    try:
        s_px, s_date = stooq_close("spy.us")
    except Exception as e:                                 # noqa: BLE001
        return None, f"Stooq unreachable ({type(e).__name__}) — cross-check skipped"
    closes = etf_panel()["Close"]
    import yfinance as yf
    live = yf.download("UPRO", period="5d", progress=False, auto_adjust=True)["Close"]
    upro = live["UPRO"] if hasattr(live, "columns") else live
    bad, checked = [], 0
    for tkr, sym_px in [("SPY", (s_px, s_date))] + [(t, None) for t in ETFS if t != "SPY"] + [("UPRO", None)]:
        try:
            px, date = sym_px if sym_px else stooq_close(f"{tkr.lower()}.us")
        except Exception:                                  # noqa: BLE001
            continue                                       # symbol-level skip, not a verdict
        series = upro.dropna() if tkr == "UPRO" else closes[tkr].dropna()
        if date not in series.index:
            continue
        dev = abs(float(series.loc[date]) / px - 1)
        checked += 1
        if dev > tolerance:
            bad.append(f"{tkr} {date.date()}: yahoo {float(series.loc[date]):.2f} vs stooq {px:.2f} ({dev:.2%})")
    if bad:
        return False, "VENDOR DISAGREEMENT — " + "; ".join(bad)
    return True, f"cross-vendor OK: {checked} tickers matched within {tolerance:.0%} (as of {s_date.date()})"


def run(refresh=True):
    checks = [("panel", lambda: check_panel(refresh)),
              ("live tickers", check_live_tickers),
              ("t-bills", check_tbills),
              ("cross-vendor", check_cross_vendor)]
    failed = 0
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:                             # noqa: BLE001
            ok, msg = False, f"check crashed: {type(e).__name__}: {e}"
        tag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        failed += (ok is False)
        print(f"  {tag:4s}  {name}: {msg}")
    print("\npreflight:", "CLEAR TO LOCK" if failed == 0 else f"{failed} FAILURE(S) — do NOT lock")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
