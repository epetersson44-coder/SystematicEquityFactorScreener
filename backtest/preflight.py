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
#   3. Live tickers priceable — UPRO / SGOV / SPLG / IAU / PDBC (every real-account
#      leg incl. the REAL_SUBS substitutes) return a quote.
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


def check_live_tickers(tickers=None):
    """Default covers EVERY ticker real money actually trades: UPRO/SGOV plus the
    REAL_SUBS substitutes (SPLG/IAU/PDBC). Before this the docstring said 'the
    real-account legs' while checking only UPRO/SGOV — an unpriceable IAU surfaced as a
    RuntimeError mid-shopping_list at lock time instead of here (ninth review, F4)."""
    from backtest.data import get_prices
    if tickers is None:
        from backtest.tracker import REAL_SUBS
        tickers = ("UPRO", "SGOV", *REAL_SUBS.values())
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
    px = stooq_history(symbol)
    return float(px.iloc[-1]), px.index[-1]


def stooq_history(symbol="spy.us"):
    """Full daily close history from Stooq (one request — the CSV is always complete).
    Raises on failure."""
    import io
    import requests
    r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    return df["Close"].dropna()


def check_cross_vendor(tolerance=0.01, hist_days=500, ret_corr_min=0.98, ret_med_bps=10):
    """EVERY live signal input (6 sleeve ETFs + UPRO) vs Stooq — latest price within 1%
    AND the last ~2 years of DAILY RETURNS in agreement (corr + median |diff|). The
    history leg (seventh review) catches the failure the spot check cannot: a silently
    corrupted adjusted HISTORY (bad dividend/split factor) that leaves today's price
    fine while every trend signal computed from the past 12 months is wrong.
    Return comparison is robust to the known benign difference (Stooq closes are
    dividend-UNadjusted; Yahoo's are adjusted): ex-div days are a handful of outliers
    per year, so corr and MEDIAN tolerate them while a shifted history does not.
    WARN (not FAIL) if Stooq is unreachable — it rate-limits; SPY probes first."""
    from backtest.trend_sleeve import etf_panel, ETFS
    try:
        spy_hist = stooq_history("spy.us")
    except Exception as e:                                 # noqa: BLE001
        return None, f"Stooq unreachable ({type(e).__name__}) — cross-check skipped"
    closes = etf_panel()["Close"]
    import yfinance as yf
    live = yf.download("UPRO", period="2y", progress=False, auto_adjust=True)["Close"]
    upro = live["UPRO"] if hasattr(live, "columns") else live
    bad, checked = [], 0
    for tkr, pre in [("SPY", spy_hist)] + [(t, None) for t in ETFS if t != "SPY"] + [("UPRO", None)]:
        try:
            s_hist = pre if pre is not None else stooq_history(f"{tkr.lower()}.us")
        except Exception:                                  # noqa: BLE001
            continue                                       # symbol-level skip, not a verdict
        y_ser = (upro if tkr == "UPRO" else closes[tkr]).dropna()
        s_date = s_hist.index[-1]
        if s_date in y_ser.index:                          # leg 1: spot within 1%
            dev = abs(float(y_ser.loc[s_date]) / float(s_hist.iloc[-1]) - 1)
            if dev > tolerance:
                bad.append(f"{tkr} spot {s_date.date()} off {dev:.2%}")
        both = pd.concat([y_ser.rename("y"), s_hist.rename("s")], axis=1).dropna().tail(hist_days)
        if len(both) >= 250:                               # leg 2: return-history agreement
            r = both.pct_change().dropna()
            corr = float(r["y"].corr(r["s"]))
            med = float((r["y"] - r["s"]).abs().median()) * 1e4
            if corr < ret_corr_min or med > ret_med_bps:
                bad.append(f"{tkr} HISTORY divergence (corr {corr:.3f}, med {med:.1f}bps "
                           f"over {len(r)}d) — adjusted history suspect")
        checked += 1
    if bad:
        return False, "VENDOR DISAGREEMENT — " + "; ".join(bad)
    return True, (f"cross-vendor OK: {checked} tickers, spot within {tolerance:.0%} and "
                  f"~{hist_days}d return history in agreement")


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
