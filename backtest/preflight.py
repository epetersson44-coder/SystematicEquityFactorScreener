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
#   5. Independent-vendor cross-check — every live ticker's close vs Cboe's exchange
#      feed (keyless), plus ~2yr of adjusted return history vs Tiingo when TIINGO_KEY
#      is set (free key). Catches a corrupted/adjusted-weird vendor feed — the one
#      Data-grade improvement available without paying for data: we can't buy breadth,
#      but we can verify what we get. Network-tolerant (WARN if unreachable).
#      (Stooq retired 2026-07-15: its CSV endpoint went behind a JS bot wall.)

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

try:                                                       # secrets live in repo-root .env
    from dotenv import load_dotenv                         # (gitignored): TIINGO_KEY=...
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:                                        # dotenv optional — env vars still work
    pass

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


def cboe_spot(symbol):
    """(current_price, prev_close) from Cboe's keyless delayed-quotes JSON — an
    EXCHANGE-operated source, independent of any market-data reseller. Raises on
    failure. (Replaced Stooq 2026-07-15: Stooq put its CSV endpoint behind a
    JavaScript bot-verification wall; we do not script around bot checks.)"""
    import requests
    r = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/quotes/{symbol}.json",
                     timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = r.json()["data"]
    cur = float(d["current_price"])
    prev = cur - float(d.get("price_change") or 0.0)
    return cur, prev


def tiingo_history(symbol, token, days=520):
    """Adjusted daily closes from Tiingo (free API key, tiingo.com). Raises on failure."""
    import datetime as _dt
    import requests
    start = (_dt.date.today() - _dt.timedelta(days=int(days * 1.6))).isoformat()
    r = requests.get(f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
                     params={"startDate": start, "token": token, "columns": "date,adjClose"},
                     timeout=20)
    r.raise_for_status()
    rows = r.json()
    s = pd.Series({pd.Timestamp(x["date"]).tz_localize(None).normalize(): float(x["adjClose"])
                   for x in rows}).sort_index()
    return s.dropna()


def check_cross_vendor(tolerance=0.01, hist_days=500, ret_corr_min=0.98, ret_med_bps=10):
    """EVERY live signal input (6 sleeve ETFs + UPRO) vs independent vendors.
    Leg 1 (SPOT, keyless, always on): latest close vs Cboe's exchange feed — deviation
    measured against the CLOSER of Cboe's current and previous close (our panel ends at
    a close; Cboe quotes move intraday). Leg 2 (HISTORY, needs TIINGO_KEY env var —
    free key): last ~2 years of DAILY RETURNS vs Tiingo adjusted closes (corr + median
    |diff|). The history leg (seventh review) catches what spot cannot: a silently
    corrupted adjusted HISTORY (bad dividend/split factor) that leaves today's price
    fine while every trend signal computed from the past 12 months is wrong — the
    go-live-day SPLG junk quote was this failure class's little sibling.
    WARN (not FAIL) when a vendor is unreachable or the key is absent."""
    import os
    from backtest.trend_sleeve import etf_panel, ETFS
    closes = etf_panel()["Close"]
    import yfinance as yf
    live = yf.download("UPRO", period="2y", progress=False, auto_adjust=True)["Close"]
    upro = live["UPRO"] if hasattr(live, "columns") else live
    token = os.environ.get("TIINGO_KEY", "").strip()
    tickers = list(ETFS) + ["UPRO"]
    bad, spot_ok, hist_ok, unreachable = [], 0, 0, 0
    for tkr in tickers:
        y_ser = (upro if tkr == "UPRO" else closes[tkr]).dropna()
        ours = float(y_ser.iloc[-1])
        try:                                               # leg 1: Cboe spot
            cur, prev = cboe_spot(tkr)
            dev = min(abs(ours / cur - 1), abs(ours / prev - 1) if prev else 9)
            if dev > tolerance:
                bad.append(f"{tkr} spot off {dev:.2%} vs Cboe")
            else:
                spot_ok += 1
        except Exception:                                  # noqa: BLE001
            unreachable += 1
        if token:                                          # leg 2: Tiingo history
            try:
                t_hist = tiingo_history(tkr, token, hist_days)
            except Exception:                              # noqa: BLE001
                unreachable += 1
                continue
            both = pd.concat([y_ser.rename("y"), t_hist.rename("t")], axis=1).dropna().tail(hist_days)
            if len(both) >= 250:
                r = both.pct_change().dropna()
                corr = float(r["y"].corr(r["t"]))
                med = float((r["y"] - r["t"]).abs().median()) * 1e4
                if corr < ret_corr_min or med > ret_med_bps:
                    bad.append(f"{tkr} HISTORY divergence (corr {corr:.3f}, med {med:.1f}bps "
                               f"over {len(r)}d) — adjusted history suspect")
                else:
                    hist_ok += 1
    if bad:
        return False, "VENDOR DISAGREEMENT — " + "; ".join(bad)
    if spot_ok == 0:
        return None, "no independent vendor reachable — cross-check skipped"
    msg = f"cross-vendor OK: {spot_ok}/{len(tickers)} spots within {tolerance:.0%} (Cboe)"
    if token:
        msg += f", {hist_ok} return histories in agreement (Tiingo)"
    else:
        msg += " — HISTORY leg off: set TIINGO_KEY (free key, tiingo.com) to enable the strongest data check"
    return True, msg


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
