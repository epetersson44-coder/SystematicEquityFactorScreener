# backtest/tests/test_tracker.py — the live factor book's pick selection.
#
# Pins the logic Path A increment 5 added: _screen_picks walks DOWN the scrubbed
# ranking and keeps the top-N names that have a LIVE price, SKIPPING names that have
# delisted / been acquired since SimFin's ~12-month-lagged data vintage (FARO, GLT,
# KAMN in the real run). Dropping an untradeable name is not survivorship bias — you
# can't buy a stock that no longer trades; the pick is made now and held forward.
#
# Offline + synthetic — run_screen and download_panel are stubbed, so no network and
# no SimFin key. Runs via pytest or `python -m backtest.tests.test_tracker`.

import os

import pandas as pd

import screen
import backtest.tracker as tracker


def _ranked(tickers):
    """A ranking frame shaped like run_screen()'s output (already sorted best-first)."""
    n = len(tickers)
    return pd.DataFrame({"ticker": tickers, "composite": [float(n - i) for i in range(n)]})


def _panel_factory(prices):
    """download_panel stub. `prices` maps ticker -> price; a None price means the name
    is delisted, so (like real yfinance) it's simply absent from the returned columns."""
    def _dl(tickers, *a, **k):
        idx = pd.to_datetime(["2026-06-11", "2026-06-12"])
        data = {t: [prices[t], prices[t]] for t in tickers if prices.get(t) is not None}
        return {"Close": pd.DataFrame(data, index=idx)}
    return _dl


def _run(ranked_tickers, prices, top_n=5):
    """Call _screen_picks with run_screen / download_panel stubbed, then restore."""
    orig_screen, orig_dl = screen.run_screen, tracker.download_panel
    screen.run_screen = lambda *a, **k: _ranked(ranked_tickers)
    tracker.download_panel = _panel_factory(prices)
    try:
        return tracker._screen_picks(top_n=top_n)
    finally:
        screen.run_screen, tracker.download_panel = orig_screen, orig_dl


# scenario: two delisted names sit high in the ranking; the walk must skip them
_RANK = ["AAA", "BBB", "DEAD1", "CCC", "DEAD2", "DDD", "EEE", "FFF"]
_PRICES = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0,
           "EEE": 50.0, "FFF": 60.0, "DEAD1": None, "DEAD2": None}


def test_skips_delisted_and_preserves_rank_order():
    weights, prices, asof = _run(_RANK, _PRICES, top_n=5)
    assert list(weights.index) == ["AAA", "BBB", "CCC", "DDD", "EEE"]   # dead ones skipped, order kept
    assert asof == "2026-06-12"                                          # asof = last price date


def test_equal_weight_sums_to_one():
    weights, _, _ = _run(_RANK, _PRICES, top_n=5)
    assert all(abs(w - 0.2) < 1e-12 for w in weights.values)
    assert abs(float(weights.sum()) - 1.0) < 1e-12


def test_prices_are_for_the_picked_names_only():
    _, prices, _ = _run(_RANK, _PRICES, top_n=3)
    assert list(prices.index) == ["AAA", "BBB", "CCC"]
    assert prices["AAA"] == 10.0 and prices["CCC"] == 30.0


def test_fewer_than_top_n_when_priceable_names_run_out():
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DEAD1": None, "DEAD2": None}
    weights, _, _ = _run(["AAA", "DEAD1", "BBB", "DEAD2", "CCC"], prices, top_n=5)
    assert list(weights.index) == ["AAA", "BBB", "CCC"]                  # only 3 priceable, no crash
    assert abs(float(weights.sum()) - 1.0) < 1e-12                       # still fully invested


def test_raises_when_nothing_prices():
    try:
        _run(["DEAD1", "DEAD2"], {"DEAD1": None, "DEAD2": None})
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when no top name has a live price")


def test_raises_on_empty_screen():
    orig_screen = screen.run_screen
    screen.run_screen = lambda *a, **k: pd.DataFrame(columns=["ticker", "composite"])
    try:
        tracker._screen_picks()
    except RuntimeError:
        return
    finally:
        screen.run_screen = orig_screen
    raise AssertionError("expected RuntimeError on empty screen")


# ----------------------------------------------- long-short book (factor_ls)
def _short_ranked(tickers, legs=None):
    """A short-screen ranking frame: best-first by short_score, with a short_legs col."""
    n = len(tickers)
    legs = legs if legs is not None else [3] * n
    return pd.DataFrame({"ticker": tickers,
                         "short_score": [float(n - i) for i in range(n)],
                         "short_legs": legs})


def _run_ls(longs, shorts, prices, top_n=5, short_legs=None, min_legs=2):
    """Call factor_ls_picks with run_screen / run_short_screen / download_panel stubbed."""
    orig = (screen.run_screen, screen.run_short_screen, tracker.download_panel)
    screen.run_screen = lambda *a, **k: _ranked(longs)
    screen.run_short_screen = lambda *a, **k: _short_ranked(shorts, short_legs)
    tracker.download_panel = _panel_factory(prices)
    try:
        return tracker.factor_ls_picks(top_n=top_n, min_legs=min_legs)
    finally:
        screen.run_screen, screen.run_short_screen, tracker.download_panel = orig


_LS_PX = {t: 10.0 for t in ["L1", "L2", "L3", "L4", "L5", "L6",
                            "S1", "S2", "S3", "S4", "S5", "S6"]}


def test_ls_book_is_dollar_neutral():
    w, prices, _ = _run_ls(["L1", "L2", "L3", "L4", "L5", "L6"],
                           ["S1", "S2", "S3", "S4", "S5", "S6"], _LS_PX, top_n=5)
    assert abs(float(w.sum())) < 1e-12                       # net ~ 0
    assert abs(float(w.abs().sum()) - 1.0) < 1e-12           # gross 1.0
    assert (w[w > 0] == 0.1).all() and (w[w < 0] == -0.1).all()
    assert len(w[w > 0]) == 5 and len(w[w < 0]) == 5


def test_ls_overlap_resolved_long():
    # "A" tops both screens -> it must be LONG only, never short.
    px = {t: 10.0 for t in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}
    w, _, _ = _run_ls(["A", "B", "C", "D", "E"], ["A", "F", "G", "H", "I"], px, top_n=5)
    assert w["A"] > 0                                        # kept long
    assert "A" not in w[w < 0].index                        # not also shorted
    assert set(w[w < 0].index) == {"F", "G", "H", "I"}      # A dropped from shorts


def test_ls_min_legs_filters_thin_shorts():
    # S1, S3 have only 1 signal leg -> excluded as shorts (need >= 2 by default).
    px = {t: 10.0 for t in ["L1", "L2", "L3", "S1", "S2", "S3", "S4", "S5"]}
    w, _, _ = _run_ls(["L1", "L2", "L3"], ["S1", "S2", "S3", "S4", "S5"], px,
                      top_n=3, short_legs=[1, 3, 1, 2, 3])
    shorts = set(w[w < 0].index)
    assert "S1" not in shorts and "S3" not in shorts        # thin (1-leg) shorts dropped
    assert shorts == {"S2", "S4", "S5"}


def test_ls_raises_when_no_shorts_survive():
    # Every short name has only 1 leg -> none pass min_legs -> no short side -> raise.
    px = {t: 10.0 for t in ["L1", "L2", "L3", "S1", "S2", "S3"]}
    try:
        _run_ls(["L1", "L2", "L3"], ["S1", "S2", "S3"], px, top_n=3, short_legs=[1, 1, 1])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when no short names survive the leg filter")


# ----------------------------------------------- momentum trend-filter failsafe
def test_market_risk_on_above_and_below_ma():
    orig = tracker.get_prices
    rising = pd.Series(range(1, 300), index=pd.bdate_range("2020-01-01", periods=299))
    falling = pd.Series(range(300, 1, -1), index=pd.bdate_range("2020-01-01", periods=299))
    try:
        tracker.get_prices = lambda *a, **k: pd.DataFrame({"Close": rising})
        assert tracker._market_risk_on() is True            # last > 200-day mean -> risk on
        tracker.get_prices = lambda *a, **k: pd.DataFrame({"Close": falling})
        assert tracker._market_risk_on() is False           # last < 200-day mean -> risk off
    finally:
        tracker.get_prices = orig


def test_simulate_cash_month_is_flat():
    # A risk-off lock has empty picks; the managed portfolio must hold flat through it.
    recs = [{"data_asof": "2026-01-02", "picks": {}}]
    dates = pd.to_datetime(["2026-01-02", "2026-02-02"])
    closes = pd.DataFrame({"A": [100.0, 130.0]}, index=dates)   # A soared, but we held cash
    spy = pd.Series([400.0, 440.0], index=dates)
    _, s = tracker._simulate(recs, closes, spy, 10_000.0)
    assert abs(s["final"] - 10_000.0) < 1e-6                # cash -> dead flat, missed the move


# ----------------------------------------------- entry-price basis (dividend drift)
def test_entry_prices_use_current_panel_basis():
    # yfinance RE-SCALES the whole adjusted history when a new dividend is paid. A stored
    # lock price then sits on a stale basis and corrupts "since lock" returns; reading the
    # entry from the current panel keeps both ends of the return on one basis.
    dates = pd.to_datetime(["2026-01-02", "2026-06-01"])
    stored = {"A": 100.0}                                   # snapshotted at lock time
    re_adj = pd.DataFrame({"A": [90.0, 108.0]}, index=dates)  # history re-scaled x0.9 since
    entry = tracker._entry_prices(re_adj, "2026-01-02", {"A": 1.0}, stored)["A"]
    assert abs(entry - 90.0) < 1e-12                        # read from the panel itself
    true_ret = 108.0 / entry - 1
    assert abs(true_ret - 0.20) < 1e-12                     # the real +20% is preserved
    stale_ret = 108.0 / stored["A"] - 1                     # the old bug: +8% out of thin air
    assert abs(stale_ret - true_ret) > 0.05


def test_entry_prices_fallback_to_stored_when_unpriced():
    dates = pd.to_datetime(["2026-01-02", "2026-06-01"])
    closes = pd.DataFrame({"A": [50.0, 60.0]}, index=dates)   # B missing from the panel
    e = tracker._entry_prices(closes, "2026-01-02", {"A": 1.0, "B": 1.0}, {"B": 20.0})
    assert abs(e["A"] - 50.0) < 1e-12
    assert abs(e["B"] - 20.0) < 1e-12                       # best-effort fallback


# ----------------------------------------------- desk: cash slice must not vanish
def _dash_run(picks, lock_prices, panel, spy):
    """Run dashboard._book on ONE synthetic lock with disk/network stubbed out."""
    import tempfile
    import json as _json
    import backtest.dashboard as dash
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "booktest"))
    rec = {"lock_date": "2026-01-02", "data_asof": "2026-01-02", "strategy": "booktest",
           "n": len(picks), "picks": picks, "lock_prices": lock_prices, "spy_lock": 100.0}
    with open(os.path.join(tmp, "booktest", "2026-01-02.json"), "w") as f:
        _json.dump(rec, f)
    orig = (tracker.PICKS_DIR, dash.download_panel, dash.get_prices)
    tracker.PICKS_DIR = tmp
    dash.download_panel = lambda *a, **k: {"Close": panel}
    dash.get_prices = lambda *a, **k: pd.DataFrame({"Close": spy})
    try:
        return dash._book("booktest", {})
    finally:
        tracker.PICKS_DIR, dash.download_panel, dash.get_prices = orig


def test_desk_book_keeps_the_cash_slice():
    # A blend-style lock that's only 50% invested: flat prices must mean a FLAT $10k book
    # (the old bug valued only the invested half -> a fake -50%).
    dates = pd.to_datetime(["2026-01-02", "2026-02-02"])
    panel = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    spy = pd.Series([500.0, 510.0], index=dates)
    b = _dash_run({"A": 0.5}, {"A": 100.0}, panel, spy)
    assert abs(b["final"] - 10_000.0) < 1e-6
    assert abs(b["ret"]) < 1e-9


def test_desk_book_dollar_neutral_pnl_rides_on_cash():
    # Long-short lock (net ~0): book value = $10k cash + spread P&L, not ~$0.
    dates = pd.to_datetime(["2026-01-02", "2026-02-02"])
    panel = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0]}, index=dates)
    spy = pd.Series([500.0, 510.0], index=dates)
    b = _dash_run({"A": 0.5, "B": -0.5}, {"A": 100.0, "B": 50.0}, panel, spy)
    assert abs(b["final"] - 10_500.0) < 1e-6                # +10% on the $5k long leg


if __name__ == "__main__":
    import sys
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {name}")
        except Exception as e:                              # noqa: BLE001
            failed.append(name); print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
