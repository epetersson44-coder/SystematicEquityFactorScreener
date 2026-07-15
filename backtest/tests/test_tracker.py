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


# ----------------------------------------------- blend book: SGOV cash slice
def _blend_run(trend_up=("TLT", "IEF", "GLD"), cash_etf="SGOV", eq_weight=0.4):
    """blend_picks on a synthetic ETF panel (etf_panel + SGOV pricing stubbed)."""
    import numpy as np
    import backtest.trend_sleeve as ts
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(11)
    cols = {}
    for t in ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]:
        drift = 0.005 if t in trend_up else -0.005                    # unambiguous trends...
        cols[t] = 100 * np.cumprod(1 + drift + rng.normal(0, 0.015, n))  # ...with REAL vol, so
    panel = pd.DataFrame(cols, index=idx)                             # the 10% vol target binds
    assert all((panel[t].iloc[-1] > panel[t].iloc[0]) == (t in trend_up) for t in cols)
    orig_panel, orig_gp = ts.etf_panel, tracker.get_prices
    ts.etf_panel = lambda *a, **k: {"Close": panel}
    tracker.get_prices = lambda *a, **k: pd.DataFrame(
        {"Close": pd.Series([100.25], index=[idx[-1]])})
    try:
        return tracker.blend_picks(eq_weight=eq_weight, cash_etf=cash_etf)
    finally:
        ts.etf_panel, tracker.get_prices = orig_panel, orig_gp


def test_blend_residual_cash_goes_to_tbill_etf():
    w, prices, _ = _blend_run()
    assert "SGOV" in w.index                                # the cash slice is invested
    assert abs(float(w.sum()) - 1.0) < 1e-9                 # fully allocated incl. T-bills
    assert w["SGOV"] > 0.005
    assert prices["SGOV"] == 100.25                         # priced for the lock file


def test_blend_all_risk_off_parks_in_tbills():
    # nothing trending up -> the whole trend sleeve is cash -> SPY leg + SGOV only
    w, _, _ = _blend_run(trend_up=())
    assert set(w.index) == {"SPY", "SGOV"}
    assert abs(w["SPY"] - 0.4) < 1e-9
    assert abs(w["SGOV"] - 0.6) < 1e-9


def test_blend_cash_etf_optional():
    w, _, _ = _blend_run(cash_etf=None)
    assert "SGOV" not in w.index                            # plain-cash behaviour preserved
    assert float(w.sum()) < 1.0 - 0.005


def _sso_run(trend_up=("TLT", "IEF", "GLD"), cash_etf="SGOV"):
    """sso_stack_picks on a synthetic panel (etf_panel + SSO/SGOV pricing stubbed)."""
    import numpy as np
    import backtest.trend_sleeve as ts
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(11)
    panel = pd.DataFrame({t: 100 * np.cumprod(1 + (0.005 if t in trend_up else -0.005)
                                              + rng.normal(0, 0.015, n))
                          for t in ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]}, index=idx)
    px = {"UPRO": 60.5, "SGOV": 100.25}
    orig = (ts.etf_panel, tracker.get_prices)
    ts.etf_panel = lambda *a, **k: {"Close": panel}
    tracker.get_prices = lambda t, *a, **k: pd.DataFrame(
        {"Close": pd.Series([px[t]], index=[idx[-1]])})
    try:
        return tracker.sso_stack_picks(cash_etf=cash_etf)
    finally:
        ts.etf_panel, tracker.get_prices = orig


def test_sso_stack_third_upro_rest_sleeve_and_tbills():
    w, prices, _ = _sso_run()
    assert abs(w["UPRO"] - 1 / 3) < 1e-9                    # the 3x-wrapped 100% SPY leg
    assert prices["UPRO"] == 60.5                           # priced for the lock file
    sleeve = w.drop(["UPRO", "SGOV"], errors="ignore")
    assert len(sleeve) > 0 and (sleeve > 0).all()           # scaled trend sleeve present
    assert abs(float(w.sum()) - 1.0) < 1e-9                 # fully allocated incl. T-bills
    assert w.get("SGOV", 0) > 0.005                         # vol-target residual in T-bills


def test_sso_stack_risk_off_is_equity_leg_plus_tbills():
    # nothing trending -> the sleeve slice sits entirely in T-bills, equity leg unchanged
    w, _, _ = _sso_run(trend_up=())
    assert set(w.index) == {"UPRO", "SGOV"}
    assert abs(w["UPRO"] - 1 / 3) < 1e-9 and abs(w["SGOV"] - 2 / 3) < 1e-9


def test_blend_mf_etf_third_leg():
    # mf_etf="DBMF": three-way inverse-vol split, DBMF priced into the lock, sum <= 1.
    import numpy as np
    import backtest.trend_sleeve as ts
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(21)
    panel = pd.DataFrame({t: 100 * np.cumprod(1 + (0.004 if t in ("SPY", "TLT") else -0.004)
                                              + rng.normal(0, 0.015, n))
                          for t in ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]}, index=idx)
    mf = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0003, 0.006, n))}, index=idx)
    trend_eq = pd.Series(10_000 * np.cumprod(1 + rng.normal(0.0003, 0.007, n)), index=idx)
    orig = (ts.etf_panel, ts.run_trend, tracker.get_prices)
    ts.etf_panel = lambda *a, **k: {"Close": panel}
    ts.run_trend = lambda *a, **k: trend_eq
    tracker.get_prices = lambda *a, **k: mf
    try:
        w, prices, _ = tracker.blend_picks(cash_etf=None, mf_etf="DBMF")
    finally:
        ts.etf_panel, ts.run_trend, tracker.get_prices = orig
    assert "DBMF" in w.index and w["DBMF"] > 0.05           # a real third leg
    assert "SPY" in w.index
    assert float(w.sum()) <= 1.0 + 1e-9                     # still unleveraged
    assert prices["DBMF"] == float(mf["Close"].iloc[-1])    # priced for the lock file


# ----------------------------------------------- shopping list (real-account orders)
def _shop(capital, fractional, weights, px_map):
    orig = (tracker.PICKERS.get("shoptest"), tracker.get_prices)
    tracker.PICKERS["shoptest"] = lambda refresh=False: (
        pd.Series(weights),
        pd.Series({t: px_map.get(t, float("nan")) for t in weights}),   # substituted names
        "2026-07-01")                                                    # are priced via get_prices
    tracker.get_prices = lambda t, *a, **k: pd.DataFrame(
        {"Close": pd.Series([px_map[t]], index=[pd.Timestamp("2026-07-01")])})
    try:
        return tracker.shopping_list(capital, book="shoptest", refresh=False,
                                     fractional=fractional)
    finally:
        tracker.get_prices = orig[1]
        if orig[0] is None:
            tracker.PICKERS.pop("shoptest", None)


def test_shopping_list_fractional_fills_exactly():
    px = {"UPRO": 141.39, "IEF": 94.03, "IAU": 75.96, "SGOV": 100.40}
    df = _shop(8000.0, True, {"UPRO": 1 / 3, "IEF": 0.5, "GLD": 1 / 6}, px)
    assert list(df["ticker"])[:3] == ["IEF", "UPRO", "IAU"]     # sorted by weight, GLD->IAU
    assert abs(df["est_cost_$"].sum() - 8000.0) < 1.5           # fully deployed
    assert abs(df.attrs["leftover_cash"]) < 1.5


def test_shopping_list_whole_shares_sweep_to_sgov():
    px = {"UPRO": 141.39, "IEF": 94.03, "IAU": 75.96, "SGOV": 100.40}
    df = _shop(8000.0, False, {"UPRO": 1 / 3, "IEF": 0.5, "GLD": 1 / 6}, px)
    assert (df["shares"] == df["shares"].astype(int)).all()     # whole shares only
    assert "SGOV" in set(df["ticker"])                          # residue swept
    assert df.attrs["leftover_cash"] < px["SGOV"]               # less than one SGOV share


def test_shopping_list_substitutions_priced_live():
    px = {"UPRO": 100.0, "SPLG": 85.76, "PDBC": 15.78, "SGOV": 100.40}
    df = _shop(6000.0, True, {"UPRO": 0.5, "SPY": 0.3, "DBC": 0.2}, px).set_index("ticker")
    assert "SPY" not in df.index and "SPLG" in df.index         # SPY slice -> SPLG
    assert "DBC" not in df.index and "PDBC" in df.index         # DBC -> PDBC (no K-1)
    assert abs(df.loc["SPLG", "price"] - 85.76) < 1e-9          # substitute priced live


# ----------------------------------------------- red-team regressions (2026-07-01)
def test_picker_refuses_partial_quote_row():
    # A transient feed hiccup (one ETF's last bar NaN) must BLOCK the lock, not silently
    # drop the asset and renormalize the book (red-team attack #1).
    import numpy as np
    import backtest.trend_sleeve as ts
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(11)
    panel = pd.DataFrame({t: 100 * np.cumprod(1 + 0.004 + rng.normal(0, 0.012, n))
                          for t in ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]}, index=idx)
    panel.iloc[-1, panel.columns.get_loc("GLD")] = float("nan")
    orig = (ts.etf_panel, tracker.get_prices)
    ts.etf_panel = lambda *a, **k: {"Close": panel}
    tracker.get_prices = lambda t, *a, **k: pd.DataFrame(
        {"Close": pd.Series([100.0], index=[idx[-1]])})
    try:
        try:
            tracker.sso_stack_picks(cash_etf=None)
        except RuntimeError as e:
            assert "GLD" in str(e)
            return
        raise AssertionError("expected RuntimeError on a partial quote row")
    finally:
        ts.etf_panel, tracker.get_prices = orig


def test_shadow_survives_broken_rf_feed():
    # A stale/broken T-bill series must degrade to spread-only financing with a warning,
    # never print $nan (red-team attack #2).
    import numpy as np
    import tempfile
    import json as _json
    import backtest.leverage_study as ls
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "shadowrf"))
    for d, px in zip(["2026-01-02", "2026-02-02"], [100.0, 110.0]):
        rec = {"lock_date": d, "data_asof": d, "strategy": "shadowrf", "n": 1,
               "picks": {"A": 1.0}, "lock_prices": {"A": px}, "spy_lock": 100.0}
        with open(os.path.join(tmp, "shadowrf", d + ".json"), "w") as f:
            _json.dump(rec, f)
    idx = pd.to_datetime(["2026-01-02", "2026-02-02"])
    panel = pd.DataFrame({"A": [100.0, 110.0]}, index=idx)
    spy = pd.Series([500.0, 505.0], index=idx)
    orig = (tracker.PICKS_DIR, tracker.download_panel, tracker.get_prices, ls.tbill_series)
    tracker.PICKS_DIR = tmp
    tracker.download_panel = lambda *a, **k: {"Close": panel}
    tracker.get_prices = lambda *a, **k: pd.DataFrame({"Close": spy})
    ls.tbill_series = lambda index, **k: pd.Series(np.nan, index=index)   # broken feed
    try:
        eq = tracker.report_shadow("shadowrf", leverage=2.0, spread=0.0, cost_bps=0.0)
    finally:
        tracker.PICKS_DIR, tracker.download_panel, tracker.get_prices, ls.tbill_series = orig
    assert np.isfinite(float(eq.iloc[-1]))
    assert abs(float(eq.iloc[-1]) - 12_000.0) < 1e-6       # 2x(+10%), rf fell back to 0


def test_simulate_marks_vanished_price_at_last_trade():
    # A name whose price disappears mid-month is scored at its LAST traded price,
    # not flat 0% (red-team attack #4: the optimism leak).
    recs = [{"data_asof": "2026-01-02", "picks": {"A": 0.5, "B": 0.5}}]
    dates = pd.to_datetime(["2026-01-02", "2026-01-20", "2026-02-02"])
    closes = pd.DataFrame({"A": [100.0, 100.0, 100.0],
                           "B": [50.0, 25.0, float("nan")]}, index=dates)  # B halves, then gone
    spy = pd.Series([400.0, 405.0, 410.0], index=dates)
    _, s = tracker._simulate(recs, closes, spy, 10_000.0, cost_bps=0.0)
    assert abs(s["ret"] - (0.5 * 0.0 + 0.5 * -0.5)) < 1e-12   # -25% book, not 0%


# ----------------------------------------------- shadow levered book
def test_shadow_lev_arithmetic():
    # Two locks a month apart, base book +10% in the month, rf=0 -> shadow 2x = +20%
    # (financing is zero at rf=0/spread=0, so the shadow is exactly L x the base).
    import tempfile
    import json as _json
    import backtest.leverage_study as ls
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "shadowtest"))
    dates = ["2026-01-02", "2026-02-02"]
    for d, px in zip(dates, [100.0, 110.0]):
        rec = {"lock_date": d, "data_asof": d, "strategy": "shadowtest", "n": 1,
               "picks": {"A": 1.0}, "lock_prices": {"A": px}, "spy_lock": 100.0}
        with open(os.path.join(tmp, "shadowtest", d + ".json"), "w") as f:
            _json.dump(rec, f)
    idx = pd.to_datetime(dates)
    panel = pd.DataFrame({"A": [100.0, 110.0]}, index=idx)
    spy = pd.Series([500.0, 505.0], index=idx)
    orig = (tracker.PICKS_DIR, tracker.download_panel, tracker.get_prices, ls.tbill_series)
    tracker.PICKS_DIR = tmp
    tracker.download_panel = lambda *a, **k: {"Close": panel}
    tracker.get_prices = lambda *a, **k: pd.DataFrame({"Close": spy})
    ls.tbill_series = lambda index, **k: pd.Series(0.0, index=index)
    try:
        eq = tracker.report_shadow("shadowtest", leverage=2.0, spread=0.0, cost_bps=0.0)
    finally:
        (tracker.PICKS_DIR, tracker.download_panel, tracker.get_prices, ls.tbill_series) = orig
    assert abs(float(eq.iloc[-1]) - 12_000.0) < 1e-6        # 2 x (+10%) on $10k


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


# ----------------------------------------------- net-of-cost simulation
_DATES3 = pd.to_datetime(["2026-01-02", "2026-02-02", "2026-03-02"])
_SPY3 = pd.Series([400.0, 410.0, 420.0], index=_DATES3)


def test_simulate_charges_the_initial_buy_in():
    recs = [{"data_asof": "2026-01-02", "picks": {"A": 1.0}}]
    closes = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=_DATES3)
    _, s = tracker._simulate(recs, closes, _SPY3, 10_000.0, cost_bps=10.0)
    assert abs(s["final"] - 10_000.0 * (1 - 0.001)) < 1e-6  # 100% turnover x 10bps
    assert abs(s["turnovers"][0] - 1.0) < 1e-12


def test_simulate_identical_relock_flat_prices_costs_nothing():
    recs = [{"data_asof": "2026-01-02", "picks": {"A": 0.5, "B": 0.5}},
            {"data_asof": "2026-02-02", "picks": {"A": 0.5, "B": 0.5}}]
    closes = pd.DataFrame({"A": [100.0] * 3, "B": [50.0] * 3}, index=_DATES3)
    _, s = tracker._simulate(recs, closes, _SPY3, 10_000.0, cost_bps=10.0)
    assert abs(s["turnovers"][1]) < 1e-12                   # no drift, no trade
    assert abs(s["cum_cost"] - 0.001) < 1e-12               # only the buy-in


def test_simulate_full_swap_costs_two_way_turnover():
    recs = [{"data_asof": "2026-01-02", "picks": {"A": 1.0}},
            {"data_asof": "2026-02-02", "picks": {"B": 1.0}}]
    closes = pd.DataFrame({"A": [100.0] * 3, "B": [50.0] * 3}, index=_DATES3)
    _, s = tracker._simulate(recs, closes, _SPY3, 10_000.0, cost_bps=10.0)
    assert abs(s["turnovers"][1] - 2.0) < 1e-12             # sell 100% + buy 100%
    assert abs(s["final"] - 10_000.0 * (1 - 0.001) * (1 - 0.002)) < 1e-6


def test_simulate_drift_makes_held_names_cost_a_trim():
    # Same picks re-locked, but A doubled while B was flat: A drifted 50%->2/3 of the
    # book, so rebalancing back to 50/50 trades |2/3-1/2|*2 = 1/3 two-way.
    recs = [{"data_asof": "2026-01-02", "picks": {"A": 0.5, "B": 0.5}},
            {"data_asof": "2026-02-02", "picks": {"A": 0.5, "B": 0.5}}]
    closes = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [50.0, 50.0, 50.0]}, index=_DATES3)
    _, s = tracker._simulate(recs, closes, _SPY3, 10_000.0, cost_bps=10.0)
    assert abs(s["turnovers"][1] - 1.0 / 3.0) < 1e-12


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


def test_lots_derive_hifo_and_terms():
    # The lot ledger's pure core: HIFO consumption, ST/LT classification (>1yr),
    # oversell detection.
    from backtest.lots import derive
    journal = [
        {"date": "2025-01-10", "side": "BUY", "ticker": "X", "shares": 10.0, "dollars": 1000.0},
        {"date": "2026-06-01", "side": "BUY", "ticker": "X", "shares": 10.0, "dollars": 1500.0},
        {"date": "2026-07-01", "side": "SELL", "ticker": "X", "shares": 5.0, "dollars": 800.0},
    ]
    open_lots, realized = derive(journal)
    # HIFO: the 2026 lot (basis 150/sh) is consumed first
    assert len(realized) == 1 and realized[0]["term"] == "ST"       # held < 1yr
    assert abs(realized[0]["basis"] - 750.0) < 1e-9                 # 5 sh x $150
    assert abs(realized[0]["gain"] - 50.0) < 1e-9                   # 800 - 750
    assert abs(sum(l["shares"] for l in open_lots) - 15.0) < 1e-9
    # selling the old lot after >1yr is LT
    journal2 = journal + [{"date": "2026-07-02", "side": "SELL", "ticker": "X",
                           "shares": 10.0, "dollars": 1600.0}]
    _, realized2 = derive(journal2)
    terms = sorted(r["term"] for r in realized2[1:])                # 5 sh 2026-lot + 5 sh 2025-lot
    assert terms == ["LT", "ST"]
    # oversell raises
    journal3 = journal + [{"date": "2026-07-03", "side": "SELL", "ticker": "X",
                           "shares": 99.0, "dollars": 9.9}]
    try:
        derive(journal3)
        raise AssertionError("expected oversell ValueError")
    except ValueError:
        pass


def test_mechanism_book_beta_identities():
    # The crisis gauge's beta: SPY-only book -> beta 1, UPRO-only -> 3 (3x SPY),
    # SGOV ignored, a 50/50 SPY/cash book -> 0.5.
    import numpy as np
    from backtest.mechanism import book_beta
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-01", periods=80)
    spy_ret = rng.normal(0.0005, 0.01, len(idx))
    closes = pd.DataFrame({"SPY": 100 * np.cumprod(1 + spy_ret),
                           "IEF": 100 * np.cumprod(1 + rng.normal(0, 0.003, len(idx)))},
                          index=idx)
    i = len(closes) - 1
    assert abs(book_beta({"SPY": 1.0}, closes, i) - 1.0) < 1e-9
    assert abs(book_beta({"UPRO": 1.0}, closes, i) - 3.0) < 1e-9
    assert abs(book_beta({"SPY": 0.5, "SGOV": 0.5}, closes, i) - 0.5) < 1e-9


def test_impl_gap_twin_scales_flows_from_anchors():
    # The pure core of the implementation-gap twin: each flow compounds from its own
    # anchor date, so a later deposit doesn't masquerade as performance.
    import numpy as np
    from backtest.impl_gap import _twin_from
    idx = pd.bdate_range("2026-07-13", periods=40)
    curve = pd.Series(10_000.0 * (1.01 ** np.arange(40)), index=idx)   # +1%/day
    flows = [("2026-07-13", 10_000.0)]
    # single flow: twin == flow grown by the curve
    assert abs(_twin_from(curve, flows) / (10_000.0 * 1.01 ** 39) - 1) < 1e-12
    # second flow half-way: grows only from ITS anchor
    flows2 = flows + [(idx[20].date().isoformat(), 5_000.0)]
    expect = 10_000.0 * 1.01 ** 39 + 5_000.0 * 1.01 ** 19
    assert abs(_twin_from(curve, flows2) / expect - 1) < 1e-12
    # asof clipping: value at an intermediate date uses that date's curve level
    mid = _twin_from(curve, flows, asof=idx[10])
    assert abs(mid / (10_000.0 * 1.01 ** 10) - 1) < 1e-12


def test_band_trades_buffered_rebalance():
    # The pure core of rebalance_orders (adopted 2026-07-13, buffer_frac=0.10):
    # within-band legs hold, outside-band legs trade to the NEAREST EDGE, dropped
    # legs exit fully, new legs enter at the lower edge.
    cur = {"UPRO": 4300.0, "IEF": 2000.0, "TLT": 1150.0, "GLD": 400.0}
    tgt = {"UPRO": 4000.0, "IEF": 2020.0, "TLT": 0.0, "EFA": 1700.0}
    tr = tracker._band_trades(cur, tgt, frac=0.10)
    assert "UPRO" not in tr                                 # band [3600,4400], cur 4300 inside
    assert "IEF" not in tr                                  # band [1818,2222], cur 2000 inside
    assert abs(tr["TLT"] + 1150.0) < 1e-9                   # dropped -> full exit
    assert abs(tr["GLD"] + 400.0) < 1e-9                    # absent from targets -> full exit
    assert abs(tr["EFA"] - 1530.0) < 1e-9                   # new leg enters at LOWER edge
    tr2 = tracker._band_trades({"A": 5000.0}, {"A": 4000.0}, frac=0.10)
    assert abs(tr2["A"] + 600.0) < 1e-9                     # to the nearest edge (4400), not 4000


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
