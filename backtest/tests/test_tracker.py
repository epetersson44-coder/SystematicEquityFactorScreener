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
