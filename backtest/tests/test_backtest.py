# backtest/tests/test_backtest.py — the permanent stress suite.
#
# This is the regression net the old /tmp harness should have been: every
# correctness property we proved by hand in Phase 1, frozen so Phase 2 (and the
# shorting/multi-asset rework in Phase 3) can't silently break them.
#
# Runs two ways:
#   pytest backtest/tests/                       (if pytest is installed)
#   python -m backtest.tests.test_backtest       (plain runner, no deps)
#
# All tests use SYNTHETIC data — deterministic, offline, fast. The one property
# that matters most lives in test_lookahead_*: corrupt the future, prove the past
# can't feel it.

import numpy as np
import pandas as pd

from backtest import metrics, baseline, costs
from backtest.engine import Portfolio, run
from backtest.strategy import BuyAndHold, SMACrossover
from backtest.tests._helpers import make_df, rising, ConstantWeight
import backtest.data as data


# ---------------------------------------------------------------- metrics
def test_total_return():
    eq = pd.Series([100.0, 150.0], index=pd.to_datetime(["2020-01-01", "2020-06-01"]))
    assert abs(metrics.total_return(eq) - 0.5) < 1e-12


def test_cagr_known():
    # 100 -> 1600 over exactly 1461 days = 4.0 years => CAGR = 16^(1/4) - 1 = 1.0
    idx = pd.to_datetime(["2016-01-01"]) ; idx = idx.append(
        pd.to_datetime(["2016-01-01"]) + pd.Timedelta(days=1461))
    eq = pd.Series([100.0, 1600.0], index=idx)
    assert abs(metrics.cagr(eq) - 1.0) < 1e-9


def test_max_drawdown():
    eq = pd.Series([100.0, 120.0, 60.0, 90.0],
                   index=pd.bdate_range("2020-01-01", periods=4))
    assert abs(metrics.max_drawdown(eq) - (-0.5)) < 1e-12   # 120 -> 60


def test_max_drawdown_duration():
    # peak at day 0 (100), dips, reclaims 100 at index 10 -> 10 trading days,
    # but duration is CALENDAR days between those bdate indices.
    vals = [100, 95, 90, 92, 94, 96, 98, 99, 99, 99, 101, 102]
    eq = pd.Series(map(float, vals), index=pd.bdate_range("2020-01-01", periods=len(vals)))
    # underwater from index1..index9; reclaimed at index10.
    expected = (eq.index[10] - eq.index[0]).days
    assert metrics.max_drawdown_duration(eq) == expected


def test_drawdown_duration_unrecovered_counts():
    # never recovers -> duration runs to the last bar
    eq = pd.Series([100.0, 90.0, 80.0, 70.0], index=pd.bdate_range("2020-01-01", periods=4))
    assert metrics.max_drawdown_duration(eq) == (eq.index[-1] - eq.index[0]).days


def test_sharpe_zero_vol_is_nan():
    eq = pd.Series([100.0] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    assert np.isnan(metrics.sharpe(eq))


def test_calmar():
    eq = pd.Series([100.0, 120.0, 60.0, 130.0], index=pd.bdate_range("2020-01-01", periods=4))
    assert abs(metrics.calmar(eq) - metrics.cagr(eq) / abs(metrics.max_drawdown(eq))) < 1e-12


def test_sortino_no_downside_is_nan():
    eq = pd.Series(rising(20), index=pd.bdate_range("2020-01-01", periods=20))
    assert np.isnan(metrics.sortino(eq))


# ---------------------------------------------------------------- accounting
def test_buyhold_matches_analytic_baseline():
    # The conservation proof: engine fill=close buy&hold == closed-form curve.
    df = make_df(rising(300))
    analytic = baseline.buy_and_hold(df, initial_capital=10_000)
    engine = run(df, BuyAndHold(), initial_capital=10_000, fill="close")
    assert np.allclose(analytic.to_numpy(), engine.to_numpy(), rtol=0, atol=1e-6)


def test_flat_strategy_holds_capital():
    df = make_df(rising(50))
    eq = run(df, ConstantWeight(0.0), initial_capital=10_000)
    assert np.allclose(eq.to_numpy(), 10_000.0)            # 0 weight, no cash rate


def test_partial_weight_half_exposure():
    df = make_df(rising(100, daily=0.002))
    full = run(df, ConstantWeight(1.0), fill="close")
    half = run(df, ConstantWeight(0.5), fill="close")
    # half-invested gains roughly half the excess over starting capital
    assert 0.45 < (half.iloc[-1] - 10_000) / (full.iloc[-1] - 10_000) < 0.55


# ---------------------------------------------------------------- look-ahead
def test_lookahead_impossible_corrupt_the_future():
    df = make_df(rising(120, daily=0.003))
    strat = SMACrossover(fast=3, slow=8)
    base = run(df, strat, fill="next_open")

    T = 60
    corrupt = df.copy()
    corrupt.iloc[T + 1:] = corrupt.iloc[T + 1:] * 5.0      # blow up ALL future bars
    after = run(corrupt, SMACrossover(fast=3, slow=8), fill="next_open")

    # equity[0..T] must be byte-identical — the past cannot see the future.
    assert np.array_equal(base.to_numpy()[:T + 1], after.to_numpy()[:T + 1])


# ---------------------------------------------------------------- guards
def test_weight_guard_above_one():
    pf = Portfolio(10_000)
    try:
        pf.rebalance(1.5, 100.0)
    except ValueError:
        return
    raise AssertionError("weight 1.5 should raise (no leverage)")


def test_weight_guard_below_zero():
    pf = Portfolio(10_000)
    try:
        pf.rebalance(-0.1, 100.0)
    except ValueError:
        return
    raise AssertionError("weight -0.1 should raise (no shorting)")


def test_bad_fill_mode_raises():
    df = make_df(rising(10))
    try:
        run(df, BuyAndHold(), fill="teleport")
    except ValueError:
        return
    raise AssertionError("unknown fill mode should raise")


# ---------------------------------------------------------------- cash rate
def test_cash_rate_default_unchanged():
    # Regression guard: cash_rate=0 must equal not setting it (validation invariant).
    df = make_df(rising(80))
    a = run(df, SMACrossover(fast=3, slow=8))
    b = run(df, SMACrossover(fast=3, slow=8), cash_rate=0.0)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_cash_rate_accrues_on_idle_cash():
    # 0 weight = all cash; with a rate it should compound at (1 + r/252) per bar.
    n = 60
    df = make_df(rising(n))
    eq = run(df, ConstantWeight(0.0), initial_capital=10_000, cash_rate=0.04)
    expected = 10_000 * (1 + 0.04 / 252) ** n
    assert abs(eq.iloc[-1] - expected) < 1e-6


def test_cash_rate_barely_touches_buyhold():
    # Buy&hold holds ~0 cash, so a cash rate shouldn't materially move it.
    df = make_df(rising(200))
    flat = run(df, BuyAndHold(), fill="close")
    paid = run(df, BuyAndHold(), fill="close", cash_rate=0.04)
    assert abs(paid.iloc[-1] - flat.iloc[-1]) / flat.iloc[-1] < 1e-3


# ---------------------------------------------------------------- costs
def test_cost_zero_bps_is_free():
    c = costs.proportional(bps=0)
    assert c(100, 50.0) == 0.0


def test_cost_linear_in_size():
    c = costs.proportional(bps=10)
    assert abs(c(200, 50.0) - 2 * c(100, 50.0)) < 1e-12


def test_cost_symmetric_buy_vs_sell():
    c = costs.proportional(bps=10)
    assert c(100, 50.0) == c(-100, 50.0)


def test_cost_reduces_terminal_equity():
    df = make_df(rising(120, daily=0.003))
    free = run(df, SMACrossover(fast=3, slow=8), fill="close")
    paid = run(df, SMACrossover(fast=3, slow=8), fill="close", cost=costs.proportional(bps=50))
    assert paid.iloc[-1] < free.iloc[-1]


# ---------------------------------------------------------------- strategy
def test_sma_flat_during_warmup():
    df = make_df(rising(10))
    s = SMACrossover(fast=3, slow=8)
    assert s.target_weight(df.iloc[:5]) == 0.0            # < slow bars -> flat


def test_sma_output_is_binary():
    df = make_df(rising(50))
    s = SMACrossover(fast=3, slow=8)
    for i in range(8, 50):
        assert s.target_weight(df.iloc[:i + 1]) in (0.0, 1.0)


def test_sma_invalid_windows_raise():
    try:
        SMACrossover(fast=200, slow=50)
    except ValueError:
        return
    raise AssertionError("fast >= slow should raise")


# ---------------------------------------------------------------- determinism
def test_determinism():
    df = make_df(rising(100, daily=0.002))
    a = run(df, SMACrossover(fast=5, slow=20), cost=costs.proportional())
    b = run(df, SMACrossover(fast=5, slow=20), cost=costs.proportional())
    assert np.array_equal(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------- data hygiene
def _clean():
    return make_df(rising(20))


def test_validate_passes_clean():
    data._validate(_clean(), "T")                          # should not raise


def test_validate_rejects_duplicate_dates():
    df = _clean()
    df = pd.concat([df, df.iloc[[-1]]])
    _expect_valueerror(lambda: data._validate(df, "T"), "duplicate")


def test_validate_rejects_nonmonotonic():
    df = _clean().iloc[::-1]                                # reversed dates
    _expect_valueerror(lambda: data._validate(df, "T"), "non-monotonic")


def test_validate_rejects_nonpositive():
    df = _clean(); df.iloc[5, df.columns.get_loc("Close")] = 0.0
    _expect_valueerror(lambda: data._validate(df, "T"), "non-positive")


def test_validate_rejects_high_below_low():
    df = _clean(); df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] - 1
    _expect_valueerror(lambda: data._validate(df, "T"), "High<Low")


def _expect_valueerror(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: expected ValueError")


# ---------------------------------------------------------------- runner
if __name__ == "__main__":
    import sys
    tests = sorted(
        (n, f) for n, f in globals().items()
        if n.startswith("test_") and callable(f)
    )
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:                              # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
