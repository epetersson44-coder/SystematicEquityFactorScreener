# backtest/tests/test_leverage_study.py — LETF/LRS/stacking arithmetic, offline.

import numpy as np
import pandas as pd

from backtest.leverage_study import (letf_returns, lrs_returns, stacked_returns,
                                     levered_returns, SPREAD, ER2)


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_letf_daily_formula_known_answer():
    spy = _series([0.01, -0.02])
    rf = _series([0.04, 0.04])
    r = letf_returns(spy, 2, 0.0089, rf)
    expect0 = 2 * 0.01 - (0.04 + SPREAD) / 252 - 0.0089 / 252
    assert abs(r.iloc[0] - expect0) < 1e-15
    expect1 = 2 * -0.02 - (0.04 + SPREAD) / 252 - 0.0089 / 252
    assert abs(r.iloc[1] - expect1) < 1e-15


def test_letf_vol_drag_emerges_from_compounding():
    # underlying flat over 2 days via +10%/-9.09..%; the 2x compounds to a LOSS
    spy = _series([0.10, -0.10 / 1.10])
    rf = _series([0.0, 0.0])
    two_x = (1 + letf_returns(spy, 2, 0.0, rf)).prod() - 1
    underlying = (1 + spy).prod() - 1
    assert abs(underlying) < 1e-12                          # flat underlying
    assert two_x < -0.01                                    # levered path decayed


def test_lrs_switches_to_tbills_below_ma():
    n = 260
    up = np.linspace(100.0, 150.0, n // 2)
    down = np.linspace(150.0, 90.0, n - n // 2)             # decisive breakdown
    px = _series(np.concatenate([up, down]))
    ret = px.pct_change().dropna()
    rf = pd.Series(0.05, index=ret.index)
    r = lrs_returns(px, ret, 2, ER2, rf, ma=50, switch_bps=0)
    # late in the decline SPY is far below its 50d MA -> position = T-bills
    tail = r.iloc[-10:]
    assert np.allclose(tail.values, 0.05 / 252, atol=1e-12)
    # early in the rally (after MA warmup) it holds the 2x
    on_day = r.index[80]
    expect = 2 * ret.loc[on_day] - (0.05 + SPREAD) / 252 - ER2 / 252
    assert abs(r.loc[on_day] - expect) < 1e-12


def test_lrs_charges_switch_cost_on_transitions():
    px = _series(np.concatenate([np.linspace(100, 140, 130),
                                 np.linspace(140, 100, 130)]))
    ret = px.pct_change().dropna()
    rf = pd.Series(0.0, index=ret.index)
    free = lrs_returns(px, ret, 2, 0.0, rf, ma=50, switch_bps=0)
    paid = lrs_returns(px, ret, 2, 0.0, rf, ma=50, switch_bps=10)
    n_switches = round(float((free - paid).sum() / (10 / 10_000 * 2)))
    assert n_switches >= 1                                   # at least the breakdown exit
    assert (paid <= free + 1e-15).all()                      # costs only ever subtract


def test_stack_and_lever_arithmetic():
    spy = _series([0.01])
    ov = _series([0.002])
    rf = _series([0.03])
    st = stacked_returns(spy, ov, 0.5, rf)
    assert abs(st.iloc[0] - (0.01 + 0.5 * 0.002 - 0.5 * (0.03 + SPREAD) / 252)) < 1e-15
    lv = levered_returns(spy, 2.0, rf)
    assert abs(lv.iloc[0] - (0.02 - (0.03 + SPREAD) / 252)) < 1e-15
    # lam=0 / L=1 degrade to plain SPY (no phantom financing)
    assert abs(stacked_returns(spy, ov, 0.0, rf).iloc[0] - 0.01) < 1e-15
    assert abs(levered_returns(spy, 1.0, rf).iloc[0] - 0.01) < 1e-15


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
