# backtest/tests/test_timing_luck.py — offset gating + tranching arithmetic.
#
# Offline + synthetic. Pins: (1) the offset knob actually moves the rebalance bar (and
# defaults to the historical behaviour), (2) a tranched curve is the mean of its
# sub-curves, (3) blend_curve compounds a constant-mix return stream correctly.

import numpy as np
import pandas as pd

from backtest.trend_sleeve import TSMOM, VolTargetTSMOM
from backtest.timing_luck import tranched_curve, blend_curve


def _panel(n=320, cols=("A", "B")):
    idx = pd.bdate_range("2020-01-01", periods=n)
    data = {c: np.linspace(100.0, 150.0, n) * (1 + 0.01 * k) for k, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


def test_offset_moves_the_rebalance_bar():
    closes = _panel()
    for strat in (TSMOM(offset=3), VolTargetTSMOM(offset=3)):
        fires = [i for i in range(len(closes)) if strat.target_weights(closes, i) is not None]
        assert fires, "strategy never rebalanced"
        assert all(i % 21 == 3 for i in fires)            # trades only on its offset
        assert all(i >= 252 for i in fires)               # warmup respected


def test_offset_zero_is_the_historical_default():
    closes = _panel()
    old_style = [i for i in range(len(closes))
                 if i >= 252 and i % 21 == 0]              # the pre-offset gating
    strat = VolTargetTSMOM()                               # default offset=0
    fires = [i for i in range(len(closes)) if strat.target_weights(closes, i) is not None]
    assert fires == old_style


def test_tranched_curve_is_mean_of_subcurves():
    idx = pd.bdate_range("2020-01-01", periods=5)
    curves = {0: pd.Series([100.0, 110, 120, 130, 140], index=idx),
              5: pd.Series([100.0, 90, 100, 110, 120], index=idx)}
    tr = tranched_curve(curves, offsets=(0, 5))
    assert np.allclose(tr.values, [100, 100, 110, 120, 130])


def test_blend_curve_constant_mix_known_answer():
    idx = pd.bdate_range("2020-01-01", periods=3)
    spy = pd.Series([100.0, 110.0, 121.0], index=idx)      # +10%/day
    trend = pd.Series([100.0, 100.0, 100.0], index=idx)    # flat
    b = blend_curve(trend, spy, w_spy=0.5)                 # 50/50 -> +5%/day
    assert np.allclose(b.values, [10_500.0, 11_025.0])


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
