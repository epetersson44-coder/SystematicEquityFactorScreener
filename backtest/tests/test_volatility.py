# backtest/tests/test_volatility.py — Yang-Zhang estimator + strategy integration.

import numpy as np
import pandas as pd

from backtest.volatility import yang_zhang, close_to_close
from backtest.trend_sleeve import VolTargetTSMOM


def _gbm_ohlc(n=1500, ann_vol=0.16, seed=0):
    """Synthetic OHLC with known volatility: continuous GBM sampled intraday so
    O/H/L/C are internally consistent (H >= max(O,C), L <= min(O,C))."""
    rng = np.random.default_rng(seed)
    steps = 130                # intraday sub-steps — coarse sampling understates H/L
                               # (discretization bias), so sample finely
    sd = ann_vol / np.sqrt(252 * steps)
    path = np.cumsum(rng.normal(0, sd, n * steps)).reshape(n, steps)
    base = np.concatenate([[0.0], path[:-1, -1]])[:, None] # yesterday's close level
    day = base + (path - base)                             # absolute log-price intraday
    o = np.exp(day[:, 0]); c = np.exp(day[:, -1])
    h = np.exp(day.max(axis=1)); l = np.exp(day.min(axis=1))
    idx = pd.bdate_range("2018-01-01", periods=n)
    f = lambda x: pd.DataFrame({"A": x}, index=idx)
    return f(o), f(h), f(l), f(c)


def test_yz_recovers_true_vol_on_gbm():
    o, h, l, c = _gbm_ohlc(ann_vol=0.16, seed=1)
    yz = yang_zhang(o, h, l, c, window=63)["A"].dropna()
    assert abs(yz.mean() - 0.16) < 0.02                    # unbiased-ish at true vol


def test_yz_less_noisy_than_close_to_close():
    # Same window, same data: the range estimator's sampling noise must be smaller
    # (that's the efficiency claim the turnover reduction rests on).
    o, h, l, c = _gbm_ohlc(ann_vol=0.16, seed=2)
    yz = yang_zhang(o, h, l, c, window=63)["A"].dropna()
    cc = close_to_close(c, window=63)["A"].reindex(yz.index)
    assert yz.std() < cc.std() * 0.8


def test_vol_df_none_is_bit_identical_to_old_path():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=700)
    closes = pd.DataFrame({k: 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 700))
                           for k in ["A", "B", "C"]}, index=idx)
    a, b = VolTargetTSMOM(), VolTargetTSMOM(vol_df=None)
    for i in range(0, 700, 21):
        wa, wb = a.target_weights(closes, i), b.target_weights(closes, i)
        assert (wa is None) == (wb is None)
        if wa is not None:
            assert np.allclose(wa.sort_index().values, wb.sort_index().values, atol=1e-15)


def test_vol_df_misaligned_index_raises():
    # Same-length but date-shifted vol panel would silently size off the wrong days'
    # vols via positional lookup (red-team attack #3) — must raise instead.
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2018-01-01", periods=700)
    closes = pd.DataFrame({k: 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.01, 700))
                           for k in ["A", "B"]}, index=idx)
    shifted = pd.DataFrame(0.2, index=pd.bdate_range("2018-02-01", periods=700),
                           columns=["A", "B"])                # same length, wrong dates
    try:
        VolTargetTSMOM(vol_df=shifted).target_weights(closes, 630)
    except ValueError:
        return
    raise AssertionError("expected ValueError on misaligned vol_df")


def test_vol_df_changes_sizing_not_selection():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2018-01-01", periods=700)
    closes = pd.DataFrame({k: 100 * np.cumprod(1 + 0.0006 + rng.normal(0, 0.012, 700))
                           for k in ["A", "B"]}, index=idx)
    vol_df = pd.DataFrame(0.20, index=idx, columns=["A", "B"])   # flat external estimate
    w_cc = VolTargetTSMOM().target_weights(closes, 630)
    w_yz = VolTargetTSMOM(vol_df=vol_df).target_weights(closes, 630)
    assert w_cc is not None and w_yz is not None
    assert set(w_cc.index) == set(w_yz.index)              # same names selected...
    assert not np.allclose(w_cc.sort_index().values, w_yz.sort_index().values)  # ...sized differently


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
