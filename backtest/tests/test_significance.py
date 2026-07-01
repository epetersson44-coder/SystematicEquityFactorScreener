# backtest/tests/test_significance.py — PSR/DSR + block-bootstrap sanity, offline.

import numpy as np
import pandas as pd

from backtest.significance import (sharpe_daily, psr, expected_max_sharpe, dsr,
                                   block_bootstrap_sharpe_diff)


def _normal_rets(mu, sd, n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n))


def test_psr_high_for_clear_edge_and_half_at_own_sharpe():
    r = _normal_rets(0.0008, 0.01, 5000)                 # daily SR ~0.08, 20yr
    assert psr(r) > 0.99                                  # clearly > 0
    assert abs(psr(r, sr_benchmark_daily=sharpe_daily(r)) - 0.5) < 1e-9


def test_psr_penalizes_bad_higher_moments():
    rng = np.random.default_rng(1)
    n = 5000
    base = rng.normal(0.0006, 0.008, n)
    crashy = base.copy()
    crashy[rng.integers(0, n, 25)] -= 0.06                # rare big losses (neg skew, fat tails)
    # re-match mean AND sd exactly (standardize, then rescale) -> identical Sharpe
    crashy = (crashy - crashy.mean()) / crashy.std(ddof=1) * base.std(ddof=1) + base.mean()
    assert abs(sharpe_daily(crashy) - sharpe_daily(base)) < 1e-9
    assert psr(pd.Series(crashy)) < psr(pd.Series(base))  # confidence must drop anyway


def test_expected_max_sharpe_grows_with_trials():
    v = 0.03 ** 2
    hurdles = [expected_max_sharpe(n, v) for n in (2, 10, 50, 200)]
    assert hurdles[0] > 0
    assert all(b > a for a, b in zip(hurdles, hurdles[1:]))
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_below_psr_and_falls_with_more_trials():
    r = _normal_rets(0.0005, 0.01, 5000, seed=2)
    trials = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, -0.3]
    d10, _ = dsr(r, 10, trial_sharpes_annual=trials)
    d100, _ = dsr(r, 100, trial_sharpes_annual=trials)
    assert d10 < psr(r)                                   # deflation is a higher hurdle
    assert d100 < d10                                     # more trials -> less confidence


def test_bootstrap_no_edge_means_pvalue_near_half():
    # two INDEPENDENT series with the same return distribution (identical series would
    # give diff == 0 in every resample, which is p = 1 by construction, not "no edge")
    a = _normal_rets(0.0004, 0.01, 4000, seed=3)
    b = _normal_rets(0.0004, 0.01, 4000, seed=13)
    bb = block_bootstrap_sharpe_diff(a, b, n_boot=400, seed=4)
    assert 0.2 < bb["p_value_luck"] < 0.8


def test_bootstrap_detects_a_real_edge():
    rng = np.random.default_rng(5)
    n = 5000
    b = pd.Series(rng.normal(0.0003, 0.012, n))           # SPY-ish
    a = pd.Series(b.values * 0.5 + rng.normal(0.0006, 0.005, n))  # higher SR, correlated
    bb = block_bootstrap_sharpe_diff(a, b, n_boot=400, seed=6)
    assert bb["observed_diff"] > 0.3
    assert bb["p_value_luck"] < 0.05
    assert bb["ci95"][0] < bb["observed_diff"] < bb["ci95"][1]


def test_bootstrap_needs_enough_data():
    r = _normal_rets(0.0, 0.01, 50)
    try:
        block_bootstrap_sharpe_diff(r, r, block=21)
    except ValueError:
        return
    raise AssertionError("expected ValueError on too-short series")


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
