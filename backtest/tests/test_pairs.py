# backtest/tests/test_pairs.py — the pairs-trading stats + strategy (Phase 3).
#
# Pins the hand-rolled cointegration math (no statsmodels) against known answers, and the
# PairsTrade state machine end-to-end through the engine: a stretched-then-converging
# spread must be entered and profit. Synthetic + offline.
#
# Run:  python -m backtest.tests.test_pairs   (or pytest)

import numpy as np
import pandas as pd

from backtest.pairs import (engle_granger, hedge_ratio, half_life, adf_tstat,
                            PairsTrade, EG_CRIT_5)
from backtest.engine_xs import run_xs
from backtest.tests._helpers import make_panel


# ----------------------------------------------------------------- stats
def test_adf_white_noise_is_stationary():
    s = np.random.default_rng(1).normal(0, 1, 500)
    assert adf_tstat(s) < -5            # very negative -> reject unit root -> stationary


def test_adf_random_walk_has_unit_root():
    rw = np.cumsum(np.random.default_rng(2).normal(0, 1, 500))
    assert adf_tstat(rw) > EG_CRIT_5    # near zero -> unit root, NOT stationary


def test_engle_granger_detects_cointegration():
    rng = np.random.default_rng(3)
    x = 100 + np.cumsum(rng.normal(0, 1, 600))
    y = 3 + 2.0 * x + rng.normal(0, 1.5, 600)        # y, x cointegrated with beta 2
    t, beta = engle_granger(y, x)
    assert t < EG_CRIT_5                              # cointegrated
    assert abs(beta - 2.0) < 0.1                      # recovers the true hedge ratio


def test_engle_granger_rejects_independent_walks():
    rng = np.random.default_rng(4)
    x = 100 + np.cumsum(rng.normal(0, 1, 600))
    y = 100 + np.cumsum(rng.normal(0, 1, 600))        # independent -> not cointegrated
    t, _ = engle_granger(y, x)
    assert t > EG_CRIT_5


def test_hedge_ratio_known():
    rng = np.random.default_rng(5)
    x = rng.normal(50, 5, 400)
    y = 1.0 + 1.5 * x + rng.normal(0, 0.5, 400)
    assert abs(hedge_ratio(y, x) - 1.5) < 0.05


def test_half_life_reverting_vs_not():
    rng = np.random.default_rng(6)
    # AR(1) with phi<1 mean-reverts (finite half-life); a random walk does not.
    n = 1000
    ar = np.zeros(n)
    for k in range(1, n):
        ar[k] = 0.9 * ar[k - 1] + rng.normal(0, 1)    # phi=0.9 -> hl = -ln2/ln0.9 ~ 6.6
    assert 3 < half_life(ar) < 12                  # fast mean reversion
    assert half_life(np.cumsum(rng.normal(0, 1, n))) > 100   # random walk: huge/inf, no reversion


# ----------------------------------------------------------------- strategy
def test_pairs_enters_and_profits_on_convergence():
    # Construct a pair whose spread is stretched then reverts: the strategy should SHORT the
    # rich leg, then profit as the spread collapses back to its mean.
    rng = np.random.default_rng(7)
    n = 140
    b = 50 + rng.normal(0, 0.3, n)
    dev = np.zeros(n)
    for t in range(42, n):
        dev[t] = 6.0 * np.exp(-(t - 42) / 8.0)        # spike to +6 at t=42, decays back to 0
    a = 50 + dev + rng.normal(0, 0.3, n)              # spread (a-b) ~ dev: stretched then reverts
    panels = make_panel(np.column_stack([a, b]), tickers=["A", "B"])
    strat = PairsTrade("A", "B", beta=1.0, window=20, entry=2.0, exit=0.5, stop=6.0)
    eq = run_xs(panels, strat, fill="close", allow_short=True, gross_max=1.0)
    assert np.all(np.isfinite(eq.to_numpy()))
    assert not np.allclose(eq.to_numpy(), 10_000.0)   # a position WAS taken (not flat all run)
    assert eq.iloc[-1] > 10_000.0                     # profited on the convergence


def test_pairs_stays_flat_without_signal():
    # A spread that never leaves its band -> never enters -> equity dead flat.
    rng = np.random.default_rng(8)
    n = 120
    b = 50 + rng.normal(0, 0.3, n)
    a = 50 + rng.normal(0, 0.3, n)                    # tiny noise, |z| never reaches entry=4
    panels = make_panel(np.column_stack([a, b]), tickers=["A", "B"])
    strat = PairsTrade("A", "B", beta=1.0, window=20, entry=4.0, exit=0.5, stop=8.0)
    eq = run_xs(panels, strat, fill="close", allow_short=True, gross_max=1.0)
    assert np.allclose(eq.to_numpy(), 10_000.0)       # never traded


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
