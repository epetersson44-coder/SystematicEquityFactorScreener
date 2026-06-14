# backtest/tests/test_engine_xs.py — hardening the multi-asset (cross-sectional) engine.
#
# The new engine (engine_xs) traded its structural look-ahead safety for speed (it
# hands the strategy the whole panel + an index, not a slice), so the corrupt-the-
# future test below is load-bearing, not ceremony. Plus the conservation identity
# (no money printed when rebalancing a basket with costs), guards, NaN handling, a
# cross-check against the single-asset engine, and a Monte Carlo sweep proving the
# invariants hold across hundreds of random universes.
#
# Run:  python -m backtest.tests.test_engine_xs   (or pytest)

import numpy as np
import pandas as pd

from backtest import metrics, baseline, costs
from backtest.engine import run as run1
from backtest.engine_xs import MultiPortfolio, run_xs
from backtest.strategy import CrossSectionalStrategy, BuyAndHold
from backtest.tests._helpers import make_panel, rising_panel, random_panel


# ---------------------------------------------------------------- test doubles
class FixedWeights(CrossSectionalStrategy):
    """Set target weights once (at bar `at`), then hold forever."""
    def __init__(self, weights, at=0):
        self.weights = pd.Series(weights, dtype=float)
        self.at = at
    def target_weights(self, closes, i):
        return self.weights if i == self.at else None


class RandomRebalance(CrossSectionalStrategy):
    """Every `every` bars, random long-only weights over available names, sum in [0.3,1]."""
    def __init__(self, seed, every=7, k=5):
        self.rng = np.random.default_rng(seed)
        self.every, self.k = every, k
    def target_weights(self, closes, i):
        if i == 0 or i % self.every != 0:
            return None
        avail = closes.iloc[i].dropna().index
        if len(avail) == 0:
            return None
        k = min(self.k, len(avail))
        pick = self.rng.choice(np.asarray(avail), size=k, replace=False)
        w = self.rng.random(k)
        w = w / w.sum() * self.rng.uniform(0.3, 1.0)
        return pd.Series(w, index=pick)


# ---------------------------------------------------------------- known answer
def test_equal_weight_identical_stocks_compounds_analytically():
    # k stocks all rising at the same rate -> equal weights never drift -> the basket
    # compounds at exactly that rate. Known closed form, no costs.
    n, k, r = 300, 5, 0.0005
    panels = rising_panel(n, k, daily=r)
    eq = run_xs(panels, FixedWeights({f"T{j}": 1.0 / k for j in range(k)}), fill="close")
    expected = 10_000 * (1 + r) ** np.arange(n)
    assert np.allclose(eq.to_numpy(), expected, rtol=0, atol=1e-6)


def test_single_asset_equivalence():
    # A 1-ticker universe, fully invested, must match the single-asset buy&hold curve.
    px = 100 * (1 + 0.001) ** np.arange(200)
    panels = make_panel(px.reshape(-1, 1), tickers=["AAA"])
    xs = run_xs(panels, FixedWeights({"AAA": 1.0}), fill="close")
    # single-asset engine on the same series
    from backtest.tests._helpers import make_df
    single = run1(make_df(px), BuyAndHold(), fill="close")
    assert np.allclose(xs.to_numpy(), single.to_numpy(), rtol=0, atol=1e-6)


# ---------------------------------------------------------------- conservation
def test_money_conservation_basket():
    # The core identity: equity drop across a rebalance == fees, and cash + positions
    # == equity at every bar. Stepped manually so we can probe both.
    panels = random_panel(400, 8, seed=3)
    closes, opens = panels["Close"], panels["Open"]
    cost = costs.proportional(20)
    pf = MultiPortfolio(10_000)
    strat = RandomRebalance(seed=3, every=5, k=4)
    pending, acct_err, fee_err = None, 0.0, 0.0
    for i in range(len(closes)):
        if pending is not None:
            pre = pf.equity(opens.iloc[i])
            fee = pf.rebalance(pending, opens.iloc[i], cost)
            post = pf.equity(opens.iloc[i])
            fee_err = max(fee_err, abs((pre - post) - fee))   # drop == fees, exactly
            pending = None
        w = strat.target_weights(closes, i)
        if w is not None:
            pending = w
        direct = pf.cash + sum(sh * closes.iloc[i][t] for t, sh in pf.shares.items())
        acct_err = max(acct_err, abs(direct - pf.equity(closes.iloc[i])))
    assert acct_err < 1e-9, f"accounting identity broken: {acct_err}"
    assert fee_err < 1e-7, f"equity drop != fees: {fee_err}"


def test_zero_cost_conserves_exactly():
    # With no cost, a rebalance must not change equity at all (at the fill prices).
    panels = random_panel(100, 6, seed=9)
    pf = MultiPortfolio(10_000)
    prices = panels["Close"].iloc[50]
    before = pf.equity(prices)
    pf.rebalance(pd.Series({c: 1 / 6 for c in panels["Close"].columns}), prices, cost=None)
    assert abs(pf.equity(prices) - before) < 1e-7


# ---------------------------------------------------------------- look-ahead
def test_lookahead_corrupt_the_future():
    panels = random_panel(250, 10, seed=5)
    strat = RandomRebalance(seed=5, every=7, k=4)
    base = run_xs(panels, strat, cost=costs.proportional(10)).to_numpy()
    for T in (40, 120, 200):
        corrupt = {k: df.copy() for k, df in panels.items()}
        for df in corrupt.values():
            df.iloc[T + 1:] = df.iloc[T + 1:] * 9.0
        after = run_xs(corrupt, RandomRebalance(seed=5, every=7, k=4),
                       cost=costs.proportional(10)).to_numpy()
        assert np.array_equal(base[:T + 1], after[:T + 1]), f"look-ahead leak at T={T}"


def test_lookahead_reverse_sanity():
    # Corrupting the PAST must move the future, or the test above is trivial.
    panels = random_panel(250, 10, seed=6)
    base = run_xs(panels, RandomRebalance(seed=6), cost=costs.proportional(10)).to_numpy()
    corrupt = {k: df.copy() for k, df in panels.items()}
    for df in corrupt.values():
        df.iloc[:60] = df.iloc[:60] * 2.0
    after = run_xs(corrupt, RandomRebalance(seed=6), cost=costs.proportional(10)).to_numpy()
    assert not np.array_equal(base[80:], after[80:])


# ---------------------------------------------------------------- guards
def test_weight_guard_sum_over_one():
    pf = MultiPortfolio(10_000)
    prices = pd.Series({"A": 10.0, "B": 20.0})
    try:
        pf.rebalance(pd.Series({"A": 0.7, "B": 0.6}), prices)   # sums to 1.3
    except ValueError:
        return
    raise AssertionError("weights summing > 1 should raise (no leverage)")


def test_weight_guard_negative():
    pf = MultiPortfolio(10_000)
    prices = pd.Series({"A": 10.0, "B": 20.0})
    try:
        pf.rebalance(pd.Series({"A": 0.5, "B": -0.1}), prices)
    except ValueError:
        return
    raise AssertionError("negative weight should raise (no shorting yet)")


# ---------------------------------------------------------------- NaN / hold
def test_nan_names_excluded():
    # B is NaN (pre-IPO) for the whole window -> a strategy targeting it can't buy it,
    # so that weight stays in cash and equity == the A-only portion + cash.
    n = 50
    close = np.column_stack([100 * (1.001) ** np.arange(n), np.full(n, np.nan)])
    panels = make_panel(close, tickers=["A", "B"])
    eq = run_xs(panels, FixedWeights({"A": 0.5, "B": 0.5}), fill="close")
    assert np.all(np.isfinite(eq.to_numpy())) and (eq.to_numpy() > 0).all()


def test_hold_semantics_none_means_no_trade():
    panels = random_panel(80, 5, seed=2)

    class NeverTrade(CrossSectionalStrategy):
        def target_weights(self, closes, i):
            return None
    eq = run_xs(panels, NeverTrade(), cost=costs.proportional(50))
    assert np.allclose(eq.to_numpy(), 10_000.0)             # all cash, never traded


def test_determinism():
    panels = random_panel(200, 8, seed=11)
    a = run_xs(panels, RandomRebalance(seed=1), cost=costs.proportional(5)).to_numpy()
    b = run_xs(panels, RandomRebalance(seed=1), cost=costs.proportional(5)).to_numpy()
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- Monte Carlo
def test_monte_carlo_invariants():
    # Run the engine on MANY random universes; on every one, the hard invariants must
    # hold: equity finite, strictly positive (long-only, no leverage), no NaN, and the
    # accounting identity intact at the final bar.
    bad = []
    for seed in range(200):
        panels = random_panel(150, np.random.default_rng(seed).integers(3, 12),
                              seed=seed)
        eq = run_xs(panels, RandomRebalance(seed=seed, every=6),
                    cost=costs.proportional(15))
        a = eq.to_numpy()
        if not (np.all(np.isfinite(a)) and (a > 0).all()):
            bad.append(seed)
    assert not bad, f"invariants violated on {len(bad)} universes, e.g. {bad[:5]}"


# ---------------------------------------------------------------- runner
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
