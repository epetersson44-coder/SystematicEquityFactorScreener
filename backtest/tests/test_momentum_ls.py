# backtest/tests/test_momentum_ls.py — momentum L/S strategy + the crash failsafes.
#
# The failsafe is only trustworthy if it's provably LOOK-AHEAD-FREE (it must use only past
# vol/trend) and the strategy is provably DOLLAR-NEUTRAL. Synthetic + offline.
#
# Run:  python -m backtest.tests.test_momentum_ls

import numpy as np
import pandas as pd

from backtest.momentum_ls import MomentumLS, vol_managed, trend_filtered
from backtest.constants import TRADING_DAYS
from backtest.tests._helpers import make_panel


def _first_weights(strat, panel):
    closes = panel["Close"]
    for i in range(len(closes)):
        w = strat.target_weights(closes, i)
        if w is not None:
            return w
    return None


# ----------------------------------------------------------------- strategy
def test_momentum_ls_is_dollar_neutral():
    # 30 names with dispersed trends -> a clear long top / short bottom decile.
    rng = np.random.default_rng(0)
    n, k = 320, 30
    drifts = np.linspace(-0.001, 0.001, k)                  # spread of momentum
    px = 100 * np.cumprod(1 + drifts + rng.normal(0, 0.005, (n, k)), axis=0)
    panel = make_panel(px, tickers=[f"T{j}" for j in range(k)])
    w = _first_weights(MomentumLS(), panel)
    assert w is not None
    assert abs(float(w.sum())) < 1e-9                       # net ~ 0 (dollar-neutral)
    assert abs(float(w.abs().sum()) - 1.0) < 1e-9           # gross 1.0
    assert (w[w > 0] > 0).all() and (w[w < 0] < 0).all()    # genuine long + short legs


# ----------------------------------------------------------------- failsafe 1: vol-managed
def _equity(n=600, seed=1, vol=0.012):
    r = np.random.default_rng(seed).normal(0.0003, vol, n)
    return pd.Series(10_000 * np.cumprod(1 + r), index=pd.bdate_range("2010-01-01", periods=n))


def test_vol_managed_no_lookahead():
    # Corrupt the FUTURE of the equity curve; the managed curve up to that point must NOT move
    # (the leverage at day t uses trailing vol through t-1 — shift(1)).
    eq = _equity()
    base = vol_managed(eq)
    corrupt = eq.copy(); corrupt.iloc[400:] *= 5.0
    after = vol_managed(corrupt)
    assert np.allclose(base.iloc[:390].to_numpy(), after.iloc[:390].to_numpy())


def test_vol_managed_targets_vol():
    # On a constant-vol series (~20% ann), targeting 10% should roughly halve realized vol.
    daily = 0.20 / np.sqrt(TRADING_DAYS)
    r = np.random.default_rng(2).normal(0, daily, 1500)
    eq = pd.Series(10_000 * np.cumprod(1 + r), index=pd.bdate_range("2010-01-01", periods=1500))
    managed = vol_managed(eq, target_vol=0.10, window=126, max_leverage=5.0)
    realized = managed.pct_change().dropna().iloc[200:].std() * np.sqrt(TRADING_DAYS)
    assert 0.07 < realized < 0.13                           # lands near the 10% target


def test_vol_managed_delevers_into_a_vol_spike():
    # Calm, then a violent stretch: managed exposure (|managed ret| / |raw ret|) must FALL
    # during the spike — that's the crash dodge.
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.006, 400)
    spike = rng.normal(0, 0.05, 120)                        # vol ~8x higher
    r = np.concatenate([calm, spike])
    eq = pd.Series(10_000 * np.cumprod(1 + r), index=pd.bdate_range("2010-01-01", periods=len(r)))
    managed = vol_managed(eq, target_vol=0.12, window=100, max_leverage=3.0)
    raw_ret = eq.pct_change().dropna()
    man_ret = managed.pct_change().dropna()
    expo = (man_ret.abs() / raw_ret.abs().replace(0, np.nan))
    calm_expo = expo.iloc[200:380].median()
    spike_expo = expo.iloc[440:510].median()
    assert spike_expo < calm_expo * 0.6                     # de-levered into the spike


# ----------------------------------------------------------------- failsafe 2: trend filter
def test_trend_filter_flat_below_ma():
    # Market rises (above its MA) then crashes (below). The filtered book must go FLAT once
    # the market is below its average — equity dead-flat through that stretch.
    up = np.linspace(100, 200, 300)
    down = np.linspace(200, 90, 200)
    market = pd.Series(np.concatenate([up, down]), index=pd.bdate_range("2010-01-01", periods=500))
    eq = pd.Series(10_000 * np.cumprod(1 + np.random.default_rng(4).normal(0.001, 0.01, 500)),
                   index=market.index)
    filt = trend_filtered(eq, market, ma=100)
    tail = filt.iloc[-60:].to_numpy()                       # deep in the crash, well below MA
    assert np.allclose(tail, tail[0])                       # flat — no exposure below the MA


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
