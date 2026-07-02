# backtest/tests/test_synthetic.py — the synthetic-market lab's own correctness.
#
# Small/fast versions of the generators (seeded, offline except none — moments are
# stubbed) + the two directional claims the study rests on:
#   - generators produce structurally valid panels (positive prices, aligned, no NaN)
#   - injected AR-drift markets ARE more trend-timeable than iid ones for the sleeve
#     (positive control beats null on average across seeds)

import numpy as np
import pandas as pd

import backtest.synthetic as syn


_MU = np.full(6, 0.0003)
_COV = (np.full((6, 6), 0.2) + 0.8 * np.eye(6)) * (0.01 ** 2)


def test_gbm_panel_structure():
    p = syn.gbm_panel(_MU, _COV, n_days=400, seed=1)
    close, open_ = p["Close"], p["Open"]
    assert list(close.columns) == syn.ETFS
    assert len(close) == 400 and len(open_) == 400
    assert (close > 0).all().all() and (open_ > 0).all().all()
    assert not close.isna().any().any()
    assert np.allclose(open_.iloc[5], close.iloc[4])        # Open[t] = Close[t-1]


def test_trending_panel_has_more_autocorrelation():
    # The injected AR drift must show up as higher lag-1 autocorrelation of monthly
    # returns than the iid null (that's the entire point of the positive control).
    def monthly_ac1(panels):
        m = panels["Close"].resample("ME").last().pct_change().dropna()
        return float(np.mean([m[c].autocorr(1) for c in m.columns]))
    ac_null = np.mean([monthly_ac1(syn.gbm_panel(_MU, _COV, 2500, seed=s)) for s in range(3)])
    ac_trend = np.mean([monthly_ac1(syn.trending_panel(_MU, _COV, 2500, seed=s)) for s in range(3)])
    assert ac_trend > ac_null + 0.05


def test_sleeve_harvests_injected_trend_but_not_noise():
    # Directional claim on tiny panels (3 seeds, 2000 days): mean excess Sharpe on
    # trending markets must clearly beat mean excess on iid markets.
    ex_null = [syn.sleeve_vs_static(syn.gbm_panel(_MU, _COV, 2000, seed=s))[2] for s in range(3)]
    ex_trend = [syn.sleeve_vs_static(syn.trending_panel(_MU, _COV, 2000, seed=s))[2] for s in range(3)]
    assert np.mean(ex_trend) > np.mean(ex_null) + 0.15


def test_bootstrap_panel_reuses_real_rows(monkeypatch=None):
    # Stub the real-panel loader so the test stays offline.
    real = pd.DataFrame(
        100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, (600, 6)), axis=0),
        index=pd.bdate_range("2015-01-01", periods=600), columns=syn.ETFS)
    import backtest.trend_sleeve as ts
    orig = ts.etf_panel
    ts.etf_panel = lambda *a, **k: {"Close": real}
    try:
        p = syn.bootstrap_panel(block=21, seed=3, n_days=300)
    finally:
        ts.etf_panel = orig
    close = p["Close"]
    assert len(close) == 300 and (close > 0).all().all()
    # every synthetic daily return must be one of the real panel's daily returns
    real_rets = real.pct_change().dropna().round(12)
    syn_rets = close.pct_change().dropna().round(12)
    real_set = {tuple(r) for r in real_rets.to_numpy()}
    assert all(tuple(r) in real_set for r in syn_rets.to_numpy())


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
