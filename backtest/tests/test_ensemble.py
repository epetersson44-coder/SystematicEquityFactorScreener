# backtest/tests/test_ensemble.py — the lookback-ensemble + cash-hurdle refactor.
#
# The single most important test here is the REGRESSION one: VolTargetTSMOM was rewritten
# to support signal ensembles, and the default single-lookback path must reproduce the OLD
# algorithm bit-for-bit (the Sharpe-0.87 headline rests on it). The old body is copied
# below verbatim as the reference implementation and compared on a random panel.

import numpy as np
import pandas as pd

from backtest.trend_sleeve import VolTargetTSMOM


class _RefVolTargetTSMOM:
    """The pre-ensemble implementation, verbatim (single 252d lookback, no hurdle)."""

    def __init__(self, look=252, vol_lb=63, target_vol=0.10, every=21, max_gross=1.0,
                 long_short=False, offset=0):
        self.look, self.vol_lb, self.target_vol, self.every, self.max_gross = (
            look, vol_lb, target_vol, every, max_gross)
        self.long_short = long_short
        self.offset = offset % every

    def target_weights(self, closes, i):
        if i < self.look or i % self.every != self.offset:
            return None
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        sign = {}
        for t in closes.columns:
            p0, pm = closes.iloc[i].get(t), closes.iloc[i - self.look].get(t)
            v = rets[t].std() if t in rets else np.nan
            if not (p0 and pm and np.isfinite(p0) and np.isfinite(pm) and np.isfinite(v) and v > 0):
                continue
            mom = p0 / pm - 1
            if mom > 0:
                sign[t] = 1.0
            elif self.long_short and mom < 0:
                sign[t] = -1.0
        if not sign:
            return pd.Series(dtype=float)
        on = list(sign)
        invvol = 1.0 / (rets[on].std() * np.sqrt(252))
        w = pd.Series({t: sign[t] * invvol[t] for t in on})
        w = w / w.abs().sum()
        cov = rets[on].cov() * 252
        pvol = float(np.sqrt(w.values @ cov.values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


def _random_panel(n=700, cols=("A", "B", "C", "D"), seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    data = {c: 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)) for c in cols}
    df = pd.DataFrame(data, index=idx)
    df.iloc[:60, 3] = np.nan                                # one late-starting column
    return df


def test_single_look_reproduces_the_old_algorithm_exactly():
    closes = _random_panel()
    for ls in (False, True):
        new = VolTargetTSMOM(long_short=ls)
        ref = _RefVolTargetTSMOM(long_short=ls)
        fired = 0
        for i in range(len(closes)):
            wn, wr = new.target_weights(closes, i), ref.target_weights(closes, i)
            assert (wn is None) == (wr is None)
            if wn is None:
                continue
            fired += 1
            assert sorted(wn.index) == sorted(wr.index)
            assert np.allclose(wn.sort_index().values, wr.sort_index().values, atol=1e-14)
        assert fired > 10                                    # the comparison actually ran


def test_ensemble_scales_positions_in_thirds():
    # A: up over every lookback -> strength 1. B: up over 252d only -> strength 1/3.
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    a = np.linspace(100, 200, n)                             # rises all the way
    b = np.concatenate([np.linspace(100, 180, n - 140),      # long-run up...
                        np.linspace(180, 120, 140)])         # ...but down over 1m AND 3m
    closes = pd.DataFrame({"A": a, "B": b}, index=idx)
    strat = VolTargetTSMOM(looks=(21, 63, 252))
    w = strat.target_weights(closes, 252)
    assert w is not None and "A" in w and "B" in w
    # equal vols would make |w_B| ~ 1/3 of |w_A|; vols differ, so just pin the ordering
    # and the exact strength ratio after backing the inverse-vol part out
    rets = closes.iloc[252 - 63:253].pct_change().iloc[1:]
    strength_a = w["A"] * rets["A"].std()
    strength_b = w["B"] * rets["B"].std()
    assert abs(strength_b / strength_a - 1 / 3) < 1e-9


def test_hurdle_column_gates_weak_trends_and_is_never_traded():
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    weak = 100 * (1 + 0.02) ** (np.arange(n) / 252)          # +2%/yr uptrend
    strong = 100 * (1 + 0.20) ** (np.arange(n) / 252)        # +20%/yr uptrend
    cash = 100 * (1 + 0.05) ** (np.arange(n) / 252)          # T-bills at 5%
    closes = pd.DataFrame({"WEAK": weak, "STRONG": strong, "BIL": cash}, index=idx)
    no_gate = VolTargetTSMOM().target_weights(closes, 252)
    gated = VolTargetTSMOM(hurdle_col="BIL").target_weights(closes, 252)
    assert "BIL" in no_gate.index                            # ungated, BIL is (wrongly) tradeable
    assert "BIL" not in gated.index                          # hurdle leg is reference-only
    assert "WEAK" in no_gate.index                           # +2% > 0 -> raw gate lets it in
    assert "WEAK" not in gated.index                         # +2% < cash 5% -> excess gate drops it
    assert "STRONG" in gated.index                           # +20% clears the hurdle


def test_hurdle_missing_data_falls_back_to_raw_gate():
    n = 300
    idx = pd.bdate_range("2019-01-01", periods=n)
    up = np.linspace(100, 120, n)
    closes = pd.DataFrame({"A": up, "BIL": np.full(n, np.nan)}, index=idx)
    w = VolTargetTSMOM(hurdle_col="BIL").target_weights(closes, 252)
    assert w is not None and "A" in w.index                  # 0-hurdle fallback, not a crash


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
