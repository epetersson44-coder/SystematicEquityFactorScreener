# backtest/tests/stress_test.py — adversarial / rough stress harness.
#
# The correctness suite (test_backtest.py) checks clean inputs. THIS file tries to
# BREAK the engine: degenerate sizes, extreme scales, hostile strategies, NaNs,
# turnover bombs, money-conservation under costs, and the look-ahead property at
# many split points (plus the reverse sanity check — corrupt the PAST and confirm
# the future DOES move, so the look-ahead test isn't passing trivially).
#
# Run:  python -m backtest.tests.stress_test
# Each probe prints  ok / WARN / BUG  with detail. Exit nonzero if any BUG.

import time
import warnings
import numpy as np
import pandas as pd

from backtest import metrics, baseline, costs
from backtest.engine import Portfolio, run
from backtest.strategy import Strategy, BuyAndHold, SMACrossover
from backtest.tests._helpers import make_df, random_walk as make_rw, ConstantWeight as ConstW
import backtest.data as data

warnings.simplefilter("ignore")          # we WANT to see numeric edge behavior raw

RESULTS = []
def record(status, name, detail=""):
    RESULTS.append((status, name, detail))
    print(f"  {status:4s}  {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------- stress-only test doubles (shared ones in _helpers)
class Flipper(Strategy):
    """Turnover bomb: alternates fully-in / fully-out every bar."""
    def target_weight(self, h): return float(len(h) % 2)

class NaNStrat(Strategy):
    def target_weight(self, h): return float("nan")

class DriftStrat(Strategy):
    """Weight creeps slightly above 1.0 — should be rejected."""
    def target_weight(self, h): return 1.0 + 1e-9

def expect_raise(fn, exc=Exception):
    try:
        fn(); return None
    except exc as e:
        return e

def finite(series):
    a = series.to_numpy()
    return np.all(np.isfinite(a))


# ================================================================ probes

def probe_degenerate_sizes():
    # empty, 1-bar, 2-bar
    for n in (0, 1, 2):
        df = make_df(np.linspace(100, 110, n)) if n else make_df([])
        try:
            eq = run(df, BuyAndHold())
            ok = (len(eq) == n) and (n == 0 or finite(eq))
            record("ok" if ok else "BUG", f"run() n={n}", f"len={len(eq)}")
        except Exception as e:
            record("WARN", f"run() n={n}", f"raised {type(e).__name__}: {e}")

def probe_empty_metrics():
    # metrics on an empty curve — should raise cleanly, not return garbage
    eq = pd.Series([], dtype=float)
    e = expect_raise(lambda: metrics.summary(eq))
    record("ok" if e else "WARN", "metrics.summary(empty)",
           f"raised {type(e).__name__}" if e else "returned without error")

def probe_single_point_metrics():
    eq = pd.Series([100.0], index=pd.to_datetime(["2020-01-01"]))
    cagr = metrics.cagr(eq); dd = metrics.max_drawdown(eq)
    ok = np.isnan(cagr) and dd == 0.0
    record("ok" if ok else "BUG", "metrics single point", f"cagr={cagr} maxdd={dd}")

def probe_extreme_scales():
    for p0 in (1e-4, 1e2, 1e9):
        df = make_df(p0 * (1 + 0.001) ** np.arange(500))
        eq = run(df, SMACrossover(5, 20), cost=costs.proportional(2))
        record("ok" if finite(eq) and (eq > 0).all() else "BUG",
               f"extreme price scale p0={p0:g}", f"final={eq.iloc[-1]:.3e}")

def probe_price_crash():
    # price collapses 1000x then recovers; engine must stay finite
    p = np.concatenate([np.full(50, 100.0), np.full(50, 0.1), np.full(50, 100.0)])
    df = make_df(p)
    eq = run(df, BuyAndHold(), fill="close")
    record("ok" if finite(eq) and (eq >= 0).all() else "BUG",
           "1000x crash & recover", f"min={eq.min():.2f} final={eq.iloc[-1]:.2f}")

def probe_gap_jump():
    p = np.full(100, 100.0); p[50:] = 1e6           # overnight 10,000x gap
    df = make_df(p)
    eq = run(df, BuyAndHold(), fill="next_open")
    record("ok" if finite(eq) else "BUG", "10,000x overnight gap", f"final={eq.iloc[-1]:.3e}")

def probe_constant_prices():
    df = make_df(np.full(300, 100.0))
    eqb = run(df, BuyAndHold(), fill="close")
    eqs = run(df, SMACrossover(50, 200))
    okb = np.allclose(eqb.to_numpy(), 10_000)        # flat price -> flat equity
    oks = np.allclose(eqs.to_numpy(), 10_000)        # SMA never crosses -> flat/cash
    record("ok" if okb and oks else "BUG", "constant prices",
           f"buyhold_flat={okb} sma_flat={oks}")

def probe_nan_weight():
    df = make_df(make_rw(100))
    e = expect_raise(lambda: run(df, NaNStrat()))
    record("ok" if isinstance(e, ValueError) else "BUG", "strategy returns NaN weight",
           f"raised {type(e).__name__}" if e else "ACCEPTED nan — silent corruption!")

def probe_weight_boundaries():
    df = make_df(make_rw(50))
    good = True
    for w in (0.0, 1.0):
        try: run(df, ConstW(w), fill="close")
        except Exception: good = False
    over = expect_raise(lambda: run(df, DriftStrat(), fill="close"), ValueError)
    record("ok" if good and over else "BUG", "weight boundaries",
           f"0&1 ok={good}; 1+1e-9 rejected={bool(over)}")

def probe_money_conservation():
    # After every bar, equity must equal cash + shares*close — no money printed,
    # and total drop across a trade must equal exactly the fee charged.
    df = make_df(make_rw(400))
    cost = costs.proportional(25)
    pf = Portfolio(10_000)
    opens, closes = df["Open"].to_numpy(), df["Close"].to_numpy()
    strat = SMACrossover(5, 20)
    pending = None; max_err = 0.0; fee_err = 0.0
    for i in range(len(df)):
        if pending is not None:
            pre = pf.equity(opens[i])
            d, fee = pf.rebalance(pending, opens[i], cost)
            post = pf.equity(opens[i])
            fee_err = max(fee_err, abs((pre - post) - fee))   # equity drop == fee
        pending = strat.target_weight(df.iloc[:i+1])
        eq_direct = pf.cash + pf.shares * closes[i]
        max_err = max(max_err, abs(eq_direct - pf.equity(closes[i])))
    ok = max_err < 1e-9 and fee_err < 1e-7
    record("ok" if ok else "BUG", "money conservation under costs",
           f"acct_err={max_err:.1e} fee_err={fee_err:.1e}")

def probe_turnover_bomb():
    df = make_df(make_rw(500))
    free = run(df, Flipper(), fill="close")
    paid = run(df, Flipper(), fill="close", cost=costs.proportional(50))
    ok = finite(paid) and paid.iloc[-1] < free.iloc[-1] and paid.iloc[-1] > 0
    record("ok" if ok else "BUG", "turnover bomb (flip every bar)",
           f"free={free.iloc[-1]:.0f} paid={paid.iloc[-1]:.0f}")

def probe_extreme_costs():
    df = make_df(make_rw(200))
    eq = run(df, Flipper(), fill="close", cost=costs.proportional(1_000_000))  # 100% / trade
    neg = (eq < 0).any()
    record("WARN" if neg else "ok", "absurd cost (1e6 bps)",
           f"equity goes negative={neg} (uncapped fees — garbage-in)")

def probe_cash_rate_extremes():
    df = make_df(make_rw(300))
    hi = run(df, ConstW(0.0), cash_rate=10.0)        # 1000% rf, all cash
    neg = run(df, ConstW(0.0), cash_rate=-0.5)       # negative rate
    ok = finite(hi) and finite(neg) and hi.iloc[-1] > 10_000 and neg.iloc[-1] < 10_000
    record("ok" if ok else "BUG", "cash_rate extremes (+1000% / -50%)",
           f"hi={hi.iloc[-1]:.2e} neg={neg.iloc[-1]:.0f}")

def probe_init_capital_edge():
    df = make_df(make_rw(50))
    z = run(df, BuyAndHold(), initial_capital=0)
    okz = np.allclose(z.to_numpy(), 0.0)
    neg = run(df, BuyAndHold(), initial_capital=-1000)   # nonsense, but shouldn't crash
    record("ok" if okz and finite(neg) else "WARN", "initial_capital 0 / negative",
           f"zero_stays_zero={okz} neg_finite={finite(neg)}")

def probe_lookahead_many_splits():
    df = make_df(make_rw(200))
    base = run(df, SMACrossover(5, 20), fill="next_open").to_numpy()
    worst = 0.0
    for T in (1, 25, 50, 99, 150, 198):
        c = df.copy(); c.iloc[T+1:] = c.iloc[T+1:] * 7.0
        after = run(c, SMACrossover(5, 20), fill="next_open").to_numpy()
        worst = max(worst, np.max(np.abs(base[:T+1] - after[:T+1])))
    record("ok" if worst == 0.0 else "BUG", "look-ahead: corrupt future @ 6 splits",
           f"max past-diff={worst:.2e}")

def probe_lookahead_reverse_sanity():
    # Corrupt the PAST -> the future MUST change, else the test is trivial.
    df = make_df(make_rw(200))
    base = run(df, SMACrossover(5, 20), fill="next_open").to_numpy()
    c = df.copy(); c.iloc[:50] = c.iloc[:50] * 3.0
    after = run(c, SMACrossover(5, 20), fill="next_open").to_numpy()
    changed = not np.array_equal(base[60:], after[60:])
    record("ok" if changed else "BUG", "reverse sanity: past change moves future",
           "future responded" if changed else "future frozen — test is trivial!")

def probe_determinism_many():
    df = make_df(make_rw(300))
    ref = run(df, SMACrossover(10, 50), cost=costs.proportional(), cash_rate=0.03).to_numpy()
    same = all(np.array_equal(ref, run(df, SMACrossover(10, 50),
               cost=costs.proportional(), cash_rate=0.03).to_numpy()) for _ in range(25))
    record("ok" if same else "BUG", "determinism x25", "identical" if same else "diverged")

def probe_sma_window_edges():
    df = make_df(make_rw(60))
    try:
        a = run(df, SMACrossover(1, 2))                    # minimal windows
        b = run(df, SMACrossover(50, 100))                 # slow > data length -> all flat
        ok = finite(a) and finite(b) and np.allclose(b.to_numpy(), 10_000)
        record("ok" if ok else "BUG", "SMA window edges (1/2 ; slow>len)",
               f"flat_when_starved={np.allclose(b.to_numpy(),10_000)}")
    except Exception as e:
        record("BUG", "SMA window edges", f"{type(e).__name__}: {e}")

def probe_monotonic_decline():
    df = make_df(100 * 0.99 ** np.arange(400))            # bleed to ~near zero
    eq = run(df, BuyAndHold(), fill="close")
    s = metrics.summary(eq)
    ok = finite(eq) and s["max_drawdown"] < 0 and s["calmar"] < 0
    record("ok" if ok else "BUG", "monotonic decline to ~0",
           f"maxDD={s['max_drawdown']*100:.1f}% calmar={s['calmar']:.2f}")

def probe_validate_catches_garbage():
    base = make_df(make_rw(30))
    cases = {
        "dupe date": pd.concat([base, base.iloc[[-1]]]),
        "reversed":  base.iloc[::-1],
        "neg price": _poke(base, 10, "Close", -5.0),
        "zero price":_poke(base, 10, "Open", 0.0),
        "nan":       _poke(base, 10, "Close", np.nan),
        "high<low":  _poke(base, 10, "High", base.iloc[10]["Low"] - 1),
    }
    caught = {k: bool(expect_raise(lambda d=v: data._validate(d, "T"), ValueError))
              for k, v in cases.items()}
    allc = all(caught.values())
    record("ok" if allc else "BUG", "data._validate catches garbage",
           ", ".join(f"{k}={'Y' if c else 'N'}" for k, c in caught.items()))

def probe_scale_performance():
    n = 50_000
    df = make_df(make_rw(n))
    t0 = time.time()
    eq = run(df, SMACrossover(50, 200), cost=costs.proportional())
    dt = time.time() - t0
    ok = finite(eq) and (eq > 0).all()
    record("ok" if ok else "BUG", f"scale {n:,} bars",
           f"{dt:.2f}s, final={eq.iloc[-1]:.0f}, finite={finite(eq)}")

def probe_real_spy_if_cached():
    try:
        spy = data.get_prices("SPY", start="2000-01-01")
    except Exception as e:
        record("WARN", "real SPY load", f"unavailable: {type(e).__name__}")
        return
    eq = run(spy, SMACrossover(50, 200), cost=costs.proportional(2), cash_rate=0.04)
    s = metrics.summary(eq, rf=0.04)
    ok = finite(eq) and (eq > 0).all()
    record("ok" if ok else "BUG", "real SPY end-to-end",
           f"CAGR {s['cagr']*100:.2f}% Sharpe {s['sharpe']:.2f} DDdur {s['max_dd_duration_days']}d")


# small utility used above (make_df / make_rw / ConstW come from _helpers)
def _poke(df, i, col, val):
    d = df.copy(); d.iloc[i, d.columns.get_loc(col)] = val; return d


# ================================================================ run all
if __name__ == "__main__":
    import sys
    probes = [v for k, v in sorted(globals().items()) if k.startswith("probe_") and callable(v)]
    print(f"\nRunning {len(probes)} adversarial probes...\n")
    for p in probes:
        try:
            p()
        except Exception as e:
            record("BUG", p.__name__, f"PROBE CRASHED: {type(e).__name__}: {e}")
    n_bug = sum(1 for s, *_ in RESULTS if s == "BUG")
    n_warn = sum(1 for s, *_ in RESULTS if s == "WARN")
    n_ok = sum(1 for s, *_ in RESULTS if s == "ok")
    print(f"\n{'='*60}\n{n_ok} ok | {n_warn} WARN | {n_bug} BUG   ({len(RESULTS)} checks)\n{'='*60}")
    sys.exit(1 if n_bug else 0)
