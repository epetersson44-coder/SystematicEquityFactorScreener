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
from backtest.engine_xs import MultiPortfolio, run_xs
from backtest.strategy import Strategy, BuyAndHold, SMACrossover, CrossSectionalStrategy
from backtest.tests._helpers import (make_df, random_walk as make_rw, ConstantWeight as ConstW,
                                     make_panel, random_panel)
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


# ================================================================ long-short (xs) probes
# Try to BREAK the cross-sectional long-short engine: short blowups, the gross cap at its
# exact boundary, market-neutral immunity under extreme moves, absurd borrow, a short on a
# delisting name, sign-flips through zero, signed conservation, look-ahead in short mode.

class XSFixed(CrossSectionalStrategy):
    """Set signed target weights once (at bar `at`), then hold."""
    def __init__(self, weights, at=0):
        self.w = pd.Series(weights, dtype=float); self.at = at
    def target_weights(self, closes, i):
        return self.w if i == self.at else None

class XSLongShortRandom(CrossSectionalStrategy):
    """Random SIGNED weights every `every` bars at a target gross exposure."""
    def __init__(self, seed, every=7, k=6, gross=1.5):
        self.rng = np.random.default_rng(seed); self.every = every; self.k = k; self.gross = gross
    def target_weights(self, closes, i):
        if i == 0 or i % self.every != 0:
            return None
        avail = closes.iloc[i].dropna().index
        if len(avail) == 0:
            return None
        k = min(self.k, len(avail))
        pick = self.rng.choice(np.asarray(avail), size=k, replace=False)
        w = self.rng.normal(0, 1, k); w = w / np.abs(w).sum() * self.gross
        return pd.Series(w, index=pick)


def probe_xs_all_short_book():
    panels = random_panel(300, 6, seed=11)
    eq = run_xs(panels, XSFixed({c: -1 / 6 for c in panels["Close"].columns}),
                fill="close", allow_short=True, gross_max=1.0)
    record("ok" if finite(eq) else "BUG", "xs 100% short book", f"final={eq.iloc[-1]:.0f}")

def probe_xs_short_blowup():
    # short a name that rips 100x: the loss is UNBOUNDED. Equity SHOULD go deeply negative
    # but MUST stay finite — the engine must report a blown short honestly, not clamp at 0.
    n = 120
    p = np.concatenate([np.full(20, 100.0), np.linspace(100, 10_000, n - 20)])
    panels = make_panel(p.reshape(-1, 1), tickers=["A"])
    eq = run_xs(panels, XSFixed({"A": -1.0}), fill="close", allow_short=True, gross_max=1.0)
    went_neg = (eq.to_numpy() < 0).any()
    record("ok" if finite(eq) and went_neg else "BUG", "xs short blowup (name 100x)",
           f"finite={finite(eq)} went_negative={went_neg} min={eq.min():,.0f}")

def probe_xs_gross_cap_boundary():
    prices = pd.Series({"A": 10., "B": 20., "C": 5.})
    at = expect_raise(lambda: MultiPortfolio(10_000, allow_short=True, gross_max=2.0)
                      .rebalance(pd.Series({"A": 1.0, "B": -1.0}), prices))          # gross 2.0 exactly
    over = expect_raise(lambda: MultiPortfolio(10_000, allow_short=True, gross_max=2.0)
                        .rebalance(pd.Series({"A": 1.0, "B": -1.0, "C": -0.01}), prices), ValueError)  # 2.01
    ok = (at is None) and isinstance(over, ValueError)
    record("ok" if ok else "BUG", "xs gross cap boundary (2.0 ok, >2 rejected)",
           f"at_cap_ok={at is None} over_rejected={bool(over)}")

def probe_xs_long_only_rail():
    # default (no allow_short) must STILL reject a negative weight — the safety rail.
    e = expect_raise(lambda: MultiPortfolio(10_000).rebalance(
        pd.Series({"A": 0.5, "B": -0.1}), pd.Series({"A": 10., "B": 20.})), ValueError)
    record("ok" if e else "BUG", "xs long-only rail rejects short",
           "rejected" if e else "ACCEPTED negative — rail down!")

def probe_xs_dollar_neutral_market_immune():
    # identical names, dollar-neutral, with a ~50x market rip: equity must stay flat.
    n = 200
    col = 100 * (1 + 0.02) ** np.arange(n)
    panels = make_panel(np.repeat(col.reshape(-1, 1), 4, axis=1))
    w = {"T0": 0.5, "T1": 0.5, "T2": -0.5, "T3": -0.5}
    eq = run_xs(panels, XSFixed(w), fill="close", allow_short=True, gross_max=2.0)
    dev = float(np.max(np.abs(eq.to_numpy() - 10_000)))
    record("ok" if finite(eq) and dev < 1e-2 else "BUG", "xs dollar-neutral immune to 50x market",
           f"max_dev={dev:.2e}")

def probe_xs_longs_pay_no_borrow():
    # borrow charges SHORTS only — a long book with absurd borrow_bps must stay flat.
    panels = make_panel(np.full((100, 1), 100.0), tickers=["A"])
    eq = run_xs(panels, XSFixed({"A": 1.0}), fill="close", allow_short=True, gross_max=1.0, borrow_bps=1e6)
    flat = np.allclose(eq.to_numpy(), 10_000)
    record("ok" if flat else "BUG", "xs longs pay no borrow", f"flat={flat}")

def probe_xs_borrow_extremes():
    panels = make_panel(np.full((100, 1), 100.0), tickers=["A"])
    huge = run_xs(panels, XSFixed({"A": -1.0}), fill="close", allow_short=True, gross_max=1.0, borrow_bps=1e6)
    zero = run_xs(panels, XSFixed({"A": -1.0}), fill="close", allow_short=True, gross_max=1.0, borrow_bps=0)
    ok = finite(huge) and np.allclose(zero.to_numpy(), 10_000) and huge.iloc[-1] < zero.iloc[-1]
    record("ok" if ok else "BUG", "xs borrow extremes (1e6 bps vs 0)",
           f"huge_final={huge.iloc[-1]:,.0f} zero_flat={np.allclose(zero.to_numpy(), 10_000)}")

def probe_xs_short_delisting():
    # a SHORT whose name delists (NaN) mid-hold: equity must stay finite (carry-forward).
    n = 60
    a = 100 * (1.001) ** np.arange(n)
    b = np.concatenate([np.full(30, 100.0), np.full(n - 30, np.nan)])
    panels = make_panel(np.column_stack([a, b]), tickers=["A", "B"])
    eq = run_xs(panels, XSFixed({"A": 0.5, "B": -0.5}), fill="close", allow_short=True, gross_max=1.0)
    record("ok" if finite(eq) else "BUG", "xs short on a delisting name",
           f"finite={finite(eq)} final={eq.iloc[-1]:,.0f}")

def probe_xs_single_name_short():
    panels = make_panel((100 * 0.999 ** np.arange(150)).reshape(-1, 1), tickers=["A"])
    eq = run_xs(panels, XSFixed({"A": -1.0}), fill="close", allow_short=True, gross_max=1.0)
    record("ok" if finite(eq) and eq.iloc[-1] > 10_000 else "BUG",
           "xs single-name short (declining)", f"final={eq.iloc[-1]:,.0f}")

def probe_xs_sign_flip_turnover():
    # flip a name long<->short every rebalance: trades THROUGH zero. Cost must drag, equity finite.
    panels = random_panel(300, 3, seed=21)
    class Flip(CrossSectionalStrategy):
        def target_weights(self, closes, i):
            if i % 5:
                return None
            return pd.Series({closes.columns[0]: 1.0 if (i // 5) % 2 == 0 else -1.0})
    free = run_xs(panels, Flip(), fill="close", allow_short=True, gross_max=1.0)
    paid = run_xs(panels, Flip(), fill="close", allow_short=True, gross_max=1.0, cost=costs.proportional(50))
    ok = finite(paid) and paid.iloc[-1] < free.iloc[-1]
    record("ok" if ok else "BUG", "xs sign-flip turnover (long<->short thru 0)",
           f"free={free.iloc[-1]:,.0f} paid={paid.iloc[-1]:,.0f}")

def probe_xs_signed_conservation():
    # cash + sum(shares*close) == equity every bar, and drop == fee, WITH shorts + cost.
    panels = random_panel(400, 8, seed=31)
    closes, opens = panels["Close"], panels["Open"]
    cost = costs.proportional(25)
    pf = MultiPortfolio(10_000, allow_short=True, gross_max=2.0)
    strat = XSLongShortRandom(seed=31, every=5, k=5, gross=1.6)
    pending, acct, feee = None, 0.0, 0.0
    for i in range(len(closes)):
        if pending is not None:
            pre = pf.equity(opens.iloc[i]); fee = pf.rebalance(pending, opens.iloc[i], cost)
            post = pf.equity(opens.iloc[i]); feee = max(feee, abs((pre - post) - fee)); pending = None
        w = strat.target_weights(closes, i)
        if w is not None:
            pending = w
        direct = pf.cash + sum(sh * closes.iloc[i][t] for t, sh in pf.shares.items())
        acct = max(acct, abs(direct - pf.equity(closes.iloc[i])))
    ok = acct < 1e-9 and feee < 1e-7
    record("ok" if ok else "BUG", "xs signed money conservation under cost",
           f"acct_err={acct:.1e} fee_err={feee:.1e}")

def probe_xs_lookahead_short_many():
    panels = random_panel(220, 10, seed=41)
    mk = lambda: XSLongShortRandom(seed=41, every=7, k=5)
    base = run_xs(panels, mk(), cost=costs.proportional(10), allow_short=True, borrow_bps=150).to_numpy()
    worst = 0.0
    for T in (5, 40, 90, 140, 200):
        c = {k: df.copy() for k, df in panels.items()}
        for df in c.values():
            df.iloc[T + 1:] = df.iloc[T + 1:] * 8.0
        after = run_xs(c, mk(), cost=costs.proportional(10), allow_short=True, borrow_bps=150).to_numpy()
        worst = max(worst, float(np.max(np.abs(base[:T + 1] - after[:T + 1]))))
    record("ok" if worst == 0.0 else "BUG", "xs look-ahead (short) @ 5 splits", f"max past-diff={worst:.2e}")

def probe_xs_monte_carlo_long_short():
    bad = []; t0 = time.time()
    for seed in range(150):
        k = int(np.random.default_rng(seed).integers(4, 12))
        panels = random_panel(150, k, vol=0.03, seed=seed)            # high vol -> stress
        eq = run_xs(panels, XSLongShortRandom(seed=seed, every=6, gross=1.8),
                    cost=costs.proportional(20), allow_short=True, borrow_bps=200)
        if not finite(eq):
            bad.append(seed)
    record("ok" if not bad else "BUG", "xs Monte Carlo 150 long-short universes (hi-vol)",
           f"{time.time() - t0:.1f}s, nonfinite={len(bad)}")


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
