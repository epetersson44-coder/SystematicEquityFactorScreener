# PRE-REGISTERED (2026-07-13, committed BEFORE running — two-commit proof).
# Fourth and FINAL item of the post-go-live sweep queue (canary PASSED -> Aug design
# call; HRP FAILED -> closed; buffering ADOPTED as execution policy).
#
# HYPOTHESIS (RSRS — 阻力支撑相对强度, Everbright Securities 2017 timing report series;
# the best export of the Chinese 研报复现 ecosystem per the 2026-07-12 sweep): the
# rolling High-on-Low regression slope, standardized, is a support/resistance-strength
# timing signal for the blend's equity leg. Mechanism story: beta > 1 means highs are
# advancing faster than lows (support firm, resistance yielding) — an order-pressure
# proxy readable from free OHLC data.
#
# PORT SPEC (ORIGINAL published rule verbatim — deliberately NOT the modified/R2/
# right-skew variants from later reports; picking among four versions would be
# selection): beta_t = OLS slope of daily HIGH on daily LOW over N=18 days
# (rolling cov/var); z_t = 600-day rolling z-score of beta. HYSTERESIS gate on the
# blend's SPY leg: e=1 when z > 0.7, e=0 when z < -0.7, else hold prior state
# (initial state long). Evaluated at the sleeve's rebalance dates (every 21d at
# offset o — the honest monthly port of a daily A-share signal, stated as such),
# next-day apply, 5bps on |delta e|, displaced weight earns rf. Sleeve untouched.
# Blend level, all 21 offsets, HONEST convention — the exact canary harness with the
# gate swapped, so the two gates are directly comparable.
#
# ADOPTION BAR (house distribution dominance): median blend exSharpe > baseline AND
# worst offset >= baseline's worst AND maxDD median no deeper. Stage-1 pass ->
# pre-registered OOS annex on the 1999-2006 panel REQUIRED (SPY is real there — the
# gate needs no proxy) before any further claim. Anything less -> ledger + close.
# EXPECTATION ON RECORD: FAIL. Priors stacked against it: (a) every US timing trial
# on this ledger died (SMA walk-forward 0.44, macro 0.59, Gayed cadence washes);
# (b) this is a cross-MARKET and cross-FREQUENCY port (A-share daily -> US monthly)
# — decay squared; (c) a binary gate whipsaws harder than canary's 3-state b/2.
# The reason it still gets its slot: the mechanism is genuinely distinct from
# anything on the ledger (order-pressure geometry, not price momentum), Erik asked,
# and a documented kill closes the Chinese-timing question with receipts.
# LEDGER: ONE trial (naive-convention gated-blend median in the results commit).
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from backtest.data import get_prices
from backtest.trend_sleeve import etf_panel, VolTargetTSMOM, ENSEMBLE_LOOKS
from backtest.engine_xs import run_xs
from backtest.timing_luck import blend_curve
from backtest.leverage_study import tbill_series
from backtest import costs, metrics

N_REG, N_STD, Z_HI, Z_LO = 18, 600, 0.7, -0.7

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)
rf_d = (rf / 252.0).reindex(spy.index).ffill().fillna(0.0)

ohlc = get_prices("SPY")
hi = ohlc["High"].reindex(spy.index).ffill()
lo = ohlc["Low"].reindex(spy.index).ffill()
beta = hi.rolling(N_REG).cov(lo) / lo.rolling(N_REG).var()
z = ((beta - beta.rolling(N_STD).mean()) / beta.rolling(N_STD).std()).reindex(spy.index)


def gated_spy_curve(offset, cost_bps=5):
    spy_ret = spy.pct_change().fillna(0.0)
    e_series = np.ones(len(spy))
    e = 1.0
    for i in range(len(spy)):
        if i % 21 == offset and np.isfinite(z.iloc[i]):
            if z.iloc[i] > Z_HI:
                e = 1.0
            elif z.iloc[i] < Z_LO:
                e = 0.0                                     # else: hold prior state
        e_series[i] = e
    e_s = pd.Series(e_series, index=spy.index).shift(1).fillna(1.0)
    turn_cost = e_s.diff().abs().fillna(0.0) * (cost_bps / 10_000.0)
    ret = e_s * spy_ret + (1 - e_s) * rf_d - turn_cost
    return 10_000 * (1 + ret).cumprod()


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def naive_sharpe(eq):
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(252))


rows = []
for off in range(21):
    strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    sleeve = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open",
                    cash_rate=rf)
    base = blend_curve(sleeve, spy)
    gated = blend_curve(sleeve, gated_spy_curve(off))
    rows.append({"base_exs": ex_sharpe(base), "base_dd": metrics.max_drawdown(base),
                 "rsrs_exs": ex_sharpe(gated), "rsrs_dd": metrics.max_drawdown(gated),
                 "rsrs_naive": naive_sharpe(gated)})
df = pd.DataFrame(rows)

print("HONEST convention, 21 offsets, blend level (RSRS z 18/600, 0.7/-0.7 hysteresis on the equity leg):")
print(f"baseline blend   exSharpe med {df.base_exs.median():.3f} "
      f"[{df.base_exs.min():.3f},{df.base_exs.max():.3f}]  maxDD med {df.base_dd.median()*100:5.1f}%")
print(f"RSRS-gated       exSharpe med {df.rsrs_exs.median():.3f} "
      f"[{df.rsrs_exs.min():.3f},{df.rsrs_exs.max():.3f}]  maxDD med {df.rsrs_dd.median()*100:5.1f}%  "
      f"naive med {df.rsrs_naive.median():.3f}")

ok = (df.rsrs_exs.median() > df.base_exs.median()
      and df.rsrs_exs.min() >= df.base_exs.min()
      and df.rsrs_dd.median() >= df.base_dd.median())
print(f"\nVERDICT vs pre-registered bar: "
      f"{'STAGE-1 PASS -> pre-register the OOS annex' if ok else 'FAIL -> ledger + close (expected)'}")

# RESULTS (run 2026-07-13, unmodified from pre-registration 77e0743; honest convention):
#   baseline blend   exSharpe med 0.767 [0.720,0.818]  maxDD med -16.9%
#   RSRS-gated       exSharpe med 0.638 [0.549,0.770]  maxDD med -18.3%  naive med 0.848
#   FAIL on all three legs, decisively — the WORST gate variant tested this summer
#   (-0.13 median exSharpe; even the worst offset of the baseline beats the MEDIAN
#   RSRS offset). The binary hysteresis whipsaws out of US bull legs and the signal
#   adds no crash protection the sleeve didn't already have (DD deepens). The
#   pre-run expectation (FAIL, priors a+b+c) held with room to spare. Direct
#   comparison, same harness: canary 3-state breadth gate +0.077, RSRS binary
#   geometry gate -0.129 — the gate SLOT is real (canary proved it), this signal
#   for it is not. VERDICT: CLOSED [EMPIRICAL]. The Chinese-timing question is
#   answered with receipts; no RSRS variant hunting (the modified/R2 versions would
#   be selection after a failed original — exactly what the protocol forbids).
#   Ledger: naive 0.85 -> TRIAL_SHARPES. The 2026-07-12 sweep queue is now fully
#   resolved: 1 two-stage survivor (canary), 1 ops adoption (buffering), 2 kills
#   (HRP, RSRS).
