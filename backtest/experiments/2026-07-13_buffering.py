# PRE-REGISTERED (2026-07-13, committed BEFORE running — two-commit proof).
# Third of the post-go-live queue (canary PASSED->Aug design call; HRP FAILED->closed).
#
# OPS-TIER MEASUREMENT, not a signal candidate — NO LEDGER SLOT (pre-agreed 2026-07-12:
# buffering changes execution only; the signal path is untouched, so nothing is being
# selected from a strategy space and the DSR hurdle is not implicated).
#
# QUESTION (Carver position buffering, from the repo sweep): does a no-trade band —
# hold when the drifted weight is within +-10% of target, else trade to the band edge —
# cut the sleeve's turnover at ~zero performance cost? Payoffs if adopted: (a) ~1-3bps/yr
# saved in modeled costs (small, known), (b) the REAL prize: fewer/smaller real-account
# orders to hand-type each month at Chase, and fewer taxable micro-realizations.
#
# SPEC (fixed): buffer_frac = 0.10 (Carver's heuristic scale) is THE decision variant.
# 0.05 and 0.20 are reported DESCRIPTIVELY for sensitivity — the adoption decision reads
# ONLY the 0.10 row (no picking the best band; that would be selection).
# Engine support (engine_xs.run_xs buffer_frac + _buffered) added and unit-tested in
# this same commit — trade-to-band-edge, composition-basis drift, empty-book buys to
# target. All 21 offsets, blend level, HONEST convention. Turnover measured directly
# by a recording cost wrapper (sum of |traded notional| / mean equity / years).
#
# ADOPTION BAR (ops bar, pre-registered): adopt buffer_frac=0.10 into the live sleeve
# config only if, across 21 offsets: median annual turnover falls >= 20% vs baseline
# AND median blend honest exSharpe is within 0.005 of baseline (no meaningful
# performance cost, in EITHER direction — a big gain would mean the band is doing
# signal work by accident and needs its own study) AND median blend maxDD no more than
# 0.5pt deeper. Anything else -> close as "measured, not worth it."
# EXPECTATION ON RECORD: turnover -30 to -60% (most monthly trades are small
# drift-corrections well inside a 10% band), exSharpe delta within +-0.005 (wash),
# maxDD ~unchanged. Lean ADOPT — this is the rare test whose expected outcome is a
# free operational simplification, which is exactly why it must clear a "changes
# nothing else" bar rather than a dominance bar.
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from backtest.trend_sleeve import etf_panel, VolTargetTSMOM, ENSEMBLE_LOOKS
from backtest.engine_xs import run_xs
from backtest.timing_luck import blend_curve
from backtest.leverage_study import tbill_series
from backtest import costs, metrics

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)
rf_d = (rf / 252.0).reindex(spy.index).ffill().fillna(0.0)


def recording_cost(bps, box):
    base = costs.proportional(bps)

    def cost(delta_shares, price):
        box[0] += abs(delta_shares * price)
        return base(delta_shares, price)

    return cost


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def run_variant(off, frac):
    box = [0.0]
    strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    eq = run_xs(panels, strat, cost=recording_cost(5, box), fill="next_open",
                cash_rate=rf, buffer_frac=frac)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    turnover = box[0] / float(eq.mean()) / yrs               # two-way, x/yr
    b = blend_curve(eq, spy)
    return {"exs": ex_sharpe(b), "dd": metrics.max_drawdown(b), "turn": turnover}


res = {}
for name, frac in (("baseline", 0.0), ("buffer 0.05", 0.05),
                   ("buffer 0.10", 0.10), ("buffer 0.20", 0.20)):
    df = pd.DataFrame([run_variant(off, frac) for off in range(21)])
    res[name] = df
    print(f"{name:12s} blend exSharpe med {df.exs.median():.3f} "
          f"[{df.exs.min():.3f},{df.exs.max():.3f}]  maxDD med {df.dd.median()*100:5.1f}%  "
          f"sleeve turnover med {df.turn.median():.2f}x/yr")

base, ten = res["baseline"], res["buffer 0.10"]
turn_cut = 1 - ten.turn.median() / base.turn.median()
ok = (turn_cut >= 0.20
      and abs(ten.exs.median() - base.exs.median()) <= 0.005
      and ten.dd.median() >= base.dd.median() - 0.005)
print(f"\nturnover cut at 0.10: {turn_cut*100:.0f}%   "
      f"exSharpe delta: {ten.exs.median() - base.exs.median():+.3f}   "
      f"maxDD delta: {(ten.dd.median() - base.dd.median())*100:+.2f}pt")
print(f"VERDICT vs pre-registered ops bar: "
      f"{'ADOPT buffer_frac=0.10 (live sleeve config + runbook)' if ok else 'CLOSE — measured, not worth it'}")
