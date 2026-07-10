# PRE-REGISTERED (2026-07-10, committed BEFORE running — two-commit proof).
#
# HYPOTHESIS (ninth external review, F2 — the last live-relevant design closure resting
# on a discredited convention): the sleeve's max_gross=1.0 verdict ("removing the old 2x
# cap IMPROVED the blend — financing cost + amplified drawdowns outweighed it") was
# priced at FLAT financing_bps=400, the exact convention leverage_study.py later proved
# overcharged the ZIRP-era bulls by ~3%/yr and whose correction FLIPPED the external
# blend-leverage verdict. The correction was never applied to the sleeve-INTERNAL gross
# question (the engine couldn't even express it until 03a3aa0 added Series financing).
# Honest financing (^IRX + 40bps) may flip this verdict too.
#
# DIAGNOSTIC (measured 2026-07-10 pre-registration; mechanism context, NOT the outcome
# variable): across 228 monthly rebalances 2006-2026, the gross cap binds on 80% of
# them — the vol target asks for median gross 1.43 (p25 1.09, p75 1.88, max 3.07) and
# the capped sleeve realizes only 7.5% vol vs the 10% design. The sleeve chronically
# under-delivers its designed risk in exactly the calm, diversified regimes where trend
# risk is cheapest (the standard managed-futures construction levers precisely there).
# CASH-ACCOUNT COROLLARY, settled by the same measurement: raising target_vol in the
# no-margin account adds exposure only in the ~20% of months where the TARGET binds —
# the concentrated/crisis regimes — the opposite of free. No in-cash-account headroom
# exists; this experiment is a MARGIN-ERA design question (2.3x-shadow / leverage-era
# playbook), banked either way, not implementable in the live cash account.
#
# DESIGN: sleeve gross cap G ∈ {1.0 baseline, 1.5, 2.0}; financing at ^IRX + 40bps as a
# time-varying Series; cash at ^IRX; costs 5bps; adopted 1/3/12 ENSEMBLE_LOOKS; all 21
# rebalance offsets; HONEST convention (excess-return Sharpes). Compared at BLEND level
# (risk-parity SPY + sleeve via blend_curve) — where the design decision lives.
#
# ADOPTION BAR (the house distribution-dominance standard, same as DMOM/defensive):
#   adopt G>1 into the margin-era design only if, across 21 offsets, blend honest
#   excess-Sharpe MEDIAN(G) > median(baseline) AND worst-offset(G) >= baseline's worst
#   AND blend maxDD median is no deeper. Anything less -> ledger + bank with note.
# EXPECTATION ON RECORD: the reviewer leans "pass on Sharpe, coin-flip on maxDD". Mine:
# honest financing turns the old clear loss into a close call on Sharpe (the 80%
# cap-binding says the extra gross lands mostly in cheap-risk regimes), but the
# amplified-drawdown half of the old verdict was NEVER a financing artifact — I expect
# G=2.0 to FAIL on the maxDD leg and G=1.5 to be the live question. Full-bar pass:
# uncertain, lean no.
# LEDGER: two candidate trials (G=1.5, G=2.0) -> TRIAL_SHARPES (naive-convention
# Sharpes for ledger comparability) in the results commit.
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
rf = tbill_series(spy.index)                    # annualized decimal, by date
rf_d = rf / 252.0
fin_bps = rf * 10_000.0 + 40.0                  # honest financing: ^IRX + 40bps, in bps


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).ffill().fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def naive_sharpe(eq):                            # ledger convention (rf=0)
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(252))


def sweep(gross):
    rows = []
    for off in range(21):
        strat = VolTargetTSMOM(max_gross=gross, looks=ENSEMBLE_LOOKS, offset=off)
        eq = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open",
                    leverage=gross, gross_max=gross,
                    financing_bps=fin_bps, cash_rate=rf)
        b = blend_curve(eq, spy)
        rows.append({"blend_exs": ex_sharpe(b), "blend_dd": metrics.max_drawdown(b),
                     "sleeve_exs": ex_sharpe(eq), "sleeve_naive": naive_sharpe(eq)})
    return pd.DataFrame(rows)


print("HONEST convention (cash ^IRX, financing ^IRX+40bps), 21 offsets, blend level:")
res = {}
for name, g in (("baseline G=1.0", 1.0), ("levered G=1.5", 1.5), ("levered G=2.0", 2.0)):
    df = sweep(g)
    res[name] = df
    print(f"{name:15s} blend exSharpe med {df.blend_exs.median():.3f} "
          f"[{df.blend_exs.min():.3f},{df.blend_exs.max():.3f}]  "
          f"maxDD med {df.blend_dd.median()*100:5.1f}%  "
          f"sleeve exSharpe med {df.sleeve_exs.median():.3f}  "
          f"sleeve naive med {df.sleeve_naive.median():.3f}")

base = res["baseline G=1.0"]
print("\nVERDICT vs pre-registered bar (median AND worst-offset AND maxDD dominance):")
for name in ("levered G=1.5", "levered G=2.0"):
    df = res[name]
    ok = (df.blend_exs.median() > base.blend_exs.median()
          and df.blend_exs.min() >= base.blend_exs.min()
          and df.blend_dd.median() >= base.blend_dd.median())
    print(f"  {name}: {'ADOPT bar MET (margin-era design)' if ok else 'FAIL -> ledger + bank'}")
