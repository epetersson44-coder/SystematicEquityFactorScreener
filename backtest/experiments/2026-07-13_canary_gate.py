# PRE-REGISTERED (2026-07-13, committed BEFORE running — two-commit proof).
# First post-go-live experiment; queue agreed 2026-07-12 (canary -> HRP -> buffering ->
# RSRS, one at a time, each closed before the next opens).
#
# HYPOTHESIS (from the 2026-07-12 global repo sweep — Keller & Keuning's VAA/DAA family,
# the one genuinely NEW mechanism found): gating the blend's naked EQUITY LEG by the
# fast momentum BREADTH of two "canary" assets (emerging-market equity + US aggregate
# bonds — global risk appetite leading indicators) improves the blend over the current
# ungated SPY leg. Mechanism: the sleeve already self-protects via per-asset trend; the
# equity leg is the book's only ungated beta, and canary breadth is claimed to detect
# risk-off regimes FASTER than own-asset 200d-class trend (13612W is ~1-month-weighted).
#
# PORT SPEC (fixed before the run; Keller's rule verbatim, no variant sweep):
#   - Canary universe: EEM + AGG (AGG substitutes the paper's BND for history — same
#     US aggregate-bond index class; EEM 2003-04+, AGG 2003-09+, full coverage of the
#     2006-07+ blend window incl. 12m warmup).
#   - Signal: 13612W momentum, W = (12*r1m + 4*r3m + 2*r6m + 1*r12m)/19, months = 21
#     trading days (lab convention: 21/63/126/252).
#   - Gate: b = count of canaries with W <= 0. Equity exposure e = 1 - b/2 (1 / 0.5 / 0).
#     Displaced weight earns rf (^IRX) daily. Evaluated at the sleeve's own rebalance
#     dates (every 21 days at offset o), applied from the NEXT day (fill convention),
#     5bps cost on |delta e|.
#   - Sleeve untouched. Compared at BLEND level: blend_curve(sleeve, gated_SPY_curve)
#     vs baseline blend_curve(sleeve, SPY). All 21 offsets, HONEST convention.
#
# ADOPTION BAR (house distribution-dominance standard): candidate passes stage 1 only
# if blend honest excess-Sharpe MEDIAN(canary) > median(baseline) AND worst-offset >=
# baseline's worst AND blend maxDD median no deeper. If stage 1 passes, a SEPARATE
# pre-registered stage-2 OOS on the 1999-2006 dot-com proxy panel (VEIEX/VBMFX as
# canary proxies) is REQUIRED before any adoption talk (the DMOM lesson). Adoption
# itself would then be a design call at a FUTURE lock. Anything less -> ledger + bank.
# EXPECTATION ON RECORD: moderate-LOW prior, lean FAIL on the full bar. Expected shape:
# maxDD improves (the gate will dodge some of 2008/2020/2022), median Sharpe wash to
# slightly up, but bull-market whipsaw (2011/2015/2018 corrections) costs participation
# on the ~44%-weight equity leg — rhyming with the ssoB defensive step-down, which
# halved DD and still failed on bull lag. Published 2017/18 -> post-publication decay
# applies. If it DOES pass, the OOS gate decides whether it was ever real.
# LEDGER: ONE trial (single pre-specified rule, no gate-strength mining).
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

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)
rf_d = (rf / 252.0).reindex(spy.index).ffill().fillna(0.0)

canaries = pd.DataFrame({t: get_prices(t)["Close"] for t in ("EEM", "AGG")})
canaries = canaries.reindex(spy.index).ffill()
LOOKS_D = {21: 12.0, 63: 4.0, 126: 2.0, 252: 1.0}          # days -> 13612W weight


def w13612(px, i):
    if i < 252 or not np.isfinite(px.iloc[i - 252]):
        return np.nan
    num = sum(w * (px.iloc[i] / px.iloc[i - d] - 1) for d, w in LOOKS_D.items())
    return num / 19.0


def gated_spy_curve(offset, cost_bps=5):
    """SPY equity-leg curve with the canary b/2 gate, rebalanced every 21d at `offset`."""
    spy_ret = spy.pct_change().fillna(0.0)
    e_series = np.ones(len(spy))
    e = 1.0
    for i in range(len(spy)):
        if i % 21 == offset and i >= 252:
            b = sum(1 for t in ("EEM", "AGG")
                    if (w := w13612(canaries[t], i)) == w and w <= 0)
            # if a canary has no 12m history yet, w is NaN -> treated as risk-ON
            e = 1.0 - b / 2.0
        e_series[i] = e
    e_s = pd.Series(e_series, index=spy.index).shift(1).fillna(1.0)   # next-day apply
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
                 "can_exs": ex_sharpe(gated), "can_dd": metrics.max_drawdown(gated),
                 "can_naive": naive_sharpe(gated)})
df = pd.DataFrame(rows)

print("HONEST convention, 21 offsets, blend level (canary = EEM+AGG 13612W, b/2 gate on the equity leg):")
print(f"baseline blend   exSharpe med {df.base_exs.median():.3f} "
      f"[{df.base_exs.min():.3f},{df.base_exs.max():.3f}]  maxDD med {df.base_dd.median()*100:5.1f}%")
print(f"canary-gated     exSharpe med {df.can_exs.median():.3f} "
      f"[{df.can_exs.min():.3f},{df.can_exs.max():.3f}]  maxDD med {df.can_dd.median()*100:5.1f}%  "
      f"naive med {df.can_naive.median():.3f}")

ok = (df.can_exs.median() > df.base_exs.median()
      and df.can_exs.min() >= df.base_exs.min()
      and df.can_dd.median() >= df.base_dd.median())
print(f"\nVERDICT vs pre-registered bar: "
      f"{'STAGE-1 PASS -> pre-register the OOS annex before ANY further claim' if ok else 'FAIL -> ledger + bank'}")
