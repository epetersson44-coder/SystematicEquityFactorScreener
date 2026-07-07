# MEASUREMENT (2026-07-05, single-commit: no hypothesis, no bar, no selection — a
# re-scoring of FROZEN constructions under the honest convention, closing the one gap
# the convention migration left: blend and ssoB got honest rows on 2026-07-05; the
# 2.3xL shadow (and the ladder rungs around it) did not. The ~$110k decision should be
# priced with the same ruler as everything else.
#
# Constructions (all frozen, all previously ledgered under rf=0): blend levered at
# L in {1.0, 1.5, 2.0, 2.3}, financing on (L-1) at real ^IRX + 40bps (levered_returns),
# on the HONEST-cash blend tranche (cash_rate = ^IRX in the sleeve engine). Metrics:
# raw rf=0 Sharpe (old ruler, for continuity) and HONEST EXCESS Sharpe (quotable),
# CAGR, maxDD, $10k terminal, 2006-07 -> present.
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import pandas as pd

from backtest.trend_sleeve import etf_panel
from backtest.timing_luck import sweep, tranched_curve, blend_curve
from backtest.leverage_study import tbill_series, levered_returns
from backtest import metrics

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)
rf_d = rf / 252.0

_, curves_h = sweep(panels=panels, cash_rate=rf)
blend_h = blend_curve(tranched_curve(curves_h, tuple(range(21))), spy).dropna()
b_ret = blend_h.pct_change().dropna()


def stats(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).ffill().fillna(0.0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
            metrics.sharpe(eq),
            float(ex.mean() / ex.std() * math.sqrt(252)),
            metrics.max_drawdown(eq),
            eq.iloc[-1] / eq.iloc[0] * 10000)


rows = {"SPY": spy.reindex(blend_h.index).dropna(), "blend 1.0x": blend_h}
for L in (1.5, 2.0, 2.3):
    rows[f"blend {L}x"] = (1 + levered_returns(b_ret, L, rf.reindex(b_ret.index))).cumprod()

print(f"{'':14s}{'CAGR':>8s}{'raw Sh':>8s}{'HONEST exSh':>12s}{'maxDD':>8s}{'$10k ->':>12s}")
for name, eq in rows.items():
    c, s, x, d, t = stats(eq.dropna())
    print(f"{name:14s}{c*100:7.2f}%{s:8.2f}{x:12.2f}{d*100:7.1f}%{t:>12,.0f}")

# RESULTS (2026-07-05):
#                  CAGR  raw Sh  HONEST exSh   maxDD     $10k ->
#   SPY           11.28%   0.65      0.57     -55.2%    $84,645
#   blend 1.0x     8.15%   0.97      0.78     -16.1%    $47,883
#   blend 1.5x    11.05%   0.89      0.77     -23.6%    $81,274
#   blend 2.0x    13.83%   0.85      0.76     -30.8%   $133,229
#   blend 2.3x    15.43%   0.84      0.76     -34.8%   $176,164
# INSIGHT: under the honest convention leverage is ~Sharpe-free (0.78 -> 0.76 across the
# whole ladder — only the 40bps spread on (L-1) drags, ~0.02 Sharpe at 2.3x). The ladder's
# price list is exact: each rung buys terminal wealth and pays in DRAWDOWN DEPTH, not
# efficiency. Honest cash accounting RAISED the 2.3x terminal ($159k -> $176k: the
# sleeve's cash earns T-bills before being levered). 2.3xL honest 0.76 vs ssoB 0.62 with
# +3.1%/yr CAGR and a shallower crash — the future construction dominates the current one
# on every honest axis; purely capital-gated.
