# PRE-REGISTERED (2026-07-13, committed BEFORE running — two-commit proof).
# Second experiment of the post-go-live queue (canary -> HRP -> buffering -> RSRS).
#
# HYPOTHESIS (from the 2026-07-12 global repo sweep — Lopez de Prado's Hierarchical
# Risk Parity, the skfolio/PyPortfolioOpt standard): replacing the sleeve's
# inverse-vol risk allocation with HRP — cluster-aware allocation via hierarchical
# clustering + recursive bisection — improves the blend. Mechanism claim: inverse-vol
# double-counts correlated assets (the 2-equity and 2-duration pairs each soak up
# 2x their diversification-adjusted share); HRP allocates across CLUSTERS first.
#
# SURGICAL A/B (one change only): baseline = production VolTargetTSMOM (ensemble
# strength x inverse-vol, normalized, vol-target scaled). Variant = identical signal
# layer and vol targeting, but strength x HRP-weight instead of strength x inverse-vol.
# HRP implemented natively in numpy (no scipy dep): correlation-distance
# d = sqrt((1-rho)/2), greedy single-linkage merge order as the seriation (exact
# enough at n<=6 — noted; re-derive with a proper dendrogram leaf order if this is
# ever adopted), recursive bisection splitting variance inversely between halves.
# All 21 offsets, blend level (blend_curve), HONEST convention. 5bps costs.
#
# ADOPTION BAR (house distribution-dominance standard): variant passes stage 1 only if
# blend honest excess-Sharpe MEDIAN > baseline's AND worst-offset >= baseline's worst
# AND blend maxDD median no deeper. Stage-1 pass -> pre-registered OOS annex required
# (dot-com proxy panel) before any further claim. Anything less -> ledger + close.
# EXPECTATION ON RECORD: WASH, bar FAIL on the strict median-> condition. At 4-6
# trending assets with one obvious cluster structure (SPY/EFA equities, TLT/IEF
# duration, GLD, DBC), HRP approximately reproduces cluster-aware inverse-vol; the
# literature's HRP edge lives at large N (20+ assets) and in estimation-noise
# robustness we don't stress at monthly cadence. The strength multiplier already
# tilts allocation by signal confirmation, further shrinking the delta. This is a
# documented-closure experiment: cheap, expected wash, kills the "should we use HRP"
# question with receipts either way.
# LEDGER: ONE trial (naive-convention blend median in the results commit).
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


def _single_linkage_order(dist):
    """Greedy single-linkage merge order (seriation stand-in, n<=6)."""
    clusters = [[i] for i in range(len(dist))]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(dist[i, j] for i in clusters[a] for j in clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters[0]


def _hrp_weights(cov, order):
    """Recursive bisection over the seriated order (Lopez de Prado)."""
    w = pd.Series(1.0, index=order)
    stack = [list(order)]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]

        def cluster_var(sub):
            c = cov.loc[sub, sub].values
            ivp = 1.0 / np.diag(c)
            ivp /= ivp.sum()
            return float(ivp @ c @ ivp)

        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1.0 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack += [left, right]
    return w


class HRPTSMOM(VolTargetTSMOM):
    """Production signal layer + vol targeting; HRP replaces inverse-vol only."""

    def target_weights(self, closes, i):
        if i < max(self.looks) or i % self.every != self.offset:
            return None
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        strength = {}
        for t in closes.columns:
            p0 = closes.iloc[i].get(t)
            v = rets[t].std() if t in rets else np.nan
            if not (p0 and np.isfinite(p0) and np.isfinite(v) and v > 0):
                continue
            sigs = []
            for lk in self.looks:
                pm = closes.iloc[i - lk].get(t)
                if not (pm and np.isfinite(pm)):
                    sigs = None
                    break
                sigs.append(1.0 if p0 / pm - 1 > 0 else 0.0)
            if sigs is None:
                continue
            s = float(np.mean(sigs))
            if s > 0:
                strength[t] = s
        if not strength:
            return pd.Series(dtype=float)
        on = list(strength)
        rets_on = rets[on].dropna()
        cov = rets_on.cov() * 252
        if len(on) == 1:
            w_hrp = pd.Series(1.0, index=on)
        else:
            corr = rets_on.corr().values
            dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
            order = [on[k] for k in _single_linkage_order(dist)]
            w_hrp = _hrp_weights(cov, order)
        w = pd.Series({t: strength[t] * w_hrp[t] for t in on})
        w = w / w.abs().sum()
        pvol = float(np.sqrt(w.values @ cov.loc[w.index, w.index].values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def naive_sharpe(eq):
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(252))


rows = []
for off in range(21):
    base_strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    hrp_strat = HRPTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    kw = dict(cost=costs.proportional(5), fill="next_open", cash_rate=rf)
    base = blend_curve(run_xs(panels, base_strat, **kw), spy)
    hrp = blend_curve(run_xs(panels, hrp_strat, **kw), spy)
    rows.append({"base_exs": ex_sharpe(base), "base_dd": metrics.max_drawdown(base),
                 "hrp_exs": ex_sharpe(hrp), "hrp_dd": metrics.max_drawdown(hrp),
                 "hrp_naive": naive_sharpe(hrp)})
df = pd.DataFrame(rows)

print("HONEST convention, 21 offsets, blend level (HRP vs inverse-vol in the sleeve):")
print(f"baseline (inv-vol) exSharpe med {df.base_exs.median():.3f} "
      f"[{df.base_exs.min():.3f},{df.base_exs.max():.3f}]  maxDD med {df.base_dd.median()*100:5.1f}%")
print(f"HRP sleeve         exSharpe med {df.hrp_exs.median():.3f} "
      f"[{df.hrp_exs.min():.3f},{df.hrp_exs.max():.3f}]  maxDD med {df.hrp_dd.median()*100:5.1f}%  "
      f"naive med {df.hrp_naive.median():.3f}")

ok = (df.hrp_exs.median() > df.base_exs.median()
      and df.hrp_exs.min() >= df.base_exs.min()
      and df.hrp_dd.median() >= df.base_dd.median())
print(f"\nVERDICT vs pre-registered bar: "
      f"{'STAGE-1 PASS -> pre-register the OOS annex' if ok else 'FAIL/WASH -> ledger + close (expected)'}")
