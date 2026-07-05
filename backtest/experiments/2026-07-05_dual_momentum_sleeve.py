# PRE-REGISTERED (2026-07-05, committed BEFORE running — two-commit proof).
#
# HYPOTHESIS (Erik's, via the DMOM/TRIMOM literature — Antonacci's dual momentum):
# combining the sleeve's TIME-SERIES gate with a CROSS-SECTIONAL selection — among
# trending assets, hold only the top-k by relative momentum — improves the blend over
# the current hold-all-trending construction. The lab's sleeve already carries a SOFT
# CS tilt (ensemble strength 0/1/3/2/3/1 x inverse-vol grades assets by trend
# confirmation); THIS tests the HARD version, which is genuinely absent from the
# 45-trial ledger.
#
# CONSTRUCTION FIXES vs the external proposal (errors the test must not inherit):
#   - "top 20% of winners" on a 6-asset universe = ~1 asset; honest variants: top-2, top-3.
#   - TRIMOM's bear-rebound inversion targets the SHORT leg of stock momentum (the
#     Daniel-Moskowitz crash); both our books are long-only w/ cash failsafes — N/A.
#   - The stock momentum book cannot be the CS leg (survivor-biased data); the honest
#     cross-section is the 6-ETF panel itself.
#
# DESIGN: DualMomTSMOM = same 1/3/12 ensemble gate, same vol targeting, but among
# assets with positive ensemble strength keep only the TOP-K by CS score (mean of
# per-look total returns), inverse-vol weighted. Variants: k=2, k=3; baseline = the
# live construction (k=all). All 21 offsets; HONEST convention (cash at ^IRX, excess
# Sharpes) — the lab's quotable standard since 2026-07-05. Compared at BLEND level
# (risk-parity SPY + sleeve), where the live decision lives.
#
# ADOPTION BAR (the ensemble standard — distribution dominance):
#   adopt k-variant only if its blend honest-excess-Sharpe MEDIAN across 21 offsets
#   exceeds baseline's AND its WORST offset >= baseline's worst AND blend maxDD median
#   is no deeper. Anything less -> ledger + close (or bank if the ride/pile trade is
#   interesting).
# EXPECTATION ON RECORD: FAILS on breadth — concentrating ~4-6 trending assets into
# 2-3 cuts independent bets (Grinold); every prior result (lookback ensemble, offset
# tranche, 10-ETF test) says signal diversification is where the Sharpe lives. The
# concentrated sleeve may show higher standalone CAGR in some offsets; the blend
# distribution should be worse or a wash.
# LEDGER: two candidate trials (k=2, k=3) -> TRIAL_SHARPES in the results commit.
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from backtest.trend_sleeve import VolTargetTSMOM, ENSEMBLE_LOOKS, etf_panel
from backtest.engine_xs import run_xs
from backtest.timing_luck import blend_curve
from backtest.leverage_study import tbill_series
from backtest import costs, metrics


class DualMomTSMOM(VolTargetTSMOM):
    """TS gate (ensemble strength > 0) -> CS selection (top-k by mean per-look return)
    -> inverse-vol weights -> vol-target scale. k=None reproduces hold-all."""

    def __init__(self, top_k=None, **kw):
        super().__init__(**kw)
        self.top_k = top_k

    def target_weights(self, closes, i):
        if i < max(self.looks) or i % self.every != self.offset:
            return None
        rets = closes.iloc[i - self.vol_lb:i + 1].pct_change().iloc[1:]
        strength, cs = {}, {}
        for t in closes.columns:
            p0 = closes.iloc[i].get(t)
            v = rets[t].std() if t in rets else np.nan
            if not (p0 and np.isfinite(p0) and np.isfinite(v) and v > 0):
                continue
            sigs, moms = [], []
            for lk in self.looks:
                pm = closes.iloc[i - lk].get(t)
                if not (pm and np.isfinite(pm)):
                    sigs = None
                    break
                r = p0 / pm - 1
                sigs.append(1.0 if r > 0 else 0.0)
                moms.append(r)
            if sigs is None:
                continue
            s = float(np.mean(sigs))
            if s > 0:
                strength[t] = s
                cs[t] = float(np.mean(moms))                 # CS score: mean of look returns
        if not strength:
            return pd.Series(dtype=float)
        on = list(strength)
        if self.top_k is not None and len(on) > self.top_k:
            on = sorted(on, key=lambda t: cs[t], reverse=True)[:self.top_k]
        invvol = 1.0 / (rets[on].std() * np.sqrt(252))
        w = pd.Series({t: strength[t] * invvol[t] for t in on})
        w = w / w.abs().sum()
        cov = rets[on].cov() * 252
        pvol = float(np.sqrt(w.values @ cov.values @ w.values))
        scale = min(self.target_vol / pvol, self.max_gross) if pvol > 0 else 1.0
        return w * scale


panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)
rf_d = rf / 252.0


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).ffill().fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def sweep_variant(top_k):
    rows = []
    for off in range(21):
        strat = DualMomTSMOM(top_k=top_k, max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
        eq = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open",
                    cash_rate=rf)
        b = blend_curve(eq, spy)
        rows.append({"blend_exs": ex_sharpe(b), "blend_dd": metrics.max_drawdown(b),
                     "sleeve_exs": ex_sharpe(eq)})
    return pd.DataFrame(rows)


print("HONEST convention, 21 offsets each, blend level:")
res = {}
for name, k in (("baseline (hold-all)", None), ("DMOM top-3", 3), ("DMOM top-2", 2)):
    df = sweep_variant(k)
    res[name] = df
    print(f"{name:20s} blend exSharpe med {df.blend_exs.median():.3f} "
          f"[{df.blend_exs.min():.3f},{df.blend_exs.max():.3f}]  "
          f"maxDD med {df.blend_dd.median()*100:5.1f}%  "
          f"sleeve exSharpe med {df.sleeve_exs.median():.3f}")

base = res["baseline (hold-all)"]
print("\nVERDICT vs pre-registered bar (median AND worst-offset AND maxDD dominance):")
for name in ("DMOM top-3", "DMOM top-2"):
    df = res[name]
    ok = (df.blend_exs.median() > base.blend_exs.median()
          and df.blend_exs.min() >= base.blend_exs.min()
          and df.blend_dd.median() >= base.blend_dd.median())
    print(f"  {name}: {'ADOPT bar MET' if ok else 'FAIL -> ledger + close/bank'}")

# RESULTS (run 2026-07-05, unmodified from pre-registration f048580; honest convention):
#   baseline (hold-all)  blend exSharpe med 0.768 [0.719,0.820]  maxDD med -16.8%
#   DMOM top-3           blend exSharpe med 0.803 [0.731,0.844]  maxDD med -16.6%  BAR MET
#   DMOM top-2           blend exSharpe med 0.807 [0.690,0.872]  maxDD med -19.5%  FAIL
#   The pre-run expectation (fails on breadth) was WRONG for top-3 in-sample — the
#   protocol exists precisely to catch the author being wrong in either direction.
# ARC CONCLUSION: the OOS annex (2026-07-05_dmom_robustness.py, pre-reg ff5279e) then
#   FAILED top-3 on the 1999-2006 proxy panel (med 0.986 vs 1.018, maxDD -11.7% vs
#   -9.1%) — in the one regime where many assets trend for years (oil/gold/bonds
#   2001-06), concentration bites, exactly where the sleeve earns its keep. VERDICT:
#   in-sample-only dominance -> BANKED, live sleeve unchanged. Had adoption happened
#   on the first bar alone, a 2006-26 artifact would have shipped into the live book.
