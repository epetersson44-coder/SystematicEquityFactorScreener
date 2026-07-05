# PRE-REGISTERED (2026-07-05, committed BEFORE running — two-commit proof; annex to
# 2026-07-05_dual_momentum_sleeve.py, where DMOM top-3 MET its adoption bar in-sample).
#
# PURPOSE: the top-3 dominance was measured on the same 2006-26 panel that motivated
# the idea. Before it can be an ADOPTION CANDIDATE at the July-13 lock it must show
# the same character on a regime it has never seen: the 1999-2006 dot-com proxy panel
# (VUSTX/VFITX/FDIVX/GC=F/FRED-WTI + SPY — the validated extension universe).
#
# BAR (pre-registered): on the extension panel, all 21 offsets, blend level, rf=0
# metric (no reliable pre-2001 ^IRX intraday alignment worry — the COMPARISON is
# paired, same convention both variants):
#   top-3 blend Sharpe MEDIAN > hold-all's AND top-3 WORST offset >= hold-all's worst
#   AND median maxDD no deeper.
#   PASS -> top-3 becomes an adoption CANDIDATE presented for the July-13 lock
#           (structural sleeve change = Erik's call, per zero-tinker terms).
#   FAIL -> BANKED as in-sample-only dominance (regime-sensitive concentration);
#           live sleeve unchanged.
# EXPECTATION ON RECORD: genuinely uncertain — the mechanism (selection only binds
# when many assets trend) suggests it should carry, but 1999-2006 had long stretches
# of exactly that (oil/bonds/gold all trending 2001-06). No confident prediction.
import sys, io, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest.trend_sleeve import VolTargetTSMOM, ENSEMBLE_LOOKS
from backtest.engine_xs import run_xs
from backtest.timing_luck import blend_curve
from backtest import costs, metrics

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
EXT_END, REPORT_START = "2006-06-30", "1999-07-01"


class DualMomTSMOM(VolTargetTSMOM):
    """Same class as the main experiment (kept in-file: experiment scripts are
    evidence, not importable modules)."""

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
                cs[t] = float(np.mean(moms))
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


def fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     timeout=30, headers=UA)
    return pd.read_csv(io.StringIO(r.text), parse_dates=[0], index_col=0,
                       na_values=".").iloc[:, 0].dropna()


yfd = yf.download("SPY FDIVX VUSTX VFITX GC=F", start="1996-01-01",
                  progress=False, auto_adjust=True)["Close"]
base = yfd.join(fred("DCOILWTICO").rename("WTI"), how="left")
base = base[base.index <= EXT_END].ffill(limit=3)
panels = {"Close": base, "Open": base}
spy_px = base["SPY"].dropna()


def sweep_variant(top_k):
    rows = []
    for off in range(21):
        strat = DualMomTSMOM(top_k=top_k, max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
        eq = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open")
        b = blend_curve(eq, spy_px)
        b = b[b.index >= REPORT_START]
        rows.append({"blend_s": metrics.sharpe(b), "blend_dd": metrics.max_drawdown(b)})
    return pd.DataFrame(rows)


print(f"extension panel {REPORT_START} -> {EXT_END}, 21 offsets, blend level (rf=0 paired):")
res = {}
for name, k in (("hold-all", None), ("DMOM top-3", 3)):
    df = sweep_variant(k)
    res[name] = df
    print(f"{name:12s} blend Sharpe med {df.blend_s.median():.3f} "
          f"[{df.blend_s.min():.3f},{df.blend_s.max():.3f}]  "
          f"maxDD med {df.blend_dd.median()*100:5.1f}%")

b, d = res["hold-all"], res["DMOM top-3"]
ok = (d.blend_s.median() > b.blend_s.median() and d.blend_s.min() >= b.blend_s.min()
      and d.blend_dd.median() >= b.blend_dd.median())
print(f"\nVERDICT: {'PASS — top-3 is an ADOPTION CANDIDATE for the July-13 lock (Erik decides)'
      if ok else 'FAIL — BANKED as in-sample-only; live sleeve unchanged'}")
