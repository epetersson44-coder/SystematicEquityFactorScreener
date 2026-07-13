# PRE-REGISTERED (2026-07-13, committed BEFORE running — two-commit proof).
# STAGE 2 (required OOS annex) of the canary-gate candidate: stage 1
# (2026-07-13_canary_gate.py, pre-reg 64f6f88) PASSED the full dominance bar
# in-sample (median exSharpe 0.844 vs 0.767, worst offset 0.740 vs 0.720, maxDD med
# -13.0% vs -16.9%) — a BIGGER margin than DMOM top-3 showed before the dot-com panel
# killed it. Per the stage-1 header, no claim exists until this annex rules.
#
# QUESTION: does the canary gate's dominance survive the 1999-2006 proxy era — the one
# out-of-window regime with a multi-year equity bear (2000-02) plus a strong subsequent
# bull (2003-06)? This is where in-sample-fit gates go to die (DMOM precedent).
#
# PANEL: the established dot-com proxy panel (2026-07-04_dotcom_proxy_extension.py):
# SPY real; FDIVX->EFA, VUSTX->TLT, VFITX->IEF, GC=F->GLD (joins late), FRED WTI->DBC.
# CANARY PROXIES: VEIEX (Vanguard EM index fund, 1994+) -> EEM; VBMFX (Vanguard Total
# Bond, 1986+) -> AGG. Both validated below against their ETFs on the 2003+ overlap
# (report corr; the gate needs SIGN of 13612W, so corr ~0.9+ is ample). rf = real ^IRX.
# Same 13612W, same b/2 rule, same 5bps, same next-day apply. Window 1999-07-01 ->
# 2006-06-30, all 21 offsets, honest convention.
#
# PRE-REGISTERED BAR (same three legs as stage 1, on this panel): canary-gated blend
# median exSharpe > baseline median AND worst-offset >= baseline's worst AND median
# maxDD no deeper. PASS -> the candidate graduates to a DESIGN CALL at a future lock
# (August at the earliest; taxes/turnover priced there; NO auto-adoption). FAIL any
# leg -> in-sample-only dominance, banked exactly like DMOM, live book unchanged.
# EXPECTATION ON RECORD: coin flip, lean pass-on-DD / uncertain-on-Sharpe. Mechanism
# cuts both ways here: 2000-02 should suit the gate (EM negative while bonds rallied ->
# e=0.5 through the bear -> shallower DD), but the 1999-2000 melt-up and 2003-06 EM
# whipsaws can bleed the Sharpe legs. DMOM's corpse is the reminder that a plausible
# mechanism story guarantees nothing.
# LEDGER: no new entry (stage 2 of the SAME candidate; stage 1's naive 1.05 enters
# TRIAL_SHARPES in the results commit).
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
EXT_END = "2006-06-30"
REPORT_START = "1999-07-01"


def fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     timeout=30, headers=UA)
    s = pd.read_csv(io.StringIO(r.text), parse_dates=[0], index_col=0, na_values=".").iloc[:, 0]
    return s.dropna()


print("fetching proxies ...")
yfd = yf.download("SPY FDIVX VUSTX VFITX GC=F VEIEX VBMFX EEM AGG ^IRX",
                  start="1996-01-01", progress=False, auto_adjust=True)["Close"]
wti = fred("DCOILWTICO")

print("canary-proxy validation (daily-return corr, 2003+ overlap):")
for pxy, real in (("VEIEX", "EEM"), ("VBMFX", "AGG")):
    both = yfd[[pxy, real]].dropna()
    r2 = both.pct_change().dropna()
    print(f"  {pxy} -> {real}: corr {r2.corr().iloc[0, 1]:.2f}  n={len(r2)}")

ext = yfd[["SPY", "FDIVX", "VUSTX", "VFITX", "GC=F"]].join(wti.rename("WTI"), how="left")
ext = ext[ext.index <= EXT_END].ffill(limit=3)
panels = {"Close": ext, "Open": ext}
spy = ext["SPY"].dropna()
rf = (yfd["^IRX"].reindex(spy.index).ffill() / 100.0).fillna(0.0)
rf_d = rf / 252.0
canaries = yfd[["VEIEX", "VBMFX"]].reindex(spy.index).ffill()
LOOKS_D = {21: 12.0, 63: 4.0, 126: 2.0, 252: 1.0}


def w13612(px, i):
    if i < 252 or not np.isfinite(px.iloc[i - 252]):
        return np.nan
    return sum(w * (px.iloc[i] / px.iloc[i - d] - 1) for d, w in LOOKS_D.items()) / 19.0


def gated_spy_curve(offset, cost_bps=5):
    spy_ret = spy.pct_change().fillna(0.0)
    e_series = np.ones(len(spy))
    e = 1.0
    for i in range(len(spy)):
        if i % 21 == offset and i >= 252:
            b = sum(1 for t in ("VEIEX", "VBMFX")
                    if (w := w13612(canaries[t], i)) == w and w <= 0)
            e = 1.0 - b / 2.0
        e_series[i] = e
    e_s = pd.Series(e_series, index=spy.index).shift(1).fillna(1.0)
    turn_cost = e_s.diff().abs().fillna(0.0) * (cost_bps / 10_000.0)
    ret = e_s * spy_ret + (1 - e_s) * rf_d - turn_cost
    return 10_000 * (1 + ret).cumprod()


def ex_sharpe(eq, start=REPORT_START):
    eq = eq[eq.index >= start]
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def dd(eq, start=REPORT_START):
    return metrics.max_drawdown(eq[eq.index >= start])


rows = []
for off in range(21):
    strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    sleeve = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open", cash_rate=rf)
    base = blend_curve(sleeve, spy)
    gated = blend_curve(sleeve, gated_spy_curve(off))
    rows.append({"base_exs": ex_sharpe(base), "base_dd": dd(base),
                 "can_exs": ex_sharpe(gated), "can_dd": dd(gated)})
df = pd.DataFrame(rows)

print(f"\nOOS 1999-07 -> 2006-06, 21 offsets, honest convention (^IRX of the era):")
print(f"baseline blend   exSharpe med {df.base_exs.median():.3f} "
      f"[{df.base_exs.min():.3f},{df.base_exs.max():.3f}]  maxDD med {df.base_dd.median()*100:5.1f}%")
print(f"canary-gated     exSharpe med {df.can_exs.median():.3f} "
      f"[{df.can_exs.min():.3f},{df.can_exs.max():.3f}]  maxDD med {df.can_dd.median()*100:5.1f}%")

ok = (df.can_exs.median() > df.base_exs.median()
      and df.can_exs.min() >= df.base_exs.min()
      and df.can_dd.median() >= df.base_dd.median())
print(f"\nVERDICT vs pre-registered bar: "
      f"{'OOS PASS -> design call at a FUTURE lock (no auto-adoption)' if ok else 'FAIL -> in-sample-only, banked (the DMOM outcome)'}")

# RESULTS (run 2026-07-13, unmodified from pre-registration e9829df):
#   canary-proxy validation: VEIEX->EEM corr 0.94; VBMFX->AGG corr 0.79 (below the
#   ~0.9 the header hoped for — acceptable for a SIGN-based gate, noted honestly).
#   baseline blend   exSharpe med 0.615 [0.486,0.682]  maxDD med -9.1%
#   canary-gated     exSharpe med 0.623 [0.496,0.690]  maxDD med -7.7%
#   OOS PASS on all three legs — the first candidate to survive BOTH stages (DMOM died
#   here). HONEST MAGNITUDE READ: in-sample +0.077 exSharpe/-3.9pts DD decays to
#   +0.008/-1.4pts OOS. Direction robust, size regime-dependent: the 2006-26 margin is
#   flattered by canary-friendly crashes (2008/2020/2022); the floor looks like
#   "harmless, mildly DD-protective." VERDICT per pre-registration: graduates to a
#   DESIGN CALL at the AUGUST 2026 lock, no auto-adoption. SCOPE NOTE for that call:
#   what passed is a BLEND-construction upgrade (equity-leg gate) — the shadow 2.3x
#   would inherit it via the locks. It is NOT evidence for gating ssoB's UPRO leg:
#   that book's thesis is beat-SPY-RAW, and e<1 months cut raw return in bulls — the
#   ssoB defensive step-down failed exactly there. An ssoB-level gate would need its
#   own experiment against its own thesis.
