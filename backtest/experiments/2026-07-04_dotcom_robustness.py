# PRE-REGISTERED (2026-07-04, committed BEFORE running — two-commit proof, annex to
# 2026-07-04_dotcom_proxy_extension.py which PASSED both its bars).
#
# PURPOSE: robustness of the dot-com validation to its two weakest inputs, per the
# post-pass external review. Three checks, bars fixed before the run:
#   (1) LEAVE-OUT-GOLD: rebuild the proxy sleeve without GC=F (weakest *coverage* — joins
#       2000-08 mid-extension; overlap corr 0.89).
#   (2) LEAVE-OUT-WTI: rebuild without the FRED WTI spot column (weakest *validation* —
#       overlap corr 0.41 vs DBC, contaminated by Apr-2020; the reviewer suggested only
#       the gold test; WTI is the honest weakest link).
#   (3) NAV-SMOOTHING BOUND: mutual-fund NAVs smooth returns and inflate Sharpe. Apply
#       Geltner AR(1) unsmoothing r_u(t) = (r(t) - phi*r(t-1)) / (1 - phi) with phi =
#       lag-1 autocorrelation, to the BLEND's daily returns over the extension window.
#       Conservative by construction: it strips ALL observed autocorrelation, including
#       any genuine trend persistence, so it is a LOWER bound on the true Sharpe. SPY is
#       an exchange-traded fund with real price discovery and is left raw.
#
# BARS (validation-of-a-validation; live books change on no outcome):
#   (a) no-gold blend:  Sharpe > SPY's AND dot-com-bear maxDD shallower than SPY's.
#   (b) no-WTI blend:   same two conditions.
#   (c) unsmoothed blend Sharpe (extension window) still > SPY's raw Sharpe.
#   All three hold -> the dot-com PASS is robust to its weakest proxies and to NAV
#   smoothing; note appended to CURRENT_STATE. Any fail -> the dot-com note gets
#   qualified with the specific dependence found.
import sys, io, warnings
sys.path.insert(0, "/Users/erik.petersson/SystematicEquityFactorScreener")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest.trend_sleeve import VolTargetTSMOM, ENSEMBLE_LOOKS
from backtest.engine_xs import run_xs
from backtest.timing_luck import blend_curve, tranched_curve
from backtest import costs, metrics

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
EXT_END, REPORT_START = "2006-06-30", "1999-07-01"
BEAR = ("2000-03-24", "2002-10-09")


def fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     timeout=30, headers=UA)
    return pd.read_csv(io.StringIO(r.text), parse_dates=[0], index_col=0,
                       na_values=".").iloc[:, 0].dropna()


yfd = yf.download("SPY FDIVX VUSTX VFITX GC=F", start="1996-01-01",
                  progress=False, auto_adjust=True)["Close"]
wti = fred("DCOILWTICO")
base = yfd.join(wti.rename("WTI"), how="left")
base = base[base.index <= EXT_END].ffill(limit=3)
spy_px = base["SPY"].dropna()


def sleeve_blend(cols):
    ext = base[cols]
    panels = {"Close": ext, "Open": ext}
    curves = {off: run_xs(panels, VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off),
                          cost=costs.proportional(5), fill="next_open") for off in range(21)}
    return blend_curve(tranched_curve(curves, tuple(range(21))), spy_px)


def stats(eq, a=None, b=None):
    seg = eq
    if a: seg = seg[seg.index >= a]
    if b: seg = seg[seg.index <= b]
    return metrics.sharpe(seg), metrics.max_drawdown(seg)


ALL = ["SPY", "FDIVX", "VUSTX", "VFITX", "GC=F", "WTI"]
variants = {"full (committed)": ALL,
            "no-gold": [c for c in ALL if c != "GC=F"],
            "no-WTI": [c for c in ALL if c != "WTI"]}

spy_r = spy_px[spy_px.index >= REPORT_START]
spy_sr, _ = stats(spy_r)
_, spy_bear_dd = stats(spy_r, *BEAR)
print(f"SPY reference: ext Sharpe {spy_sr:.2f}, bear maxDD {spy_bear_dd*100:.1f}%\n")

results, blends = {}, {}
for name, cols in variants.items():
    b = sleeve_blend(cols)
    b = b[b.index >= REPORT_START]
    blends[name] = b
    sr, _ = stats(b)
    _, bear_dd = stats(b, *BEAR)
    ok = sr > spy_sr and bear_dd > spy_bear_dd
    results[name] = ok
    print(f"{name:18s} blend Sharpe {sr:.2f}  bear maxDD {bear_dd*100:6.1f}%  "
          f"{'PASS' if ok else 'FAIL'}")

# (3) Geltner unsmoothing on the full-variant blend returns
r = blends["full (committed)"].pct_change().dropna()
phi = float(r.autocorr(1))
ru = (r - phi * r.shift(1)) / (1 - phi)
ru = ru.dropna()
sr_u = ru.mean() / ru.std() * np.sqrt(252)
print(f"\nNAV-smoothing bound: blend lag-1 autocorr {phi:+.3f}")
for col in ["FDIVX", "VUSTX", "VFITX"]:
    pr = base[col].pct_change().dropna()
    pr = pr[pr.index >= REPORT_START]
    print(f"  {col} proxy lag-1 autocorr {pr.autocorr(1):+.3f}  (smoothing evidence)")
print(f"  blend Sharpe raw {stats(blends['full (committed)'])[0]:.2f} -> "
      f"unsmoothed {sr_u:.2f}  vs SPY {spy_sr:.2f}  "
      f"{'PASS' if sr_u > spy_sr else 'FAIL'}")

print(f"\nVERDICT: bars (a) {'PASS' if results['no-gold'] else 'FAIL'}  "
      f"(b) {'PASS' if results['no-WTI'] else 'FAIL'}  "
      f"(c) {'PASS' if sr_u > spy_sr else 'FAIL'}")

# RESULTS (run 2026-07-04, unmodified from pre-registration commit 209ffa8):
#   SPY reference: ext Sharpe 0.11, bear maxDD -47.5%
#   full (committed)  blend Sharpe 1.04  bear maxDD  -7.6%  PASS
#   no-gold           blend Sharpe 0.94  bear maxDD  -8.6%  PASS
#   no-WTI            blend Sharpe 0.99  bear maxDD  -7.2%  PASS
#   NAV smoothing: blend lag-1 autocorr +0.063; proxy autocorrs FDIVX +0.178 (real
#     smoothing, small slice), VUSTX +0.005 / VFITX +0.011 (treasuries mark liquid
#     paper — essentially unsmoothed). Geltner bound: 1.04 -> 0.96, far above SPY 0.11.
#     The reviewer's guessed 0.2-0.3 Sharpe haircut was ~3x too big; measured ~0.08.
#   VERDICT: (a) PASS (b) PASS (c) PASS — the dot-com verdict survives its weakest
#   proxy, its weakest coverage, and a conservative unsmoothing bound.
#   Honesty note: this rebuild shows full-variant Sharpe 1.04 vs the original run's
#   0.97 — ~±0.07 construction sensitivity from panel row-grid details (which calendar
#   days survive the join). Reinforces the standing caveat: the SIGN is the finding,
#   the second decimal is noise.
