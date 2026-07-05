# PRE-REGISTERED (2026-07-04, committed BEFORE running — see git history: this header's
# commit precedes the results commit; first use of the two-commit proof).
#
# HYPOTHESIS UNDER TEST (DeepSeek review's one genuinely new test; the "your window
# starts in 2006" critique has now come from three independent reviewers): the blend's
# claim "beats SPY over long periods" extends to the 2000-2002 dot-com bear — a slow-grind
# equity crash NOT in the ETF-era panel. Trend lore says 2000-02 was a GOOD trend regime
# (bonds rallied on Fed cuts, gold bottomed 2001, oil trended); the published century
# evidence (Hurst-Ooi-Pedersen, Lemperiere) says trend cleared it. This tests OUR exact
# construction (1/3/12 ensemble, vol-target 10%, monthly, all-21-offset tranche) on proxy
# data, not the literature's.
#
# PROXY PANEL (free-data constraints; each validated against its live ETF on the 2006+
# overlap, reported below):
#   SPY   -> SPY (real, 1993+)             EFA -> FDIVX (Fidelity Diversified Intl, 1991+)
#   TLT   -> VUSTX (Vanguard LT Treasury)  IEF -> VFITX (Vanguard IT Treasury)
#   GLD   -> GC=F (COMEX gold cont., 2000-08+ — JOINS LATE; conservative: gold's 2001-02
#            uptrend is partially missed)  DBC -> FRED DCOILWTICO (WTI SPOT — energy-only,
#            no roll/collateral return; DBC is ~55% energy)
# Mutual-fund NAVs are total-return (yfinance auto_adjust). Open=Close for all (NAV/spot
# have no real opens): next_open fills become next-CLOSE fills — one day later than the
# live convention, no look-ahead. This tests the strategy's REGIME BEHAVIOR, not retail
# implementability in 2000 (mutual-fund frictions of that era are out of scope).
#
# PRE-REGISTERED BAR (a VALIDATION of a claim, not an adoption test — live books do not
# change on any outcome):
#   (a) blend (risk-parity SPY + proxy sleeve, same construction as the headline) posts a
#       HIGHER Sharpe than SPY over the full extension window 1999-07-01 -> 2006-06-30, AND
#   (b) blend's max drawdown over the dot-com bear (2000-03-24 -> 2002-10-09) is SHALLOWER
#       than SPY's.
#   PASS both -> the unqualified claim stands, extension noted in CURRENT_STATE.
#   FAIL either -> the claim gets qualified to "since 2007" in CURRENT_STATE + the wiki.
# SECONDARY (descriptive, no bar): simulated ssoB (33% UPRO-sim via letf_returns w/ real
# ^IRX financing + 67% proxy sleeve, monthly rebalance) cumulative return vs SPY over the
# extension — recorded either way as the "what would Erik's book have done" number.
# LEDGER: recorded as an out-of-sample VALIDATION, not a candidate trial (no selection
# happens here; the bar was committed before the run).
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
from backtest.leverage_study import letf_returns, ER3
from backtest import costs, metrics

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
EXT_END = "2006-06-30"
REPORT_START = "1999-07-01"
BEAR = ("2000-03-24", "2002-10-09")


def fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
                     timeout=30, headers=UA)
    s = pd.read_csv(io.StringIO(r.text), parse_dates=[0], index_col=0, na_values=".").iloc[:, 0]
    return s.dropna()


print("=" * 74)
print("fetching proxies + validation ETFs ...")
yfd = yf.download("SPY FDIVX VUSTX VFITX GC=F TLT IEF EFA GLD DBC ^IRX",
                  start="1996-01-01", progress=False, auto_adjust=True)["Close"]
wti = fred("DCOILWTICO")

PROXY_OF = {"FDIVX": "EFA", "VUSTX": "TLT", "VFITX": "IEF", "GC=F": "GLD", "WTI": "DBC"}
print("\nproxy validation (daily-return corr + ann-vol ratio, 2006-07 -> present overlap):")
wide = yfd.join(wti.rename("WTI"), how="outer")
for pxy, real in PROXY_OF.items():
    both = wide[[pxy, real]].dropna()
    both = both[both.index >= "2006-07-01"]
    r2 = both.pct_change().dropna()
    corr = r2.corr().iloc[0, 1]
    vr = r2[pxy].std() / r2[real].std()
    print(f"  {pxy:6s} -> {real:4s}  corr {corr:5.2f}   vol ratio {vr:4.2f}   n={len(r2)}")

# extension panel (proxy names; strategy is name-agnostic)
ext = wide[["SPY", "FDIVX", "VUSTX", "VFITX", "GC=F"]].join(wti.rename("WTI"), how="left")
ext = ext[ext.index <= EXT_END].ffill(limit=3)
panels = {"Close": ext, "Open": ext}
spy_px = ext["SPY"].dropna()

print(f"\nextension panel {ext.index[0].date()} -> {ext.index[-1].date()}; "
      f"gold joins {ext['GC=F'].first_valid_index().date()}")

curves = {}
for off in range(21):
    strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=off)
    curves[off] = run_xs(panels, strat, cost=costs.proportional(5), fill="next_open")
sleeve_eq = tranched_curve(curves, tuple(range(21)))
blend_eq = blend_curve(sleeve_eq, spy_px)

# secondary: simulated ssoB on the extension (UPRO sim financed at real ^IRX + spread)
rf = (yfd["^IRX"] / 100.0).reindex(spy_px.index).ffill()
spy_ret = spy_px.pct_change().dropna()
upro_ret = letf_returns(spy_ret, 3, ER3, rf.reindex(spy_ret.index))
sl_ret = sleeve_eq.pct_change().reindex(spy_ret.index).fillna(0.0)
month = spy_ret.index.to_period("M")
eq_v, sl_v, vals = 1 / 3, 2 / 3, []
for i in range(len(spy_ret)):
    if i and month[i] != month[i - 1]:
        tot = eq_v + sl_v
        cost = (abs(tot / 3 - eq_v) + abs(2 * tot / 3 - sl_v)) * 10 / 1e4
        tot -= cost
        eq_v, sl_v = tot / 3, 2 * tot / 3
    eq_v *= 1 + upro_ret.iloc[i]
    sl_v *= 1 + sl_ret.iloc[i]
    vals.append(eq_v + sl_v)
ssob_eq = pd.Series(vals, index=spy_ret.index)


def stats(eq, a=None, b=None):
    seg = eq
    if a: seg = seg[seg.index >= a]
    if b: seg = seg[seg.index <= b]
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    return ((seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1,
            metrics.sharpe(seg), metrics.max_drawdown(seg))


print(f"\nRESULTS ({REPORT_START} -> {EXT_END}; bear window {BEAR[0]} -> {BEAR[1]}):")
rows = {}
for name, eq in (("SPY", spy_px), ("sleeve (proxy)", sleeve_eq),
                 ("blend (proxy)", blend_eq), ("ssoB-sim (proxy)", ssob_eq)):
    eq = eq[eq.index >= REPORT_START]
    c, s, d = stats(eq)
    cb, sb, db = stats(eq, *BEAR)
    rows[name] = (c, s, d, cb, db)
    print(f"  {name:16s} FULL-EXT: CAGR {c*100:6.2f}%  Sharpe {s:5.2f}  maxDD {d*100:6.1f}%"
          f"   | BEAR: CAGR {cb*100:7.2f}%  maxDD {db*100:6.1f}%")

print("\nVERDICT vs pre-registered bar:")
a_pass = rows["blend (proxy)"][1] > rows["SPY"][1]
b_pass = rows["blend (proxy)"][4] > rows["SPY"][4]          # maxDD is negative: greater = shallower
print(f"  (a) blend Sharpe {rows['blend (proxy)'][1]:.2f} > SPY {rows['SPY'][1]:.2f} over extension: "
      f"{'PASS' if a_pass else 'FAIL'}")
print(f"  (b) blend bear maxDD {rows['blend (proxy)'][4]*100:.1f}% shallower than SPY "
      f"{rows['SPY'][4]*100:.1f}%: {'PASS' if b_pass else 'FAIL'}")
print(f"  => claim {'STANDS unqualified' if a_pass and b_pass else 'gets QUALIFIED to since-2007'}")
print(f"  secondary: ssoB-sim ${ssob_eq.iloc[-1]/ssob_eq[ssob_eq.index >= REPORT_START].iloc[0]*10:,.2f}k "
      f"vs SPY ${spy_px.iloc[-1]/spy_px[spy_px.index >= REPORT_START].iloc[0]*10:,.2f}k per $10k over extension")

# RESULTS (run 2026-07-04, unmodified from the pre-registration commit 2f2bf84):
#   proxy validation (2006+ overlap): FDIVX~EFA 0.97, VUSTX~TLT 0.98, VFITX~IEF 0.95,
#     GC=F~GLD 0.89, WTI~DBC 0.41 (WTI overlap stat contaminated by Apr-2020 negative
#     prices; inverse-vol sizing inside the sleeve contains it by construction)
#   1999-07 -> 2006-06:            CAGR    Sharpe   maxDD   | bear CAGR   bear maxDD
#     SPY                           0.34%   0.11    -47.5%  |  -22.4%      -47.5%
#     sleeve (proxy)                9.51%   1.39     -9.7%  |   +7.9%       -5.5%
#     blend (proxy)                 7.27%   0.97     -9.4%  |   -1.0%       -9.4%
#     ssoB-sim (proxy)              2.53%   0.23    -47.2%  |  -22.2%      -47.2%
#   VERDICT: (a) PASS, (b) PASS -> the beat-SPY claim STANDS unqualified; it now spans
#   the dot-com bear as well as 2008/2020/2022. ssoB-sim beat SPY over the extension
#   ($11.9k vs $10.2k per $10k) while eating the full SPY-shaped -47% — the pile thesis
#   exactly as designed. Level caveat: mutual-fund NAV smoothing flatters proxy Sharpes
#   (vol ratios 0.70-0.91); the SIGN of the verdict is the finding, not the 1.39.
