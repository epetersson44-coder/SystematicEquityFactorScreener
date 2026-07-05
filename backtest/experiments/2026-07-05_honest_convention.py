# PRE-REGISTERED (2026-07-05, committed BEFORE running — two-commit proof).
#
# F2 FROM THE SIXTH EXTERNAL REVIEW — the most serious finding of the gauntlet, and it
# quotes our own standard back at us: backtest/SCOREBOARD.md line 6 declared, in Phase 1,
# "Two rows per strategy — the naive rf=0/cash 0% and the honest rf=4%/cash 4% ... The
# honest one is the truth," after the rf=0 convention produced a WRONG verdict on the SMA
# study (idle cash scored at 0% flipped 'wins on return' to 'loses'). The headline claim
# (blend Sharpe 0.94 vs SPY 0.64, DSR 0.93, bootstrap p=1.6%) runs entirely on the naive
# side: cash_rate=0 curves, rf=0 Sharpes. The two corrections fight each other (cash
# credit raises the cash-heavy blend's return; rf in the numerator penalizes the
# lower-vol blend's Sharpe MORE than SPY's), so the net direction is genuinely unknown
# until run.
#
# ALSO FOLDED IN (same review):
#   F4: the IMPLEMENTABLE book — a live lock is ONE offset (not the 21-tranche) with an
#       EXPANDING-window risk-parity weight (what blend_picks actually computes) and
#       monthly mix costs. The honest live expectation is that distribution's median,
#       not the tranche's 0.94.
#   F1: ssoB (the real-money book) gets its own significance page: honest excess Sharpe,
#       tracking error vs SPY, paired block bootstrap, and a POWER ANALYSIS of the
#       pre-committed 3-5yr pile metric (years needed to resolve the edge at t=1, t=2).
#   F7b: bootstrap block-length sensitivity 21/63/126d (crisis clustering > 1 month).
#
# CONVENTION UNDER TEST ("honest"): engine credits idle cash at the REAL ^IRX series
# (run_xs cash_rate now accepts a Series); all Sharpes are EXCESS-return Sharpes
# (daily return minus same-day ^IRX/252). SPY gets the identical treatment.
#
# PRE-REGISTERED RE-HEADLINE RULE (the bar, fixed before the numbers):
#   If the honest blend-vs-SPY EXCESS-Sharpe gap (tranche construction, apples-to-apples
#   with the old headline) falls below +0.15, OR the 21d-block bootstrap p(luck) exceeds
#   0.10 -> the 0.94-vs-0.64 headline is RETRACTED and re-stated under the honest
#   convention in CURRENT_STATE, picks.md, and the wiki. Otherwise the headline gains a
#   measured honest-convention row next to it. Either way, the implementable-book median
#   becomes the number displayed beside the live book.
# EXPECTATION ON RECORD: the gap survives at roughly +0.20 to +0.30 with p < 5%; the
# implementable median lands ~0.05-0.10 below the tranche; ssoB's pile metric is
# confirmed statistically undecidable inside 5 years (expected ~25+ years for t=1) —
# which is WHY the powered falsifier is the blend tripwire, not ssoB's raw-return gap.
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from backtest.trend_sleeve import etf_panel
from backtest.timing_luck import sweep, tranched_curve, blend_curve
from backtest.leverage_study import tbill_series, letf_returns, ER3
from backtest.significance import block_bootstrap_sharpe_diff, dsr, TRIAL_SHARPES
from backtest import metrics

panels = etf_panel()
spy_px = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy_px.index)                     # annualized decimals, daily index
rf_d = (rf / 252.0)

print("sweeping 21 offsets under HONEST convention (cash_rate = ^IRX series)...")
_, curves_h = sweep(panels=panels, cash_rate=rf)
sleeve_h = tranched_curve(curves_h, tuple(range(21)))


def ex_sharpe(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).ffill().fillna(0.0)
    return float(ex.mean() / ex.std() * math.sqrt(252))


def ex_rets(eq):
    r = eq.pct_change().dropna()
    return r - rf_d.reindex(r.index).ffill().fillna(0.0)


def raw_sharpe(eq):
    return metrics.sharpe(eq)


# ---- headline apples-to-apples: tranche + full-sample RP, honest cash + excess Sharpe
blend_h = blend_curve(sleeve_h, spy_px)
common = blend_h.dropna().index
spy_c = spy_px.reindex(common)

print("\n== HEADLINE CONVENTION COMPARISON (tranche + full-sample RP) ==")
print(f"{'':28s}{'raw rf=0':>10s}{'honest excess':>15s}")
print(f"{'blend Sharpe':28s}{raw_sharpe(blend_h):>10.3f}{ex_sharpe(blend_h):>15.3f}")
print(f"{'SPY Sharpe':28s}{raw_sharpe(spy_c):>10.3f}{ex_sharpe(spy_c):>15.3f}")
gap_h = ex_sharpe(blend_h) - ex_sharpe(spy_c)
print(f"{'gap':28s}{raw_sharpe(blend_h)-raw_sharpe(spy_c):>10.3f}{gap_h:>15.3f}")

# bootstrap on paired EXCESS returns, block sensitivity
print("\nblock bootstrap of EXCESS-Sharpe diff (blend - SPY):")
bb21 = None
for blk in (21, 63, 126):
    bb = block_bootstrap_sharpe_diff(ex_rets(blend_h), ex_rets(spy_c), block=blk)
    if blk == 21:
        bb21 = bb
    print(f"  block {blk:3d}d: diff {bb['observed_diff']:+.3f}  "
          f"CI [{bb['ci95'][0]:+.3f},{bb['ci95'][1]:+.3f}]  p(luck) {bb['p_value_luck']:.4f}")

# DSR under mixed convention (conservative: hurdle from the rf=0 ledger)
d_val, hurdle = dsr(ex_rets(blend_h), len(TRIAL_SHARPES), trial_sharpes_annual=TRIAL_SHARPES)
print(f"\nDSR of honest-excess blend vs rf=0-ledger hurdle (CONSERVATIVE — hurdle "
      f"overstated): {d_val:.4f} (hurdle {hurdle:.2f})")

# ---- F4: the IMPLEMENTABLE book — single offset, expanding RP, monthly mix @10bps
def implementable_blend(sleeve_eq, cost_bps=10, min_obs=252):
    df = pd.DataFrame({"s": spy_c.pct_change(), "t": sleeve_eq.reindex(common).pct_change()}).dropna()
    es, et = df["s"].expanding(min_obs).std(), df["t"].expanding(min_obs).std()
    w = (1 / es) / (1 / es + 1 / et)                 # weight on SPY, point-in-time
    month = df.index.to_period("M")
    sv, tv, vals = 0.5, 0.5, []
    for i in range(len(df)):
        if i and month[i] != month[i - 1]:
            wi = w.iloc[i - 1]
            if np.isfinite(wi):
                tot = sv + tv
                cost = (abs(tot * wi - sv) + abs(tot * (1 - wi) - tv)) * cost_bps / 1e4
                tot -= cost
                sv, tv = tot * wi, tot * (1 - wi)
        sv *= 1 + df["s"].iloc[i]
        tv *= 1 + df["t"].iloc[i]
        vals.append(sv + tv)
    return pd.Series(vals, index=df.index)


print("\n== IMPLEMENTABLE BOOK (per-offset, expanding RP, monthly mix @10bps, honest) ==")
impl = [ex_sharpe(implementable_blend(curves_h[o])) for o in range(21)]
impl = pd.Series(impl)
print(f"honest excess Sharpe across 21 offsets: median {impl.median():.3f}  "
      f"[{impl.min():.3f}, {impl.max():.3f}]   (vs SPY {ex_sharpe(spy_c):.3f})")

# ---- F1: ssoB's own significance page (honest convention)
spy_ret = spy_c.pct_change().dropna()
upro = letf_returns(spy_ret, 3, ER3, rf.reindex(spy_ret.index))
sl_ret = sleeve_h.pct_change().reindex(spy_ret.index).fillna(0.0)
month = spy_ret.index.to_period("M")
ev, sv2, vals = 1 / 3, 2 / 3, []
for i in range(len(spy_ret)):
    if i and month[i] != month[i - 1]:
        tot = ev + sv2
        cost = (abs(tot / 3 - ev) + abs(2 * tot / 3 - sv2)) * 10 / 1e4
        tot -= cost
        ev, sv2 = tot / 3, 2 * tot / 3
    ev *= 1 + upro.iloc[i]
    sv2 *= 1 + sl_ret.iloc[i]
    vals.append(ev + sv2)
ssob_h = pd.Series(vals, index=spy_ret.index)

print("\n== ssoB SIGNIFICANCE PAGE (the real-money book, honest convention) ==")
yrs = (ssob_h.index[-1] - ssob_h.index[0]).days / 365.25
cagr_s = (ssob_h.iloc[-1] / ssob_h.iloc[0]) ** (1 / yrs) - 1
cagr_spy = (spy_c.dropna().iloc[-1] / spy_c.dropna().iloc[0]) ** (1 / yrs) - 1
print(f"ssoB honest excess Sharpe {ex_sharpe(ssob_h):.3f}  vs SPY {ex_sharpe(spy_c):.3f}")
bb_s = block_bootstrap_sharpe_diff(ex_rets(ssob_h), ex_rets(spy_c), block=21)
print(f"bootstrap Sharpe diff {bb_s['observed_diff']:+.3f}  "
      f"CI [{bb_s['ci95'][0]:+.3f},{bb_s['ci95'][1]:+.3f}]  p(luck) {bb_s['p_value_luck']:.4f}")
diff = (ssob_h.pct_change() - spy_c.pct_change()).dropna()
te = float(diff.std() * math.sqrt(252))
edge = cagr_s - cagr_spy
print(f"raw edge {edge*100:+.2f}%/yr   tracking error {te*100:.2f}%/yr")
if edge > 0:
    print(f"POWER: years for t=1 (weak evidence): {(te/edge)**2:5.1f}   "
          f"t=2 (conventional): {4*(te/edge)**2:5.1f}")
print("=> the pile metric CANNOT be statistically resolved on the 3-5yr horizon; the")
print("   powered falsifier remains the blend tripwire + live mechanism checks.")

# ---- verdict vs the pre-registered rule
print("\n== VERDICT vs pre-registered re-headline rule ==")
fail = gap_h < 0.15 or bb21["p_value_luck"] > 0.10
print(f"honest gap {gap_h:+.3f} (rule: >= +0.15)   p(luck) {bb21['p_value_luck']:.4f} "
      f"(rule: <= 0.10)")
print("RETRACT & RE-STATE HEADLINE" if fail else
      "HEADLINE SURVIVES the honest convention — gains a measured honest row")
