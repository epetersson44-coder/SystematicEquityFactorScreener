# PRE-REGISTERED (2026-07-05, committed BEFORE running — two-commit proof).
#
# HYPOTHESIS (the US-bias falsification, proposed via external review): the live
# constructions are NOT artifacts of the US equity bull. Transplant the exact
# constructions onto non-US equity cores — a US investor whose core is Japan (EWJ)
# or the Eurozone (EZU) instead of the S&P — and the construction should still beat
# ITS OWN core, because the mechanism (full cheap beta + uncorrelated global trend
# sleeve bonus) does not know what the core is. Japan is the deliberate nightmare
# pick: a core that spent 2006-2012 losing ~60% and two decades going nowhere.
#
# DESIGN (no FX fudging, no fabricated data):
#   Cores: EWJ (Japan, USD fund, 1996+), EZU (Eurozone, USD fund, 2000+), SPY (ref).
#   ssoB-transplant:  33% x simulated 3x daily-reset on the core (letf_returns, real
#                     ^IRX + spread financing — CONSERVATIVE for EWJ/EZU since local
#                     rates were below USD) + 67% x the SAME global 6-ETF trend
#                     sleeve (all-21-offset tranche), monthly rebalance, 10bps.
#   blend-transplant: risk-parity core + the same sleeve (blend_curve).
#   2.3xL-transplant: blend-transplant levered 2.3x at rf+spread (DESCRIPTIVE, no bar).
#   PRIMARY window 2006-07 -> 2026 (ETF-era sleeve, the validated machinery).
#
# PRE-REGISTERED BARS (validation of the construction, not an adoption test; live
# books change on NO outcome):
#   For EACH non-US core (EWJ, EZU), over the primary window:
#   (a) ssoB-transplant CUMULATIVE RAW RETURN > its own core's buy-and-hold (the
#       pile thesis, transplanted), AND
#   (b) blend-transplant Sharpe > its own core's buy-and-hold Sharpe (the ride
#       thesis, transplanted).
#   PASS all four -> the construction generalizes; US-bias concern closed, noted in
#   CURRENT_STATE. Any FAIL -> recorded as a real dependence of the construction on
#   the equity core's health — a caveat on the live book's premise, written into
#   CURRENT_STATE exactly as found.
# EXPECTATION ON RECORD (before the run): all four PASS, driven by the sleeve; the
# interesting open number is ssoB-on-EWJ's ABSOLUTE CAGR — 3x daily-reset decay on
# a sideways core is brutal, and beating a dead core is not the same as being good.
# LEDGER: OOS validation of frozen constructions — no trial entries.
import sys, warnings
sys.path.insert(0, "/Users/erik.petersson/SystematicEquityFactorScreener")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.trend_sleeve import etf_panel
from backtest.timing_luck import sweep, tranched_curve, blend_curve
from backtest.leverage_study import tbill_series, letf_returns, levered_returns, ER3
from backtest import metrics

panels = etf_panel()
_, curves = sweep(panels=panels)
sleeve = tranched_curve(curves, tuple(range(21)))
sl_ret_full = sleeve.pct_change().dropna()

cores_px = yf.download("SPY EWJ EZU", start="1996-01-01", progress=False,
                       auto_adjust=True)["Close"]
rf = tbill_series(cores_px.index)


def ssob_mix(core_ret, sl_ret, rf_ser, w_eq=1 / 3, cost_bps=10):
    df = pd.DataFrame({"c": core_ret, "s": sl_ret}).dropna()
    lev3 = letf_returns(df["c"], 3, ER3, rf_ser.reindex(df.index))
    month = df.index.to_period("M")
    eq_v, sl_v, vals = w_eq, 1 - w_eq, []
    for i in range(len(df)):
        if i and month[i] != month[i - 1]:
            tot = eq_v + sl_v
            cost = (abs(tot * w_eq - eq_v) + abs(tot * (1 - w_eq) - sl_v)) * cost_bps / 1e4
            tot -= cost
            eq_v, sl_v = tot * w_eq, tot * (1 - w_eq)
        eq_v *= 1 + lev3.iloc[i]
        sl_v *= 1 + df["s"].iloc[i]
        vals.append(eq_v + sl_v)
    return pd.Series(vals, index=df.index)


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, metrics.sharpe(eq),
            metrics.max_drawdown(eq), eq.iloc[-1] / eq.iloc[0] * 10000)


START = "2006-07-01"
print(f"PRIMARY window {START} -> {cores_px.index[-1].date()}   (same global sleeve for all cores)")
print(f"{'':22s}{'CAGR':>8s}{'Sharpe':>8s}{'maxDD':>8s}{'$10k ->':>12s}")
verdicts = {}
for core in ["SPY", "EWJ", "EZU"]:
    px = cores_px[core].dropna()
    px = px[px.index >= START]
    ret = px.pct_change().dropna()
    rows = {}
    rows[f"{core} buy-and-hold"] = px / px.iloc[0]
    rows[f"ssoB-{core}"] = ssob_mix(ret, sl_ret_full, rf)
    b = blend_curve(sleeve.reindex(px.index).dropna(), px)
    rows[f"blend-{core}"] = b
    b_ret = b.pct_change().dropna()
    rows[f"2.3xL-{core}"] = (1 + levered_returns(b_ret, 2.3, rf.reindex(b_ret.index))).cumprod()
    res = {}
    for name, eq in rows.items():
        c, s, d, term = stats(eq.dropna())
        res[name] = (c, s, d, term)
        print(f"{name:22s}{c*100:7.2f}%{s:8.2f}{d*100:7.1f}%{term:>12,.0f}")
    if core != "SPY":
        a_pass = res[f"ssoB-{core}"][3] > res[f"{core} buy-and-hold"][3]
        b_pass = res[f"blend-{core}"][1] > res[f"{core} buy-and-hold"][1]
        verdicts[core] = (a_pass, b_pass)
        print(f"  -> bars: (a) ssoB beats core pile: {'PASS' if a_pass else 'FAIL'}   "
              f"(b) blend beats core Sharpe: {'PASS' if b_pass else 'FAIL'}")
    print()

allp = all(v for pair in verdicts.values() for v in pair)
print(f"VERDICT: {'ALL FOUR BARS PASS — construction generalizes; US-bias concern closed'
          if allp else 'AT LEAST ONE FAIL — record the dependence in CURRENT_STATE'}")
