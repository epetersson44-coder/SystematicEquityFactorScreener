# PRE-REGISTERED (2026-07-04) — Codex-proposed ssoB variants, tested against the book's bar.
#
# Codex proposed: (1) DEFENSIVE: equity leg UPRO -> SPY when SPY < 200d SMA, sleeve
# unchanged; (2) GROWTH: 33% UPRO + 1.25x-levered sleeve. (2) is ruled out at Erik's scale
# without a run: a cash account cannot exceed 100% of capital (0.33 + 0.67*1.25 = 1.17),
# and the levered-wrapper route was already measured at ~2.2%/yr drag (capital-ladder
# study) > the claimed +0.8%/yr gain. Only (1) is testable and is a NEW twist on the
# rejected C/D Gayed-filter variants (those de-risked fully; this steps 100% -> 33%
# notional, keeping 33% SPY).
#
# ADOPTION BAR (fixed before running; ssoB is the PILE book, thesis = beat SPY's RAW
# return everywhere incl. crisis-free bulls):
#   (a) defensive variant must beat SPY's CAGR in BOTH bull windows (2011-19, 2023-25), AND
#   (b) beat baseline ssoB's full-cycle terminal wealth.
# Sharpe/maxDD improvements alone do NOT displace the live book — the ride-optimized
# construction already exists (blend). If it fails the bar but shows a materially better
# ride, it gets LEDGERED + noted as a ride-vs-pile preference option (Erik's call, not a
# backtest call).
#
# Cadences: MONTHLY regime check (what the live lock cadence can actually implement) is
# primary; DAILY check (Gayed-style, what Codex likely ran) reported for reference.
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.trend_sleeve import etf_panel
from backtest.timing_luck import sweep, tranched_curve
from backtest.leverage_study import tbill_series, letf_returns, ER3
from backtest import metrics

COST_BPS = 10  # on traded dollars at rebalance/switch, matching tracker._simulate

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
spy_ret = spy.pct_change().dropna()
rf = tbill_series(spy.index)

_, curves = sweep(panels=panels)                      # ensemble sleeve, 21 offsets, net 5bps
sleeve_eq = tranched_curve(curves, tuple(range(21)))
sleeve_ret = sleeve_eq.pct_change().dropna()

upro_ret = letf_returns(spy_ret, 3, ER3, rf.reindex(spy_ret.index))

df = pd.DataFrame({"spy": spy_ret, "upro": upro_ret, "sleeve": sleeve_ret}).dropna()
sma = spy.rolling(200).mean()
risk_on = (spy >= sma).reindex(df.index).ffill().astype(bool)
month = df.index.to_period("M")
rebal = pd.Series(True, index=df.index)
rebal.iloc[1:] = month[1:] != month[:-1]              # first trading day of each month
rebal.iloc[0] = True


def run_mix(regime_daily=None, regime_monthly=None, w_eq=1 / 3):
    """33/67 EQ/sleeve, monthly rebalance. EQ instrument = UPRO when risk-on else SPY,
    per the given regime series (None -> always UPRO). Regime uses PRIOR day's close vs
    SMA (no same-day peek). Returns the equity curve."""
    eq_v, sl_v = w_eq, 1 - w_eq
    inst = "upro"
    vals = []
    reg = regime_daily if regime_daily is not None else regime_monthly
    for i, d in enumerate(df.index):
        if i > 0:                                     # decide/trade at today's open, using yesterday's signal
            want = inst
            if reg is not None:
                sig = bool(reg.iloc[i - 1])
                check = regime_daily is not None or rebal.iloc[i]
                if check:
                    want = "upro" if sig else "spy"
            total = eq_v + sl_v
            traded = 0.0
            if want != inst:                          # swap the whole EQ slice
                traded += 2 * eq_v
                inst = want
            if rebal.iloc[i]:
                t_eq, t_sl = w_eq * total, (1 - w_eq) * total
                traded += abs(t_eq - eq_v) + abs(t_sl - sl_v)
                eq_v, sl_v = t_eq, t_sl
            cost = traded * COST_BPS / 1e4
            eq_v -= cost * (eq_v / total)
            sl_v -= cost * (sl_v / total)
        eq_v *= 1 + df[inst].iloc[i]
        sl_v *= 1 + df["sleeve"].iloc[i]
        vals.append(eq_v + sl_v)
    return pd.Series(vals, index=df.index)


curves_out = {
    "SPY": (1 + df["spy"]).cumprod(),
    "ssoB (live)": run_mix(),
    "def-MONTHLY": run_mix(regime_monthly=risk_on),
    "def-DAILY": run_mix(regime_daily=risk_on),
}

WINDOWS = [("FULL", None, None),
           ("GFC 07-09", "2007-10-09", "2009-03-09"),
           ("BULL 2011-19", "2011-01-01", "2019-12-31"),
           ("COVID 2020", "2020-02-19", "2020-12-31"),
           ("BEAR 2022", "2022-01-01", "2022-12-31"),
           ("BULL 2023-25", "2023-01-01", "2025-12-31")]

print(f"common window {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days), costs {COST_BPS}bps")
for name, eq in curves_out.items():
    n_sw = ""
    print(f"\n{name}:  full Sharpe {metrics.sharpe(eq):.3f}  maxDD {metrics.max_drawdown(eq)*100:.1f}%  $10k -> ${eq.iloc[-1]/eq.iloc[0]*10000:,.0f}")
    for wname, a, b in WINDOWS:
        seg = eq
        if a: seg = seg[seg.index >= a]
        if b: seg = seg[seg.index <= b]
        yrs = (seg.index[-1] - seg.index[0]).days / 365.25
        cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
        print(f"   {wname:14s} CAGR {cagr*100:6.2f}%  maxDD {metrics.max_drawdown(seg)*100:6.1f}%")

# switch counts (churn check)
for label, monthly in (("monthly-check", True), ("daily-check", False)):
    sig = risk_on.shift(1).dropna()
    if monthly:
        sig = sig[rebal.reindex(sig.index).fillna(False)]
    print(f"\n{label}: {int((sig != sig.shift(1)).sum()) - 1} instrument switches over {sig.index[0].year}-{sig.index[-1].year}")
