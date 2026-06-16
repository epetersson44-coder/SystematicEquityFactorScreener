# backtest/factor_analysis.py — does each factor actually work, and are they redundant?
#
# The quant's two core questions about a multi-factor model, answered honestly on the
# point-in-time, survivorship-free history (same data as factor_backtest):
#
#   1. EFFICACY (Information Coefficient) — for each factor, rank the eligible names by
#      its GOOD direction at each rebalance, then rank-correlate that with the forward
#      return. Positive mean IC => the factor predicts. ~0 => dead. Negative => it's
#      hurting you. IR = mean(IC)/std(IC); t = IR*sqrt(periods).
#   2. REDUNDANCY — the cross-sectional rank-correlation BETWEEN factor signals. Two
#      factors at 0.7+ are the same bet double-counted; "5 factors" may be ~3 real ones.
#
# HONEST CAVEAT: ~14 quarterly periods is a tiny sample — ICs are noisy and t-stats won't
# clear 2. This finds DIRECTION and REDUNDANCY, not statistical certainty.
#
# Run:  python -m backtest.factor_analysis

import numpy as np
import pandas as pd

from fundamentals import get_fundamentals_asof
from factors import calculate_factors
from screen import EXCLUDE_SECTORS, MIN_CAP, MAX_CAP
from backtest.factor_backtest import daily_panel
from config import WEIGHTS

FACTORS = list(WEIGHTS.keys())                       # ev_ebit, price_fcf, roic, gm_stability, net_debt_ebitda
LOWER_BETTER = ["ev_ebit", "price_fcf", "gm_stability", "net_debt_ebitda"]


def collect(start="2021-07-01", end="2025-03-01", freq="QS"):
    """Per eligible name per rebalance: the 5 factor values + the forward return to the
    next rebalance (survivorship-honest — a name that delists is marked at its last price)."""
    close = daily_panel()["Close"]
    dates = close.index
    universe = list(close.columns)
    rb = []
    for tgt in pd.date_range(start, end, freq=freq):
        if tgt > dates[-1]:
            break
        rb.append(dates[dates.searchsorted(tgt)])

    rows = []
    for k in range(len(rb) - 1):                     # need a forward window
        d, d1 = rb[k], rb[k + 1]
        crow, c1 = close.loc[d], close.loc[d1]
        seg = close[(close.index > d) & (close.index <= d1)]
        for t in universe:
            p0 = crow.get(t)
            if p0 is None or not np.isfinite(p0) or p0 <= 0:
                continue
            f = get_fundamentals_asof(t, d, price=float(p0))
            mc = f.get("market_cap")
            if mc is None or not (MIN_CAP <= mc <= MAX_CAP):
                continue
            if f.get("sector") in EXCLUDE_SECTORS:
                continue
            p1 = c1.get(t)
            if p1 is None or not np.isfinite(p1):     # delisted mid-quarter -> last trade
                s = seg[t].dropna()
                p1 = float(s.iloc[-1]) if len(s) else np.nan
            if not np.isfinite(p1) or p1 <= 0:
                continue
            rec = calculate_factors(f)
            rec["date"] = d
            rec["fwd"] = float(p1) / float(p0) - 1
            rows.append(rec)
    return pd.DataFrame(rows)


def signal(df):
    """Good-direction percentile per factor within each rebalance date (so high = good,
    matching the composite), plus the forward-return rank. Spearman = Pearson on ranks,
    and these columns ARE ranks — so downstream we use plain Pearson (no scipy needed)."""
    out = pd.DataFrame(index=df.index)
    for f in FACTORS:
        asc = f not in LOWER_BETTER                  # higher-is-better ranks ascending
        out[f] = df.groupby("date")[f].rank(ascending=asc, pct=True)
    out["fwd_rank"] = df.groupby("date")["fwd"].rank(pct=True)
    out["date"], out["fwd"] = df["date"], df["fwd"]
    return out


def ic_table(sig):
    res = {}
    for f in FACTORS:                                # Pearson(signal_rank, fwd_rank) == Spearman IC
        ics = sig.groupby("date").apply(
            lambda g: g[f].corr(g["fwd_rank"]), include_groups=False).dropna()
        m, s = ics.mean(), ics.std()
        res[f] = {"IC": round(m, 4), "IC_std": round(s, 4),
                  "IR": round(m / s, 3) if s else np.nan,
                  "t_stat": round(m / s * np.sqrt(len(ics)), 2) if s else np.nan,
                  "hit_rate": round((ics > 0).mean(), 2), "periods": len(ics)}
    return pd.DataFrame(res).T


def corr_matrix(sig):                                # Pearson on the rank columns == Spearman
    mats = [g[FACTORS].corr() for _, g in sig.groupby("date")]
    return (sum(mats) / len(mats)).round(2)


def quintile_spread(sig):
    """Top-quintile minus bottom-quintile forward return per factor (avg over dates, %)."""
    res = {}
    for f in FACTORS:
        sp = []
        for _, g in sig.groupby("date"):
            top = g.loc[g[f] >= 0.8, "fwd"].mean()
            bot = g.loc[g[f] <= 0.2, "fwd"].mean()
            if np.isfinite(top) and np.isfinite(bot):
                sp.append(top - bot)
        res[f] = round(np.mean(sp) * 100, 2) if sp else np.nan
    return res


if __name__ == "__main__":
    df = collect()
    sig = signal(df)
    n_obs, n_dates = len(df), df["date"].nunique()
    print(f"\ncollected {n_obs:,} name-quarter observations across {n_dates} rebalances\n")

    ic = ic_table(sig)
    sp = quintile_spread(sig)
    ic["Q5-Q1_%"] = pd.Series(sp)
    print("=== FACTOR EFFICACY (does it predict forward returns?) ===")
    print(ic.to_string())
    print("\n  IC>0 = predicts · IC~0 = dead · IC<0 = HURTS · |t|<2 = not significant (small sample)")

    print("\n=== FACTOR REDUNDANCY (signal-vs-signal rank correlation) ===")
    print(corr_matrix(sig).to_string())
    print("\n  >0.6 between two factors = largely the same bet, double-counted.")
