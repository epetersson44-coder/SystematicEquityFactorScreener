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

import os

import numpy as np
import pandas as pd

from fundamentals import get_fundamentals_asof
from factors import calculate_factors
from screen import EXCLUDE_SECTORS, MIN_CAP, MAX_CAP
from backtest.factor_backtest import daily_panel
from config import WEIGHTS

FACTORS = list(WEIGHTS.keys())                       # ev_ebit, price_fcf, roic, gm_stability, net_debt_ebitda
LOWER_BETTER = ["ev_ebit", "price_fcf", "gm_stability", "net_debt_ebitda"]
OBS_CSV = os.path.join(os.path.dirname(__file__), "_factor_obs.csv")


def collect(start="2021-07-01", end="2025-03-01", freq="QS", refresh=False):
    """Per eligible name per rebalance: the 5 factor values + the forward return to the
    next rebalance (survivorship-honest — a name that delists is marked at its last price).
    Cached to _factor_obs.csv so weight-scheme bake-offs are instant."""
    if not refresh and os.path.exists(OBS_CSV):
        return pd.read_csv(OBS_CSV, parse_dates=["date"])
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
    out = pd.DataFrame(rows)
    out.to_csv(OBS_CSV, index=False)
    return out


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


# Pre-committed weight schemes — NOT tuned to the data (that would be overfitting). Each is
# a principled rule decided in advance from the IC finding (quality carries it, 2 factors dead).
SCHEMES = {
    "current":   WEIGHTS,                                              # arbitrary status quo
    "equal":     {f: 0.20 for f in FACTORS},                          # robust default
    "drop_dead": {"ev_ebit": 1 / 3, "roic": 1 / 3, "gm_stability": 1 / 3,
                  "price_fcf": 0.0, "net_debt_ebitda": 0.0},           # drop the 2 dead, equal-weight rest
    "quality":   {"roic": 0.5, "gm_stability": 0.5, "ev_ebit": 0.0,
                  "price_fcf": 0.0, "net_debt_ebitda": 0.0},           # only the 2 significant factors
}


def bakeoff(sig, schemes=SCHEMES, n_top=20, min_factors=4):
    """Race weight schemes on a FIXED, production-consistent universe (names with >=
    `min_factors` of the 5 factors). Each scheme's composite is the weighted average over
    the factors a name actually has (re-normalized, exactly like score()), so missing data
    can't bias one scheme's universe vs another's. Benchmark = equal-weight eligible universe."""
    S = sig[FACTORS]
    base = sig[S.notna().sum(axis=1) >= min_factors].copy()       # one universe for all schemes
    Sb = base[FACTORS]
    uni = base.groupby("date")["fwd"].mean()
    out = {}
    for name, w in schemes.items():
        W = pd.Series({f: w.get(f, 0.0) for f in FACTORS})
        avail_w = Sb.notna().mul(W, axis=1).sum(axis=1)           # weight of the factors present
        comp = Sb.mul(W, axis=1).sum(axis=1) / avail_w.replace(0, np.nan)   # re-normalized avg
        rets = base.assign(comp=comp).dropna(subset=["comp"]).groupby("date").apply(
            lambda g: g.nlargest(n_top, "comp")["fwd"].mean(), include_groups=False)
        ex = rets - uni
        out[name] = {"cum_%": round(((1 + rets).prod() - 1) * 100, 1),
                     "mean_q_%": round(rets.mean() * 100, 2),
                     "sharpe": round(rets.mean() / rets.std() * np.sqrt(4), 2) if rets.std() else np.nan,
                     "excess_ann_%": round(ex.mean() * 4 * 100, 2),
                     "beat_univ": round((ex > 0).mean(), 2)}
    out["[universe EW]"] = {"cum_%": round(((1 + uni).prod() - 1) * 100, 1),
                            "mean_q_%": round(uni.mean() * 100, 2),
                            "sharpe": round(uni.mean() / uni.std() * np.sqrt(4), 2),
                            "excess_ann_%": 0.0, "beat_univ": np.nan}
    return pd.DataFrame(out).T


def _comp(sig, weights, min_factors=4):
    """Re-normalized composite on the production-consistent universe; returns base df + comp."""
    S = sig[FACTORS]
    base = sig[S.notna().sum(axis=1) >= min_factors].copy()
    Sb, W = base[FACTORS], pd.Series({f: weights.get(f, 0.0) for f in FACTORS})
    aw = Sb.notna().mul(W, axis=1).sum(axis=1)
    base["comp"] = Sb.mul(W, axis=1).sum(axis=1) / aw.replace(0, np.nan)
    return base.dropna(subset=["comp"])


def _stat(rets):
    return {"cum_%": round(((1 + rets).prod() - 1) * 100, 1),
            "mean_q_%": round(rets.mean() * 100, 2),
            "sharpe": round(rets.mean() / rets.std() * np.sqrt(4), 2) if rets.std() else np.nan,
            "ann_%": round(rets.mean() * 4 * 100, 2)}


def breadth(sig, weights=WEIGHTS, min_factors=4):
    """Does BREADTH harvest the factor spread the concentrated book can't? Compares long-only
    baskets of widening size against the universe, and pure long-short (top vs bottom by
    composite, dollar-neutral) — which strips out the market and isolates the factor spread."""
    base = _comp(sig, weights, min_factors)
    uni = base.groupby("date")["fwd"].mean()

    def long_n(n):
        return base.groupby("date").apply(
            lambda g: g.nlargest(n if n >= 1 else max(1, int(len(g) * n)), "comp")["fwd"].mean(),
            include_groups=False)

    def ls(frac):                                            # long top frac, short bottom frac
        def f(g):
            k = max(1, int(len(g) * frac)); s = g.sort_values("comp", ascending=False)
            return s["fwd"].iloc[:k].mean() - s["fwd"].iloc[-k:].mean()
        return base.groupby("date").apply(f, include_groups=False)

    rows = {"long top-5": _stat(long_n(5)), "long top-20": _stat(long_n(20)),
            "long top-quintile": _stat(long_n(0.2)), "long top-half": _stat(long_n(0.5)),
            "[universe EW]": _stat(uni),
            "L/S decile (10/10)": _stat(ls(0.1)), "L/S quintile (20/20)": _stat(ls(0.2))}
    return pd.DataFrame(rows).T


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

    print("\n=== WEIGHT-SCHEME BAKE-OFF (top-20 basket, point-in-time, survivorship-free) ===")
    print(bakeoff(sig).to_string())
    print("\n  excess_ann_% = annualized return OVER the same-style eligible universe (the fair test);")
    print("  beat_univ = share of quarters ahead of it. Pre-committed schemes, NOT tuned.")
    print("  14 quarters in a value-hostile regime — directional, not statistically decisive.")

    print("\n=== BREADTH TEST (does a wider / long-short book harvest the spread?) ===")
    print(breadth(sig).to_string())
    print("\n  long rows carry the universe's beta (compare to [universe EW]); the L/S rows are")
    print("  MARKET-NEUTRAL (ann_% is the harvested factor spread vs cash). Positive L/S Sharpe")
    print("  = the factor edge the concentrated top-5/20 book is too narrow to capture.")
