# backtest/factor_research.py — incremental factor decomposition (research plan Step 1 + 3).
#
# The honest upgrade over factor_analysis (which measured STANDALONE IC): does each factor
# carry signal that is NOT already in the others? We collect an 8-factor point-in-time panel
# on the survivorship-free EDGAR data (the 5 current value/quality legs + 3 evidence-backed
# additions — Novy-Marx gross profitability, asset growth, net issuance) and compare:
#
#   * STANDALONE IC   — rank-correlation of each factor's good-direction percentile vs the
#                       forward return (what factor_analysis did).
#   * INCREMENTAL IC  — Fama-MacBeth: each rebalance, OLS-regress the forward-return rank on
#                       ALL factor ranks jointly; the per-factor coefficient is its signal
#                       AFTER controlling for the others. A factor that's significant
#                       standalone but ~0 in FM is redundant; a factor significant in FM
#                       carries unique information. (Harvey-Liu-Zhu: demand a high bar.)
#   * SUBPERIOD       — 2011-2016 / 2017-2020 / 2021-2026, to catch one-regime mirages.
#
# Hand-rolled (numpy lstsq for the cross-sectional OLS; Spearman = Pearson on ranks). Built on
# the EDGAR backtest plumbing (price panels + screen_asof eligibility). Run heavy (re-screens
# the universe quarterly); see scratchpad runner.

import numpy as np
import pandas as pd

import edgar
from backtest.edgar_backtest import price_panels, sp600_tickers, build_schedules, screen_asof, START  # noqa: F401

# 8 factors under test, with direction (True = LOWER raw value is better -> ranked descending)
FACTORS = ["ev_ebit", "price_fcf", "roic", "gm_stability", "net_debt_ebitda",
           "gross_prof", "asset_growth", "net_issuance"]
LOWER_BETTER = {"ev_ebit", "price_fcf", "gm_stability", "net_debt_ebitda", "asset_growth", "net_issuance"}


def collect_panel(end="2025-09-01", freq="QS", tickers=None, tag="sp600"):
    """Per eligible name per quarterly rebalance: the 8 raw factor values + the forward return
    to the next rebalance, point-in-time on EDGAR fundamentals (band+ex-financials+scrubs via
    screen_asof). The 5 current legs come from the screen; gross profitability / asset growth /
    net issuance are computed from the raw EDGAR extraction (gross_profit, total_assets[_prior],
    shares[_prior]). Forward returns use adjusted prices; a name delisted mid-quarter is marked
    at its last trade (survivorship-honest)."""
    if tickers is None:
        tickers = sp600_tickers()
    panels = price_panels(tickers, tag=tag)
    adj, raw_px = panels["adj_close"], panels["raw_close"]
    dates = adj.index
    universe = list(adj.columns)
    rb = [dates[dates.searchsorted(t)] for t in pd.date_range(START, end, freq=freq) if t <= dates[-1]]
    rows = []
    for k in range(len(rb) - 1):
        d, d1 = rb[k], rb[k + 1]
        ranked = screen_asof(d, raw_px.loc[d] if d in raw_px.index else pd.Series(dtype=float),
                             universe, sector_neutral=False, source="edgar")
        if ranked.empty:
            continue
        seg = adj[(adj.index > d) & (adj.index <= d1)]
        c0, c1 = adj.loc[d], adj.loc[d1]
        for _, r in ranked.iterrows():
            t = r["ticker"]
            p0, p1 = c0.get(t), c1.get(t)
            if not (p0 is not None and np.isfinite(p0) and p0 > 0):
                continue
            if not (p1 is not None and np.isfinite(p1)):
                s = seg[t].dropna()
                p1 = float(s.iloc[-1]) if len(s) else np.nan
            if not (np.isfinite(p1) and p1 > 0):
                continue
            f = edgar.fundamentals_asof(t, str(d.date())) or {}
            gp, ta, tap = f.get("gross_profit"), f.get("total_assets"), f.get("total_assets_prior")
            sh, shp = f.get("shares"), f.get("shares_prior")
            rows.append({
                "date": d, "ticker": t, "fwd": p1 / p0 - 1,
                "ev_ebit": r.get("ev_ebit"), "price_fcf": r.get("price_fcf"), "roic": r.get("roic"),
                "gm_stability": r.get("gm_stability"), "net_debt_ebitda": r.get("net_debt_ebitda"),
                "gross_prof": (gp / ta) if (gp is not None and ta) else np.nan,
                "asset_growth": (ta / tap - 1) if (ta and tap) else np.nan,
                "net_issuance": (sh / shp - 1) if (sh and shp) else np.nan,
            })
    return pd.DataFrame(rows)


def _rank(obs):
    """Add good-direction percentile ranks per rebalance (higher = better) + forward rank."""
    out = obs.copy()
    for f in FACTORS:
        asc = f not in LOWER_BETTER                                  # lower-better -> rank descending
        out[f + "_r"] = out.groupby("date")[f].rank(ascending=asc, pct=True)
    out["fwd_r"] = out.groupby("date")["fwd"].rank(pct=True)
    return out


def standalone_ic(ranked):
    """{factor: (mean_IC, t_stat, n_periods)} — each factor's rank vs forward rank, alone."""
    res = {}
    for f in FACTORS:
        ics = ranked.groupby("date").apply(lambda g: g[f + "_r"].corr(g["fwd_r"]),
                                           include_groups=False).dropna()
        m, sd = ics.mean(), ics.std()
        res[f] = (round(m, 4), round(m / sd * np.sqrt(len(ics)), 2) if sd else np.nan, len(ics))
    return res


def fama_macbeth(ranked, min_names=20):
    """Cross-sectional FM regression: each rebalance, OLS forward-rank ~ all 8 factor ranks
    (missing ranks filled at 0.5 = neutral, so a sparse factor doesn't shrink the sample).
    Returns {factor: (mean_coef, FM_t_stat)} + n_periods. The coef is incremental signal."""
    cols = [f + "_r" for f in FACTORS]
    coefs = []
    for _, g in ranked.groupby("date"):
        sub = g[cols + ["fwd_r"]].copy()
        sub[cols] = sub[cols].fillna(0.5)
        sub = sub.dropna(subset=["fwd_r"])
        if len(sub) < min_names:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[cols].values])
        b, *_ = np.linalg.lstsq(X, sub["fwd_r"].values, rcond=None)
        coefs.append(b[1:])
    C = np.array(coefs)
    mean, sd = C.mean(0), C.std(0)
    t = mean / (sd / np.sqrt(len(C)))
    return {f: (round(float(mean[i]), 4), round(float(t[i]), 2)) for i, f in enumerate(FACTORS)}, len(C)


def subperiods(ranked):
    """Standalone IC per factor in three regimes (catches one-period mirages)."""
    cuts = {"2011-2016": ("2011-01-01", "2016-12-31"),
            "2017-2020": ("2017-01-01", "2020-12-31"),
            "2021-2026": ("2021-01-01", "2026-12-31")}
    out = {}
    for name, (a, b) in cuts.items():
        sub = ranked[(ranked["date"] >= a) & (ranked["date"] <= b)]
        out[name] = {f: standalone_ic(sub)[f][0] for f in FACTORS} if len(sub) else {}
    return out
