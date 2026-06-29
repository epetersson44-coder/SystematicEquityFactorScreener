# backtest/edgar_backtest.py — the FULL-CYCLE factor backtest on EDGAR fundamentals.
#
# factor_backtest.py is capped at ~2021-2025 by SimFin-free's price+fundamentals wall. This
# runs the SAME screen funnel on EDGAR fundamentals (survivorship-free, ~2010+; see edgar.py),
# extending the window to ~2011-2025 — through the 2020 COVID crash, the 2022 bear, the 2015-16
# and 2018-Q4 selloffs (three real drawdowns SimFin-free never sees).
#
# THE PRICE CAVEAT (state it loudly): fundamentals are survivorship-free, but free PRICE history
# (yfinance) only covers names that STILL trade — delisted names get no prices and silently
# leave the tradeable universe. So this backtest is survivorship-free on FUNDAMENTALS but
# survivor-biased on the PRICE/UNIVERSE side: the absolute return is INFLATED (the losers that
# went to zero aren't here). The relative reads — factor SIGNAL vs the same-style universe, and
# behaviour THROUGH the crashes — survive the bias; the headline CAGR does not. (Stooq, which
# carries delisted names, is the honest fix if this looks worth hardening.)
#
# Two price views, both needed and both from yfinance:
#   * adjusted (split+dividend) Close/Open -> RETURNS (the engine; total-return equity curve)
#   * UNADJUSTED Close                     -> MARKET CAP at screen time (unadj price x EDGAR
#                                             as-filed shares; using adjusted price would
#                                             understate historical caps and break the band)
#
# Run:  python -m backtest.edgar_backtest

import os
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest.data import CACHE_DIR
from backtest.factor_backtest import screen_asof, _run, _stats, _spy_curve, daily_panel  # noqa: F401
from backtest import metrics, costs
from edgar import HEADERS

START = "2011-01-01"                                     # ~where small-cap XBRL coverage firms up


def sp600_tickers(refresh=False):
    """Today's S&P 600 small-cap constituents (Wikipedia), cached to sp600_tickers.txt.
    SURVIVORSHIP CAVEAT: today's members only — a real small-cap universe but survivor-biased."""
    path = os.path.join(os.path.dirname(__file__), "sp600_tickers.txt")
    if not refresh and os.path.exists(path):
        return [t.strip() for t in open(path) if t.strip()]
    # requests (certifi) not urllib — macOS urllib hits SSL CERT_VERIFY (see fundamentals/simfin)
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                        headers=HEADERS, timeout=30).text
    tables = pd.read_html(StringIO(html))
    tbl = next(t for t in tables if "Symbol" in t.columns)
    tickers = [str(s).replace(".", "-").strip().upper() for s in tbl["Symbol"]]
    with open(path, "w") as f:
        f.write("\n".join(tickers) + "\n")
    return tickers


def _panel_path(tag, field):
    return os.path.join(CACHE_DIR, f"edgar_uni_{tag}_{field}.csv")


def price_panels(tickers, tag="sp600", refresh=False):
    """{'adj_close','adj_open','raw_close'} (date x ticker) for `tickers`, from yfinance,
    cached. adj_* are split+dividend adjusted (returns); raw_close is unadjusted (market cap)."""
    fields = {"adj_close": ("adj", "Close"), "adj_open": ("adj", "Open"), "raw_close": ("raw", "Close")}
    if not refresh and all(os.path.exists(_panel_path(tag, k)) for k in fields):
        return {k: pd.read_csv(_panel_path(tag, k), index_col=0, parse_dates=True) for k in fields}
    os.makedirs(CACHE_DIR, exist_ok=True)
    adj = yf.download(list(tickers), start=START, auto_adjust=True, progress=False)
    raw = yf.download(list(tickers), start=START, auto_adjust=False, progress=False)
    out = {
        "adj_close": adj["Close"].sort_index(),
        "adj_open": adj["Open"].sort_index(),
        "raw_close": raw["Close"].sort_index(),
    }
    for k, panel in out.items():
        panel.index.name = "Date"
        panel.to_csv(_panel_path(tag, k))
    return out


def build_schedules(panels, top_n=20, end="2025-09-01", freq="QS", sector_neutral=True):
    """At each quarterly rebalance, rank the universe by the EDGAR-sourced composite (band ->
    ex-financials -> Altman -> Beneish -> rank, point-in-time), using UNADJUSTED price x EDGAR
    shares for the market cap. Returns (top_schedule, universe_schedule) — top-N picks vs the
    whole eligible same-style universe (the fair factor-signal benchmark)."""
    raw_close, adj_close = panels["raw_close"], panels["adj_close"]
    dates = adj_close.index
    universe = list(adj_close.columns)
    top_sched, uni_sched = {}, {}
    for target in pd.date_range(START, end, freq=freq):
        if target > dates[-1]:
            break
        d = dates[dates.searchsorted(target)]                # first trading day on/after
        raw_row = raw_close.loc[d] if d in raw_close.index else pd.Series(dtype=float)
        ranked = screen_asof(d, raw_row, universe, sector_neutral=sector_neutral, source="edgar")
        if ranked.empty:
            continue
        names = ranked["ticker"].tolist()
        picks = names[:top_n]
        top_sched[d] = pd.Series(1.0 / len(picks), index=picks)
        uni_sched[d] = pd.Series(1.0 / len(names), index=names)
        print(f"  {d.date()}: ranked {len(names):3d} -> top {len(picks)}: {', '.join(picks[:6])}...")
    return top_sched, uni_sched


def run_edgar_backtest(top_n=20, end="2025-09-01", freq="QS", cost_bps=30, sector_neutral=True,
                       tickers=None, tag="sp600"):
    """Full-cycle EDGAR-fundamentals factor backtest. Returns (factor_eq, uni_eq, spy_eq, stats).
    Survivorship-free fundamentals; survivor-biased prices (see module header) — read the
    factor-SIGNAL line, not the absolute CAGR."""
    if tickers is None:
        tickers = sp600_tickers()
    panels = price_panels(tickers, tag=tag)
    engine_panel = {"Close": panels["adj_close"], "Open": panels["adj_open"]}
    top_sched, uni_sched = build_schedules(panels, top_n, end, freq, sector_neutral)
    if not top_sched:
        raise RuntimeError("no rebalance produced picks — check coverage / dates")
    factor_eq = _run(top_sched, engine_panel, cost_bps)
    uni_eq = _run(uni_sched, engine_panel, cost_bps)
    spy = _spy_curve(factor_eq.index)
    stats = {"factor": _stats(factor_eq, "factor top-%d" % top_n),
             "universe": _stats(uni_eq, "eligible universe (EW)"),
             "spy": _stats(spy, "SPY"),
             "n_rebalances": len(top_sched),
             "start": factor_eq.index[0].date(), "end": factor_eq.index[-1].date()}
    return factor_eq, uni_eq, spy, stats


FCOLS = ["ev_ebit", "price_fcf", "roic", "gm_stability", "net_debt_ebitda", "fscore"]


def decompose_drag(end="2025-09-01", freq="QS", tickers=None, tag="sp600"):
    """WHICH factor makes the composite anti-predictive? Per-factor Information Coefficient over
    the full EDGAR cycle: at each rebalance, take each factor's sector-neutral GOOD-direction
    percentile (exactly how the screen ranks — the {factor}_pct columns from screen_asof) and
    rank-correlate it with the cross-sectional forward return to the next rebalance. Positive
    mean IC => the factor predicts; ~0 => dead; NEGATIVE => it drags. Also reports the COMPOSITE's
    own IC and a raw 12-1 MOMENTUM positive-control (the known real edge) for contrast. Returns
    (ic_table, obs). IC is whole-cross-section, so it's free of the top-20 concentration noise."""
    if tickers is None:
        tickers = sp600_tickers()
    panels = price_panels(tickers, tag=tag)
    adj, raw = panels["adj_close"], panels["raw_close"]
    dates = adj.index
    universe = list(adj.columns)
    rb = [dates[dates.searchsorted(t)] for t in pd.date_range(START, end, freq=freq) if t <= dates[-1]]
    SKIP, LOOK = 21, 252                                  # 12-1 momentum (skip ~1mo, look back ~12mo)
    recs = []
    for k in range(len(rb) - 1):
        d, d1 = rb[k], rb[k + 1]
        ranked = screen_asof(d, raw.loc[d] if d in raw.index else pd.Series(dtype=float),
                             universe, sector_neutral=True, source="edgar")
        if ranked.empty:
            continue
        seg = adj[(adj.index > d) & (adj.index <= d1)]
        c0, c1 = adj.loc[d], adj.loc[d1]
        i = dates.get_loc(d)
        mom = (adj.iloc[i - SKIP] / adj.iloc[i - SKIP - LOOK] - 1) if i - SKIP - LOOK >= 0 else None
        for _, r in ranked.iterrows():
            t = r["ticker"]
            p0, p1 = c0.get(t), c1.get(t)
            if not (p0 is not None and np.isfinite(p0) and p0 > 0):
                continue
            if not (p1 is not None and np.isfinite(p1)):     # delisted mid-quarter -> last trade
                s = seg[t].dropna()
                p1 = float(s.iloc[-1]) if len(s) else np.nan
            if not (np.isfinite(p1) and p1 > 0):
                continue
            rec = {"date": d, "fwd": p1 / p0 - 1, "composite": r.get("composite")}
            for f in FCOLS:
                rec[f] = r.get(f + "_pct")                # sector-neutral good-direction pct
            rec["momentum"] = float(mom[t]) if (mom is not None and t in mom.index
                                                and np.isfinite(mom.get(t, np.nan))) else np.nan
            recs.append(rec)
    obs = pd.DataFrame(recs)
    obs["fwd_rank"] = obs.groupby("date")["fwd"].rank(pct=True)
    rows = {}
    big = obs.groupby("date").size()                      # skip near-empty early periods (degenerate corr)
    obs = obs[obs["date"].isin(big[big >= 5].index)]
    for f in FCOLS + ["composite", "momentum"]:
        ics = obs.groupby("date").apply(lambda g: g[f].corr(g["fwd_rank"]), include_groups=False).dropna()
        m, sd = ics.mean(), ics.std()
        rows[f] = {"IC": round(m, 4), "t_stat": round(m / sd * np.sqrt(len(ics)), 2) if sd else np.nan,
                   "hit_rate": round(float((ics > 0).mean()), 2), "periods": len(ics)}
    return pd.DataFrame(rows).T, obs


if __name__ == "__main__":
    f_eq, u_eq, spy, s = run_edgar_backtest()
    print(f"\n=== EDGAR full-cycle factor backtest, {s['start']} -> {s['end']}, "
          f"{s['n_rebalances']} quarterly rebalances ===")
    for key in ("factor", "universe", "spy"):
        r = s[key]
        line = f"  {r['label']:<24} ${r['final']:>10,.0f}  ({r['ret'] * 100:+7.1f}%)"
        if key != "spy":
            line += f"  CAGR {r['cagr'] * 100:+5.1f}%  Sharpe {r['sharpe']:+.2f}  maxDD {r['maxdd'] * 100:.0f}%"
        print(line)
    print(f"\n  factor SIGNAL  (top-N vs eligible universe): {(s['factor']['ret'] - s['universe']['ret']) * 100:+.1f}%")
    print(f"  style HEADWIND (universe vs SPY)            : {(s['universe']['ret'] - s['spy']['ret']) * 100:+.1f}%")
    print("\n  PRICE CAVEAT: survivorship-free FUNDAMENTALS, survivor-biased PRICES (yfinance has")
    print("  no delisted names) -> absolute CAGR is INFLATED. The factor-SIGNAL line (top-N vs the")
    print("  same-style eligible universe) is the bias-robust read; the headline number is not.")
