# backtest/factor_backtest.py — the point-in-time, survivorship-aware factor backtest.
#
# The first HONEST historical test of the factor screener (everything before this ran
# it only on today's data). Two foundations make it trustworthy, addressing the two
# biases that made the momentum backtest fiction (Axis B in [[cross-sectional]]):
#
#   1. LOOK-AHEAD — fundamentals come from get_fundamentals_asof(): at each rebalance
#      date only statements already PUBLISHED are visible. You can't rank on a 10-K
#      that didn't exist yet.
#   2. SURVIVORSHIP — prices come from SimFin's DAILY shareprices, which retain delisted
#      names up to their last trading day (TWTR ends 2022-10, ATVI 2023-10, ...). A name
#      that died is held through its death and drags the record down honestly; the
#      engine marks it at its last price after (the carry-forward fix).
#
# HONEST LIMITS (state them, don't hide them): SimFin free carries only ~FY2020-2025, so
# the window is ~2021-2025 (4 years) — too short for statistical confidence; figures are
# mildly restated-in-place (see fundamentals.simfin_fundamentals_asof). This SHOWS the
# process honestly; it does not PROVE the factor model works.
#
# Run:  python -m backtest.factor_backtest

import os

import numpy as np
import pandas as pd

from fundamentals import get_fundamentals_asof, piotroski_fscore_asof
from factors import calculate_factors, altman_z, beneish_m
from score import score
from screen import EXCLUDE_SECTORS, MIN_CAP, MAX_CAP, MIN_Z, MAX_M
from backtest.engine_xs import run_xs
from backtest.strategy import CrossSectionalStrategy
from backtest import metrics, costs

_PANEL = {}


def daily_panel():
    """SimFin DAILY shareprices as {'Close','Open'} (date x ticker) — survivorship-free
    (delisted names retained to their last trading day). Cached in memory."""
    if _PANEL:
        return _PANEL
    import simfin as sf
    from dotenv import load_dotenv
    load_dotenv()
    sf.set_api_key(os.environ["SIMFIN_API_KEY"])
    sf.set_data_dir(os.path.expanduser("~/simfin_data"))
    px = sf.load_shareprices(variant="daily", market="us")
    for field in ("Close", "Open"):
        panel = px[field].unstack(level=0).sort_index()      # date x ticker
        panel.columns = [str(c) for c in panel.columns]
        _PANEL[field] = panel
    return _PANEL


def screen_asof(asof, close_row, universe, sector_neutral=False, source="simfin"):
    """Rank `universe` by the composite using POINT-IN-TIME fundamentals as of `asof`,
    applying the same funnel as the live screen (band → ex-financials → Altman-Z → Beneish-M
    → rank). `close_row` is a Series ticker->price at `asof`, used for the point-in-time
    market cap. sector_neutral ranks within sector (see score()). source: 'simfin' (~2020+)
    or 'edgar' (survivorship-free, ~2010+). The F-Score is computed as a reference column
    but carries NO composite weight — same as the live screen, whose shared score() weights
    only config.WEIGHTS, from which the F-Score was dropped as window-overfit (2026-06; see
    the note above config.WEIGHTS). Returns the ranked long-side DataFrame."""
    rows = []
    for t in universe:
        price = close_row.get(t)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        f = get_fundamentals_asof(t, asof, price=float(price), source=source)
        mc = f.get("market_cap")
        if mc is None or not (MIN_CAP <= mc <= MAX_CAP):
            continue
        if f.get("sector") in EXCLUDE_SECTORS:
            continue
        z = altman_z(f)
        if z is not None and z < MIN_Z:                      # known-distressed: scrub
            continue
        m = beneish_m(f)
        if m is not None and m > MAX_M:                      # known-manipulator: scrub
            continue
        rec = calculate_factors(f)
        rec["market_cap"] = mc
        rec["sector"] = f.get("sector")                      # for sector-neutral ranking
        rec["fscore"] = piotroski_fscore_asof(t, asof, source=source)   # reference only: no weight in the composite
        rows.append(rec)
    df = pd.DataFrame(rows)
    return score(df, sector_neutral=sector_neutral).dropna(subset=["composite"]) if not df.empty else df


class ScheduledWeights(CrossSectionalStrategy):
    """Replay a precomputed {rebalance_date: weights} schedule through the engine. The
    slow fundamental ranking is done up front; the engine just simulates prices."""
    def __init__(self, schedule):
        self.sched = {pd.Timestamp(k): v for k, v in schedule.items()}
    def target_weights(self, closes, i):
        return self.sched.get(closes.index[i])


def build_schedules(top_n=20, start="2021-07-01", end="2025-03-01", freq="QS", sector_neutral=False,
                    source="simfin", universe=None):
    """At each rebalance date, compute BOTH baskets, point-in-time:
      - top: the top-`top_n` factor picks (equal weight)
      - universe: the WHOLE eligible (band+scrubbed) universe, equal weight — the FAIR,
        same-style benchmark that isolates the factor SIGNAL from the small-cap-value style.
    sector_neutral ranks within sector (see score()). source: 'simfin' or 'edgar'. `universe`
    caps the names screened each rebalance (default = all priced names; pass a list to bound
    EDGAR's per-name fetches). Returns (top_schedule, universe_schedule).
    freq: 'QS' quarterly (default), 'MS' monthly."""
    panel = daily_panel()
    close = panel["Close"]
    dates = close.index
    if universe is None:
        universe = list(close.columns)
    top_sched, uni_sched = {}, {}
    for target in pd.date_range(start, end, freq=freq):
        if target > dates[-1]:
            break
        d = dates[dates.searchsorted(target)]                # first trading day on/after
        ranked = screen_asof(d, close.loc[d], universe, sector_neutral=sector_neutral, source=source)
        if ranked.empty:
            continue
        names = ranked["ticker"].tolist()
        picks = names[:top_n]
        top_sched[d] = pd.Series(1.0 / len(picks), index=picks)
        uni_sched[d] = pd.Series(1.0 / len(names), index=names)
        print(f"  {d.date()}: ranked {len(names)} -> top {len(picks)}: {', '.join(picks[:6])}...")
    return top_sched, uni_sched


def _spy_curve(index):
    """SPY buy-and-hold rebased to $10k over the backtest dates (yfinance cache)."""
    from backtest.data import get_prices
    spy = get_prices("SPY")["Close"]
    spy = spy[(spy.index >= index[0]) & (spy.index <= index[-1])]
    return 10_000 * spy / spy.iloc[0]


def _run(schedule, panel, cost_bps):
    held = sorted({t for w in schedule.values() for t in w.index})
    sub = {"Close": panel["Close"][held], "Open": panel["Open"][held]}
    eq = run_xs(sub, ScheduledWeights(schedule), cost=costs.proportional(cost_bps), fill="next_open")
    return eq[eq.index >= min(schedule)]                     # start at first lock


def _stats(eq, label):
    return {"label": label, "final": float(eq.iloc[-1]), "ret": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "cagr": metrics.cagr(eq), "sharpe": metrics.sharpe(eq), "maxdd": metrics.max_drawdown(eq)}


def run_factor_backtest(top_n=20, start="2021-07-01", end="2025-03-01", freq="QS", cost_bps=30,
                        sector_neutral=False, source="simfin", universe=None):
    """Point-in-time, survivorship-aware backtest. Returns (factor_eq, uni_eq, spy_eq, stats)
    — the factor top-N vs the equal-weight eligible universe (fair, same-style) vs SPY.
    source: 'simfin' (~2020+) or 'edgar' (survivorship-free fundamentals, ~2010+)."""
    top_sched, uni_sched = build_schedules(top_n, start, end, freq, sector_neutral, source, universe)
    if not top_sched:
        raise RuntimeError("no rebalance produced picks — check the date window / data")
    panel = daily_panel()
    factor_eq = _run(top_sched, panel, cost_bps)
    uni_eq = _run(uni_sched, panel, cost_bps)
    spy = _spy_curve(factor_eq.index)
    stats = {"factor": _stats(factor_eq, "factor top-%d" % top_n),
             "universe": _stats(uni_eq, "eligible universe (EW)"),
             "spy": _stats(spy, "SPY"),
             "n_rebalances": len(top_sched),
             "start": factor_eq.index[0].date(), "end": factor_eq.index[-1].date()}
    return factor_eq, uni_eq, spy, stats


if __name__ == "__main__":
    f_eq, u_eq, spy, s = run_factor_backtest()
    print(f"\n=== Point-in-time factor backtest (survivorship-aware), {s['start']} -> {s['end']}, "
          f"{s['n_rebalances']} quarterly rebalances ===")
    for key in ("factor", "universe", "spy"):
        r = s[key]
        line = f"  {r['label']:<24} ${r['final']:>9,.0f}  ({r['ret'] * 100:+6.1f}%)"
        if key != "spy":
            line += f"  CAGR {r['cagr'] * 100:+5.1f}%  Sharpe {r['sharpe']:+.2f}  maxDD {r['maxdd'] * 100:.0f}%"
        print(line)
    print(f"\n  factor SIGNAL  (top-N vs eligible universe): {(s['factor']['ret'] - s['universe']['ret']) * 100:+.1f}%")
    print(f"  style HEADWIND (universe vs SPY)            : {(s['universe']['ret'] - s['spy']['ret']) * 100:+.1f}%")
    print("  (4-yr window, free-data limited — shows the process honestly, doesn't prove an edge.")
    print("   SPY is mega-cap growth; the universe-relative line is the fair test of the factor.)")
