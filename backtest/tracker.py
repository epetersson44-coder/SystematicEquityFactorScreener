# backtest/tracker.py — minimal live paper-tracker.
#
# Locks a strategy's picks to an IMMUTABLE dated file the moment they're made —
# before anyone knows the outcome — then scores them forward vs SPY. This is
# survivorship-bias-FREE BY CONSTRUCTION: the record is built going forward, so a
# name that later dies is in the picks when it was chosen and drags the record down
# honestly (unlike a backtest on today's survivors). The lock files
# (picks/<strategy>/<date>.json) are committed to git; that timestamp is the entire
# credibility of the track record — it proves the pick predated the outcome.
#
# Usage:
#   python -m backtest.tracker lock     # lock this month's picks (run monthly)
#   python -m backtest.tracker report   # score every locked pick-set vs SPY

import os
import json
import datetime as dt

import numpy as np
import pandas as pd

from backtest.universe import get_universe, download_panel
from backtest.data import get_prices
from backtest.strategy import CrossSectionalMomentum

PICKS_DIR = os.path.join(os.path.dirname(__file__), "picks")


def _market_risk_on(refresh=False, ma=200):
    """Trend-filter regime: is SPY above its `ma`-day average? Below it = downtrend."""
    spy = get_prices("SPY", refresh=refresh)["Close"]
    if len(spy) < ma:
        return True
    return float(spy.iloc[-1]) > float(spy.iloc[-ma:].mean())


def momentum_picks(refresh=False):
    """Today's cross-sectional momentum basket WITH the trend-filter failsafe: if SPY is
    below its 200-day average (downtrend), hold CASH this month instead of the book.
    Validated 2005-2026 (momentum_ls.py): halves max drawdown (-59%->-30%), dodged the 2008
    crash, lifts Sharpe 0.79->0.89. Returns (weights, prices_now, data_asof); weights is
    EMPTY when risk-off — a cash month, which the simulator/desk treat as flat."""
    closes = get_universe("sp500", refresh=refresh)["Close"]
    i = len(closes) - 1
    asof = closes.index[i].date().isoformat()
    if not _market_risk_on(refresh=refresh):
        return pd.Series(dtype=float), closes.iloc[i], asof        # risk-off -> cash
    weights = CrossSectionalMomentum().rank(closes, i)
    if weights is None:
        raise RuntimeError("not enough history to rank the universe")
    return weights, closes.iloc[i], asof


def _priceable_topn(ranked, top_n, candidate_mult=5, min_candidates=25):
    """Walk DOWN a ranking and return the top `top_n` names with a LIVE price today:
    (picks_list, last_prices Series, asof).

    Why not just `ranked.head(top_n)`? SimFin's fundamentals run ~12 months behind, so a
    top-ranked name can already have been acquired or delisted since the data vintage
    (FARO, GLT, ...). Such a name can't be held in a live book, so we skip to the next
    priceable name. NOT survivorship bias — a stock that no longer trades can't be bought
    today; the pick is made now and tracked forward, and a real future delisting (while
    held) still drags the record down."""
    if ranked.empty:
        raise RuntimeError("screen returned no names with a valid composite")
    candidates = ranked.head(max(top_n * candidate_mult, min_candidates))["ticker"].tolist()
    closes = download_panel(candidates)["Close"]
    last = closes.iloc[-1]
    priced = [t for t in candidates                          # keep ranking order
              if t in last.index and pd.notna(last[t]) and last[t] > 0]
    picks = priced[:top_n]
    if not picks:
        raise RuntimeError("no top-ranked name has a live price")
    return picks, last[picks], closes.index[-1].date().isoformat()


def _screen_picks(top_n=5, source="simfin"):
    """Top-`top_n` priceable names from the long screen, equal-weighted: (weights, prices, asof)."""
    try:
        from screen import run_screen
    except ImportError as e:                                  # needs the repo root on path
        raise RuntimeError(f"can't import the screen — run from the repo root: {e}")
    picks, prices, asof = _priceable_topn(run_screen(source=source), top_n)
    if len(picks) < top_n:
        print(f"[factor] only {len(picks)}/{top_n} top names are priceable today — locking those")
    return pd.Series(1.0 / len(picks), index=picks), prices, asof


def factor_picks(top_n=5):
    """Today's top-`top_n` names from the SCALED, scrubbed small-cap value screen,
    equal-weighted and held forward: (weights, prices, asof).

    This is screen.py over the full ~4,300-name SimFin universe — $300M–$5B band,
    ex-financials/REITs, Altman-Z distress scrub, Beneish-M manipulation scrub — ranked
    by the same 5-factor composite. It replaced the original 15-name yfinance watchlist
    on 2026-06-15; the immutable June 15-name lock stays in the record as history."""
    return _screen_picks(top_n=top_n)


def factor_ls_picks(top_n=5, source="simfin", min_legs=2):
    """Today's DOLLAR-NEUTRAL long-short book: (weights, prices, asof).

    Long the top-`top_n` priceable names of the LONG screen (scrubbed value), short the
    top-`top_n` of the inverted SHORT screen (distress/manipulation as signals). Signed
    weights sum to net ~0 with gross 1.0 — $5k long / $5k short on $10k, the unlevered
    market-neutral spread. A short must have `min_legs` signals firing (default 2) so
    it's a real distress/manipulation short, not bottom-of-factor noise. A name landing
    on BOTH sides is kept LONG (resolve the contradiction in the long book's favor).

    Benched vs CASH, not SPY (the book is market-neutral — see report()). NOTE: the
    paper chain does not yet model borrow cost on the shorts (the backtest engine does);
    at ~2%/yr on a $5k short that's ~$8/mo, a documented v1 simplification."""
    try:
        from screen import run_screen, run_short_screen
    except ImportError as e:                                  # needs the repo root on path
        raise RuntimeError(f"can't import the screen — run from the repo root: {e}")
    long_names, long_px, asof = _priceable_topn(run_screen(source=source), top_n)
    short_ranked = run_short_screen(source=source)
    if min_legs and not short_ranked.empty:
        short_ranked = short_ranked[short_ranked["short_legs"] >= min_legs]
    short_names, short_px, _ = _priceable_topn(short_ranked, top_n)
    short_names = [t for t in short_names if t not in long_names]   # long wins any overlap
    if not short_names:
        raise RuntimeError("no priceable short names after de-overlap")

    weights = pd.concat([pd.Series(0.5 / len(long_names), index=long_names),
                         pd.Series(-0.5 / len(short_names), index=short_names)])
    prices = pd.concat([long_px, short_px])
    prices = prices[~prices.index.duplicated(keep="first")].reindex(weights.index)
    return weights, prices, asof


def blend_picks(refresh=False, eq_weight=None, cash_etf="SGOV", mf_etf=None):
    """Today's UNLEVERAGED SPY + cross-asset-trend blend, as net ETF weights: (weights, prices, asof).

    The project's headline result (trend_sleeve.py + timing_luck.py, 2006-2026 vs SPY
    Sharpe 0.64/maxDD −55%): the SPY equity leg + the vol-targeted 6-ETF trend sleeve
    (SPY/EFA/TLT/IEF/GLD/DBC), combined risk-parity (inverse-vol). Signals use the ADOPTED
    1/3/12-month MOP ensemble (timing-luck-controlled blend Sharpe ~0.94 vs 0.87 single-look
    — see trend_sleeve.ENSEMBLE_LOOKS; the June 2026 lock predates this and stays immutable
    single-look history). Long-only, NO borrowing; any unallocated weight is cash. A real
    6-ETF monthly allocation you could run in a brokerage account — this is the one book
    worth tracking live (the factor screener was a proven zero-edge result). Benched vs
    SPY. The risk-parity equity/trend split is recomputed from full-history vols each lock.

    mf_etf: optional managed-futures ETF (e.g. "DBMF") added as a THIRD risk-parity leg.
    Rationale: a CTA replicator carries the 50+-market futures breadth (FX, rates,
    commodity curves) our 6-ETF sleeve can't reach on free data — measured corr to the
    homemade sleeve only ~0.3 (DBMF) / ~0.05 (KMLM), and the 3-way blend improved Sharpe
    AND maxDD on the available window. OFF BY DEFAULT: that window is 2019+/2021+ only and
    contains 2022 (the best CTA year in decades) — promising, NOT full-cycle proven, so
    turning it on is the book owner's call, not code's. When set, the SPY/sleeve/MF split
    is inverse-vol over their common history and eq_weight is ignored."""
    from backtest.trend_sleeve import etf_panel, run_trend, VolTargetTSMOM, ENSEMBLE_LOOKS
    closes = etf_panel(refresh=refresh)["Close"]
    i = len(closes) - 1
    asof = closes.index[i].date().isoformat()
    w_mf = 0.0
    if mf_etf:                                              # 3-way inverse-vol: SPY / sleeve / MF-ETF
        mf_px = get_prices(mf_etf, refresh=refresh)["Close"]
        al = pd.DataFrame({"SPY": closes["SPY"], "trend": run_trend(cash_rate=0.0),
                           "mf": mf_px}).dropna().pct_change().dropna()
        iv = 1.0 / al.std()
        w = iv / iv.sum()
        eq_weight, w_trend, w_mf = float(w["SPY"]), float(w["trend"]), float(w["mf"])
    else:
        if eq_weight is None:                               # inverse-vol risk parity: SPY vs trend sleeve
            al = pd.DataFrame({"SPY": closes["SPY"], "trend": run_trend(cash_rate=0.0)}).dropna()
            al = al.pct_change().dropna()
            ivs, ivt = 1.0 / al["SPY"].std(), 1.0 / al["trend"].std()
            eq_weight = ivs / (ivs + ivt)
        w_trend = 1.0 - eq_weight
    tw = VolTargetTSMOM(max_gross=1.0, every=1, looks=ENSEMBLE_LOOKS).target_weights(closes, i)
    tw = tw if (tw is not None and not tw.empty) else pd.Series(dtype=float)
    net = pd.Series({"SPY": eq_weight}).add(w_trend * tw, fill_value=0.0)
    if w_mf > 1e-9:
        net = net.add(pd.Series({mf_etf: w_mf}), fill_value=0.0)
    net = net[net.abs() > 1e-9]
    prices = closes.iloc[i].reindex(net.index)
    if mf_etf and mf_etf in net.index:
        prices[mf_etf] = float(mf_px.iloc[-1])

    # The unallocated slice is cash — in a real account that's T-bills, not 0%. Park it in
    # a T-bill ETF (SGOV) so the live book earns the rf its backtest counterpart is owed
    # (worth ~+0.3% CAGR at 2025-26 rates; cash_rate A/B in the sleeve backtest). If the
    # ETF can't be priced right now, the slice stays plain cash rather than blocking a lock.
    resid = 1.0 - float(net.sum())
    if cash_etf and resid > 0.005:
        try:
            px = float(get_prices(cash_etf, refresh=refresh)["Close"].iloc[-1])
            if np.isfinite(px) and px > 0:
                net[cash_etf] = resid
                prices[cash_etf] = px
        except Exception as e:                              # noqa: BLE001 — cash fallback, never fatal
            print(f"[blend] {cash_etf} unpriceable ({e}) — leaving {resid:.1%} as plain cash")
    return net, prices, asof


# Live books. momentum (real, crash-guarded edge) and blend (the headline Sharpe-0.90 trend
# allocation) are the keepers tracked monthly; factor/factor_ls stay defined so their existing
# locks still report, but are no longer locked forward (the screener was a proven zero-edge result).
PICKERS = {"momentum": momentum_picks, "blend": blend_picks,
           "factor": factor_picks, "factor_ls": factor_ls_picks}
LIVE = ("momentum", "blend")            # what /picks locks each month now
MARKET_NEUTRAL = {"factor_ls"}          # benched vs cash, not SPY (beta is hedged out)
_FRESH_PRICED = {"momentum", "blend"}   # books that pull fresh prices on lock


def lock(strategy="momentum", refresh=False):
    """Compute today's picks and write them to an immutable dated file. Refuses to
    overwrite an existing lock — picks, once made, never change."""
    if strategy not in PICKERS:
        raise ValueError(f"unknown strategy {strategy!r} (use {sorted(PICKERS)})")
    picker = PICKERS[strategy]
    weights, prices, asof = picker(refresh=refresh) if strategy in _FRESH_PRICED else picker()
    spy_close = float(get_prices("SPY", refresh=refresh)["Close"].iloc[-1])

    rec = {
        "lock_date": dt.date.today().isoformat(),
        "data_asof": asof,
        "strategy": strategy,
        "n": int(len(weights)),
        "picks": {t: round(float(w), 6) for t, w in weights.items()},
        "lock_prices": {t: round(float(prices[t]), 4) for t in weights.index},
        "spy_lock": round(spy_close, 4),
    }
    if strategy == "momentum":                           # record the trend-filter regime
        rec["regime"] = "risk_on" if rec["n"] else "risk_off (cash)"
    out_dir = os.path.join(PICKS_DIR, strategy)
    os.makedirs(out_dir, exist_ok=True)
    month = rec["lock_date"][:7]                          # one lock per calendar month
    existing = [f for f in os.listdir(out_dir) if f.startswith(month) and f.endswith(".json")]
    if existing:
        raise FileExistsError(f"already locked for {month}: {existing[0]} — one lock per month, picks are immutable")
    path = os.path.join(out_dir, f"{rec['lock_date']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    if rec["n"] == 0:
        print(f"locked {strategy}: RISK-OFF — cash this month (SPY below its 200-day) -> {os.path.relpath(path)}")
    else:
        print(f"locked {rec['n']} {strategy} picks (data as of {asof}) -> {os.path.relpath(path)}")
        print(f"  sample names: {', '.join(sorted(rec['picks'])[:12])} ...")
    return rec


def _entry_prices(closes, asof, picks, fallback=None):
    """Entry prices for a locked basket, read from the CURRENT panel at the lock's
    data_asof bar: {ticker: price}.

    Why not the stored lock_prices? Those were snapshotted from ADJUSTED closes at lock
    time, and yfinance re-scales the whole adjusted history every time a new dividend is
    paid — so months later the stored numbers sit on a DIFFERENT adjustment basis than
    today's panel, and every "since lock" return computed across the two silently drifts
    by the accumulated adjustments. Reading entry and exit from the SAME panel keeps both
    ends on one basis (and makes the return total-return-correct). The stored lock_prices
    stay in the JSON as the immutable audit record; a name with no panel price at the
    as-of bar falls back to them (best effort — right at lock time, drifts after)."""
    idx = closes.index[closes.index >= pd.to_datetime(asof)]
    row = closes.loc[idx[0]] if len(idx) else pd.Series(dtype=float)
    fallback = fallback or {}
    out = {}
    for t in picks:
        p = float(row.get(t, np.nan))
        if not (np.isfinite(p) and p > 0):
            p = float(fallback.get(t, np.nan))
        out[t] = p
    return out


def _simulate(recs, closes, spy_close, initial=10_000.0, cost_bps=10.0):
    """ONE managed paper portfolio: start with `initial`, hold each month's locked
    basket until the next lock, then rebalance into the new picks; carry the value
    forward. Returns (equity Series, summary). Prices come from the universe panel,
    so any pick's value at any date is looked up consistently.

    NET OF COSTS: each rebalance charges `cost_bps` on TRADED weight — the two-way
    turnover between the incoming basket and the previous basket's weights after they
    DRIFTED with the month's prices (a winner that grew from 20% to 24% of the book
    costs 4 points of turnover to trim back, even if the name is 'held'). The first
    lock pays a full buy-in. A record that only ever reported gross would quietly
    overstate a high-turnover book; the fall memo needs the net number.

    recs: list of {data_asof, picks} (one per monthly lock). closes: (date x ticker)
    panel. spy_close: Series of SPY close by date."""
    recs = sorted(recs, key=lambda r: r["data_asof"])
    dates = [pd.to_datetime(r["data_asof"]) for r in recs]
    end = closes.index[-1]
    bounds = dates + [end]                                # each basket runs lock_k -> lock_{k+1}
    value = initial
    curve = {dates[0]: initial}
    drifted = {}                                          # prior basket's weights, price-drifted
    turnovers, cum_cost = [], 0.0
    for k, rec in enumerate(recs):
        d0, d1 = bounds[k], bounds[k + 1]
        traded = sum(abs(rec["picks"].get(t, 0.0) - drifted.get(t, 0.0))
                     for t in set(rec["picks"]) | set(drifted))
        fee = traded * cost_bps / 10_000.0
        value *= (1.0 - fee)
        cum_cost += fee
        turnovers.append(traded)
        if d1 <= d0:
            continue
        seg_ret, name_ret = 0.0, {}
        for t, w in rec["picks"].items():
            if t not in closes.columns:
                continue
            p0, p1 = closes.at[d0, t], closes.at[d1, t]
            if p0 == p0 and p1 == p1 and p0 > 0:          # both prices present
                name_ret[t] = p1 / p0 - 1
                seg_ret += w * name_ret[t]
        value *= (1 + seg_ret)
        curve[d1] = value
        drifted = {t: w * (1 + name_ret.get(t, 0.0)) / (1 + seg_ret)
                   for t, w in rec["picks"].items()}
    eq = pd.Series(curve).sort_index()
    s0 = float(spy_close.loc[spy_close.index >= dates[0]].iloc[0])
    s1 = float(spy_close.iloc[-1])
    return eq, {"final": value, "ret": value / initial - 1,
                "spy_ret": s1 / s0 - 1, "spy_final": initial * (s1 / s0),
                "start": dates[0].date(), "end": end.date(),
                "cost_bps": cost_bps, "cum_cost": cum_cost, "turnovers": turnovers,
                "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0}


def report(strategy="momentum", initial=10_000.0, refresh=False):
    """Score the live record as ONE managed $10k paper portfolio vs SPY, plus a
    per-month breakdown of each basket since its own lock."""
    out_dir = os.path.join(PICKS_DIR, strategy)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json")) if os.path.isdir(out_dir) else []
    if not files:
        print(f"no locked picks for {strategy!r} yet — run: python -m backtest.tracker lock")
        return None

    recs = [json.load(open(os.path.join(out_dir, f))) for f in files]
    universe = sorted({t for rec in recs for t in rec["picks"]})   # exactly this book's names
    closes = download_panel(universe)["Close"]
    spy_close = get_prices("SPY", refresh=refresh)["Close"]

    eq, s = _simulate(recs, closes, spy_close, initial)
    if strategy in MARKET_NEUTRAL:
        # market-neutral: the honest benchmark is CASH, not SPY (beta is hedged out, so
        # "excess vs SPY" would be a category error). SPY shown only for context.
        print(f"Managed ${initial:,.0f} market-NEUTRAL paper book ({strategy}), {s['start']} -> {s['end']}:")
        print(f"  strategy : ${s['final']:>11,.0f}   ({s['ret'] * 100:+.1f}%)")
        print(f"  cash(0%) : ${initial:>11,.0f}   (+0.0%)   <- honest benchmark (rf would refine it)")
        print(f"  alpha    : {s['ret'] * 100:+.1f}%")
        print(f"  (SPY over span {s['spy_ret'] * 100:+.1f}% — context only, NOT the benchmark for a neutral book)")
        print(f"  ({len(recs)} lock(s) chained; dollar-neutral ~$5k long / $5k short, borrow not modeled)")
    else:
        print(f"Managed ${initial:,.0f} paper portfolio ({strategy}), {s['start']} -> {s['end']}:")
        print(f"  strategy : ${s['final']:>11,.0f}   ({s['ret'] * 100:+.1f}%)  NET of {s['cost_bps']:.0f}bps on traded weight")
        print(f"  SPY      : ${s['spy_final']:>11,.0f}   ({s['spy_ret'] * 100:+.1f}%)")
        print(f"  excess   : {(s['ret'] - s['spy_ret']) * 100:+.1f}%")
        print(f"  ({len(recs)} monthly lock(s) chained; rebalances into new picks each month)")
    print(f"  costs: cumulative drag {s['cum_cost'] * 100:.2f}% "
          f"(avg two-way turnover {s['avg_turnover'] * 100:.0f}%/lock incl. the initial buy-in; "
          f"taxes NOT modeled — monthly turnover is short-term gains in a real account)")

    now = closes.iloc[-1]
    rows = []
    for idx, rec in enumerate(sorted(recs, key=lambda r: r["data_asof"])):
        entry = _entry_prices(closes, rec["data_asof"], rec["picks"], rec.get("lock_prices"))
        ret = sum(w * (float(now.get(t, np.nan)) / entry[t] - 1)
                  for t, w in rec["picks"].items()
                  if np.isfinite(entry.get(t, np.nan)) and float(now.get(t, np.nan)) == float(now.get(t, np.nan)))
        rows.append({"lock_date": rec["lock_date"], "n": rec["n"],
                     "basket_%_since_lock": round(ret * 100, 2),
                     "turnover_%": round(s["turnovers"][idx] * 100)})
    print("\nPer-month basket (each since its own lock):")
    print(pd.DataFrame(rows).to_string(index=False))
    eq.to_csv(os.path.join(out_dir, "_equity.csv"))
    return eq


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "lock"
    strat = sys.argv[2] if len(sys.argv) > 2 else "momentum"
    # CLI = real monthly use -> pull fresh prices (the interactive funcs default to cache)
    if cmd == "lock":
        lock(strat, refresh=True)
    elif cmd == "report":
        report(strat, refresh=True)
    else:
        print("usage: python -m backtest.tracker [lock|report] [strategy]")
