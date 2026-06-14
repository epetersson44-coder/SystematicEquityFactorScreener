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

from backtest.universe import get_universe
from backtest.data import get_prices
from backtest.strategy import CrossSectionalMomentum

PICKS_DIR = os.path.join(os.path.dirname(__file__), "picks")


def momentum_picks(refresh=False):
    """Today's cross-sectional momentum basket: (weights, prices_now, data_asof)."""
    closes = get_universe("sp500", refresh=refresh)["Close"]
    i = len(closes) - 1
    weights = CrossSectionalMomentum().rank(closes, i)
    if weights is None:
        raise RuntimeError("not enough history to rank the universe")
    return weights, closes.iloc[i], closes.index[i].date().isoformat()


def lock(strategy="momentum", refresh=False):
    """Compute today's picks and write them to an immutable dated file. Refuses to
    overwrite an existing lock — picks, once made, never change."""
    if strategy != "momentum":
        raise ValueError(f"unknown strategy {strategy!r} (only 'momentum' for now)")
    weights, prices, asof = momentum_picks(refresh=refresh)
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
    out_dir = os.path.join(PICKS_DIR, strategy)
    os.makedirs(out_dir, exist_ok=True)
    month = rec["lock_date"][:7]                          # one lock per calendar month
    existing = [f for f in os.listdir(out_dir) if f.startswith(month) and f.endswith(".json")]
    if existing:
        raise FileExistsError(f"already locked for {month}: {existing[0]} — one lock per month, picks are immutable")
    path = os.path.join(out_dir, f"{rec['lock_date']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"locked {rec['n']} {strategy} picks (data as of {asof}) -> {os.path.relpath(path)}")
    top = sorted(rec["picks"], key=lambda t: -rec["lock_prices"][t])[:8]
    print(f"  sample names: {', '.join(sorted(rec['picks'])[:12])} ...")
    return rec


def _simulate(recs, closes, spy_close, initial=10_000.0):
    """ONE managed paper portfolio: start with `initial`, hold each month's locked
    basket until the next lock, then rebalance into the new picks; carry the value
    forward. Returns (equity Series, summary). Prices come from the universe panel,
    so any pick's value at any date is looked up consistently.

    recs: list of {data_asof, picks} (one per monthly lock). closes: (date x ticker)
    panel. spy_close: Series of SPY close by date."""
    recs = sorted(recs, key=lambda r: r["data_asof"])
    dates = [pd.to_datetime(r["data_asof"]) for r in recs]
    end = closes.index[-1]
    bounds = dates + [end]                                # each basket runs lock_k -> lock_{k+1}
    value = initial
    curve = {dates[0]: initial}
    for k, rec in enumerate(recs):
        d0, d1 = bounds[k], bounds[k + 1]
        if d1 <= d0:
            continue
        seg_ret = 0.0
        for t, w in rec["picks"].items():
            if t not in closes.columns:
                continue
            p0, p1 = closes.at[d0, t], closes.at[d1, t]
            if p0 == p0 and p1 == p1 and p0 > 0:          # both prices present
                seg_ret += w * (p1 / p0 - 1)
        value *= (1 + seg_ret)
        curve[d1] = value
    eq = pd.Series(curve).sort_index()
    s0 = float(spy_close.loc[spy_close.index >= dates[0]].iloc[0])
    s1 = float(spy_close.iloc[-1])
    return eq, {"final": value, "ret": value / initial - 1,
                "spy_ret": s1 / s0 - 1, "spy_final": initial * (s1 / s0),
                "start": dates[0].date(), "end": end.date()}


def report(strategy="momentum", initial=10_000.0, refresh=False):
    """Score the live record as ONE managed $10k paper portfolio vs SPY, plus a
    per-month breakdown of each basket since its own lock."""
    out_dir = os.path.join(PICKS_DIR, strategy)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json")) if os.path.isdir(out_dir) else []
    if not files:
        print(f"no locked picks for {strategy!r} yet — run: python -m backtest.tracker lock")
        return None

    closes = get_universe("sp500", refresh=refresh)["Close"]
    spy_close = get_prices("SPY", refresh=refresh)["Close"]
    recs = [json.load(open(os.path.join(out_dir, f))) for f in files]

    eq, s = _simulate(recs, closes, spy_close, initial)
    print(f"Managed ${initial:,.0f} paper portfolio ({strategy}), {s['start']} -> {s['end']}:")
    print(f"  strategy : ${s['final']:>11,.0f}   ({s['ret'] * 100:+.1f}%)")
    print(f"  SPY      : ${s['spy_final']:>11,.0f}   ({s['spy_ret'] * 100:+.1f}%)")
    print(f"  excess   : {(s['ret'] - s['spy_ret']) * 100:+.1f}%")
    print(f"  ({len(recs)} monthly lock(s) chained; rebalances into new picks each month)")

    now = closes.iloc[-1]
    rows = []
    for rec in recs:
        ret = sum(w * (float(now.get(t, np.nan)) / rec["lock_prices"][t] - 1)
                  for t, w in rec["picks"].items() if float(now.get(t, np.nan)) == float(now.get(t, np.nan)))
        rows.append({"lock_date": rec["lock_date"], "n": rec["n"], "basket_%_since_lock": round(ret * 100, 2)})
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
