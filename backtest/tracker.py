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
    path = os.path.join(out_dir, f"{rec['lock_date']}.json")
    if os.path.exists(path):
        raise FileExistsError(f"already locked today: {path} — picks are immutable")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"locked {rec['n']} {strategy} picks (data as of {asof}) -> {os.path.relpath(path)}")
    top = sorted(rec["picks"], key=lambda t: -rec["lock_prices"][t])[:8]
    print(f"  sample names: {', '.join(sorted(rec['picks'])[:12])} ...")
    return rec


def report(strategy="momentum", refresh=False):
    """Score every locked pick-set's return-since-lock vs SPY over the same span."""
    out_dir = os.path.join(PICKS_DIR, strategy)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json")) if os.path.isdir(out_dir) else []
    if not files:
        print(f"no locked picks for {strategy!r} yet — run: python -m backtest.tracker lock")
        return None

    now = get_universe("sp500", refresh=refresh)["Close"].iloc[-1]
    spy_now = float(get_prices("SPY", refresh=refresh)["Close"].iloc[-1])

    rows = []
    for fname in files:
        rec = json.load(open(os.path.join(out_dir, fname)))
        ret, missing = 0.0, []
        for t, w in rec["picks"].items():
            p1 = float(now.get(t, float("nan")))
            if p1 == p1:                                  # not NaN
                ret += w * (p1 / rec["lock_prices"][t] - 1)
            else:
                missing.append(t)                         # delisted since lock (rare)
        spy_ret = spy_now / rec["spy_lock"] - 1
        rows.append({
            "lock_date": rec["lock_date"], "n": rec["n"],
            "strat_%": round(ret * 100, 2), "spy_%": round(spy_ret * 100, 2),
            "excess_%": round((ret - spy_ret) * 100, 2),
            "missing": len(missing),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "_track_record.csv"), index=False)
    print(f"\n(saved -> {os.path.relpath(os.path.join(out_dir, '_track_record.csv'))}; "
          f"first lock {files[0][:-5]}, picks are immutable + committed)")
    return df


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "lock"
    strat = sys.argv[2] if len(sys.argv) > 2 else "momentum"
    if cmd == "lock":
        lock(strat)
    elif cmd == "report":
        report(strat)
    else:
        print("usage: python -m backtest.tracker [lock|report] [strategy]")
