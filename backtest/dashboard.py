# backtest/dashboard.py — the paper-trading "desk": data engine for the chat widget + journal.
#
# Reads every book's locked picks, marks them to TODAY's prices, and produces:
#   - desk_data.json     the blob the chat dashboard widget renders (per book: positions,
#                        equity curve vs SPY, and the month-over-month diff)
#   - desk_history.json  an APPEND-ONLY monthly dataset — our own accumulating record, so we
#                        can chart the track record and analyse it over time
#   - a printed month-over-month diff (added / held / dropped) per book
#
# Run by /picks after locking (also runnable standalone): python -m backtest.dashboard

import os
import json
import datetime as dt

import numpy as np
import pandas as pd

import backtest.tracker as T
from backtest.universe import download_panel
from backtest.data import get_prices

BOOKS = ["momentum", "factor", "factor_ls"]
HERE = os.path.dirname(__file__)
DATA_JSON = os.path.join(HERE, "desk_data.json")        # latest snapshot (for the widget)
HISTORY = os.path.join(HERE, "desk_history.json")       # accumulating monthly dataset


def _book(strategy):
    """Mark one book's latest locked basket to live prices; return the widget blob + diff."""
    d = os.path.join(T.PICKS_DIR, strategy)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        return None
    recs = [json.load(open(os.path.join(d, f))) for f in files]
    rec, prev = recs[-1], (recs[-2] if len(recs) >= 2 else None)
    names = list(rec["picks"])
    panel = download_panel(names)["Close"]
    lock = pd.to_datetime(rec["data_asof"])
    seg = panel[panel.index >= lock]
    shares = {t: 10_000 * rec["picks"][t] / rec["lock_prices"][t] for t in names}
    eq = seg.mul(pd.Series(shares)).sum(axis=1).dropna()
    now = seg.iloc[-1]

    pos = []
    for t in names:
        e = rec["lock_prices"][t]
        c = float(now.get(t, np.nan))
        if c != c:
            continue
        r = c / e - 1
        pos.append({"t": t, "e": round(e, 2), "c": round(c, 2),
                    "r": round(r * 100, 2), "p": round(10_000 * rec["picks"][t] * r, 2)})

    spy = get_prices("SPY", refresh=True)["Close"]
    ss = spy[spy.index >= lock]
    spyeq = 10_000 * ss / ss.iloc[0]

    diff = None
    if prev:                                            # month-over-month turnover
        cur, old = set(rec["picks"]), set(prev["picks"])
        diff = {"prev_lock": prev["lock_date"], "added": sorted(cur - old),
                "dropped": sorted(old - cur), "held": sorted(cur & old)}

    return {
        "strategy": strategy, "lock": rec["lock_date"], "asof": rec["data_asof"],
        "neutral": strategy in T.MARKET_NEUTRAL,
        "final": round(float(eq.iloc[-1]), 2),
        "ret": round(float(eq.iloc[-1] / 10_000 - 1) * 100, 2),
        "spyret": round(float(spyeq.iloc[-1] / 10_000 - 1) * 100, 2),
        "pos": sorted(pos, key=lambda x: -x["r"]),
        "curve": [{"d": i.strftime("%m/%d"), "v": round(float(v), 1)} for i, v in eq.items()],
        "spyc": [{"d": i.strftime("%m/%d"), "v": round(float(v), 1)} for i, v in spyeq.items()],
        "diff": diff,
    }


def build():
    """Regenerate desk_data.json and append this month to desk_history.json. Returns the blob."""
    data = {"generated": dt.date.today().isoformat()}
    for b in BOOKS:
        try:
            r = _book(b)
            if r:
                data[b] = r
        except Exception as e:                          # one bad book shouldn't sink the desk
            print(f"[desk] {b}: {e}")
    json.dump(data, open(DATA_JSON, "w"), indent=2)

    hist = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    month = data["generated"][:7]
    snap = {"month": month, "generated": data["generated"], "books": {
        b: {"lock": data[b]["lock"], "final": data[b]["final"], "ret": data[b]["ret"],
            "spyret": data[b]["spyret"], "picks": [p["t"] for p in data[b]["pos"]]}
        for b in BOOKS if b in data}}
    hist = [h for h in hist if h["month"] != month] + [snap]   # one row per month (replace)
    json.dump(sorted(hist, key=lambda h: h["month"]), open(HISTORY, "w"), indent=2)
    return data


if __name__ == "__main__":
    d = build()
    for b in BOOKS:
        if b not in d:
            continue
        x = d[b]
        bench = "cash" if x["neutral"] else "SPY"
        print(f"\n{b:9s} ${x['final']:>10,.0f}  ({x['ret']:+.2f}%)  vs {bench} {x['spyret']:+.2f}%")
        if x["diff"]:
            print(f"  vs {x['diff']['prev_lock']}: +{len(x['diff']['added'])} added "
                  f"{x['diff']['added'][:6]}, -{len(x['diff']['dropped'])} dropped "
                  f"{x['diff']['dropped'][:6]}, {len(x['diff']['held'])} held")
        else:
            print("  (first lock — month-over-month diff starts next month)")
    n = len(json.load(open(HISTORY)))
    print(f"\n-> wrote {os.path.basename(DATA_JSON)} + {os.path.basename(HISTORY)} ({n} month(s) of history)")
