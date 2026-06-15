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
DAILY_CSV = os.path.join(HERE, "desk_daily.csv")        # full DAILY equity curve per book + SPY


def _sector_map():
    """ticker -> sector (SimFin), best-effort. Empty dict if SimFin isn't available."""
    try:
        from fundamentals import _simfin_load
        return _simfin_load()["sector"]
    except Exception:
        return {}


def _book(strategy, smap):
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

    pos, sec = [], {}
    for t in names:
        e = rec["lock_prices"][t]
        c = float(now.get(t, np.nan))
        if c != c:
            continue
        r = c / e - 1
        s = smap.get(t) or "Other"
        pos.append({"t": t, "e": round(e, 2), "c": round(c, 2), "s": s,
                    "r": round(r * 100, 2), "p": round(10_000 * rec["picks"][t] * r, 2)})
        sec[s] = sec.get(s, 0.0) + abs(rec["picks"][t])         # abs => shorts count too

    spy = get_prices("SPY", refresh=True)["Close"]
    ss = spy[spy.index >= lock]
    spyeq = 10_000 * ss / ss.iloc[0]

    diff = None
    if prev:                                            # month-over-month turnover
        cur, old = set(rec["picks"]), set(prev["picks"])
        diff = {"prev_lock": prev["lock_date"], "added": sorted(cur - old),
                "dropped": sorted(old - cur), "held": sorted(cur & old),
                "turnover": round(len(cur ^ old) / max(len(cur), 1) * 100)}

    tot = sum(sec.values()) or 1
    sectors = sorted(({"s": k, "w": round(v / tot * 100, 1)} for k, v in sec.items()),
                     key=lambda x: -x["w"])
    return {
        "strategy": strategy, "lock": rec["lock_date"], "asof": rec["data_asof"],
        "neutral": strategy in T.MARKET_NEUTRAL,
        "final": round(float(eq.iloc[-1]), 2),
        "ret": round(float(eq.iloc[-1] / 10_000 - 1) * 100, 2),
        "spyret": round(float(spyeq.iloc[-1] / 10_000 - 1) * 100, 2),
        "pos": sorted(pos, key=lambda x: -x["r"]),
        "sectors": sectors,
        "curve": [{"d": i.strftime("%m/%d"), "v": round(float(v), 1)} for i, v in eq.items()],
        "spyc": [{"d": i.strftime("%m/%d"), "v": round(float(v), 1)} for i, v in spyeq.items()],
        "diff": diff,
    }


def _daily_book(strategy):
    """Reconstruct ONE book's full daily equity curve, chained across every lock (the
    managed $10k that rebalances at each lock), from the immutable locks + daily prices."""
    d = os.path.join(T.PICKS_DIR, strategy)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        return None
    recs = sorted((json.load(open(os.path.join(d, f))) for f in files), key=lambda r: r["data_asof"])
    names = sorted({t for r in recs for t in r["picks"]})
    panel = download_panel(names)["Close"]
    bounds = [pd.to_datetime(r["data_asof"]) for r in recs] + [panel.index[-1]]
    value, pieces = 10_000.0, []
    for k, rec in enumerate(recs):
        seg = panel[(panel.index >= bounds[k]) & (panel.index <= bounds[k + 1])]
        if seg.empty:
            continue
        shares = {t: value * rec["picks"][t] / rec["lock_prices"][t] for t in rec["picks"]}
        eq = seg.mul(pd.Series(shares)).sum(axis=1).dropna()
        if eq.empty:
            continue
        last = k == len(recs) - 1
        pieces.append(eq if last else eq.iloc[:-1])          # drop the shared boundary day
        value = float(eq.iloc[-1])                           # carry value across the rebalance
    return pd.concat(pieces) if pieces else None


def daily_dataset():
    """Write desk_daily.csv: every trading day's equity per book + SPY, rebuilt from source.
    Deterministic, so it fills in EVERY day even on months nothing was run — daily-resolution
    research data for free, no daily job needed."""
    cols = {}
    for b in BOOKS:
        try:
            s = _daily_book(b)
            if s is not None:
                cols[b] = s
        except Exception as e:
            print(f"[desk] daily {b}: {e}")
    if not cols:
        return 0
    first = min(s.index[0] for s in cols.values())
    spy = get_prices("SPY", refresh=True)["Close"]
    ss = spy[spy.index >= first]
    cols["SPY"] = 10_000 * ss / ss.iloc[0]
    df = pd.DataFrame(cols).sort_index().round(2)
    df.index.name = "date"
    df.to_csv(DAILY_CSV)
    return len(df)


def build():
    """Regenerate desk_data.json, append this month to desk_history.json, and rebuild the
    daily dataset (desk_daily.csv). Returns the blob."""
    smap = _sector_map()
    data = {"generated": dt.date.today().isoformat()}
    for b in BOOKS:
        try:
            r = _book(b, smap)
            if r:
                data[b] = r
        except Exception as e:                          # one bad book shouldn't sink the desk
            print(f"[desk] {b}: {e}")
    data["daily_rows"] = daily_dataset()                # full daily curve -> desk_daily.csv
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
    print(f"\n-> {os.path.basename(DATA_JSON)} + {os.path.basename(HISTORY)} ({n} month(s)) "
          f"+ {os.path.basename(DAILY_CSV)} ({d.get('daily_rows', 0)} daily rows)")
    for b in BOOKS:
        if b in d and d[b]["sectors"]:
            top = ", ".join(f"{s['s']} {s['w']:.0f}%" for s in d[b]["sectors"][:4])
            print(f"  {b} sectors: {top}")
