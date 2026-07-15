# backtest/lots.py — the tax LOT LEDGER for the real account.
#
#   .venv/bin/python -m backtest.lots                          # report
#   .venv/bin/python -m backtest.lots add BUY UPRO 28.4387 4086.50 [2026-07-13] [note]
#   .venv/bin/python -m backtest.lots add SELL IEF 5.0 470.00 [date] [note]
#
# December's tax measurement, built in month one instead of reconstructed in month
# six. The FACTS are an append-only journal of real fills (backtest/lots_journal.json,
# git-committed): date, side, ticker, shares, total dollars. Everything else — open
# lots, realized gains, ST/LT classification, the December plan — is DERIVED from the
# journal on demand (the lab's standard facts-vs-views pattern).
#
# CONVENTIONS: sells consume lots HIFO (highest basis/share first) — the standing
# order matching the Specific-ID plan at Chase; the broker's confirmations remain
# authoritative, and any deliberate non-HIFO lot pick should be journaled in the note.
# A lot turns LONG-TERM the day after one full year from acquisition (IRS holding
# rule); gains on lots sold at or under a year are SHORT-TERM = ordinary income.
# WORKFLOW: after each monthly rebalance's fills, journal each trade (`lots add ...`
# with the dollars from the confirmation). Dollar amounts are exact for dollar-based
# orders; share counts to 4dp.
import json
import os
import sys
import datetime as dt

import pandas as pd

JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "lots_journal.json")
LT_DAYS = 366                                              # held MORE than one year


def _load():
    if not os.path.exists(JOURNAL_PATH):
        return []
    return json.load(open(JOURNAL_PATH))


def add(side, ticker, shares, dollars, date=None, note=""):
    side = side.upper()
    assert side in ("BUY", "SELL"), "side must be BUY or SELL"
    journal = _load()
    journal.append({"date": date or dt.date.today().isoformat(), "side": side,
                    "ticker": ticker.upper(), "shares": float(shares),
                    "dollars": float(dollars), "note": note})
    journal.sort(key=lambda x: x["date"])
    with open(JOURNAL_PATH, "w") as f:
        json.dump(journal, f, indent=1)
    print(f"journaled: {side} {ticker} {float(shares)} sh for ${float(dollars):,.2f}")


def derive(journal=None, asof=None):
    """(open_lots, realized) from the journal. open_lots: [{ticker, acquired, shares,
    basis}], realized: [{date, ticker, shares, proceeds, basis, gain, term}]. Sells
    consume HIFO. Pure — testable without the file or network."""
    journal = _load() if journal is None else journal
    lots, realized = [], []
    for fill in sorted(journal, key=lambda x: x["date"]):
        if fill["side"] == "BUY":
            lots.append({"ticker": fill["ticker"], "acquired": fill["date"],
                         "shares": fill["shares"], "basis": fill["dollars"]})
            continue
        remaining = fill["shares"]
        px = fill["dollars"] / fill["shares"]
        open_t = sorted((l for l in lots if l["ticker"] == fill["ticker"] and l["shares"] > 1e-9),
                        key=lambda l: -(l["basis"] / l["shares"]))     # HIFO
        for lot in open_t:
            if remaining <= 1e-9:
                break
            take = min(remaining, lot["shares"])
            frac_basis = lot["basis"] * take / lot["shares"]
            held = (pd.Timestamp(fill["date"]) - pd.Timestamp(lot["acquired"])).days
            realized.append({"date": fill["date"], "ticker": fill["ticker"],
                             "shares": take, "proceeds": take * px, "basis": frac_basis,
                             "gain": take * px - frac_basis,
                             "term": "LT" if held >= LT_DAYS else "ST"})
            lot["basis"] -= frac_basis
            lot["shares"] -= take
            remaining -= take
        if remaining > 1e-6:
            raise ValueError(f"SELL {fill['ticker']} {fill['shares']} exceeds open lots "
                             f"(short by {remaining:.4f} sh) — journal out of sync")
    return [l for l in lots if l["shares"] > 1e-9], realized


def report():
    open_lots, realized = derive()
    if not open_lots and not realized:
        print("journal empty — seed it with the go-live fills")
        return
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import portfolio
    prices = portfolio.fetch_prices(sorted({l["ticker"] for l in open_lots}))
    today = pd.Timestamp(dt.date.today())
    rows, lt_by_dec = [], 0.0
    for l in sorted(open_lots, key=lambda x: (x["ticker"], x["acquired"])):
        px = prices.get(l["ticker"])
        mv = l["shares"] * px if px else float("nan")
        unreal = mv - l["basis"] if px else float("nan")
        flips = pd.Timestamp(l["acquired"]) + pd.Timedelta(days=LT_DAYS)
        term = "LT" if flips <= today else "ST"
        if flips <= pd.Timestamp(f"{today.year}-12-31") and unreal == unreal and unreal > 0:
            lt_by_dec += unreal
        rows.append({"ticker": l["ticker"], "acquired": l["acquired"],
                     "shares": round(l["shares"], 4), "basis_$": round(l["basis"], 2),
                     "mkt_$": round(mv, 2) if mv == mv else "n/a",
                     "unreal_$": round(unreal, 2) if unreal == unreal else "n/a",
                     "term": term, "goes_LT": flips.date().isoformat()})
    print("OPEN LOTS:")
    print(pd.DataFrame(rows).to_string(index=False))
    yr = str(today.year)
    st = sum(r["gain"] for r in realized if r["term"] == "ST" and r["date"].startswith(yr))
    lt = sum(r["gain"] for r in realized if r["term"] == "LT" and r["date"].startswith(yr))
    print(f"\nREALIZED {yr}: short-term ${st:+,.2f} (ordinary income) | "
          f"long-term ${lt:+,.2f} (0% bracket while it lasts)")
    print(f"DECEMBER PLAN: unrealized gains on lots that will be LONG-TERM by Dec 31: "
          f"${lt_by_dec:,.2f}"
          + ("" if lt_by_dec else " — none this year (go-live lots turn LT 2027-07;"
                                 " no 0%-bracket gain-harvest available in "
                                 f"{yr}, and ST gains should NOT be volunteered)"))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add":
        add(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
            sys.argv[6] if len(sys.argv) > 6 else None,
            " ".join(sys.argv[7:]) if len(sys.argv) > 7 else "")
    else:
        report()
