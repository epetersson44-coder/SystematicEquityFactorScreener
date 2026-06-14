# fundamentals.py — the data ROUTER / translator for the screener.
#
# ONE canonical schema; swappable SOURCE. The factor recipes (factors.py) consume
# the canonical dict and never know or care where the numbers came from. This is a
# ROUTER, not a blender: you pick ONE source per run. Never mix vintages across a
# backtest — look-ahead is a TIME problem, and normalizing units can't fix a time
# problem. (yfinance = today's data; SimFin free = ~12-mo delayed point-in-time.)
#
# Canonical schema — one dict per company (all values in the statements' native $):
#   ticker, market_cap, enterprise_value, total_debt, cash, equity,
#   ebit, ebitda, tax_rate, free_cash_flow,
#   revenue_history / gross_profit_history  (lists, newest first — for GM stability)
#
# Sources:
#   "yfinance" — per-ticker live fetch. Current, keyless, but slow/fragile at scale.
#   "simfin"   — bulk download ONCE, then per-ticker lookup. Fast at scale; free but
#                ~12-mo delayed; needs a free API key (env SIMFIN_API_KEY).

import os

import pandas as pd


# ----------------------------------------------------------------- yfinance adapter
def _yf_row(df, *labels):
    """First matching row's latest NON-NaN value — yfinance row labels shift between
    tickers, AND a present row often has a NaN latest value (so callers can fall back
    to the info dict). Treat NaN like missing."""
    for label in labels:
        if label in df.index:
            v = df.loc[label]
            val = v.iloc[0] if hasattr(v, "iloc") else v
            if val is not None and val == val:                # val == val is False for NaN
                return val
    return None


def yfinance_fundamentals(ticker):
    """Canonical fundamentals for one ticker via yfinance (reuses fetch.fetch_all)."""
    from fetch import fetch_all
    data = fetch_all(ticker)
    income, balance, cashflow, info = data["income"], data["balance"], data["cashflow"], data["info"]

    rev_hist = list(income.loc["Total Revenue"]) if "Total Revenue" in income.index else []
    gp_hist = list(income.loc["Gross Profit"]) if "Gross Profit" in income.index else []
    total_debt = _yf_row(balance, "Total Debt")
    cash = _yf_row(balance, "Cash And Cash Equivalents",
                   "Cash Cash Equivalents And Short Term Investments")

    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),          # yfinance gives it directly
        "total_debt": total_debt if total_debt is not None else info.get("totalDebt"),
        "cash": cash if cash is not None else info.get("totalCash"),
        "equity": _yf_row(balance, "Stockholders Equity", "Common Stock Equity",
                          "Total Equity Gross Minority Interest"),
        "ebit": _yf_row(income, "EBIT"),
        "ebitda": _yf_row(income, "EBITDA"),
        "tax_rate": _yf_row(income, "Tax Rate For Calcs"),
        "free_cash_flow": _yf_row(cashflow, "Free Cash Flow"),
        "revenue_history": rev_hist,
        "gross_profit_history": gp_hist,
        # distress-screen (Altman Z) inputs:
        "revenue": rev_hist[0] if rev_hist else None,
        "total_assets": _yf_row(balance, "Total Assets"),
        "total_liabilities": _yf_row(balance, "Total Liabilities Net Minority Interest", "Total Liabilities"),
        "current_assets": _yf_row(balance, "Current Assets", "Total Current Assets"),
        "current_liabilities": _yf_row(balance, "Current Liabilities", "Total Current Liabilities"),
        "retained_earnings": _yf_row(balance, "Retained Earnings"),
    }


# ------------------------------------------------------------------- simfin adapter
# NOTE: column names below are SimFin's documented standard names but are UNVERIFIED
# until we load real data with a registered key. Calibrate once SIMFIN_API_KEY works.
_SIMFIN = {}


def _simfin_load():
    """Load SimFin bulk datasets ONCE (cached to ~/simfin_data + in memory)."""
    if _SIMFIN:
        return _SIMFIN
    import simfin as sf
    from dotenv import load_dotenv
    load_dotenv()                                             # pick up .env on any entry point
    key = os.environ.get("SIMFIN_API_KEY")
    if not key:
        raise RuntimeError(
            "SimFin needs a free API key: register at simfin.com, then put "
            "SIMFIN_API_KEY=<key> in this repo's .env (and `pip install simfin`).")
    sf.set_api_key(key)
    sf.set_data_dir(os.path.expanduser("~/simfin_data"))
    _SIMFIN["income"] = sf.load_income(variant="annual", market="us")
    _SIMFIN["balance"] = sf.load_balance(variant="annual", market="us")
    _SIMFIN["cashflow"] = sf.load_cashflow(variant="annual", market="us")
    _SIMFIN["prices"] = sf.load_shareprices(variant="latest", market="us")
    # ticker -> GICS-style Sector (for excluding financials/REITs)
    companies = sf.load_companies(market="us")
    id_to_sector = sf.load_industries()["Sector"].to_dict()
    _SIMFIN["sector"] = {t: id_to_sector.get(iid) for t, iid in companies["IndustryId"].items()}
    return _SIMFIN


def _latest(df, ticker):
    """Most recent annual row for `ticker` from a SimFin (Ticker, Report Date) frame."""
    try:
        rows = df.loc[ticker]                                  # KeyError if not covered (O(log n))
    except KeyError:
        return None
    if isinstance(rows, pd.Series):                            # single report date
        return rows
    return rows.sort_index().iloc[-1]                          # latest report date


def simfin_fundamentals(ticker):
    """Canonical fundamentals for one ticker from the SimFin bulk datasets
    (column names calibrated against real SimFin US data, 2026-06-14)."""
    d = _simfin_load()
    inc, bal, cf = _latest(d["income"], ticker), _latest(d["balance"], ticker), _latest(d["cashflow"], ticker)
    if inc is None or bal is None:
        return {"ticker": ticker}                              # not covered -> factors go NaN

    def g(row, col):
        return float(row[col]) if (row is not None and col in row and row[col] == row[col]) else None

    ebit = g(inc, "Operating Income (Loss)")
    da = g(inc, "Depreciation & Amortization")                 # D&A is on SimFin's income stmt
    pretax = g(inc, "Pretax Income (Loss)")
    tax = g(inc, "Income Tax (Expense) Benefit, Net")          # negative = expense
    total_debt = (g(bal, "Short Term Debt") or 0.0) + (g(bal, "Long Term Debt") or 0.0)
    cash = g(bal, "Cash, Cash Equivalents & Short Term Investments")
    ocf = g(cf, "Net Cash from Operating Activities")
    capex = g(cf, "Change in Fixed Assets & Intangibles")      # negative
    market_cap = _simfin_market_cap(d["prices"], ticker)

    return {
        "ticker": ticker,
        "sector": d["sector"].get(ticker),
        "market_cap": market_cap,
        "enterprise_value": (market_cap + total_debt - cash) if (market_cap and cash is not None) else None,
        "total_debt": total_debt,
        "cash": cash,
        "equity": g(bal, "Total Equity"),
        "ebit": ebit,
        "ebitda": (ebit + da) if (ebit is not None and da is not None) else None,
        "tax_rate": (-tax / pretax) if (tax is not None and pretax) else None,
        "free_cash_flow": (ocf + capex) if (ocf is not None and capex is not None) else None,
        "revenue_history": _simfin_history(d["income"], ticker, "Revenue"),
        "gross_profit_history": _simfin_history(d["income"], ticker, "Gross Profit"),
        # distress-screen (Altman Z) inputs:
        "revenue": g(inc, "Revenue"),
        "total_assets": g(bal, "Total Assets"),
        "total_liabilities": g(bal, "Total Liabilities"),
        "current_assets": g(bal, "Total Current Assets"),
        "current_liabilities": g(bal, "Total Current Liabilities"),
        "retained_earnings": g(bal, "Retained Earnings"),
    }


def _simfin_market_cap(prices, ticker):
    """Latest close x shares outstanding (both from SimFin's shareprices dataset)."""
    try:
        rows = prices.loc[ticker]
    except KeyError:
        return None
    last = rows if isinstance(rows, pd.Series) else rows.sort_index().iloc[-1]
    close, shares = last.get("Close"), last.get("Shares Outstanding")
    if close == close and shares == shares and shares:        # not NaN
        return float(close) * float(shares)
    return None


def _simfin_history(df, ticker, col):
    try:
        s = df.loc[ticker][col].dropna()
        return list(s.iloc[::-1])                              # newest first
    except Exception:
        return []


# ------------------------------------------------------------------------- router
_SOURCES = {"yfinance": yfinance_fundamentals, "simfin": simfin_fundamentals}


def get_fundamentals(ticker, source="yfinance"):
    """Canonical fundamentals dict for `ticker` from the chosen source. Pick ONE
    source per run — never blend across a backtest."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r} (use {sorted(_SOURCES)})")
    return _SOURCES[source](ticker)


def enterprise_value(f):
    """EV, preferring a source-provided value, else derived (market cap + debt − cash)."""
    if f.get("enterprise_value") is not None:
        return f["enterprise_value"]
    mc, td, cash = f.get("market_cap"), f.get("total_debt"), f.get("cash")
    return (mc + td - cash) if (mc is not None and td is not None and cash is not None) else None


if __name__ == "__main__":
    f = get_fundamentals("CALM", source="yfinance")
    for k, v in f.items():
        print(f"{k:20} {v if not isinstance(v, list) else f'[{len(v)} values] {v[:3]}'}")
