# fundamentals.py — the data ROUTER / translator for the screener.
#
# ONE canonical schema; swappable SOURCE. The factor recipes (factors.py) consume
# the canonical dict and never know or care where the numbers came from. This is a
# ROUTER, not a blender: you pick ONE source per run. Never mix vintages across a
# backtest — look-ahead is a TIME problem, and normalizing units can't fix a time
# problem. (yfinance = today's data; SimFin free = ~12-mo delayed point-in-time.)
#
# The CONTRACT (enforced by get_fundamentals -> _normalize): every canonical dict
# has the FULL schema below, every scalar is a real number or None (NEVER NaN), and
# every list field is NaN-free. Downstream code only ever sees clean values — the
# source's quirks (missing rows, NaN latest values, stale reports) are scrubbed here,
# at the single boundary, so a bug can't silently poison a factor.
#
#   ticker, sector, report_date,
#   market_cap, enterprise_value, total_debt, cash, equity,
#   ebit, ebitda, tax_rate, free_cash_flow, revenue,
#   total_assets, total_liabilities, current_assets, current_liabilities,
#   retained_earnings,
#   revenue_history / gross_profit_history  (lists, newest first — for GM stability)
#
# Sources:
#   "yfinance" — per-ticker live fetch. Current, keyless, slow/fragile at scale.
#   "simfin"   — bulk download ONCE, then per-ticker lookup. Fast at scale; free but
#                ~12-mo delayed; needs a free API key (env SIMFIN_API_KEY).

import os

import pandas as pd

SCALAR_KEYS = (
    "market_cap", "enterprise_value", "total_debt", "cash", "equity",
    "ebit", "ebitda", "tax_rate", "free_cash_flow", "revenue",
    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
    "retained_earnings",
)
META_KEYS = ("ticker", "sector", "report_date")
LIST_KEYS = ("revenue_history", "gross_profit_history")
CANONICAL_KEYS = META_KEYS + SCALAR_KEYS + LIST_KEYS

TAX_RATE_MAX = 0.60                                            # clamp: real rates are 0–~40%


def _isnan(x):
    return isinstance(x, float) and x != x                    # NaN != NaN


def _blank(ticker):
    """A canonical dict with the full schema, everything empty."""
    d = {k: None for k in CANONICAL_KEYS}
    d["ticker"] = ticker
    d["revenue_history"] = []
    d["gross_profit_history"] = []
    return d


def _clamp_tax(rate):
    """A sane effective tax rate, or None. Loss-making firms give garbage -tax/pretax."""
    if rate is None or _isnan(rate):
        return None
    return min(max(float(rate), 0.0), TAX_RATE_MAX)


def _normalize(f):
    """Enforce the contract: full schema, NaN -> None for scalars, NaN stripped from
    lists, tax_rate clamped. The single place source quirks are scrubbed."""
    out = _blank(f.get("ticker"))
    for k in CANONICAL_KEYS:
        if k not in f:
            continue
        v = f[k]
        if k in LIST_KEYS:
            out[k] = [float(x) for x in (v or []) if x is not None and not _isnan(x)]
        elif k in META_KEYS:
            out[k] = None if _isnan(v) else v
        else:
            out[k] = None if (v is None or _isnan(v)) else float(v)
    out["tax_rate"] = _clamp_tax(out["tax_rate"])
    return out


def validate_canonical(f):
    """Assert a canonical dict conforms to the contract. Raises AssertionError."""
    missing = [k for k in CANONICAL_KEYS if k not in f]
    assert not missing, f"missing canonical keys: {missing}"
    for k in LIST_KEYS:
        assert isinstance(f[k], list), f"{k} must be a list"
        assert all(isinstance(x, float) and not _isnan(x) for x in f[k]), f"{k} has NaN/non-float"
    for k in SCALAR_KEYS:
        v = f[k]
        assert v is None or (isinstance(v, float) and not _isnan(v)), f"{k} must be float|None"
    return True


# ----------------------------------------------------------------- yfinance adapter
def _yf_row(df, *labels):
    """First matching row's latest NON-NaN value — yfinance row labels shift between
    tickers, and a present row often has a NaN latest value (callers fall back to the
    info dict). Treat NaN like missing."""
    for label in labels:
        if label in df.index:
            v = df.loc[label]
            val = v.iloc[0] if hasattr(v, "iloc") else v
            if val is not None and not _isnan(val):
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
    report_date = str(income.columns[0].date()) if len(getattr(income, "columns", [])) else None

    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "report_date": report_date,
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
        "revenue": _yf_row(income, "Total Revenue"),
        "revenue_history": rev_hist,
        "gross_profit_history": gp_hist,
        "total_assets": _yf_row(balance, "Total Assets"),
        "total_liabilities": _yf_row(balance, "Total Liabilities Net Minority Interest", "Total Liabilities"),
        "current_assets": _yf_row(balance, "Current Assets", "Total Current Assets"),
        "current_liabilities": _yf_row(balance, "Current Liabilities", "Total Current Liabilities"),
        "retained_earnings": _yf_row(balance, "Retained Earnings"),
    }


# ------------------------------------------------------------------- simfin adapter
_SIMFIN = {}


def _simfin_load():
    """Load SimFin bulk datasets ONCE (cached to ~/simfin_data + in memory), sorted."""
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
    for name, fn in [("income", sf.load_income), ("balance", sf.load_balance),
                     ("cashflow", sf.load_cashflow)]:
        _SIMFIN[name] = fn(variant="annual", market="us").sort_index()    # sorted => fast .loc, no warnings
    _SIMFIN["prices"] = sf.load_shareprices(variant="latest", market="us").sort_index()
    # ticker -> GICS-style Sector (for excluding financials/REITs)
    companies = sf.load_companies(market="us")
    id_to_sector = sf.load_industries()["Sector"].to_dict()
    _SIMFIN["sector"] = {t: id_to_sector.get(iid) for t, iid in companies["IndustryId"].items()}
    return _SIMFIN


def _latest(df, ticker):
    """Most recent annual row for `ticker` from a SimFin (Ticker, Report Date) frame."""
    try:
        rows = df.loc[ticker]                                  # KeyError if not covered
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
        return _blank(ticker)                                  # not covered -> full blank schema

    def g(row, col):
        return float(row[col]) if (row is not None and col in row and row[col] == row[col]) else None

    ebit = g(inc, "Operating Income (Loss)")
    da = g(inc, "Depreciation & Amortization")
    pretax = g(inc, "Pretax Income (Loss)")
    tax = g(inc, "Income Tax (Expense) Benefit, Net")          # negative = expense
    total_debt = (g(bal, "Short Term Debt") or 0.0) + (g(bal, "Long Term Debt") or 0.0)
    cash = g(bal, "Cash, Cash Equivalents & Short Term Investments")
    ocf = g(cf, "Net Cash from Operating Activities")
    capex = g(cf, "Change in Fixed Assets & Intangibles")      # negative
    market_cap = _simfin_market_cap(d["prices"], ticker)
    report_date = inc.name if hasattr(inc, "name") else None   # the latest Report Date

    return {
        "ticker": ticker,
        "sector": d["sector"].get(ticker),
        "report_date": str(report_date.date()) if hasattr(report_date, "date") else str(report_date),
        "market_cap": market_cap,
        "enterprise_value": (market_cap + total_debt - cash) if (market_cap and cash is not None) else None,
        "total_debt": total_debt,
        "cash": cash,
        "equity": g(bal, "Total Equity"),
        "ebit": ebit,
        "ebitda": (ebit + da) if (ebit is not None and da is not None) else None,
        "tax_rate": (-tax / pretax) if (tax is not None and pretax) else None,
        "free_cash_flow": (ocf + capex) if (ocf is not None and capex is not None) else None,
        "revenue": g(inc, "Revenue"),
        "revenue_history": _simfin_history(d["income"], ticker, "Revenue"),
        "gross_profit_history": _simfin_history(d["income"], ticker, "Gross Profit"),
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
    if not _isnan(close) and not _isnan(shares) and shares:
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
    """Canonical fundamentals dict for `ticker` from the chosen source — full schema,
    NaN-free, contract-enforced. Pick ONE source per run; never blend across a
    backtest."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r} (use {sorted(_SOURCES)})")
    return _normalize(_SOURCES[source](ticker))


def enterprise_value(f):
    """EV, preferring a source-provided value, else derived (market cap + debt − cash)."""
    if f.get("enterprise_value") is not None:
        return f["enterprise_value"]
    mc, td, cash = f.get("market_cap"), f.get("total_debt"), f.get("cash")
    return (mc + td - cash) if (mc is not None and td is not None and cash is not None) else None


if __name__ == "__main__":
    f = get_fundamentals("CALM", source="yfinance")
    validate_canonical(f)
    for k, v in f.items():
        print(f"{k:20} {v if not isinstance(v, list) else f'[{len(v)} values] {v[:3]}'}")
