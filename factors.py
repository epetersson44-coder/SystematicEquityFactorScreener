# factors.py — the 5 factor recipes, computed from the CANONICAL schema.
#
# Source-agnostic: these take a canonical fundamentals dict (from fundamentals.py)
# and never touch yfinance/SimFin specifics. ONE recipe; the source is swapped
# upstream by the router. The factor MATH is unchanged from the original yfinance
# version — only the input contract moved to the canonical schema.

import numpy as np

from fundamentals import enterprise_value


def get_ev_ebit(f):
    ev, ebit = enterprise_value(f), f.get("ebit")
    if ev and ebit and ebit > 0:
        return ev / ebit
    return None


def get_price_fcf(f):
    mc, fcf = f.get("market_cap"), f.get("free_cash_flow")
    if mc and fcf and fcf > 0:
        return mc / fcf
    return None


def get_roic(f):
    ebit, tax_rate, equity = f.get("ebit"), f.get("tax_rate"), f.get("equity")
    if ebit is None or tax_rate is None or equity is None:
        return None
    nopat = ebit * (1 - tax_rate)
    # Invested capital = debt + equity − cash. (Assets − liabilities is just equity
    # → that's ROE, and leverage would inflate the score.)
    invested_capital = (f.get("total_debt") or 0) + equity - (f.get("cash") or 0)
    if invested_capital and invested_capital > 0:
        return nopat / invested_capital
    return None


def get_gm_stability(f):
    rev = f.get("revenue_history") or []
    gp = f.get("gross_profit_history") or []
    n = min(len(rev), len(gp))
    # Skip NaN years (yfinance leaves gaps) — the original pandas .std() did this via
    # skipna; np.std does not. ddof=1 (sample std) matches the original too.
    margins = [gp[i] / rev[i] for i in range(n)
               if rev[i] and rev[i] == rev[i] and gp[i] == gp[i]]
    return float(np.std(margins, ddof=1)) if len(margins) >= 2 else None


def get_net_debt_ebitda(f):
    total_debt, cash, ebitda = f.get("total_debt"), f.get("cash"), f.get("ebitda")
    # Missing data must stay missing — defaulting debt to 0 made data gaps score as
    # best-in-class leverage.
    if total_debt is None or cash is None:
        return None
    if ebitda and ebitda > 0:
        return (total_debt - cash) / ebitda
    return None


def calculate_factors(f):
    """Compute all 5 factors from a canonical fundamentals dict."""
    return {
        "ticker":          f.get("ticker"),
        "ev_ebit":         get_ev_ebit(f),
        "price_fcf":       get_price_fcf(f),
        "roic":            get_roic(f),
        "gm_stability":    get_gm_stability(f),
        "net_debt_ebitda": get_net_debt_ebitda(f),
    }


if __name__ == "__main__":
    from fundamentals import get_fundamentals
    print(calculate_factors(get_fundamentals("CALM", source="yfinance")))
