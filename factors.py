# factors.py — the 5 factor recipes, computed from the CANONICAL schema.
#
# Source-agnostic: these take a canonical fundamentals dict (from fundamentals.py)
# and never touch yfinance/SimFin specifics. ONE recipe; the source is swapped
# upstream by the router. The factor MATH is unchanged from the original yfinance
# version — only the input contract moved to the canonical schema.
#
# Also here: the two distress/manipulation SCREENS (altman_z, beneish_m). These
# aren't ranked factors — they're the "margin of safety first" scrub applied before
# ranking (Quantitative Value); see screen.py.

import math

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


def altman_z(f):
    """Altman Z-score (original, public-firm form) — bankruptcy/distress gauge.
    Z = 1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MVE/TL) + 1.0·(Sales/TA).
    Zones: >2.99 safe · 1.81–2.99 grey · <1.81 distress. None if inputs missing."""
    ta, tl = f.get("total_assets"), f.get("total_liabilities")
    if not ta or not tl:
        return None
    re, ebit, mve, sales = (f.get("retained_earnings"), f.get("ebit"),
                            f.get("market_cap"), f.get("revenue"))
    if None in (re, ebit, mve, sales):
        return None
    wc = (f.get("current_assets") or 0) - (f.get("current_liabilities") or 0)
    return (1.2 * wc / ta + 1.4 * re / ta + 3.3 * ebit / ta
            + 0.6 * mve / tl + 1.0 * sales / ta)


def beneish_m(f):
    """Beneish M-score — earnings-manipulation flag (8-variable model, Beneish 1999).
    M = -4.84 + .92·DSRI + .528·GMI + .404·AQI + .892·SGI + .115·DEPI
        − .172·SGAI + 4.679·TATA − .327·LVGI
    M > -1.78 => likely manipulator (flag/exclude). Needs TWO years; None if the
    prior-year inputs are incomplete or any denominator is zero."""
    p = f.get("prior") or {}
    s_t, s_p = f.get("revenue"), p.get("revenue")
    rec_t, rec_p = f.get("receivables"), p.get("receivables")
    gp_t, gp_p = f.get("gross_profit"), p.get("gross_profit")
    ca_t, ca_p = f.get("current_assets"), p.get("current_assets")
    ppe_t, ppe_p = f.get("ppe"), p.get("ppe")
    sec_t, sec_p = (f.get("securities") or 0.0), (p.get("securities") or 0.0)
    ta_t, ta_p = f.get("total_assets"), p.get("total_assets")
    dep_t, dep_p = f.get("depreciation"), p.get("depreciation")
    sga_t, sga_p = f.get("sga"), p.get("sga")
    cl_t, cl_p = f.get("current_liabilities"), p.get("current_liabilities")
    td_t, td_p = f.get("total_debt"), p.get("total_debt")
    cfo_t, inc_t = f.get("cfo"), f.get("income_continuing")

    required = [s_t, s_p, rec_t, rec_p, gp_t, gp_p, ca_t, ca_p, ppe_t, ppe_p,
                ta_t, ta_p, dep_t, dep_p, sga_t, sga_p, cl_t, cl_p, td_t, td_p,
                cfo_t, inc_t]
    if any(x is None for x in required):
        return None
    try:
        dsri = (rec_t / s_t) / (rec_p / s_p)
        gmi = (gp_p / s_p) / (gp_t / s_t)
        aqi = (1 - (ca_t + ppe_t + sec_t) / ta_t) / (1 - (ca_p + ppe_p + sec_p) / ta_p)
        sgi = s_t / s_p
        depi = (dep_p / (dep_p + ppe_p)) / (dep_t / (dep_t + ppe_t))
        sgai = (sga_t / s_t) / (sga_p / s_p)
        lvgi = ((cl_t + td_t) / ta_t) / ((cl_p + td_p) / ta_p)
        tata = (inc_t - cfo_t) / ta_t
        m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
             + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    except ZeroDivisionError:
        return None
    return m if math.isfinite(m) else None


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
