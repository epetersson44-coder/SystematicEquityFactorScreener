# factors.py — calculates the 5 factors from raw yfinance data

import numpy as np

def get_ev_ebit(info, income):
    try:
        ev = info.get("enterpriseValue")
        ebit = income.loc["EBIT"].iloc[0]
        if ev and ebit and ebit > 0:
            return ev / ebit
    except:
        pass
    return None

def get_price_fcf(info, cashflow):
    try:
        market_cap = info.get("marketCap")
        fcf = cashflow.loc["Free Cash Flow"].iloc[0]
        if market_cap and fcf and fcf > 0:
            return market_cap / fcf
    except:
        pass
    return None

def get_roic(income, balance):
    try:
        ebit = income.loc["EBIT"].iloc[0]
        tax_rate = income.loc["Tax Rate For Calcs"].iloc[0]
        nopat = ebit * (1 - tax_rate)
        total_assets = balance.loc["Total Assets"].iloc[0]
        total_liabilities = balance.loc["Total Liabilities Net Minority Interest"].iloc[0]
        invested_capital = total_assets - total_liabilities
        if invested_capital and invested_capital > 0:
            return nopat / invested_capital
    except:
        pass
    return None

def get_gm_stability(income):
    try:
        revenue = income.loc["Total Revenue"]
        gross_profit = income.loc["Gross Profit"]
        gm = gross_profit / revenue
        return gm.std()
    except:
        pass
    return None

def get_net_debt_ebitda(info, income):
    try:
        net_debt = info.get("totalDebt", 0) - info.get("totalCash", 0)
        ebitda = income.loc["EBITDA"].iloc[0]
        if ebitda and ebitda > 0:
            return net_debt / ebitda
    except:
        pass
    return None

def calculate_factors(data):
    ticker = data["ticker"]
    info = data["info"]
    income = data["income"]
    balance = data["balance"]
    cashflow = data["cashflow"]

    return {
        "ticker":          ticker,
        "ev_ebit":         get_ev_ebit(info, income),
        "price_fcf":       get_price_fcf(info, cashflow),
        "roic":            get_roic(income, balance),
        "gm_stability":    get_gm_stability(income),
        "net_debt_ebitda": get_net_debt_ebitda(info, income),
    }

if __name__ == "__main__":
    from fetch import fetch_all
    data = fetch_all("CALM")
    factors = calculate_factors(data)
    print(factors)