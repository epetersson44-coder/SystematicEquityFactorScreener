# score.py — percentile ranking and composite scoring

import pandas as pd
import numpy as np
from fundamentals import get_fundamentals
from factors import calculate_factors
from config import TICKERS, WEIGHTS, MIN_FACTORS

def build_factor_table(tickers, source="yfinance"):
    rows = []
    for ticker in tickers:
        try:
            f = get_fundamentals(ticker, source=source)
            rows.append(calculate_factors(f))
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
    return pd.DataFrame(rows)

def score(df):
    scored = df.copy()

    # Lower is better for these — invert ranking
    lower_is_better = ["ev_ebit", "price_fcf", "gm_stability", "net_debt_ebitda"]

    for factor in WEIGHTS.keys():
        if factor not in df.columns:
            continue
        if factor in lower_is_better:
            scored[f"{factor}_pct"] = df[factor].rank(ascending=True, pct=True) * 100
        else:
            scored[f"{factor}_pct"] = df[factor].rank(ascending=False, pct=True) * 100

    # Composite weighted score — averaged over the factors each company
    # actually has, re-normalizing the weights. (A plain weighted sum lets
    # one missing factor turn the whole composite NaN, silently burying
    # the company at the bottom of the ranking.)
    pct_cols = [f"{f}_pct" for f in WEIGHTS if f"{f}_pct" in scored.columns]
    weights = pd.Series({f"{f}_pct": w for f, w in WEIGHTS.items() if f"{f}_pct" in scored.columns})
    pcts = scored[pct_cols]
    scored["composite"] = pcts.mul(weights).sum(axis=1) / pcts.notna().mul(weights).sum(axis=1)
    scored["n_factors"] = pcts.notna().sum(axis=1)
    scored.loc[scored["n_factors"] < MIN_FACTORS, "composite"] = float("nan")

    return scored.sort_values("composite", ascending=False)

if __name__ == "__main__":
    print("Building factor table for all tickers...")
    df = build_factor_table(TICKERS)
    print("\nRaw factors:")
    print(df[["ticker", "ev_ebit", "price_fcf", "roic", "gm_stability", "net_debt_ebitda"]])

    results = score(df)
    print("\nFinal rankings:")
    print(results[["ticker", "composite", "n_factors"]].to_string(index=False))