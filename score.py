# score.py — percentile ranking and composite scoring

import pandas as pd
import numpy as np
from fetch import fetch_all
from factors import calculate_factors
from config import TICKERS, WEIGHTS

def build_factor_table(tickers):
    rows = []
    for ticker in tickers:
        try:
            data = fetch_all(ticker)
            factors = calculate_factors(data)
            rows.append(factors)
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

    # Composite weighted score
    scored["composite"] = sum(
        scored[f"{factor}_pct"] * weight
        for factor, weight in WEIGHTS.items()
        if f"{factor}_pct" in scored.columns
    )

    return scored.sort_values("composite", ascending=False)

if __name__ == "__main__":
    print("Building factor table for all tickers...")
    df = build_factor_table(TICKERS)
    print("\nRaw factors:")
    print(df[["ticker", "ev_ebit", "price_fcf", "roic", "gm_stability", "net_debt_ebitda"]])

    results = score(df)
    print("\nFinal rankings:")
    print(results[["ticker", "composite"]].to_string(index=False))