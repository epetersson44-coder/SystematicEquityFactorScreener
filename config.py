# config.py — central settings for the screener

from dotenv import load_dotenv
load_dotenv()

# Factor weights — must sum to 1.0. AGNOSTIC EQUAL WEIGHT, by evidence.
# History: an earlier tilt up-weighted the F-Score to 0.25 on its +0.116 IC in the 2021-2025
# SimFin window. The full-cycle EDGAR decomposition (backtest/edgar_backtest.decompose_drag,
# S&P 600, 2011-2026, 8,387 obs) REFUTED that: the F-Score's IC flips to -0.028 (t -2.42, the
# only SIGNIFICANT factor, and negative) — its weight was overfit to a window, so the F-Score
# is DROPPED. The five value/quality factors are all statistically dead (|t|<0.6), so there is
# no evidence to tilt by: equal weight is the honest, no-conviction default. This does NOT lift
# returns (the whole composite is ~zero IC; the bake-off proved reweighting does nothing) — it
# just stops us betting on a refuted signal. The screener stands as a rigorous NEGATIVE result.
WEIGHTS = {
    "ev_ebit":         0.20,   # value
    "price_fcf":       0.20,   # value
    "roic":            0.20,   # quality
    "gm_stability":    0.20,   # quality
    "net_debt_ebitda": 0.20,   # leverage
}

TICKERS = [
    "CALM", "PRGS", "HTLD", "SKYW", "MGRC",
    "ADUS", "FIZZ", "CWST", "EPAC", "NBTB",
    "UFPI", "HCSG", "MNRO", "SRCE", "LCII"
]

MIN_YEARS_HISTORY = 3

# Companies with fewer ranked factors than this are excluded from the
# composite — a score built on 1–2 factors flatters whatever data survived
MIN_FACTORS = 4