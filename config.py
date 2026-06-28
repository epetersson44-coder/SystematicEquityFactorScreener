# config.py — central settings for the screener

from dotenv import load_dotenv
load_dotenv()

# Factor weights — must sum to 1.0. Tilted toward the factors with POSITIVE information
# coefficient in the point-in-time study (factor_analysis.py): F-Score (best IC, 0.116) and
# the quality factors (roic, gm_stability) carry more; the weak/dead value factors carry
# less. A principled evidence tilt, NOT an optimization (we did not fish weights for return).
WEIGHTS = {
    "fscore":          0.25,   # Piotroski F-Score — highest IC, the documented quality upgrade
    "roic":            0.20,   # quality (positive IC)
    "gm_stability":    0.15,   # quality (positive IC)
    "ev_ebit":         0.20,   # value (weak but kept for style)
    "price_fcf":       0.10,   # value (weak)
    "net_debt_ebitda": 0.10,   # leverage (weak)
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