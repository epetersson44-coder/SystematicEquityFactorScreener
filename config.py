# config.py — central settings for the screener

from dotenv import load_dotenv
load_dotenv()

# Factor weights — must sum to 1.0
WEIGHTS = {
    "ev_ebit":         0.25,
    "price_fcf":       0.15,
    "roic":            0.30,
    "gm_stability":    0.10,
    "net_debt_ebitda": 0.20,
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