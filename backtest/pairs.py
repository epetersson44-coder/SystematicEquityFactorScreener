# backtest/pairs.py — pairs trading (market-neutral mean reversion). Completes Phase 3.
#
# The other side of cross-sectional momentum: instead of betting winners keep winning,
# bet that two related stocks whose prices move TOGETHER will snap back when they diverge.
# Long the cheap leg, short the rich leg, profit on convergence — market-neutral, so it
# leans on the long-short engine (engine_xs, allow_short) we built and stress-tested.
#
# The honest discipline (mirrors the rest of the lab):
#   - COINTEGRATION first, correlation never. Two random walks can correlate by luck; a
#     cointegrated pair has a spread that is genuinely mean-reverting (stationary). We test
#     it with Engle-Granger (regress one on the other, ADF the residual) — hand-rolled from
#     OLS, no statsmodels.
#   - ECONOMIC RATIONALE first. Pairs are tested only WITHIN industry groups (KO/PEP, V/MA,
#     XOM/CVX, ...), not across all 125k S&P pairs — both because same-business names share
#     drivers, and to limit the multiple-testing trap (scan enough pairs and you'll "find"
#     cointegration by chance — the Phase-2 data-snooping lesson, [[walk-forward]]).
#   - OUT-OF-SAMPLE. The hedge ratio + cointegration are estimated on a FORMATION window;
#     trading happens only AFTER it, with a trailing z-score (point-in-time).
#
# Run:  python -m backtest.pairs

import numpy as np
import pandas as pd

from backtest.universe import get_universe
from backtest.engine_xs import run_xs
from backtest.strategy import CrossSectionalStrategy
from backtest import metrics, costs

# Engle-Granger residual ADF critical value, 2 variables w/ constant (~5%, MacKinnon).
# More negative t-stat than this => reject unit root => the spread is mean-reverting.
EG_CRIT_5 = -3.34

# Same-industry candidates (economic rationale first). Only within-group pairs are tested.
INDUSTRY_GROUPS = {
    "beverages":   ["KO", "PEP"],
    "mega_tech":   ["MSFT", "AAPL", "GOOGL", "ORCL"],
    "banks":       ["JPM", "BAC", "WFC", "C"],
    "oil":         ["XOM", "CVX", "COP"],
    "home_retail": ["HD", "LOW"],
    "big_retail":  ["WMT", "TGT", "COST"],
    "payments":    ["V", "MA"],
    "telecom":     ["VZ", "T"],
    "pharma":      ["PFE", "MRK", "BMY"],
    "industrials": ["CAT", "DE"],
}


# ----------------------------------------------------------------- hand-rolled stats
def _ols(y, X):
    """OLS y = X·b. Returns (coefs, t-stats). X must already include an intercept column."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tstats = np.where(se > 0, beta / se, np.nan)
    return beta, tstats


def adf_tstat(s, lags=1):
    """Augmented Dickey-Fuller test statistic (with constant) for a unit root in `s`.
    Regress Δs_t on a constant, the lagged level s_{t-1}, and `lags` lagged differences;
    the t-stat on the level coefficient is the ADF stat. More negative = more stationary."""
    s = np.asarray(s, float)
    if len(s) < lags + 4:
        return np.nan
    ds = np.diff(s)
    y = ds[lags:]
    cols = [np.ones(len(y)), s[lags:-1]]                 # const, lagged LEVEL s_{t-1}
    for j in range(1, lags + 1):
        cols.append(ds[lags - j:-j])                     # lagged differences Δs_{t-j}
    _, tstats = _ols(y, np.column_stack(cols))
    return tstats[1]                                     # t-stat on the level coefficient


def hedge_ratio(y, x):
    """OLS slope β in y ≈ α + β·x. The spread that should be stationary is y − β·x."""
    beta, _ = _ols(y, np.column_stack([np.ones(len(x)), np.asarray(x, float)]))
    return beta[1]


def engle_granger(y, x):
    """Engle-Granger cointegration test. Regress y on x, ADF the residual.
    Returns (adf_tstat, beta); tstat < EG_CRIT_5 (~ -3.34) => cointegrated at ~5%."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    beta_vec, _ = _ols(y, np.column_stack([np.ones(len(x)), x]))
    resid = y - (beta_vec[0] + beta_vec[1] * x)
    return adf_tstat(resid), beta_vec[1]


def half_life(spread):
    """Mean-reversion half-life (days) from an Ornstein-Uhlenbeck / AR(1) fit:
    Δs_t = a + θ·s_{t-1}; half-life = −ln2/θ. ∞ if not reverting (θ ≥ 0)."""
    s = np.asarray(spread, float)
    ds = np.diff(s)
    s_lag = s[:-1]
    beta_vec, _ = _ols(ds, np.column_stack([np.ones(len(s_lag)), s_lag]))
    theta = beta_vec[1]
    return (-np.log(2) / theta) if theta < 0 else np.inf


# ----------------------------------------------------------------- the strategy
class PairsTrade(CrossSectionalStrategy):
    """Trade ONE cointegrated pair (a, b) with a fixed hedge ratio β (estimated out-of-
    sample, on the formation window). spread = price_a − β·price_b; z = trailing z-score.

    State machine (hysteresis avoids whipsaw): flat until |z| > entry, then take the
    convergence bet (long the cheap leg / short the rich leg), hold until |z| < exit
    (reverted — take profit) or |z| > stop (blew out — the relationship broke, cut it).
    Legs are equal-dollar (±0.5 each, gross 1.0) — a dollar-neutral simplification; true
    β-weighted sizing is a refinement."""

    def __init__(self, a, b, beta, window=60, entry=2.0, exit=0.5, stop=4.0):
        self.a, self.b, self.beta = a, b, beta
        self.window, self.entry, self.exit, self.stop = window, entry, exit, stop
        self.position = 0                                # -1 short spread, 0 flat, +1 long spread

    def target_weights(self, closes, i):
        if i < self.window:
            return None                                  # warmup: trailing window not ready
        a = closes[self.a].iloc[:i + 1]
        b = closes[self.b].iloc[:i + 1]
        spread = a - self.beta * b
        win = spread.iloc[-self.window:]
        sd = win.std()
        if not np.isfinite(sd) or sd == 0:
            return None
        z = (spread.iloc[-1] - win.mean()) / sd
        if not np.isfinite(z):
            return None

        if self.position == 0:                           # flat -> look for an entry
            if z > self.entry:
                self.position = -1                       # spread rich -> short a, long b
            elif z < -self.entry:
                self.position = +1                       # spread cheap -> long a, short b
            else:
                return None                              # stay flat, no trade
        else:                                            # in a position -> hold or close
            if abs(z) < self.exit or abs(z) > self.stop:
                self.position = 0                        # converged (profit) or blew out (stop)
            else:
                return None                              # hold

        if self.position == +1:
            w = {self.a: 0.5, self.b: -0.5}
        elif self.position == -1:
            w = {self.a: -0.5, self.b: 0.5}
        else:
            w = {self.a: 0.0, self.b: 0.0}              # flat: weights 0 -> engine closes
        return pd.Series(w)


# ----------------------------------------------------------------- selection + backtest
def candidate_pairs(groups=INDUSTRY_GROUPS):
    """All within-group (a, b) pairs — the economically-motivated candidate set."""
    pairs = []
    for names in groups.values():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))
    return pairs


def find_pairs(form_close, groups=INDUSTRY_GROUPS, min_hl=5, max_hl=252):
    """Cointegrated pairs on the FORMATION window, sorted most-cointegrated first.
    Returns [(a, b, beta, adf_tstat, half_life)] for pairs that pass EG_CRIT_5 and whose
    reversion half-life is sane (not <5d noise, not >1yr too-slow)."""
    found = []
    for a, b in candidate_pairs(groups):
        if a not in form_close.columns or b not in form_close.columns:
            continue
        ya = form_close[a].dropna(); xb = form_close[b].dropna()
        idx = ya.index.intersection(xb.index)
        if len(idx) < 60:
            continue
        ya, xb = ya.loc[idx], xb.loc[idx]
        t, beta = engle_granger(ya.values, xb.values)
        if not np.isfinite(t) or t >= EG_CRIT_5 or beta <= 0:
            continue
        hl = half_life((ya - beta * xb).values)
        if not (min_hl <= hl <= max_hl):
            continue
        found.append((a, b, beta, float(t), float(hl)))
    return sorted(found, key=lambda r: r[3])             # most negative ADF t first


def _stats(eq, label):
    return {"label": label, "final": float(eq.iloc[-1]), "ret": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "cagr": metrics.cagr(eq), "sharpe": metrics.sharpe(eq), "maxdd": metrics.max_drawdown(eq)}


def run_pairs_backtest(formation=("2016-01-01", "2019-12-31"), trade_start="2020-01-01",
                       top_k=5, window=60, entry=2.0, exit=0.5, stop=4.0, cost_bps=10):
    """Select cointegrated pairs on the formation window, then trade each OUT-OF-SAMPLE
    from trade_start. Returns (per_pair, portfolio_eq, stats). Market-neutral, so the
    benchmark is cash (Sharpe is the number that matters)."""
    panels = get_universe("sp500")
    close, opn = panels["Close"], panels["Open"]
    form = close[(close.index >= formation[0]) & (close.index <= formation[1])]
    selected = find_pairs(form)[:top_k]
    if not selected:
        raise RuntimeError("no cointegrated pairs found on the formation window")

    curves, per_pair = [], []
    for a, b, beta, t, hl in selected:
        sub = {"Close": close[[a, b]][close.index >= trade_start],
               "Open": opn[[a, b]][opn.index >= trade_start]}
        eq = run_xs(sub, PairsTrade(a, b, beta, window, entry, exit, stop),
                    cost=costs.proportional(cost_bps), allow_short=True, gross_max=1.0)
        curves.append(eq / eq.iloc[0])
        per_pair.append((f"{a}/{b}", beta, t, hl, _stats(eq, f"{a}/{b}")))

    port = pd.concat(curves, axis=1).mean(axis=1) * 10_000.0    # equal-capital across pairs
    stats = {"portfolio": _stats(port, "equal-weight pairs book"),
             "n_pairs": len(selected), "start": port.index[0].date(), "end": port.index[-1].date()}
    return per_pair, port, stats


if __name__ == "__main__":
    per_pair, port, s = run_pairs_backtest()
    print(f"\n=== Pairs trading (market-neutral mean reversion), OOS {s['start']} -> {s['end']} ===")
    print(f"Selected {s['n_pairs']} cointegrated pairs (formation 2016-2019), traded out-of-sample:\n")
    print(f"  {'pair':<10}{'beta':>7}{'ADF t':>8}{'half-life':>11}{'return':>9}{'Sharpe':>8}")
    for name, beta, t, hl, st in per_pair:
        print(f"  {name:<10}{beta:>7.2f}{t:>8.2f}{hl:>9.0f}d{st['ret'] * 100:>+8.1f}%{st['sharpe']:>+8.2f}")
    p = s["portfolio"]
    print(f"\n  {'PORTFOLIO':<10} equal-weight: ${p['final']:>9,.0f}  ({p['ret'] * 100:+.1f}%)  "
          f"CAGR {p['cagr'] * 100:+.1f}%  Sharpe {p['sharpe']:+.2f}  maxDD {p['maxdd'] * 100:.0f}%")
    print("  (market-neutral -> Sharpe is the number that matters; benchmark is cash, not SPY.")
    print("   Pairs found by scanning within-industry candidates — mind the multiple-testing caveat.)")
