# backtest/synthetic.py — synthetic-market validation of the trend sleeve.
#
# Erik's question (2026-07-01): "test the strategy on fake generated markets, so we know
# it works and isn't just biased to our one market's history." The formal version of that
# worry is Lopez de Prado's: history is ONE path, and everything we adopted was chosen
# while looking at it. Synthetic markets can't prove the edge will persist (only the live
# record can) — but they can FALSIFY the machinery three ways:
#
#   1. NULL control — iid returns with the real assets' vol/correlation but ZERO
#      autocorrelation (nothing to time). An honest trend strategy should add ~NOTHING
#      vs just buying the same assets. If it "wins" here, we have a look-ahead bug.
#   2. POSITIVE control — same noise + slowly wandering drift (AR(1) regimes), i.e.
#      markets that genuinely trend. The sleeve should now clearly beat static holding.
#      If it can't harvest trend where trend definitely exists, it's broken.
#   3. PATH robustness — block-bootstrap scrambles of the REAL panel (keeps fat tails +
#      cross-asset correlation, breaks trend continuity beyond the block length). Shows
#      the DISTRIBUTION of outcomes across histories-that-could-have-been, and how the
#      edge decays as the very thing it harvests (long trends) is chopped up.
#
# The yardstick everywhere is EXCESS Sharpe: sleeve minus static equal-weight buy&hold
# of the SAME panel (beta is free; timing skill is what's being tested).
#
# Run:  python -m backtest.synthetic          (~10 seeds per market type, a few minutes)

import numpy as np
import pandas as pd

from backtest.engine_xs import run_xs
from backtest.trend_sleeve import VolTargetTSMOM, ENSEMBLE_LOOKS, ETFS
from backtest import costs, metrics

N_DAYS = 4800                      # ~19 years, comparable to the real window


def _real_moments():
    """(mean vector, covariance matrix) of daily returns from the real 6-ETF panel."""
    from backtest.trend_sleeve import etf_panel
    rets = etf_panel()["Close"][ETFS].pct_change().dropna()
    return rets.mean().to_numpy(), rets.cov().to_numpy()


def _to_panels(rets_df):
    """Turn a synthetic daily-returns frame into {'Close','Open'} price panels.
    Open[t] = Close[t-1] (fills happen at the prior close's level — neutral, and the
    look-ahead structure of run_xs is unchanged)."""
    close = 100.0 * (1.0 + rets_df).cumprod()
    open_ = close.shift(1).fillna(100.0)
    return {"Close": close, "Open": open_}


def gbm_panel(mu, cov, n_days=N_DAYS, seed=0):
    """NULL market: iid multivariate-normal daily returns (real vol + correlation,
    constant drift, ZERO autocorrelation — there is nothing to time)."""
    rng = np.random.default_rng(seed)
    r = rng.multivariate_normal(mu, cov, size=n_days)
    idx = pd.bdate_range("2007-01-01", periods=n_days)
    return _to_panels(pd.DataFrame(r, index=idx, columns=ETFS))


def trending_panel(mu, cov, n_days=N_DAYS, seed=0, phi=0.995, drift_vol_mult=1.0):
    """POSITIVE-control market: same noise, but each asset's drift follows a slow AR(1)
    (half-life ~ 140 days at phi=0.995) — genuine multi-month trends exist by
    construction. Unconditional mean drift = the real mu, so long-run returns are
    comparable; only the TIMEABILITY differs from the null."""
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros_like(mu), cov, size=n_days)
    k = len(mu)
    sd_d = drift_vol_mult * np.sqrt(np.diag(cov)) * 0.1     # slow, modest drift wander
    innov = rng.normal(0.0, 1.0, size=(n_days, k)) * (sd_d * np.sqrt(1 - phi ** 2))
    drift = np.empty((n_days, k))
    d = np.zeros(k)
    for t in range(n_days):
        d = phi * d + innov[t]
        drift[t] = d
    r = mu + drift + noise
    idx = pd.bdate_range("2007-01-01", periods=n_days)
    return _to_panels(pd.DataFrame(r, index=idx, columns=ETFS))


def bootstrap_panel(block=21, seed=0, n_days=None):
    """PATH-robustness market: moving-block bootstrap of the REAL panel's daily returns
    (rows resampled jointly — cross-asset correlation and fat tails survive; trend
    continuity is broken at the block seams). Longer blocks preserve more trend."""
    from backtest.trend_sleeve import etf_panel
    rets = etf_panel()["Close"][ETFS].pct_change().dropna().to_numpy()
    n = n_days or len(rets)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, len(rets) - block + 1, size=n_blocks)
    idx_rows = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    idx = pd.bdate_range("2007-01-01", periods=n)
    return _to_panels(pd.DataFrame(rets[idx_rows], index=idx, columns=ETFS))


def sleeve_vs_static(panels, cost_bps=5):
    """(sleeve Sharpe, static equal-weight buy&hold Sharpe, excess) on one panel."""
    strat = VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS)
    eq = run_xs(panels, strat, cost=costs.proportional(cost_bps), fill="next_open")
    s_sleeve = metrics.sharpe(eq)
    static = (1 + panels["Close"].pct_change().dropna().mean(axis=1)).cumprod()
    s_static = metrics.sharpe(static)
    return s_sleeve, s_static, s_sleeve - s_static


def study(n_seeds=10, n_days=N_DAYS, quiet=False):
    """The three-way falsification study. Returns {market_type: DataFrame of runs}."""
    mu, cov = _real_moments()
    cases = {
        "null (iid, nothing to time)": lambda s: gbm_panel(mu, cov, n_days, seed=s),
        "trending (AR-drift injected)": lambda s: trending_panel(mu, cov, n_days, seed=s),
        "bootstrap real, 21d blocks": lambda s: bootstrap_panel(21, seed=s, n_days=n_days),
        "bootstrap real, 126d blocks": lambda s: bootstrap_panel(126, seed=s, n_days=n_days),
    }
    out = {}
    for name, gen in cases.items():
        rows = []
        for s in range(n_seeds):
            sl, st, ex = sleeve_vs_static(gen(s))
            rows.append({"seed": s, "sleeve": round(sl, 3), "static": round(st, 3),
                         "excess": round(ex, 3)})
        df = pd.DataFrame(rows).set_index("seed")
        out[name] = df
        if not quiet:
            print(f"\n===== {name}  ({n_seeds} seeds)")
            print(df.to_string())
            print(f"  excess Sharpe: median {df['excess'].median():+.3f}   "
                  f"range [{df['excess'].min():+.3f}, {df['excess'].max():+.3f}]")
    if not quiet:
        print("\nHow to read this: NULL excess should straddle ~0 (no cheating possible);"
              "\nTRENDING excess should be clearly positive (the harvester works);"
              "\nBOOTSTRAP excess should sit between them and improve with block length"
              "\n(the edge lives in trend continuity, which longer blocks preserve).")
    return out


if __name__ == "__main__":
    study()
