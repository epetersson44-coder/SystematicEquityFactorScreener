# backtest/leverage_study.py — can anything beat SPY in CRISIS-FREE windows?
#
# The blend's known weakness: unlevered diversification mathematically cannot beat 100%
# SPY on raw return in a window where SPY is the best asset (any mix dilutes the winner).
# Every credible fix means holding MORE than 100% exposure to the bull without dying in
# the crash. Three literature-backed candidates, all priced with the REAL T-bill series
# (^IRX) — NOT the flat 4% financing of the earlier leverage test, which overcharged the
# 2011-2019 bull (actual rf ~0-1%) by ~3%/yr and produced the old "leverage loses in
# bulls" verdict. With honest financing, that verdict FLIPS (2026-07-01 study):
#
#   A. LEVERED BLEND (lever the highest-Sharpe curve; Markowitz/CML logic): the 0.94
#      ensemble blend at ~2.3x (SPY-vol-matched) beat SPY's raw return in EVERY window
#      tested — bulls included (2011-19: 14.3% vs 13.1%; 2023-25: 26.5% vs 23.1%) — and
#      the full cycle ($149k vs $68k per $10k, maxDD -35% vs -55%, Sharpe 0.82 vs 0.61).
#   B. RETURN STACKING (Hoffstein/ReSolve; RSST does this live): 100% SPY + lambda x
#      trend overlay on financed notional. Beats SPY everywhere by a moderate margin,
#      but keeps SPY's full crash depth (the SPY leg is never hedged).
#   C. GAYED "Leverage for the Long Run" (2016 Dow Award): daily-reset LETF (2x/3x)
#      when SPY > 200d MA, T-bills below. Biggest raw pile (3x: $207k full cycle) but
#      Sharpe ~= SPY and -45/-55% drawdowns — more risk on the same line, not alpha.
#
# SYNTHETIC-MARKET RECALIBRATION (2026-07-01, backtest/synthetic.py generators, 40
# alternate histories): the real path's -35% maxDD for the 2.3x blend was a FAVORABLE
# DRAW. Across null/trending/bootstrapped worlds the levered book's median worst
# drawdown is -41%..-48%, tails -72% (bootstrapped real returns) to -78% (adverse
# trending world). Leverage created edge NOWHERE (lev-blend Sharpe gap ~-0.02 = pure
# financing drag, every world) — the ranking stands, but size any future leverage off
# the DISTRIBUTION (-45% typical adverse, -70%+ possible at 2.3x; ~1.8x for mid-30s),
# never off the single historical path.
#
# The LETF simulator (daily reset, financing on (L-1) at rf+spread, expense ratio) is
# VALIDATED against real SSO/UPRO: corr 0.996+, tracking gap -0.3/-0.6/yr%. Caveats that
# stay attached to any use of these numbers: retail margin costs more than rf+40bps (the
# implementable vehicles are LETFs/RSST-style funds which embed institutional rates);
# MA-switching in a taxable account costs ~1-2%/yr in taxes; a levered book's drawdowns
# arrive levered (behavioral survival is the binding constraint); leverage choice per
# Kelly, not per appetite.
#
# Run:  python -m backtest.leverage_study

import os

import numpy as np
import pandas as pd

from backtest.data import CACHE_DIR
from backtest import metrics

SPREAD = 0.004                    # financing spread over T-bills (swap / LETF embed)
ER2, ER3 = 0.0089, 0.0091         # SSO / UPRO expense ratios
IRX_CACHE = os.path.join(CACHE_DIR, "irx_daily.csv")


def tbill_series(index, refresh=False):
    """Annualized 13-week T-bill yield (^IRX/100) aligned to `index`, ffilled, cached.
    Fetched directly (not via data.get_prices — its positive-price gate rejects the
    near-zero yields of 2020-21, which are real)."""
    if not refresh and os.path.exists(IRX_CACHE):
        s = pd.read_csv(IRX_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        s = yf.download("^IRX", start="2006-01-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        os.makedirs(CACHE_DIR, exist_ok=True)
        s.to_csv(IRX_CACHE)
    return (s.reindex(index).ffill() / 100.0).fillna(0.0)


def letf_returns(spy_ret, L, er, rf_annual):
    """Daily-reset leveraged-ETF returns from underlying returns: L x underlying minus
    financing on the borrowed (L-1) at rf+SPREAD minus the expense ratio. Daily reset
    means vol drag emerges from compounding — no extra term needed."""
    return L * spy_ret - (L - 1) * (rf_annual / 252.0 + SPREAD / 252.0) - er / 252.0


def lrs_returns(spy_px, spy_ret, L, er, rf_annual, ma=200, switch_bps=5):
    """Gayed Leveraged Rotation Strategy: hold the L-x LETF while SPY > its `ma`-day
    SMA (signal lagged one day), T-bills below; switch cost charged on transition days."""
    on = (spy_px > spy_px.rolling(ma).mean()).astype(float).shift(1)
    on = on.reindex(spy_ret.index).fillna(0.0)
    cost = on.diff().abs().fillna(0.0) * (switch_bps / 10_000.0) * L
    return on * letf_returns(spy_ret, L, er, rf_annual) + (1 - on) * rf_annual / 252.0 - cost


def stacked_returns(spy_ret, overlay_ret, lam, rf_annual):
    """Return stacking: 100% SPY + lam x overlay on financed notional (rf+SPREAD)."""
    return spy_ret + lam * overlay_ret - lam * (rf_annual / 252.0 + SPREAD / 252.0)


def levered_returns(ret, L, rf_annual):
    """Constant-leverage daily-rebalanced L x a return stream, financing the (L-1)."""
    return L * ret - (L - 1) * (rf_annual / 252.0 + SPREAD / 252.0)


def validate_letf(spy_ret, rf_annual):
    """Compare the simulator against the real funds (network). Returns rows of
    (ticker, corr, annualized tracking gap) — trust gate for everything above."""
    import yfinance as yf
    real = yf.download(["SSO", "UPRO"], start="2006-06-01", auto_adjust=True,
                       progress=False)["Close"]
    out = []
    for t, L, er in (("SSO", 2, ER2), ("UPRO", 3, ER3)):
        rr = real[t].pct_change().dropna()
        sim = letf_returns(spy_ret, L, er, rf_annual).reindex(rr.index).dropna()
        rr = rr.reindex(sim.index)
        gap = (1 + rr).prod() ** (252 / len(rr)) - (1 + sim).prod() ** (252 / len(sim))
        out.append({"fund": t, "corr": round(float(sim.corr(rr)), 4),
                    "ann_gap_%": round(float(gap) * 100, 2), "days": len(sim)})
    return out


WINDOWS = [("FULL 2007-2026 (3 crises in-window)", "2007-07-01", None),
           ("BULL 2011-2019", "2011-01-01", "2019-12-31"),
           ("BULL 2012-2021", "2012-01-01", "2021-12-31"),
           ("BULL 2023-2025", "2023-01-01", "2025-12-31")]


def _stats(r, label, w0=None, w1=None):
    r = r.dropna()
    if w0:
        r = r[r.index >= w0]
    if w1:
        r = r[r.index <= w1]
    eq = (1 + r).cumprod()
    return {"": label, "CAGR%": round(metrics.cagr(eq) * 100, 1),
            "Sharpe": round(metrics.sharpe(eq), 2),
            "maxDD%": round(metrics.max_drawdown(eq) * 100, 0),
            "$10k->": round(10_000 * float(eq.iloc[-1] / eq.iloc[0]))}


def report():
    from backtest.trend_sleeve import etf_panel
    from backtest.timing_luck import sweep, tranched_curve, blend_curve

    panels = etf_panel()
    spy_px = panels["Close"]["SPY"].dropna()
    spy_ret = spy_px.pct_change().dropna()
    rf = tbill_series(spy_ret.index)

    for row in validate_letf(spy_ret, rf):
        print(f"validate {row['fund']}: corr {row['corr']}, gap {row['ann_gap_%']:+.2f}%/yr")

    _, curves = sweep(panels=panels)
    trend_eq = tranched_curve(curves, tuple(range(21)))
    trend_ret = trend_eq.pct_change().reindex(spy_ret.index).fillna(0.0)
    blend_ret = blend_curve(trend_eq, spy_px).pct_change().reindex(spy_ret.index).dropna()

    cands = [("SPY buy&hold", spy_ret),
             ("blend unlevered (the 0.94)", blend_ret)]
    for L in (1.5, 2.0, spy_ret.std() / blend_ret.std()):
        cands.append((f"blend levered {L:.2f}x @ real rf", levered_returns(blend_ret, L, rf)))
    cands += [("stack SPY + 0.5x trend", stacked_returns(spy_ret, trend_ret, 0.5, rf)),
              ("stack SPY + 1.0x trend", stacked_returns(spy_ret, trend_ret, 1.0, rf)),
              ("Gayed LRS 2x", lrs_returns(spy_px, spy_ret, 2, ER2, rf)),
              ("Gayed LRS 3x", lrs_returns(spy_px, spy_ret, 3, ER3, rf))]

    for wname, w0, w1 in WINDOWS:
        print(f"\n===== {wname}")
        print(pd.DataFrame([_stats(r, n, w0, w1) for n, r in cands]).set_index("").to_string())


if __name__ == "__main__":
    report()
