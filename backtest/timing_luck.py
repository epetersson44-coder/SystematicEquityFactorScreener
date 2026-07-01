# backtest/timing_luck.py — rebalance-timing-luck study of the trend sleeve.
#
# The sleeve trades every 21 bars at ONE offset anchored to the first bar of the data —
# an arbitrary choice that was never tested. Hoffstein, Faber & Braun ("Rebalance Timing
# Luck: The Dumb (Timing) Luck of Smart Beta", 2020) show identically-managed portfolios
# that differ ONLY in rebalance date can diverge by whole percentage points a year. So the
# honest headline is the DISTRIBUTION of results across all 21 offsets — median and spread
# — not whichever single offset we happened to anchor on.
#
# The cure, if the spread is material, is TRANCHING: split capital across N staggered
# sub-sleeves (rebalance a fraction each week instead of everything on one day). That
# converges on the average offset and cuts timing luck by ~1/N.
#
# Run:  python -m backtest.timing_luck

import numpy as np
import pandas as pd

from backtest.trend_sleeve import etf_panel, run_trend
from backtest import metrics


def blend_curve(trend_eq, spy, w_spy=None):
    """Daily constant-mix SPY+trend blend, risk-parity by default — the SAME construction
    as trend_sleeve.analyze(), kept identical so these numbers are comparable with the
    headline. (Yes, the full-sample inverse-vol weight is mildly in-sample; that applies
    equally to every offset, so it cancels for THIS study's question.)"""
    df = pd.DataFrame({"SPY": spy, "trend": trend_eq}).dropna()
    rets = df.pct_change().dropna()
    if w_spy is None:
        iv_s, iv_t = 1 / rets["SPY"].std(), 1 / rets["trend"].std()
        w_spy = iv_s / (iv_s + iv_t)
    br = w_spy * rets["SPY"] + (1 - w_spy) * rets["trend"]
    return 10_000 * (1 + br).cumprod()


def sweep(every=21, cost_bps=5, panels=None, **run_kw):
    """Run the sleeve at every rebalance offset (0..every-1). Returns (stats DataFrame
    indexed by offset, {offset: sleeve equity curve})."""
    panels = panels or etf_panel()
    spy = panels["Close"]["SPY"].dropna()
    rows, curves = [], {}
    for off in range(every):
        eq = run_trend(cost_bps=cost_bps, panels=panels, offset=off, **run_kw)
        b = blend_curve(eq, spy)
        rows.append({"offset": off,
                     "sleeve_sharpe": metrics.sharpe(eq), "sleeve_cagr": metrics.cagr(eq),
                     "sleeve_maxdd": metrics.max_drawdown(eq),
                     "blend_sharpe": metrics.sharpe(b), "blend_cagr": metrics.cagr(b),
                     "blend_maxdd": metrics.max_drawdown(b)})
        curves[off] = eq
    return pd.DataFrame(rows).set_index("offset"), curves


def tranched_curve(curves, offsets=(0, 5, 10, 15)):
    """Equity of a tranched sleeve: capital split equally across staggered offsets (default
    ~weekly). Each tranche compounds independently with its own costs, so the combined
    curve is simply the mean of the sub-curves."""
    df = pd.DataFrame({o: curves[o] for o in offsets}).dropna()
    return df.mean(axis=1)


def _dist(col):
    return {"median": float(col.median()), "min": float(col.min()), "max": float(col.max()),
            "spread": float(col.max() - col.min())}


def report(every=21, cost_bps=5, **run_kw):
    """The study: per-offset table, distribution summary, and the tranched alternative."""
    panels = etf_panel()
    stats, curves = sweep(every=every, cost_bps=cost_bps, panels=panels, **run_kw)
    spy = panels["Close"]["SPY"].dropna()

    print("Per-offset results (sleeve = vol-targeted 6-ETF trend; blend = risk-parity SPY+trend):")
    print(stats.round(3).to_string())

    print("\nDistribution across the 21 offsets:")
    for name, col in (("sleeve Sharpe", stats["sleeve_sharpe"]),
                      ("blend  Sharpe", stats["blend_sharpe"]),
                      ("blend  CAGR", stats["blend_cagr"]),
                      ("blend  maxDD", stats["blend_maxdd"])):
        d = _dist(col)
        print(f"  {name:14s} median {d['median']:+.3f}   range [{d['min']:+.3f}, {d['max']:+.3f}]"
              f"   spread {d['spread']:.3f}")

    print("\n4-tranche sleeve, every possible weekly-stagger anchor (the tranche construction"
          "\nmust itself be luck-robust, or we've just moved the luck):")
    for a in range(5):
        offs = tuple(a + 5 * k for k in range(4))
        tb = blend_curve(tranched_curve(curves, offs), spy)
        print(f"  anchors {offs}: blend Sharpe {metrics.sharpe(tb):+.3f}"
              f"   maxDD {metrics.max_drawdown(tb) * 100:.1f}%")

    tb21 = blend_curve(tranched_curve(curves, tuple(range(every))), spy)
    print(f"\nAll-{every}-tranche limit (zero timing luck — the honest headline):")
    print(f"  blend Sharpe {metrics.sharpe(tb21):+.3f}   CAGR {metrics.cagr(tb21) * 100:+.2f}%"
          f"   maxDD {metrics.max_drawdown(tb21) * 100:.1f}%")
    return stats, curves


if __name__ == "__main__":
    report()
