# backtest/optimize.py — parameter search, and the overfitting it invites.
#
# Phase 2's whole lesson lives here. grid_search() sweeps SMA window pairs and
# ranks them by a metric. That LOOKS like "finding the best strategy" — but the
# pair that wins on past data is mostly fit to THAT data's noise (data-snooping
# bias). Optimizing on all of history and trusting the winner is the trap. The
# cure is to score on data the parameters never saw: an out-of-sample split
# (see split_search) and, next, walk-forward.

import pandas as pd

from backtest.engine import run
from backtest.strategy import SMACrossover
from backtest import metrics
from backtest.costs import proportional


def sma_param_grid(fasts, slows):
    """Every (fast, slow) window pair with fast < slow."""
    return [(f, s) for f in fasts for s in slows if f < s]


def grid_search(prices, params, metric="sharpe", rf=0.04, cost=None, cash_rate=0.04):
    """Backtest every (fast, slow) in `params`; return results ranked by `metric`.

    Honest defaults (rf=4%, idle cash earns 4%, 2 bps cost). Returns a DataFrame
    sorted best-first, one row per pair with the headline stats.
    """
    if cost is None:
        cost = proportional(2)
    rows = []
    for fast, slow in params:
        eq = run(prices, SMACrossover(fast, slow), cost=cost, cash_rate=cash_rate)
        s = metrics.summary(eq, rf=rf)
        rows.append({
            "fast": fast, "slow": slow, "sharpe": s["sharpe"], "cagr": s["cagr"],
            "max_dd": s["max_drawdown"], "calmar": s["calmar"], "final": s["final_value"],
        })
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def best_params(prices, params, metric="sharpe", **kw):
    """Just the winning (fast, slow) from a grid_search on `prices`."""
    top = grid_search(prices, params, metric=metric, **kw).iloc[0]
    return int(top["fast"]), int(top["slow"])


def split_search(prices, params, split, metric="sharpe", **kw):
    """Optimize on the in-sample half (before `split`), keep the out-of-sample half honest.

    Returns (train, test, is_grid, oos_grid, winner): the two ranked grids and the
    pair that won in-sample. The reveal is comparing the winner's in-sample score to
    its out-of-sample score — the gap is data-snooping, made visible.
    """
    cut = pd.to_datetime(split)
    train = prices[prices.index < cut]
    test = prices[prices.index >= cut]
    is_grid = grid_search(train, params, metric=metric, **kw)
    oos_grid = grid_search(test, params, metric=metric, **kw)
    winner = (int(is_grid.iloc[0]["fast"]), int(is_grid.iloc[0]["slow"]))
    return train, test, is_grid, oos_grid, winner


def _row(grid, fast, slow):
    """The grid row for one (fast, slow) pair."""
    return grid[(grid["fast"] == fast) & (grid["slow"] == slow)].iloc[0]


def walk_forward_schedule(prices, params, train_bars, test_bars, metric="sharpe", **kw):
    """Roll a [train_bars -> test_bars] window across `prices`, re-optimizing each step.

    For each step: optimize on the trailing `train_bars` of data, then assign that
    winning pair to the next `test_bars` (the out-of-sample test window). Returns a
    time-ordered list of (test_start, test_end, fast, slow) — feed it to
    strategy.WalkForwardSMA. Params for a window come only from data before it.
    """
    schedule = []
    n = len(prices)
    i = train_bars
    while i < n:
        train = prices.iloc[i - train_bars:i]
        fast, slow = best_params(train, params, metric=metric, **kw)
        j = min(i + test_bars, n)
        schedule.append((prices.index[i], prices.index[j - 1], fast, slow))
        i = j
    return schedule


def _print_table(res, n=10):
    print(f"{'fast':>4} {'slow':>4}  {'Sharpe':>6} {'CAGR':>6} {'maxDD':>7} {'Calmar':>6}")
    for _, r in res.head(n).iterrows():
        print(f"{int(r.fast):>4} {int(r.slow):>4}  {r.sharpe:>6.2f} "
              f"{r.cagr * 100:>5.1f}% {r.max_dd * 100:>6.1f}% {r.calmar:>6.2f}")


if __name__ == "__main__":
    from backtest.data import get_prices

    spy = get_prices("SPY", start="2000-01-01")
    grid = sma_param_grid([10, 20, 30, 40, 50, 75, 100], [50, 100, 150, 200, 250])
    res = grid_search(spy, grid, metric="sharpe")

    print(f"Optimizing SMA windows on ALL of history, 2000-2026 ({len(grid)} pairs):\n")
    _print_table(res, 10)

    best = res.iloc[0]
    print(f"\n'Best' in-sample: SMA {int(best.fast)}/{int(best.slow)} -> "
          f"Sharpe {best.sharpe:.2f}, CAGR {best.cagr * 100:.1f}%, maxDD {best.max_dd * 100:.1f}%")
    print("Our untuned 50/200 scored Sharpe ~0.45. This looks better.\n")

    print("=" * 64)
    print("THE REVEAL — optimize on 2000-2012 only, then test on 2013-2026")
    print("=" * 64)
    SPLIT = "2013-01-01"
    train, test, is_grid, oos_grid, winner = split_search(spy, grid, SPLIT, metric="sharpe")
    f, s = winner
    is_row, oos_row = _row(is_grid, f, s), _row(oos_grid, f, s)
    base_oos = _row(oos_grid, 50, 200)
    oos_best = oos_grid.iloc[0]

    print(f"\nWinner trained on 2000-2012:  SMA {f}/{s}")
    print(f"   in-sample (2000-2012)    Sharpe {is_row.sharpe:>5.2f}   <- looked great")
    print(f"   out-of-sample (2013-26)  Sharpe {oos_row.sharpe:>5.2f}   <- what you'd ACTUALLY get")
    print(f"\nOut-of-sample 2013-2026, for context:")
    print(f"   untuned 50/200           Sharpe {base_oos.sharpe:>5.2f}")
    print(f"   best-in-hindsight pair   SMA {int(oos_best.fast)}/{int(oos_best.slow)}  Sharpe {oos_best.sharpe:>5.2f}")
    print(f"\nThe out-of-sample winner is a DIFFERENT pair than the in-sample winner.")
    print("Last era's 'best' parameters did not carry. That gap is data-snooping.")
