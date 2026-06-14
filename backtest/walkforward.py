# backtest/walkforward.py — Phase 2's cure: walk-forward analysis.
#
# The honest answer to "which SMA should I trade?" The reveal and overfit_demo
# showed that the in-sample-best pair is noise and that searching harder only
# inflates appearances. Walk-forward takes that as given: it NEVER trades a
# parameter it couldn't have known. It rolls a train->test window across history,
# re-optimizes on each train slice, and trades the winner only on the *next*
# (unseen) slice — stitching those out-of-sample pieces into one trustworthy curve.
#
# We then line that honest curve up against three references:
#   - "optimize on everything" — the in-sample-best applied to all of history. The
#     fantasy. It cheats (it used the whole past to pick its params). The gap
#     between it and walk-forward is the overfitting tax.
#   - untuned 50/200 — never optimized at all. The "did the machinery earn its keep?"
#   - SPY buy & hold — the benchmark that started it all.
#
# Run:  python -m backtest.walkforward

import pandas as pd

from backtest.data import get_prices
from backtest import metrics
from backtest.engine import run
from backtest.strategy import BuyAndHold, SMACrossover, WalkForwardSMA
from backtest.costs import proportional
from backtest.optimize import sma_param_grid, best_params, walk_forward_schedule
from backtest.constants import INITIAL_CAPITAL, TRADING_DAYS

COST = proportional(2)
CASH_RATE = 0.04
RF = 0.04


def _restate(equity, eval_start):
    """Slice an equity curve to the evaluation window and rebase to INITIAL_CAPITAL.
    CAGR/Sharpe/drawdown are scale-invariant, so rebasing only fixes the final $ for
    a fair side-by-side — it doesn't flatter any strategy."""
    e = equity[equity.index >= eval_start]
    return e / e.iloc[0] * INITIAL_CAPITAL


def main():
    spy = get_prices("SPY", start="2000-01-01")
    grid = sma_param_grid([10, 20, 30, 40, 50, 75, 100], [50, 100, 150, 200, 250])

    print("Building walk-forward schedule (4-yr train -> 1-yr test, rolling)...")
    schedule = walk_forward_schedule(spy, grid, train_bars=4 * TRADING_DAYS,
                                     test_bars=TRADING_DAYS, metric="sharpe")
    eval_start = schedule[0][0]

    print(f"\nWhat the optimizer picked for each out-of-sample year "
          f"({len(schedule)} windows):")
    print(f"  {'test window':<23} {'SMA pair chosen'}")
    for start, end, f, s in schedule:
        print(f"  {start.date()} -> {end.date()}   {f}/{s}")
    distinct = sorted(set((f, s) for _, _, f, s in schedule))
    print(f"  -> {len(distinct)} distinct pairs across {len(schedule)} windows: {distinct}")

    curves = {
        "Walk-forward (honest)": run(spy, WalkForwardSMA(schedule), cost=COST, cash_rate=CASH_RATE),
        "Optimize-all (fantasy)": run(spy, SMACrossover(*best_params(spy, grid)), cost=COST, cash_rate=CASH_RATE),
        "Untuned 50/200": run(spy, SMACrossover(50, 200), cost=COST, cash_rate=CASH_RATE),
        "SPY buy & hold": run(spy, BuyAndHold(), cost=COST, cash_rate=CASH_RATE),
    }

    print(f"\nAll compared over the out-of-sample span {eval_start.date()} -> {spy.index[-1].date()}, "
          f"rf={RF:.0%}, cash earns {CASH_RATE:.0%}:\n")
    print(f"  {'strategy':<24} {'CAGR':>6} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7} {'final $10k':>11}")
    stats = {}
    for name, eq in curves.items():
        s = metrics.summary(_restate(eq, eval_start), rf=RF)
        stats[name] = s
        print(f"  {name:<24} {s['cagr']*100:>5.1f}% {s['sharpe']:>7.2f} "
              f"{s['max_drawdown']*100:>6.1f}% {s['calmar']:>7.2f} ${s['final_value']:>10,.0f}")

    tax = stats["Optimize-all (fantasy)"]["sharpe"] - stats["Walk-forward (honest)"]["sharpe"]
    edge = stats["Walk-forward (honest)"]["sharpe"] - stats["Untuned 50/200"]["sharpe"]
    print(f"\n  Overfitting tax (fantasy - walk-forward Sharpe): {tax:+.2f}")
    print(f"  Walk-forward's edge over not optimizing at all:  {edge:+.2f}")


if __name__ == "__main__":
    main()
