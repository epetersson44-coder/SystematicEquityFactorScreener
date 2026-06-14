# backtest/overfit_demo.py — Phase 2 demonstration: more knobs = more self-deception.
#
# The engine behind data-snooping is the MULTIPLE-TESTING effect: if you try K
# parameter combos and keep the one with the best IN-SAMPLE score, that best score
# drifts upward as K grows — simply because the maximum of K noisy numbers tends
# higher the more numbers you draw. The very same pick's OUT-OF-SAMPLE score does
# NOT rise, because the inflation was luck, not signal. The widening gap is the
# overfitting, quantified.
#
# We compute every pair's in-sample (2000-2012) and out-of-sample (2013-2026)
# Sharpe ONCE, then — by resampling K-sized subsets — measure how good the
# optimizer's in-sample pick looks vs how it actually performs out-of-sample.

import numpy as np
import pandas as pd

from backtest.data import get_prices
from backtest.optimize import grid_search, sma_param_grid

SPLIT = "2013-01-01"


def main():
    spy = get_prices("SPY", start="2000-01-01")
    train = spy[spy.index < pd.to_datetime(SPLIT)]
    test = spy[spy.index >= pd.to_datetime(SPLIT)]

    fasts = list(range(5, 105, 5))          # 5, 10, ..., 100
    slows = list(range(60, 300, 20))        # 60, 80, ..., 280
    grid = sma_param_grid(fasts, slows)
    print(f"Precomputing {len(grid)} SMA pairs, in-sample (2000-12) + out-of-sample (2013-26)...")

    is_s = grid_search(train, grid).set_index(["fast", "slow"])["sharpe"]
    oos_s = grid_search(test, grid).set_index(["fast", "slow"])["sharpe"].reindex(is_s.index)
    is_arr, oos_arr = is_s.to_numpy(), oos_s.to_numpy()
    n = len(is_arr)

    rng = np.random.default_rng(0)
    print(f"\n{'pairs tried (K)':>15} {'best looks (IS)':>16} {'best really is (OOS)':>21} {'overfit gap':>12}")
    for K in [1, 2, 5, 10, 25, 50, 100, n]:
        if K > n:
            continue
        trials = 1 if K == n else 400
        looks, really = [], []
        for _ in range(trials):
            idx = rng.choice(n, size=K, replace=False)
            win = idx[np.argmax(is_arr[idx])]       # pick the best by IN-SAMPLE
            looks.append(is_arr[win])
            really.append(oos_arr[win])
        lk, rl = np.mean(looks), np.mean(really)
        print(f"{K:>15} {lk:>16.2f} {rl:>21.2f} {lk - rl:>12.2f}")

    print("\nMore pairs tried -> the in-sample pick looks better and better, while its")
    print("out-of-sample reality flatlines. That growing gap is pure overfitting: the")
    print("reward for torturing the data, paid entirely in fake confidence.")


if __name__ == "__main__":
    main()
