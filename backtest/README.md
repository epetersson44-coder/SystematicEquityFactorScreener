# backtest/ — event-driven backtesting lab

A small, honest backtester for testing trading strategies against a buy-and-hold
benchmark. Independent of the factor screener in the parent repo — it shares the
repo, and (later) the universe, nothing else.

## The one idea
Walk price bars forward in time. At each bar, show the strategy **only the history
up to that bar**, let it pick a target weight (0 = all cash, 1 = fully invested),
and fill the trade at the *next* bar's open. Because the strategy can never see a
future bar, look-ahead bias is **structurally impossible** — not a rule we remember,
a thing the code physically can't do.

## Data flow

```
   data.py                strategy.py            engine.py              metrics.py
 ┌──────────┐  prices   ┌────────────┐ weight  ┌───────────┐ equity  ┌───────────┐
 │ get_     │──────────>│ target_    │────────>│  run()    │────────>│ summary() │
 │ prices() │  OHLC df  │ weight()   │ 0..1    │ Portfolio │  curve  │ Sharpe,   │
 │ (cached) │           │ per bar    │         │ + bar loop│         │ DD, ...   │
 └──────────┘           └────────────┘         └───────────┘         └───────────┘
   yfinance →                                        │                     │
   CSV cache                                         v                     v
   + _validate                              baseline.buy_and_hold   SCOREBOARD.md
                                            (analytic check)        (strategy vs SPY)
```

Every strategy, however complex, reduces to one **equity curve** (a dated Series of
portfolio value). All metrics are computed from that curve, so the yardstick is
fully decoupled from how the curve was produced.

## Files
| File | Role |
|------|------|
| `constants.py` | Shared numbers (`INITIAL_CAPITAL`, `TRADING_DAYS`) — one source of truth |
| `data.py` | Cached single-ticker price loader (yfinance → CSV once), with `_validate` gate |
| `strategy.py` | Strategies: single-asset (`BuyAndHold`, `SMACrossover`, `WalkForwardSMA`) + cross-sectional (`CrossSectionalMomentum`) |
| `engine.py` | Single-asset `Portfolio` + `run()` — the bar loop that builds the equity curve |
| `metrics.py` | CAGR, vol, Sharpe, Sortino, Calmar, max drawdown + **drawdown duration**, off a curve |
| `baseline.py` | Analytic buy-and-hold benchmark + `print_summary` reporting helper |
| **Phase 2** | |
| `optimize.py` | `grid_search`, `split_search`, `walk_forward_schedule` — parameter search + the overfitting reveal |
| `overfit_demo.py` | The multiple-testing demonstration (more knobs = more self-deception) |
| `walkforward.py` | Honest out-of-sample walk-forward vs the optimize-on-everything fantasy |
| **Phase 3** | |
| `universe.py` | Multi-ticker price *panels* (date × ticker); S&P 500 batch-fetch + cache. **Survivorship caveat in header** |
| `engine_xs.py` | Cross-sectional engine: `MultiPortfolio` (cash + basket) + `run_xs()` over a panel |
| `sp500_tickers.txt` | Saved S&P 500 constituents (today's list — see survivorship caveat) |
| `SCOREBOARD.md` | Every strategy vs SPY, naive and honest assumptions |
| `tests/` | `test_backtest.py` (31, single-asset) + `test_engine_xs.py` (12, multi-asset incl. Monte Carlo) + `stress_test.py` (adversarial) + `_helpers.py` |

## Key design choices (the honest defaults)
- **Fill timing** — `fill="next_open"` (default, honest): decide on today's close, fill
  tomorrow's open. `fill="close"` is optimistic, used only to validate accounting against
  the closed-form `baseline.buy_and_hold` (they match to ~1e-11).
- **Long-only, no leverage** — `Portfolio.rebalance` rejects any weight outside [0, 1].
  Shorting / multiple assets arrive in Phase 3 (pairs trading) and will relax this *deliberately*.
- **Idle cash earns the risk-free rate** — `run(cash_rate=0.04)`; default 0 keeps the
  analytic-baseline validation exact. Pair it with the same `rf` in `metrics.sharpe`.
- **Costs** — `costs.proportional(bps)` folds commission + spread + slippage into one
  basis-point knob on traded notional.

## Running it
```bash
python -m backtest.data            # load + cache SPY, print tail
python -m backtest.baseline        # SPY buy & hold summary
python -m backtest.engine          # prove engine == analytic baseline (validation)
python -m backtest.tests.test_backtest    # 31 correctness checks
python -m backtest.tests.stress_test      # adversarial / rough stress harness
```

## Status & roadmap
- **Phase 1 (done):** single-asset engine, metrics, costs, SMA crossover, full test + stress coverage.
- **Phase 2 (done):** the overfitting lesson — walk-forward proved optimizing the SMA added +0.00 Sharpe over a naive default. See `walkforward.py` / wiki `CODE/walk-forward`.
- **Phase 3 (in progress):** cross-sectional strategies on a real universe.
  - ✓ multi-asset engine (`engine_xs.py`), hardened (12 checks incl. conservation, look-ahead, Monte Carlo).
  - ✓ S&P 500 universe loader + price momentum. First result is **survivorship-inflated** (19% CAGR — not real; today's-constituents bias). Engine math proven correct, so the inflation is entirely the data.
  - ☐ **survivorship-free / point-in-time membership** — the real data fix; the blocker on any trustworthy result.
  - ☐ bring in the fundamental factor screener (point-in-time financials).
  - ☐ shorting + the long-only guard relaxation (pairs / long-short).
- **Phase 4:** screener goes live, monthly locked picks tracked vs SPY (start a minimal version early — the out-of-sample clock matters).

The methodology this is built against lives in the Brain2.0 wiki: `CODE/quant-trading`
(Chan's backtesting playbook) and `CODE/quant-methods` (the screener factor specs).
