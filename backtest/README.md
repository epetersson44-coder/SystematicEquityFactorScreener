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
| `data.py` | Cached daily price loader (yfinance → CSV once), with `_validate` data-hygiene gate |
| `strategy.py` | `Strategy` base + `BuyAndHold` + `SMACrossover`; each returns a target weight 0..1 |
| `engine.py` | `Portfolio` (cash + shares) and `run()` — the bar loop that builds the equity curve |
| `metrics.py` | CAGR, vol, Sharpe, Sortino, Calmar, max drawdown + **drawdown duration**, off a curve |
| `baseline.py` | Analytic buy-and-hold benchmark + `print_summary` reporting helper |
| `SCOREBOARD.md` | Every strategy vs SPY, naive and honest assumptions |
| `tests/` | `test_backtest.py` (31 correctness checks) + `stress_test.py` (adversarial) + `_helpers.py` |

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
- **Phase 1 (done):** engine, metrics, costs, SMA crossover, full test + stress coverage.
- **Phase 2 (next):** the overfitting lesson — tune SMA windows in-sample, watch them
  collapse out-of-sample; build walk-forward testing.
- **Phase 3:** mean-reversion / momentum on a real universe (needs shorting + multi-asset).
- **Phase 4:** screener goes live, tracked vs SPY.

The methodology this is built against lives in the Brain2.0 wiki: `CODE/quant-trading`
(Chan's backtesting playbook) and `CODE/quant-methods` (the screener factor specs).
