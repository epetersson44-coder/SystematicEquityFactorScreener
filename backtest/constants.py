# backtest/constants.py — the single source of truth for shared numbers.
#
# Anything used by more than one module lives here, so a change can't leave two
# files silently disagreeing (e.g. metrics annualizing by 252 while the engine
# accrues cash by a different number). Module-specific config stays in its module
# (cache paths in data.py, default SMA windows in strategy.py).

INITIAL_CAPITAL = 10_000     # starting portfolio value for every backtest
TRADING_DAYS = 252           # trading days per year — annualization + cash accrual
