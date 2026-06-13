# backtest/metrics.py — performance metrics computed from an equity curve.
#
# An "equity curve" is a pandas Series of portfolio value indexed by date.
# Every strategy, however it's run, reduces to one of these — so the yardstick
# lives here, decoupled from how the curve was produced.

import numpy as np

TRADING_DAYS = 252


def total_return(equity):
    """Cumulative return over the whole period, as a fraction."""
    return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity):
    """Geometric annualized return, using the actual calendar span."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years == 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(equity, rf=0.0, periods_per_year=TRADING_DAYS):
    """Annualized Sharpe ratio from daily simple returns.

    rf is the annual risk-free rate (default 0 — project keeps v1 simple and
    relies on the comparison vs SPY, where a constant rf cancels). The daily
    risk-free rf/periods_per_year is subtracted from each daily return.
    """
    ret = equity.pct_change().dropna()
    sd = ret.std(ddof=1)
    if sd == 0:
        return np.nan
    excess = ret - rf / periods_per_year
    return excess.mean() / sd * np.sqrt(periods_per_year)


def max_drawdown(equity):
    """Worst peak-to-trough decline, as a negative fraction (e.g. -0.55)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def summary(equity, rf=0.0):
    """All headline metrics as a dict."""
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final_value": equity.iloc[-1],
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "sharpe": sharpe(equity, rf=rf),
        "max_drawdown": max_drawdown(equity),
    }
