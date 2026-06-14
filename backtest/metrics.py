# backtest/metrics.py — performance metrics computed from an equity curve.
#
# An "equity curve" is a pandas Series of portfolio value indexed by date.
# Every strategy, however it's run, reduces to one of these — so the yardstick
# lives here, decoupled from how the curve was produced.
#
# Chan's lesson (Quantitative Trading, ch.3): risk-adjusted return is the headline,
# and drawdown DURATION — how long you sit underwater — is the killer that ends
# most strategies, not drawdown depth. So we report both, plus Calmar/Sortino.

import numpy as np
import pandas as pd

from backtest.constants import TRADING_DAYS


def _daily_returns(equity):
    """Daily simple returns of an equity curve, first (NaN) row dropped."""
    return equity.pct_change().dropna()


def total_return(equity):
    """Cumulative return over the whole period, as a fraction."""
    return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity):
    """Geometric annualized return, using the actual calendar span."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years == 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualized_volatility(equity, periods_per_year=TRADING_DAYS):
    """Annualized standard deviation of daily simple returns."""
    ret = _daily_returns(equity)
    return ret.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe(equity, rf=0.0, periods_per_year=TRADING_DAYS):
    """Annualized Sharpe ratio from daily simple returns.

    rf is the annual risk-free rate. Default 0 keeps v1 simple, but BEWARE:
    rf=0 inflates the ratio. A realistic ~4% rf cuts a buy-and-hold Sharpe
    materially; a strategy that parks in cash is hurt less (its idle cash should
    also earn rf — see engine.run(cash_rate=...)). For a fair number, pass the
    same rf to both this and the engine's cash_rate. Chan's rough bar to even
    bother trading: annualized Sharpe >= ~1.
    """
    ret = _daily_returns(equity)
    sd = ret.std(ddof=1)
    if sd == 0:
        return np.nan
    excess = ret - rf / periods_per_year
    return excess.mean() / sd * np.sqrt(periods_per_year)


def sortino(equity, rf=0.0, periods_per_year=TRADING_DAYS):
    """Like Sharpe but penalizes only DOWNSIDE volatility (downside deviation).

    Upside swings aren't risk; Sortino divides excess return by the RMS of the
    negative excess returns only. Higher = better. NaN if there's no downside.
    """
    ret = _daily_returns(equity)
    excess = ret - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return excess.mean() / downside_dev * np.sqrt(periods_per_year)


def max_drawdown(equity):
    """Worst peak-to-trough decline, as a negative fraction (e.g. -0.55)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def max_drawdown_duration(equity):
    """Longest time UNDERWATER, in calendar days: peak -> recovery of that peak.

    Chan's point: a -20% drawdown that recovers in a month is survivable; one
    that grinds for three years ends careers. Measured as the span from the peak
    that started the dip until the curve reclaims it (the recovery bar is the end
    of the stretch). A strictly rising curve is never underwater -> 0. If the
    curve never recovers by the end, that trailing stretch counts (worst case —
    still bleeding at the finish).
    """
    running_max = np.maximum.accumulate(equity.to_numpy())
    underwater = equity.to_numpy() < running_max            # strictly below peak
    dates = equity.index
    longest = pd.Timedelta(0)
    last_peak_date = dates[0]
    dip_start = None                                        # peak that began the dip
    for i in range(len(dates)):
        if underwater[i]:
            if dip_start is None:
                dip_start = last_peak_date                  # dip began at prior peak
            longest = max(longest, dates[i] - dip_start)    # extend (incl. recovery)
        else:
            if dip_start is not None:
                longest = max(longest, dates[i] - dip_start)  # peak -> recovery span
                dip_start = None
            last_peak_date = dates[i]
    return longest.days


def calmar(equity):
    """CAGR divided by the magnitude of max drawdown — return per unit of pain.

    A Chan-style risk-adjusted number that, unlike Sharpe, is rf-independent and
    speaks the language of "how much do I make for the worst loss I must stomach."
    """
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return cagr(equity) / abs(mdd)


def summary(equity, rf=0.0):
    """All headline metrics as a dict."""
    return {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "final_value": equity.iloc[-1],
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(equity),
        "sharpe": sharpe(equity, rf=rf),
        "sortino": sortino(equity, rf=rf),
        "max_drawdown": max_drawdown(equity),
        "max_dd_duration_days": max_drawdown_duration(equity),
        "calmar": calmar(equity),
    }
