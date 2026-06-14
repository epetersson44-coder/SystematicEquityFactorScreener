# backtest/baseline.py — the benchmark every strategy is measured against:
# buy SPY on day one, reinvest dividends, hold forever. The first scoreboard line.

from backtest.data import get_prices
from backtest import metrics
from backtest.constants import INITIAL_CAPITAL


def buy_and_hold(prices, initial_capital=INITIAL_CAPITAL):
    """Equity curve from putting `initial_capital` into the asset on day one and
    holding. Uses adjusted Close, so this is total return (dividends reinvested)."""
    close = prices["Close"]
    return close / close.iloc[0] * initial_capital


def print_summary(name, equity, initial_capital=INITIAL_CAPITAL, rf=0.0):
    s = metrics.summary(equity, rf=rf)
    yrs = s["max_dd_duration_days"] / 365.25
    print(f"\n=== {name} ===")
    print(f"period        {s['start']} -> {s['end']}")
    print(f"final value   ${s['final_value']:,.0f}   (from ${initial_capital:,})")
    print(f"total return  {s['total_return'] * 100:,.1f}%")
    print(f"CAGR          {s['cagr'] * 100:.2f}%")
    print(f"volatility    {s['volatility'] * 100:.1f}%   (annualized)")
    print(f"Sharpe        {s['sharpe']:.2f}   (rf={rf:.0%})")
    print(f"Sortino       {s['sortino']:.2f}")
    print(f"Calmar        {s['calmar']:.2f}   (CAGR / |max DD|)")
    print(f"max drawdown  {s['max_drawdown'] * 100:.1f}%")
    print(f"max DD length {s['max_dd_duration_days']} days   ({yrs:.1f} yr underwater)")


if __name__ == "__main__":
    # Pin the benchmark window so the scoreboard is stable even though the cache
    # now holds full history (SPY back to 1993).
    spy = get_prices("SPY", start="2000-01-01")
    equity = buy_and_hold(spy)
    print_summary("SPY buy & hold", equity)
