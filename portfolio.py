# portfolio.py — live portfolio tracker
# Run: python portfolio.py
# Updates portfolio.md in the wiki with current prices and P&L

import yfinance as yf
from datetime import datetime

# Holdings: ticker -> (shares, cost_basis_total)
HOLDINGS = {
    "VTI":  (7,    2426.88),
    "VXUS": (4,     347.56),
    "QQQM": (4.2,  1071.39),
    "AVUV": (3,     364.14),
    "NVDA": (4,     763.98),
    "GOOG": (4,    1262.40),
    "AMD":  (1,     218.58),
    "PLTR": (3,     582.24),
}

# Role of each holding in the core-satellite structure
ROLES = {
    "VTI":  "Core",
    "VXUS": "Core (intl)",
    "QQQM": "Tech tilt",
    "AVUV": "Factor sleeve",
    "NVDA": "Individual",
    "GOOG": "Individual",
    "AMD":  "Individual",
    "PLTR": "Speculative",
}

CASH = 137.17
WIKI_PORTFOLIO = "/Users/erik.petersson/Library/Mobile Documents/iCloud~md~obsidian/Documents/ClaudeBrain2.0/Brain2.0/wiki/PERSONAL/portfolio.md"


def fetch_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            price = t.fast_info["last_price"]
            prices[ticker] = round(price, 2)
        except Exception as e:
            print(f"  Could not fetch {ticker}: {e}")
            prices[ticker] = None
    return prices


def build_report(prices):
    rows = []
    total_cost = sum(cost for _, cost in HOLDINGS.values())
    total_value = CASH

    for ticker, (shares, cost) in HOLDINGS.items():
        price = prices.get(ticker)
        if price is None:
            rows.append((ticker, shares, cost, cost/shares, "N/A", "N/A", "N/A", "N/A"))
            continue
        market_val = round(shares * price, 2)
        unreal = round(market_val - cost, 2)
        unreal_pct = round((unreal / cost) * 100, 2)
        total_value += market_val
        rows.append((ticker, shares, cost, round(cost/shares, 2), price, market_val, unreal, unreal_pct))

    total_unreal = round(total_value - CASH - total_cost, 2)
    total_unreal_pct = round((total_unreal / total_cost) * 100, 2)
    return rows, total_value, total_cost, total_unreal, total_unreal_pct


def print_report(rows, total_value, total_cost, total_unreal, total_unreal_pct):
    print(f"\n{'='*80}")
    print(f"  PORTFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"  Total Value:      ${total_value:>10,.2f}")
    print(f"  Total Cost:       ${total_cost:>10,.2f}")
    print(f"  Unrealized G/L:   ${total_unreal:>+10,.2f}  ({total_unreal_pct:+.2f}%)")
    print(f"  Cash:             ${CASH:>10,.2f}")
    print(f"{'='*80}")
    print(f"  {'Ticker':<6} {'Shares':>7} {'Cost/sh':>9} {'Price':>9} {'Mkt Val':>10} {'G/L $':>10} {'G/L %':>8}")
    print(f"  {'-'*65}")
    for ticker, shares, cost, cost_per, price, mkt_val, unreal, unreal_pct in rows:
        if price == "N/A":
            print(f"  {ticker:<6} {shares:>7} {cost_per:>9.2f}  {'N/A':>9} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
        else:
            sign = "+" if unreal >= 0 else ""
            print(f"  {ticker:<6} {shares:>7} {cost_per:>9.2f} {price:>9.2f} {mkt_val:>10.2f} {sign}{unreal:>9.2f} {sign}{unreal_pct:>6.2f}%")
    print(f"{'='*80}\n")


def update_wiki(rows, total_value, total_cost, total_unreal, total_unreal_pct):
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M")

    table_lines = [
        "| Ticker | Role | Shares | Cost Basis | Cost/Share | Current Price | Market Value | Unrealized G/L | Unrealized % |",
        "|--------|------|--------|-----------|------------|---------------|--------------|----------------|--------------|",
    ]
    for ticker, shares, cost, cost_per, price, mkt_val, unreal, unreal_pct in rows:
        role = ROLES.get(ticker, "")
        if price == "N/A":
            table_lines.append(f"| {ticker} | {role} | {shares} | ${cost:,.2f} | ${cost_per:.2f} | N/A | N/A | N/A | N/A |")
        else:
            sign = "+" if unreal >= 0 else ""
            table_lines.append(
                f"| {ticker} | {role} | {shares} | ${cost:,.2f} | ${cost_per:.2f} | ${price:,.2f} | ${mkt_val:,.2f} | {sign}${unreal:,.2f} | {sign}{unreal_pct:.2f}% |"
            )

    summary = f"""---
tags: [personal, finance, portfolio, investing, stocks]
sources: [Memory.md, portfolio.py auto-update]
---

# Portfolio

Last updated: **{date} {time}** (auto-updated by portfolio.py)

See also: [[PERSONAL/profile]], [[PERSONAL/career-plan]], [[FINANCE/dcf-valuation]], [[FINANCE/what-works-wall-street]], [[CODE/factor-screener]], [[synthesis]]

---

## Investment Policy (set 2026-06-17)

Goal: **steady, durable compounding** — not a playground for one-off theses. Structure is **core-satellite**:
- **Core (~65-70%)** — broad index funds held for decades, fed by ongoing contributions. VTI-led, plus international (VXUS).
- **Satellite (~25-35%)** — individual names + factor tilts where I can take real risk and learn. Cap any single name at ~8-10% so one blowup is a bruise, not a crater. Diversify the *type* of bet, not just the size.
- **Roth first** once earning — contributions stay withdrawable; only growth is age-locked. Tax-free compounding is the priority vehicle.
- Cost basis and recent performance are **irrelevant** to hold/sell decisions. The only question: is this the best home for the money going forward?

---

## Summary

| Metric | Value |
|--------|-------|
| Total value | ${total_value:,.2f} |
| Cash | ${CASH:,.2f} |
| Unrealized G/L | +${total_unreal:,.2f} ({total_unreal_pct:+.2f}%) |
| Last updated | {date} {time} |

---

## Holdings

{chr(10).join(table_lines)}

---

## Position Notes

### VTI — Core
- The compounding engine — future contributions feed this first. Largest holding by dollar value.

### VXUS — Core, International
- First non-US exposure (opened 2026-06-17, funded from the SCHD sale). Fills the geography gap — international trades ~40% cheaper than the US (~15x fwd P/E vs ~22x).

### AVUV — Factor Sleeve
- Avantis US Small-Cap Value (opened 2026-06-17). The small-cap value factor — the most documented premium in academic finance (see [[FINANCE/what-works-wall-street]]), tied to [[CODE/factor-screener]].
- **Discipline note:** permanent, decades-long sleeve, NOT a momentum chase. It WILL have ugly multi-year stretches — do not panic-sell when it lags; rebalance *into* it when it's down.

### AMD — Best Performer
- 1 share, up massively from $218.58 cost. Small in dollars despite the outsized % gain. No thesis beyond the run — decide whether to trim into strength.

### NVDA — Individual
- 4 shares, blended cost $191.00. Related: [[FINANCE/dcf-valuation]].

### PLTR — Holding, Check Back in a Year
- $194.08/share (3 shares). Decision made 2026-06-15: hold and reassess ~June 2027. No action until then.

### Tech Concentration Issue
QQQM (~25% NVDA, ~7% GOOG, ~5% AMD) + individual NVDA/GOOG/AMD + VTI's tech weight = one big correlated large-cap tech bet. VXUS + AVUV are the first deliberate offset. QQQM is now consciously a **tech tilt within the satellite budget**, not stable core.

---

## Open Decisions

1. **VTI vs AVUS for the core** — whether to shift the core fund to Avantis US Equity (mild value/profitability/size tilt at 0.15%) over time.
2. **AMD** — trim the winner into strength, or hold? No active thesis.
3. **Roth IRA** — open and prioritize once earned income starts.
"""

    with open(WIKI_PORTFOLIO, "w") as f:
        f.write(summary)
    print(f"  Wiki updated: {WIKI_PORTFOLIO}")


if __name__ == "__main__":
    print("Fetching prices...")
    prices = fetch_prices(list(HOLDINGS.keys()))
    rows, total_value, total_cost, total_unreal, total_unreal_pct = build_report(prices)
    print_report(rows, total_value, total_cost, total_unreal, total_unreal_pct)
    update_wiki(rows, total_value, total_cost, total_unreal, total_unreal_pct)
