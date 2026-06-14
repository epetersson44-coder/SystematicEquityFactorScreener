# Quant Lab — Scoreboard

Every strategy vs SPY buy-and-hold, run through the same engine.
Convention: SPY 2000→2026, daily, fill = next-bar open, 2 bps cost.
Two rows per strategy — the naive `rf=0 / cash 0%` and the honest `rf=4% / cash 4%`
(idle cash earns T-bills; Sharpe charges the same rf). The honest one is the truth.

### Honest assumptions — rf = 4%, idle cash earns 4%
| Strategy | CAGR | Vol | Sharpe | Sortino | Calmar | Max DD | DD duration | Final ($10k) | Trades |
|---|---|---|---|---|---|---|---|---|---|
| **SPY buy & hold** _(benchmark)_ | 8.30% | 19.3% | 0.30 | 0.29 | 0.15 | -55.2% | **6.6 yr** | $82,271 | 1 |
| **SMA 50/200** | **9.35%** | **12.8%** | **0.45** | **0.35** | **0.28** | **-33.7%** | **1.8 yr** | **$106,387** | 27 |

### Naive assumptions — rf = 0, cash earns 0% (for reference / why it misled)
| Strategy | CAGR | Sharpe | Calmar | Max DD | DD duration | Final ($10k) |
|---|---|---|---|---|---|---|
| SPY buy & hold | 8.30% | 0.51 | 0.15 | -55.2% | 6.6 yr | $82,244 |
| SMA 50/200 | 8.08% | 0.67 | 0.24 | -33.7% | 1.9 yr | $77,945 |

**Read:** Under honest assumptions the SMA wins on **both** return *and* risk — CAGR
9.35% vs 8.30% and a far smoother ride (vol 12.8% vs 19.3%, max DD −34% vs −55%).
The earlier "loses on return" verdict was an artifact of scoring its idle cash at 0%;
it sits in cash ~29% of days, and real T-bills lift its return above the index.

**The duration metric earns its keep:** depth said −55% vs −34%; *duration* says SPY
spent **6.6 years underwater** (the 2000 top wasn't reclaimed until 2007, then 2008
erased it again) versus the SMA's 1.8. Chan's point exactly — time underwater, not
just depth, is what ends a strategy's life. A −34% / 1.8-yr hole is survivable; a
−55% / 6.6-yr one breaks most people's discipline.

**But neither clears the bar.** With a realistic rf, both Sharpes (0.30, 0.45) sit
below Chan's ~1.0 rule of thumb for "worth trading." What we've built is honest and
correct; it is not yet alpha. That's the point of Phase 2.

**Caveats (still open):** in-sample, a single untuned 50/200 choice (robustness =
Phase 2 walk-forward), single asset, no survivorship-bias-free universe. Don't read
the SMA's edge as proven — it's one parameter pair on one index over one history.

_Regenerate: `python -m backtest.tests.test_backtest` (31 checks) then the engine run._
