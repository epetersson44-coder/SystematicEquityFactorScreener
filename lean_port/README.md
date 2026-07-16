# Lean port — third-party replication of the blend

The strongest rung of the verification ladder: the lab's headline construction
re-implemented on **QuantConnect LEAN** — a different engine (their event loop, fills,
IB fee model) running on **different data** (QuantConnect's equity feed, not Yahoo).
The local clean-room replica (`backtest/replica.py`) already proved spec==engine on our
own data; this proves the result isn't an artifact of our data vendor or our simulator.

## How to run (10 minutes, free)

1. Create a free account at quantconnect.com (operator action — the lab does not
   create accounts).
2. New Algorithm → delete the template → paste `blend_qc.py` whole.
3. Run a backtest (defaults are set in the file: 2006-07-01 → 2026-07-01, $100k).
4. Compare the result to the reference table below.

## Reference numbers (computed 2026-07-15 from the lab's engine, SAME conventions:
## cash at 0, expanding-RP implementable book, 2006-07 → 2026-07)

| Metric | Lab reference | Agreement band | Breach means |
|---|---|---|---|
| CAGR | **8.08%** | 6.6% – 9.6% | investigate |
| Sharpe (rf=0) | **0.98** | 0.85 – 1.10 | investigate |
| Max drawdown | **−14.6%** | −11% to −19% | investigate |
| $100k terminal | **$473,534** | $360k – $620k | investigate |
| 2008–09 drawdown | single digits to −20% | | if ~−50%: the sleeve is not defending — a REAL breach |

The bands are wide on purpose: different data vendor, IB commissions vs 5–10bps,
different rebalance-day phase (the lab's own 21-offset naive Sharpe spread is
[0.72, 0.82] — timing luck alone moves these numbers). The test is **"same animal"**,
not byte equality: a diversified ~10%-vol book that compounds high-single-digits,
holds its 2008 drawdown to a fraction of SPY's, and lands in the bands. The local
replica (corr 1.00000) already covers exactness; this rung covers independence.

## Known convention gaps (listed so nobody chases them as bugs)

- LEAN default cash earns 0 → reference computed with cash at 0 (NOT the honest-^IRX
  headline row; do not compare against 0.78/0.57 — wrong convention).
- IB commission model vs proportional bps: a few bps/yr systematic drag.
- Day-count phase differs from the lab's offset 0: a timing-luck offset, priced above.

## If it breaches

A breach outside the bands is a FINDING (the replica's rule: investigate, never widen
the band). First suspects, in order: LEAN data mapping (check the column-matching
fallback in `OnData`), DBC/GLD data availability in early years on QC's feed, then —
only after those are excluded — the uncomfortable hypothesis the rung exists for.
