# CURRENT STATE — read this first

*Updated 2026-07-10. If you are reviewing this repo (human or AI): it is **not one product**.
It is a research lab containing live strategies, retired negative results kept on purpose, and
banked options. Reviewing a retired module as if it were the product is the #1 misread.*

**The question this lab exists to answer:** can a systematic strategy beat SPY over long
periods, net of realistic frictions — and can its operator hold it through the drawdowns?

---

## ACTIVE — the live system (6-ETF allocation + return stack)

Three paper books locked monthly via the tracker (`backtest/tracker.py`, immutable dated JSON
in `backtest/picks/`, git-committed forward — survivorship-free by construction):

| Book | Construction | Role |
|---|---|---|
| `blend` | risk-parity SPY + vol-targeted trend sleeve (SPY/EFA/TLT/IEF/GLD/DBC, 1/3/12-mo TSMOM ensemble), cash in SGOV | the headline: wins the RIDE. Naive convention (rf=0, cash 0%): Sharpe 0.94 vs SPY 0.65. **HONEST convention (cash at ^IRX, excess Sharpe — the quotable row per SCOREBOARD's own standard): 0.78 vs 0.57, gap +0.22, p(luck) 5.1%**; implementable single-offset book: median 0.78 [0.73, 0.83] — ≈ the tranche ideal. maxDD −16% vs −55%, 2006–26, net |
| `sso_stack` ("ssoB") | 33% UPRO + 67% × the same trend sleeve, residual SGOV (~167% notional, no margin) | goes for the PILE: beats SPY's raw return in every tested window incl. crisis-free bulls; accepts ~SPY crash depth |
| `momentum` | S&P 500 12-1 cross-sectional momentum + 200d trend failsafe; risk-off months lock 100% SGOV (2026-07-10 — was flat 0% cash, the one book violating SCOREBOARD's own idle-cash lesson; ninth review F1, worth ~+1%/yr at today's bills) | kept as the one surviving stock-selection record, for the memo |

Watch-only: `shadow` — the 2.3× levered blend derived from the same locks (the ~$110k
portfolio-margin era construction). Never presented as tradeable today. Honest-convention
row (2026-07-05, `experiments/2026-07-05_honest_leverage_ladder.py`): CAGR 15.4%, honest
excess Sharpe 0.76 (vs blend 0.78 — leverage is ~Sharpe-free at rf+40bps; rungs are priced
in DRAWDOWN, −16%→−35%, not efficiency), maxDD −35%, $10k→$176k. Dominates ssoB (0.62,
12.3%) on every honest axis; purely capital-gated. **Open design
question, deferred to that era's first lock: tail management at 2.3× (synthetic tails
−70%+). Known dials, cheapest first: size 1.8–2.0× (synthetic median maxDD mid-30s, per
`leverage_study.py`), vol-target the leverage (dynamic L ≤ 2.3), the banked DBMF slice.
A real put/collar overlay study requires REAL options data (OptionMetrics/ORATS-class,
paid) — simulating option prices from VIX rules-of-thumb is fabrication and will not be
done; that subscription is the first legitimate paid-data unlock of the margin era. The
same paid-data study should price buy-write/put-write overlays (the classic slow-market
income tool — three gates today: conditional deployment = regime timing [closed 0.59],
honest testing = options data, permanent deployment sells the upside ssoB buys).**

**The slow-core (Japanified-US) operating rule, from the transplant receipts:** tripwires
detect TREND death; the parallel books detect LEVERAGE death. On a dead equity core,
blend out-piles ssoB (EWJ: $32.8k vs $26.8k) because daily-reset decay eats the 3× in
chop — so a Japan decade shows up in the desk itself as the paper blend line walking away
from the real ssoB line for years while the sleeve keeps earning. Sanctioned response: a
[POLICY] mix rotation (ssoB → blend construction) at a lock, decided from the live desk
evidence, never from a forecast.

**Real money:** from 2026-07-13 the ssoB construction runs in a real ~$8k account
(substitutions SPLG/IAU/PDBC; order sheets only via `tracker.shopping_list()`). Terms:
changes only at monthly locks, tinker budget zero, pre-committed drawdown protocol in the
Brain2.0 wiki (`CODE/quant-desk.md`) — depth is never a tripwire, only premise-death is.
Forward expectation on record: ~SPY +1–1.5%/yr net of costs, PRE-TAX (current-bracket tax
drag ≈0–0.3%/yr — the headline and the tax caveat below travel together).
**Tax caveat (taxable account):** the sleeve's monthly rotation realizes mostly SHORT-TERM
gains; at the operator's current bracket (~0–12%, low income) the drag is ≈0–0.3%/yr —
small but not modeled. Mechanics that keep it small: locks trade DRIFT (a few % of the
book/month), tax hits only the gain slice of lots sold, and trend naturally rides winners
(deferral) / sells losers (auto-offset). One-time setting: cost-basis method at Chase →
Specific-ID/highest-cost-first (minimizes realized gains at every rebalance, free). It scales with income and capital: at a 24%+ bracket the same
turnover could consume 0.5–1%/yr of the expected edge. To be MEASURED, not guessed, from
the realized lock history at each year-end and written here; future new-savings
contributions should weigh a Roth IRA wrapper, where this line item vanishes.

**Success metrics, pre-committed per book (evaluation horizon 3–5 years — months are
noise):** `sso_stack` succeeds on the PILE: cumulative raw return ≥ SPY's; drawdowns count
against it only if materially deeper than SPY's (that depth is the accepted price, not the
metric). `blend` succeeds on the RIDE: Sharpe/maxDD vs SPY (trailing SPY's raw return in
bulls is by design, not failure). `momentum` is a record, not a bet — its metric is
whether the live edge matches the (survivorship-flattered) backtest at all. Falsifiers =
the drawdown-protocol tripwires: ~zero excess Sharpe on blend over 5+ live years, or
industry-wide trend-following death.

**Statistical power of these metrics, measured (2026-07-05, honest convention):** ssoB's
pile metric is FORMALLY UNDECIDABLE on the pre-committed horizon — edge +1.24%/yr at
tracking error 6.4%/yr needs ~26 years for even weak (t=1) evidence, ~106 for
conventional significance; its honest excess-Sharpe gap over SPY is +0.05, p(luck)=0.18
(statistically nothing, by design — it's mostly beta). Nobody may grade ssoB's edge off
its live P&L on any horizon we'll act on. What CAN be graded: the blend's excess Sharpe
(the powered tripwire) and the MECHANISM firing live — in a real equity crisis the
sleeve must rotate defensive (to cash/havens); a crisis it ends still fully long is
premise-death regardless of the P&L. One more falsifier added on that basis.

## RETIRED — negative results, kept as findings (do not "fix", do not re-run as live)

*Each closed item is tagged: **[EMPIRICAL]** = the math said no (a backtest verdict, in the
ledger) vs **[POLICY]** = the operator said no (implementability / tail tolerance — a design
decision that could legitimately be revisited if the constraint changes). Do not re-litigate
a POLICY closure as if it were refuted statistically, or vice versa.*

- **`factor` (the original small-cap value screener** — the repo's namesake): **zero edge**,
  proven by a survivorship-free point-in-time EDGAR backtest (`backtest/edgar_backtest.py`,
  `backtest/factor_backtest.py`) after fixing an inverted scorer. The composite's IC ≈ 0; the
  F-Score weight was window-overfit and dropped. The rigor is the deliverable. Its frozen
  paper book still `report`s monthly; no new locks. [EMPIRICAL]
- **`factor_ls`** — weak long-short variant. Frozen likewise. [EMPIRICAL]
- **Pairs trading** — negative OOS (Sharpe −0.55, 2020–26). [EMPIRICAL]
- **Macro/yield-curve regime switching, carry (free-data version), stop-loss overlays,
  faster-than-monthly cadence, small-cap anything** — tested, closed. See the trial ledger.
  [EMPIRICAL]
- **Volatility risk premium / VIX term structure (2026-07-04)** [POLICY] — closed WITHOUT a run, by
  operator decision on implementability + instrument tail: the only cash-account-holdable
  short-vol instrument (SVXY) lost ~90% in one day (Feb 2018), and that tail class is
  outside what the operator will ever hold with real money; short-vol is also
  crash-correlated (equity beta in disguise) in a book that already carries full equity
  beta. Recorded as a design verdict, not a backtest verdict — no ledger entry. (An
  external reviewer's proposed backtest for this family was rejected separately for
  fabricated pre-2009 data, a degenerate VXX/VXZ price-ratio signal, and cash-account
  shorting; the honest design — ^VIX/^VIX3M signal, long-only SVXY/VIXM, 2011+ — is
  documented here in case the implementability constraint ever changes.)

## BANKED — tested, not adopted; structural options awaiting a decision at a lock

- **DBMF/KMLM managed-futures slice** (`mf_etf=` in the tracker): full-cycle proxy says
  ~0.4%/yr cost for 2–5pt relief in slow crashes only. Insurance, not alpha.
- **BTC as 7th sleeve asset** (via IBIT): backtest (+0.18 Sharpe, 2015+) ruled *inadmissible*
  — survivor-selected asset. The trend-gating design argument stands alone; design call only.
- **Yang-Zhang OHLC vol estimator** (`backtest/volatility.py`): wash at monthly cadence.
- **BIL excess-return hurdle** (`hurdle_col`): wash.
- **Dual-momentum sleeve (DMOM top-3)** — TS gate + hard cross-sectional top-3 selection
  (Antonacci family; the sleeve's strength-weighting is already the soft version). Met
  the full in-sample dominance bar (blend honest exSharpe 0.803 vs 0.768, all 21
  offsets) then **FAILED the pre-registered OOS check** on the 1999–2006 proxy panel
  (0.986 vs 1.018, deeper DD): in long many-assets-trending regimes, dropping the
  weakest trender discards real diversification. Banked as in-sample-only dominance —
  the two-stage gate caught a would-be adoption artifact. Top-2 failed in-sample
  outright. (`backtest/experiments/2026-07-05_dual_momentum_sleeve.py` + `_dmom_robustness.py`)
- **ssoB defensive step-down** (equity leg UPRO→SPY when SPY < 200d SMA, checked at the
  monthly lock; `backtest/experiments/2026-07-04_ssoB_defensive.py`): halves the max
  drawdown (−29% vs −56%) at a near-tie in full-cycle terminal wealth, but loses BOTH bull
  windows to SPY itself (12.1% vs 13.1%, 21.3% vs 23.1%) — fails this book's beat-SPY-raw
  thesis, so not adopted. Recorded as the best-seen ride-vs-pile trade if that preference
  ever flips; that's a design call at a lock, not a backtest call.
- **Sleeve-internal gross >1 at honest financing** (margin era only; pre-reg `d7da27d`,
  `backtest/experiments/2026-07-10_sleeve_gross_honest.py`, ninth review F2): the old
  "leverage loses" closure was priced at flat-4% financing — re-run at ^IRX+40bps the
  **Sharpe leg flips** (blend median exSharpe 0.768→0.800 at G=2.0, worst offset improves
  too) but the **maxDD leg fails** (−16.8%→−20.8% median): the amplified-drawdown half was
  never a financing artifact. Priced menu row for the leverage era: +0.03 exSharpe per
  ~4pts maxDD. Mechanism on record: the gross cap binds on **80% of rebalances** (vol
  target wants median gross 1.43; capped sleeve realizes 7.5% vol vs the 10% design) — and
  the same measurement settles the cash account: raising `target_vol` there adds exposure
  only in the ~20% concentrated/crisis months where the TARGET binds, so **no in-cash-account
  vol-target headroom exists**. Closed [EMPIRICAL], banked for the 2.3× era's sleeve design.
- **Global repo sweep, 2026-07-12 (pre-go-live; ~30 repos, US/UK/DE/FR/CN/RU/JP/KR):**
  no construction found that dominates GTAA-class trend+RP at this lab's constraints — the
  serious labs (Carver, Keller family, AQR lineage) converge on the primitives already live
  here. Candidates banked WITH PRIORS, none touching the live book: (1) **canary-asset
  gating** (Keller VAA/DAA — gate aggression on fast momentum of EEM+BND breadth, not
  own-asset trend): the one genuinely new mechanism; moderate-low prior (published 2018,
  decay applies); test = two-stage pre-reg with dot-com proxy OOS, worth one ledger trial.
  (2) **HRP/HERC weighting** vs inverse-vol (via skfolio-style clustering, reimplemented
  natively): wash prior at 6 assets. (3) **Carver position buffering** (no-trade bands):
  wash-tier at monthly cadence, ~1–3bps. (4) **RSRS timing** (光大 support/resistance
  regression slope, QuantsPlaybook reproductions): best Chinese export, low prior — US
  timing trials all died. Tooling notes: **QuantConnect Lean** upgrades the banked
  clean-room replication (independent engine AND data vendor in one); **pysystemtrade**
  (Carver) is the Rung-3 futures-era reference implementation (in the playbook);
  **OpenSourceAP/CrossSection** (Chen–Zimmermann, 200+ replicated anomalies w/ code+data)
  is the revival reference if the retired factor path ever reopens.

## Implementation alpha — banked ops/tax/financing upgrades (signals untouched)

*Lateral pass 2026-07-05: the signal layer is closed; the remaining certain edges are
plumbing. Each fires at a capital rung, each is worth ~0.3–1%/yr with certainty — more
than any disputed signal idea of the review gauntlet.*

- **NOW (taxable account): tax-GAIN harvesting.** Erik is in the 0% LTCG bracket — each
  December, realize long-term gains up to the bracket ceiling and instantly rebuy (no
  wash-sale rule on gains): free basis step-up against the 15% rate his future self pays.
  Pairs with the Roth wrapper for new contributions (see tax caveat above).
- **Deep drawdowns, once ABOVE the 0% bracket: tax-LOSS harvesting at locks** — swap legs
  sitting at big losses into non-identical twins (UPRO→SPXL, SPLG→VOO, IAU→GLDM...),
  exposure identical, loss carried forward forever against future 15–24% gains. The one
  extra crash harvest that costs the strategy nothing. Sanctioned AT locks only (wired
  into the wiki drawdown protocol 2026-07-05); while at 0% bracket, harvest GAINS not
  losses. Drift-band rebalancing (more harvest in crashes) noted and DECLINED for the
  manual era — intra-month triggers violate the don't-watch-the-account covenant; banked
  for the automation era.
- **~$35k: one MES micro-future replaces UPRO** — same 100% S&P notional, no 0.91% ER, no
  daily-reset decay, financing at implied repo (~rf+30bps): ~0.5–1%/yr cheaper on the
  equity slice. Needs a futures-capable broker (IBKR — the ladder's destination anyway).
  Lumpy below $34k (0-or-1 contract), so it unlocks AT the rung, not before.
- **IBKR era: fully-paid securities lending on UPRO** (chronically hard-to-borrow;
  ~0.3–1%/yr on that slice for checking a box; Chase self-directed doesn't offer it).
- **~$110k portfolio-margin era: box-spread financing** — selling SPX boxes borrows at
  ~T-bills+30bps, upgrading the degraded middle-rung leverage pricing toward the clean
  2.3× the studies assume. Decide at that era's first lock alongside the tail-management
  question above.

## The evidence chain (where the proof lives)

- **Trial ledger:** `backtest/significance.py` `TRIAL_SHARPES` — every distinct "can it beat
  SPY?" construction evaluated (48 as of 2026-07-10; honest label: a LOWER BOUND on trials,
  reconstructed from committed experiments — the N=75/100 paranoia rows in `memo_report`
  are the mitigation; correlation note added 2026-07-10: trials are NOT independent, which
  makes the independent-N hurdle conservative — the two labels push in opposite directions
  and the paranoia rows bound the bad one). Naive convention: blend DSR **0.93**, bootstrap
  P(luck) **1.6%**.
- **Honest-convention memo (2026-07-05, two-commit pre-registration at `3fb1e82` — F2 of
  the sixth review, which correctly quoted SCOREBOARD.md's own "the honest one is the
  truth" back at us):** cash at real ^IRX in the engine + excess-return Sharpes. Blend
  **0.78 vs SPY 0.57 (gap +0.22)**, bootstrap p(luck) **5.1%** (stable at 63/126d blocks),
  DSR quoted as a RANGE per the lower-bound ledger honesty: **0.83 at ledger N=48 →
  0.77 @N=75 → 0.74 @N=100 paranoia** (recomputed 2026-07-10 on the median-offset
  implementable book — the same curve as the quoted 0.78 — after the ledger grew to 48;
  rf=0 ledger hurdle — conservative mix). Survived its pre-registered
  re-headline rule (gap ≥ +0.15, p ≤ 0.10) — but these honest numbers are the QUOTABLE
  ones now; the naive row stays for ledger comparability only. Implementable book (single
  offset, expanding RP, monthly mix costs): median 0.78 [0.73, 0.83] ≈ the tranche —
  the live construction sacrifices ~nothing to the design ideal.
  (`backtest/experiments/2026-07-05_honest_convention.py`)
- **Ninth review response (2026-07-10, `03a3aa0` + the sleeve-gross experiment):** 11
  findings verified against code, 10 real — the gauntlet's best hit rate (rounds 7–8 had
  decayed to recycled/fabricated claims; this reviewer cited real internals). Fixed: momentum
  risk-off → SGOV (F1, above), ONE delisting convention across all three views of a record
  (_simulate last-trade carry == report table == dashboard render — a mid-hold delisting
  used to show three different numbers), preflight now covers every real-account leg incl.
  REAL_SUBS, one-row-per-ticker order sheets, Sortino moved to the standard full-n
  target-downside convention, SimFin debt None-propagation (missing stays missing), Series
  financing in the engine, DSR correlation note, PII redaction in the repo pack.
- **Timing luck:** `backtest/timing_luck.py` — all-21-offset sweeps; adopted numbers are
  all-offset medians/tranches, not a lucky calendar day.
- **Leverage & LETF mechanics:** `backtest/leverage_study.py` — daily-reset simulation with
  real T-bill financing, validated vs real SSO/UPRO (corr 0.996+).
- **Synthetic falsification:** `backtest/synthetic.py` — null (iid) markets: sleeve earns
  ~zero minus costs (no machinery bias); AR-drift markets: strong positive; block-bootstrap
  real-correlation markets: wash. The system finds trend when it exists and nothing when it
  doesn't.
- **Dot-com extension (2026-07-04, two-commit pre-registration — bar committed at
  `2f2bf84` BEFORE the run):** proxy panel (VUSTX/VFITX/FDIVX/GC=F/WTI-spot, each
  validated on the 2006+ overlap) extends the exact construction through 1999–2006.
  Blend: Sharpe 0.97 vs SPY 0.11, dot-com-bear maxDD −9.4% vs −47.5% — PASSED both
  pre-registered conditions, so the blend's RIDE claim (Sharpe/maxDD dominance — its
  pinned metric, not raw return) spans the 2000–02 bear as well as 2008/2020/2022. ssoB-sim beat SPY over the lost half-decade too ($11.9k vs $10.2k per
  $10k) while eating the full SPY-shaped −47% — the pile thesis as designed. Level
  caveat: NAV smoothing flatters proxy Sharpes; the SIGN of the verdict is the finding.
  (`backtest/experiments/2026-07-04_dotcom_proxy_extension.py`; recorded as an OOS
  validation, not a ledger candidate — no selection occurred.) **Robustness annex (same
  day, pre-registered at `209ffa8`): PASS without gold, PASS without WTI (the weakest
  proxy, overlap corr 0.41), and the NAV-smoothing haircut MEASURED at ~0.08 Sharpe via
  Geltner unsmoothing (blend lag-1 autocorr +0.06; smoothing concentrated in FDIVX) —
  the verdict stands under all three. Rebuilds vary ~±0.07 Sharpe on panel-construction
  details: the sign is the finding, the second decimal is noise.**
- **International transplant (2026-07-05, two-commit pre-registration at `51e56bc`):**
  the exact ssoB/blend constructions transplanted onto Japan (EWJ — a core that went
  ~nowhere for two decades) and the Eurozone (EZU), same global sleeve, US financing
  (conservative). ALL FOUR pre-registered bars PASS: each transplant beats its own core
  on pile (ssoB) and ride (blend). Scope, stated honestly (per the sixth review's F5):
  this closes the CORE-choice bias question — the construction degrades gracefully on
  weak cores — but it does NOT test the sleeve's own selection-sample dependence (its
  universe/lookbacks/vol target were chosen on 2006–26 data and it still holds SPY as
  one of six assets). That residual is covered by the synthetic falsification and,
  ultimately, the live record — not by this test. Honest
  lesson inside the pass: leverage cannot resurrect a dead core (ssoB's edge over its
  core shrinks +1.1%/yr → +0.65%/yr on Japan; blend out-piles ssoB on weak cores) —
  which is why the tripwire watches the BLEND's excess Sharpe, not ssoB's. The 2.3×L
  transplant tripled the dead cores' terminals at −37/−40% maxDD.
  (`backtest/experiments/2026-07-05_intl_transplant.py`; OOS validation, no ledger entry.)
- **Ops guards:** `backtest/preflight.py` (mandatory before any lock), cross-vendor price
  check, complete-row guards in the tracker.
- **Conventions, now MEASURED (supersedes the old "known conservatisms" note):** the
  cash_rate=0 understatement and the rf=0 flattery were both quantified 2026-07-05 (see
  the honest-convention memo above): cash credit lifts the blend +0.03 Sharpe; charging
  rf costs the low-vol blend more than SPY; net honest gap +0.22 vs naive +0.32. Ledger
  trials remain rf=0/cash-0 for internal comparability — headline claims use the honest
  row. The momentum book's backtest is survivorship-flattered the other way — which is
  why its verdict is assigned to the forward record, not the backtest. LETF financing
  spread sensitivity (asked by three reviewers, answered analytically): ssoB's edge moves
  by w_eq×(L−1) = 0.67 × any spread change — a stress +60bps over the modeled 40bps costs
  ~0.4%/yr; the realized average is bounded by the SSO/UPRO validation (tracking gap
  −0.3/−0.6%/yr including 2008/2020).
- **Named UNTESTED premise (no honest test exists with free data): the fast inflationary
  crash.** The sleeve's crisis alpha in 2008/2020 was mostly long duration in a
  disinflationary regime; 2022 (slow inflationary bear) was survived via cash/commodities.
  A 1970s-style FAST joint crash — stocks and bonds gapping down together quicker than a
  1/3/12-month ensemble de-risks, commodities whipsawing — is outside every falsification
  tool in this repo (synthetics don't generate it; both proxy extensions are
  disinflationary bears). Published century evidence (Hurst-Ooi-Pedersen) covers the
  1970s favorably, but with futures shorts we don't have. This is the live blend's real
  premise risk; it stands alongside the trend-death tripwire and is why the mechanism
  check (does the sleeve rotate?) is a falsifier in its own right.

## Experiment protocol (the ritual — follow it or the ledger lies)

1. **Pre-specify before running:** hypothesis, exact construction, and the adoption bar,
   written in the experiment script's header — and the script is committed under
   `backtest/experiments/` (dated filename) so the evidence outlives the session. When
   feasible, commit the header (hypothesis + bar) BEFORE the run and the results in a
   second commit — the git hash chain then *proves* the bar predated the outcome instead
   of asserting it, and `python -m backtest.verify_prereg` machine-checks every claim
   (currently: 6 verified, 0 failed). No bar, no run.
2. **Control timing luck:** anything cadence-sensitive runs all 21 offsets; compare
   distributions, not single curves.
3. **Ledger in the same commit:** every trial's full-cycle annualized Sharpe is appended to
   `TRIAL_SHARPES` in `backtest/significance.py` with a one-line comment — including (
   especially) the failures. The DSR is only honest if the ledger is complete.
4. **Verdict is recorded, not remembered:** ADOPTED (code + A/B note in the module header),
   BANKED (listed above), or CLOSED (listed above) — update this file in the same commit.
5. **Live behavior changes only at monthly locks.** Between locks the answer is no.

## Reproducibility

- Env: Python 3.13 venv, pinned in `requirements.txt` (recreate command in its header).
- Tests: `.venv/bin/python -m pytest -q test_fundamentals.py backtest/tests/` — 161 tests.
  Stress probes: `.venv/bin/python backtest/tests/stress_test.py` (39 WARN-level probes).
- Monthly ritual: the `/picks` command file (`~/.claude/commands/picks.md`) is the runbook.
- Full methodology narrative: Brain2.0 wiki, `CODE/` pages (quant-desk, live-tracker,
  factor-backtest) + the research journal entries there.
