# CURRENT STATE — read this first

*Updated 2026-07-04. If you are reviewing this repo (human or AI): it is **not one product**.
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
| `blend` | risk-parity SPY + vol-targeted trend sleeve (SPY/EFA/TLT/IEF/GLD/DBC, 1/3/12-mo TSMOM ensemble), cash in SGOV | the headline: wins the RIDE (Sharpe ~0.94 vs SPY 0.64, maxDD −16% vs −55%, 2006–26, net of costs) |
| `sso_stack` ("ssoB") | 33% UPRO + 67% × the same trend sleeve, residual SGOV (~167% notional, no margin) | goes for the PILE: beats SPY's raw return in every tested window incl. crisis-free bulls; accepts ~SPY crash depth |
| `momentum` | S&P 500 12-1 cross-sectional momentum + 200d trend failsafe | kept as the one surviving stock-selection record, for the memo |

Watch-only: `shadow` — the 2.3× levered blend derived from the same locks (the ~$110k
portfolio-margin era construction). Never presented as tradeable today.

**Real money:** from 2026-07-13 the ssoB construction runs in a real ~$8k account
(substitutions SPLG/IAU/PDBC; order sheets only via `tracker.shopping_list()`). Terms:
changes only at monthly locks, tinker budget zero, pre-committed drawdown protocol in the
Brain2.0 wiki (`CODE/quant-desk.md`) — depth is never a tripwire, only premise-death is.
Forward expectation on record: ~SPY +1–1.5%/yr net (haircut from the +2.8% backtest edge).

**Success metrics, pre-committed per book (evaluation horizon 3–5 years — months are
noise):** `sso_stack` succeeds on the PILE: cumulative raw return ≥ SPY's; drawdowns count
against it only if materially deeper than SPY's (that depth is the accepted price, not the
metric). `blend` succeeds on the RIDE: Sharpe/maxDD vs SPY (trailing SPY's raw return in
bulls is by design, not failure). `momentum` is a record, not a bet — its metric is
whether the live edge matches the (survivorship-flattered) backtest at all. Falsifiers =
the drawdown-protocol tripwires: ~zero excess Sharpe on blend over 5+ live years, or
industry-wide trend-following death.

## RETIRED — negative results, kept as findings (do not "fix", do not re-run as live)

- **`factor` (the original small-cap value screener** — the repo's namesake): **zero edge**,
  proven by a survivorship-free point-in-time EDGAR backtest (`backtest/edgar_backtest.py`,
  `backtest/factor_backtest.py`) after fixing an inverted scorer. The composite's IC ≈ 0; the
  F-Score weight was window-overfit and dropped. The rigor is the deliverable. Its frozen
  paper book still `report`s monthly; no new locks.
- **`factor_ls`** — weak long-short variant. Frozen likewise.
- **Pairs trading** — negative OOS (Sharpe −0.55, 2020–26).
- **Macro/yield-curve regime switching, carry (free-data version), stop-loss overlays,
  faster-than-monthly cadence, small-cap anything** — tested, closed. See the trial ledger.
- **Volatility risk premium / VIX term structure (2026-07-04)** — closed WITHOUT a run, by
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
- **ssoB defensive step-down** (equity leg UPRO→SPY when SPY < 200d SMA, checked at the
  monthly lock; `backtest/experiments/2026-07-04_ssoB_defensive.py`): halves the max
  drawdown (−29% vs −56%) at a near-tie in full-cycle terminal wealth, but loses BOTH bull
  windows to SPY itself (12.1% vs 13.1%, 21.3% vs 23.1%) — fails this book's beat-SPY-raw
  thesis, so not adopted. Recorded as the best-seen ride-vs-pile trade if that preference
  ever flips; that's a design call at a lock, not a backtest call.

## The evidence chain (where the proof lives)

- **Trial ledger:** `backtest/significance.py` `TRIAL_SHARPES` — every distinct "can it beat
  SPY?" construction evaluated (43 as of 2026-07-02). Headline blend: **DSR 0.93** against
  the best-of-ledger luck hurdle; block-bootstrap P(edge = luck) **1.6%**.
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
  pre-registered conditions, so the beat-SPY claim spans the 2000–02 bear as well as
  2008/2020/2022. ssoB-sim beat SPY over the lost half-decade too ($11.9k vs $10.2k per
  $10k) while eating the full SPY-shaped −47% — the pile thesis as designed. Level
  caveat: NAV smoothing flatters proxy Sharpes; the SIGN of the verdict is the finding.
  (`backtest/experiments/2026-07-04_dotcom_proxy_extension.py`; recorded as an OOS
  validation, not a ledger candidate — no selection occurred.)
- **Ops guards:** `backtest/preflight.py` (mandatory before any lock), cross-vendor price
  check, complete-row guards in the tracker.
- **Known modeling conservatisms (deliberate, direction = backtest UNDERSTATES live):**
  headline sleeve/blend backtests run `cash_rate=0` — the vol-targeted sleeve's idle cash
  earns nothing in the backtest while the live books sweep it into SGOV at ~T-bill yield
  (engine supports `cash_rate=`; kept at 0 so all ledger trials share one convention).
  Expect small positive live-vs-backtest drift in high-rate regimes. The momentum book's
  backtest is survivorship-flattered the other way — which is why its verdict is assigned
  to the forward record, not the backtest.

## Experiment protocol (the ritual — follow it or the ledger lies)

1. **Pre-specify before running:** hypothesis, exact construction, and the adoption bar,
   written in the experiment script's header — and the script is committed under
   `backtest/experiments/` (dated filename) so the evidence outlives the session. When
   feasible, commit the header (hypothesis + bar) BEFORE the run and the results in a
   second commit — the git hash chain then *proves* the bar predated the outcome instead
   of asserting it. No bar, no run.
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
