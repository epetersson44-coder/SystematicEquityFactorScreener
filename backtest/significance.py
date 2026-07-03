# backtest/significance.py — is the edge real, or did we just test until something stuck?
#
# Two hand-rolled tools (no scipy; Phi from stdlib statistics.NormalDist):
#
# 1. PSR / DSR (Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio"). The
#    Probabilistic Sharpe Ratio asks: given n observations and the return series' actual
#    skew/kurtosis, what's the probability the TRUE Sharpe exceeds a benchmark SR*? The
#    DEFLATED Sharpe raises that hurdle to the Sharpe you'd expect the BEST of N unskilled
#    trials to post by luck — the correction for "we tried leverage, stops, macro timing,
#    carry, blends, universes ... and kept the winner". A strategy whose PSR against that
#    inflated hurdle stays high is a real finding, not the champion of a lottery.
#
# 2. Moving-block bootstrap of a Sharpe DIFFERENCE: resample paired daily returns in
#    ~monthly blocks (preserving cross-correlation and short-range autocorrelation),
#    recompute Sharpe(A) - Sharpe(B) each time, and read the p-value / CI off the
#    distribution. Answers "is the blend's Sharpe actually distinguishable from SPY's
#    on one 20-year sample?" — a question the point estimates alone cannot answer.
#
# Everything works in PER-PERIOD (daily) Sharpe units internally; annualized numbers are
# for display only.

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest.constants import TRADING_DAYS

_PHI = NormalDist()
_EULER = 0.5772156649015329


def _moments(ret):
    """(mean, std, skew, kurtosis) of a return series; kurtosis is Pearson (normal=3)."""
    x = np.asarray(ret, dtype=float)
    x = x[np.isfinite(x)]
    m, s = x.mean(), x.std(ddof=1)
    z = (x - m) / s
    return m, s, float((z ** 3).mean()), float((z ** 4).mean())


def sharpe_daily(ret, rf_daily=0.0):
    """Per-period (daily) Sharpe of a return series."""
    x = np.asarray(ret, dtype=float)
    x = x[np.isfinite(x)]
    return (x.mean() - rf_daily) / x.std(ddof=1)


def psr(ret, sr_benchmark_daily=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark), skew/kurtosis-aware."""
    sr = sharpe_daily(ret)
    _, _, g3, g4 = _moments(ret)
    n = np.isfinite(np.asarray(ret, dtype=float)).sum()
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark_daily) * math.sqrt(n - 1) / denom
    return _PHI.cdf(z)


def expected_max_sharpe(n_trials, trial_sr_var_daily):
    """E[max daily SR] across n_trials unskilled strategies whose trial SRs have the given
    variance — the luck hurdle a survivor must clear (BLdP eq. for E[max] of Gaussians)."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(trial_sr_var_daily) * (
        (1.0 - _EULER) * _PHI.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER * _PHI.inv_cdf(1.0 - 1.0 / (n_trials * math.e)))


def dsr(ret, n_trials, trial_sharpes_annual=None, trial_sr_var_daily=None):
    """Deflated Sharpe Ratio: PSR against the expected-max-of-N-trials hurdle.

    The trial-SR variance can be given directly (daily units) or estimated from a list of
    the ANNUALIZED Sharpes actually observed across the trials — the honest bookkeeping
    BLdP ask for ('keep track of the number of backtests conducted')."""
    if trial_sr_var_daily is None:
        if not trial_sharpes_annual or len(trial_sharpes_annual) < 2:
            raise ValueError("need trial_sr_var_daily or >= 2 recorded trial Sharpes")
        daily = np.asarray(trial_sharpes_annual, dtype=float) / math.sqrt(TRADING_DAYS)
        trial_sr_var_daily = float(daily.var(ddof=1))
    hurdle = expected_max_sharpe(n_trials, trial_sr_var_daily)
    return psr(ret, sr_benchmark_daily=hurdle), hurdle * math.sqrt(TRADING_DAYS)


def block_bootstrap_sharpe_diff(ret_a, ret_b, block=21, n_boot=2000, seed=0):
    """Moving-block bootstrap of Sharpe(A) - Sharpe(B) on PAIRED daily returns.

    Rows are resampled in contiguous blocks of `block` days (both series take the SAME
    blocks, preserving their correlation — essential: the diff's variance depends on it).
    Returns dict with the observed diff (annualized), bootstrap CI, and p_value =
    P(diff <= 0) — the probability A's Sharpe edge over B is luck."""
    df = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()
    a, b = df["a"].to_numpy(), df["b"].to_numpy()
    n = len(a)
    if n < block * 5:
        raise ValueError(f"too few paired observations ({n}) for block={block}")
    ann = math.sqrt(TRADING_DAYS)
    observed = (sharpe_daily(a) - sharpe_daily(b)) * ann
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts_max = n - block
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[k] = (sharpe_daily(a[idx]) - sharpe_daily(b[idx])) * ann
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"observed_diff": observed, "ci95": (float(lo), float(hi)),
            "p_value_luck": float((diffs <= 0).mean()), "n_days": n, "n_boot": n_boot}


# ----------------------------------------------------------------------------- the memo run
# The trial LEDGER: annualized Sharpes of the distinct "can it beat SPY?" constructions this
# lab actually evaluated over summer 2026, reconstructed from the committed experiments.
# BLdP's deflation needs exactly this bookkeeping — the survivor must beat the best of THESE.
TRIAL_SHARPES = [
    0.67,   # SMA 50/200 on SPY (Phase 1)
    0.44,   # walk-forward optimized SMA (Phase 2)
    0.79,   # long-only xs momentum, raw (S&P panel)
    0.89,   # momentum + 200d trend filter (the live book)
    1.05,   # momentum L/S 5/5 (2021-25 window, the regime mirage)
    0.23,   # momentum L/S with both failsafes, full cycle
    -0.55,  # pairs portfolio (2020-26 OOS)
    0.66,   # value screener top-20 (EDGAR full cycle)
    0.59,   # screener monthly (compare_configs)
    0.57,   # screener + 20% stop-loss
    0.59,   # screener levered 1.5x
    0.51,   # momentum levered 1.5x
    0.59,   # macro / yield-curve regime switch
    0.36,   # EFA buy-and-hold
    0.71,   # trend sleeve standalone (6 ETF, single-look)
    0.59,   # trend sleeve, 10-ETF universe
    0.54,   # trend sleeve, long-short
    0.82,   # SPY+trend+carry 3-way blend
    0.76,   # 50/50 SPY+trend
    0.81,   # 60/40 SPY+trend
    0.85,   # risk-parity SPY+trend (levered-cap era)
    0.90,   # risk-parity SPY+trend, unleveraged (the prior headline)
    0.93,   # vol-managed blend (Moreira-Muir overlay)
    0.94,   # 1/3/12 ensemble blend, all-offset average (the current headline)
    # leverage_study.py (2026-07-01, real-rf financing) — bull-window candidates:
    0.87,   # blend levered 1.5x
    0.83,   # blend levered 2.0x
    0.82,   # blend levered 2.3x (SPY-vol-matched)
    0.69,   # return stack SPY + 0.5x trend overlay
    0.73,   # return stack SPY + 1.0x trend overlay
    0.65,   # Gayed LRS 2x (200d MA LETF rotation)
    0.63,   # Gayed LRS 3x
    # SSO-mix (retail return-stack) sensitivity, full cycle:
    0.82,   # 25% SSO + 75% trend
    0.71,   # 40% SSO + 60% trend
    0.66,   # 50% SSO + 50% trend (the original sso_stack book)
    0.61,   # 65% SSO + 35% trend
    # sso_stack optimization round (pre-specified A-E, 2026-07-01), full cycle:
    0.69,   # B: 33% UPRO + 67% trend (ADOPTED -> the live sso_stack construction)
    0.79,   # C: 50% SSO w/ 200d Gayed filter + 50% trend (rejected: kills bull revenue)
    0.80,   # D: 33% UPRO w/ 200d Gayed filter + 67% trend (rejected: same)
    0.75,   # E: 25% UPRO + 75% trend, 75/75 balanced (rejected: loses beat-SPY thesis)
    0.93,   # Yang-Zhang vol estimator in the sleeve (wash: identical Sharpe, -0.8% turnover)
    0.93,   # signal-TYPE ensemble (return-sign+SMA200+Donchian x looks): wash vs looks-only
    1.13,   # BTC as 7th sleeve asset, 2015+ — SURVIVOR-SELECTED window (banked, not adopted:
            # ~+2.3%/yr is beta rental of the decade's best asset found ex post; the design
            # argument (trend-gated 3.5% slot, self-liquidated in both -70% winters) stands
            # on its own and is a structural call, not a backtest call)
]


def memo_report(n_boot=2000):
    """The honest significance read on the headline blend vs SPY. Slow-ish (runs the
    all-21-offset ensemble sweep first): python -m backtest.significance"""
    from backtest.trend_sleeve import etf_panel
    from backtest.timing_luck import sweep, tranched_curve, blend_curve

    panels = etf_panel()
    spy_px = panels["Close"]["SPY"].dropna()
    _, curves = sweep(panels=panels)                      # adopted ensemble, all 21 offsets
    blend_eq = blend_curve(tranched_curve(curves, tuple(range(21))), spy_px)
    df = pd.DataFrame({"blend": blend_eq, "spy": spy_px}).dropna().pct_change().dropna()

    ann = math.sqrt(TRADING_DAYS)
    sr_b, sr_s = sharpe_daily(df["blend"]) * ann, sharpe_daily(df["spy"]) * ann
    print(f"window {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days)")
    print(f"annualized Sharpe (rf=0):  blend {sr_b:.3f}   SPY {sr_s:.3f}\n")

    print(f"PSR (P[true Sharpe > 0]):            blend {psr(df['blend']):.4f}")
    n_led = len(TRIAL_SHARPES)
    for n_trials in (n_led, 50, 100):
        d, hurdle = dsr(df["blend"], n_trials, trial_sharpes_annual=TRIAL_SHARPES)
        print(f"DSR, N={n_trials:3d} trials (luck hurdle SR {hurdle:.2f} ann.):  {d:.4f}")
    print(f"  (ledger has {n_led} recorded constructions; 50/100 = paranoia sensitivity)\n")

    bb = block_bootstrap_sharpe_diff(df["blend"], df["spy"], n_boot=n_boot)
    print(f"block bootstrap, Sharpe(blend) - Sharpe(SPY), {bb['n_boot']} resamples:")
    print(f"  observed {bb['observed_diff']:+.3f}   95% CI [{bb['ci95'][0]:+.3f}, {bb['ci95'][1]:+.3f}]")
    print(f"  P(edge is luck) = {bb['p_value_luck']:.4f}")


if __name__ == "__main__":
    memo_report()
