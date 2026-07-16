# PRE-REGISTERED (2026-07-15, committed BEFORE running — two-commit proof).
#
# HYPOTHESIS (the one UNTESTED dial in the margin-era decision package — playbook item
# (e), listed since 2026-07-05 but never run): VOL-TARGETED LEVERAGE — scaling the
# blend's external leverage down when its realized vol spikes (Moreira-Muir class,
# "volatility-managed portfolios") — dominates CONSTANT 2.3x for the ~$110k book:
# materially shallower drawdowns and tails at comparable efficiency and acceptable
# terminal cost. Mechanism: leverage x vol is the ruin arithmetic; vol clusters and is
# forecastable at 1-2 month horizons (the same fact the sleeve's vol target exploits);
# de-levering INTO vol spikes cuts exactly the tail the margin book fears — and
# structurally widens the margin-call cushion (breach at 2.3x PM needs a -33.5%
# intra-month blend move; dynamic-L makes high-L coincide with LOW vol regimes).
#
# SPEC (fixed before the run, no parameter mining):
#   Base curve: the honest tranched blend (cash at ^IRX, all-21-offset tranche) — the
#   SAME curve as the 2026-07-05 honest leverage ladder, so rows are comparable.
#   CONSTANT book: L=2.3 daily-rebalanced, financing (L-1) at ^IRX+40bps
#   (levered_returns — reproduces the ladder row: 15.4% CAGR, -34.8% maxDD, $176k).
#   DYNAMIC book: L_t = clip(V*/sigma_t, 1.0, 2.3), updated every 21 trading days;
#   sigma_t = trailing-63d annualized vol of the UNLEVERED blend (the sleeve's own
#   vol_lb, no new parameter); V* = 2.3 x expanding-MEDIAN of sigma (>=252d warmup,
#   lookahead-free; constant 2.3 until warm). Floor 1.0 (the margin book never goes
#   below unlevered — design choice, stated). Financing (L_t-1) at ^IRX+40bps daily;
#   5bps charged on |delta L| at each update (leverage changes trade the whole book).
#   TAIL ANNEX (pre-specified): block bootstrap of the unlevered blend's daily returns
#   (21d blocks, full-length paths, N=500, seed 7), BOTH overlays applied to every
#   path; report median and p95 of max drawdown.
#
# ADOPTION BAR (playbook default flips to dynamic-L only if ALL legs pass):
#   history: honest exSharpe(dyn) >= exSharpe(const) - 0.01 AND maxDD(dyn) shallower
#   by >= 3pts AND terminal(dyn) >= 80% of terminal(const);
#   bootstrap: p95 maxDD(dyn) shallower than p95 maxDD(const) by >= 5pts.
#   PASS -> pre-registered stage-2 OOS annex (1999-2006 dot-com proxy panel) REQUIRED
#   before the playbook flips (the DMOM lesson). FAIL any leg -> banked as a priced row
#   in the tail-management menu; constant-L stays the reference.
# EXPECTATION ON RECORD: lean PASS. Vol management is among the robust published
# results, the mechanism is direct, and the sleeve already proves vol forecastability
# at this horizon on this panel. Expected shape: maxDD cut 5-10pts, p95 tail cut more,
# terminal cost 5-20% of the constant book's pile, Sharpe wash-to-slightly-up. The
# honest risk: vol-managed leverage lags in V-shaped recoveries (de-levered at the
# bottom) — 2009 and 2020 will price that.
# LEDGER: ONE trial (naive-convention dynamic-book Sharpe in the results commit).
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from backtest.trend_sleeve import etf_panel
from backtest.timing_luck import sweep, tranched_curve, blend_curve
from backtest.leverage_study import tbill_series, SPREAD
from backtest import metrics

L_MAX, L_MIN, VOL_LB, UPDATE, WARM = 2.3, 1.0, 63, 21, 252
BOOT_N, BOOT_BLOCK, BOOT_SEED = 500, 21, 7

panels = etf_panel()
spy = panels["Close"]["SPY"].dropna()
rf = tbill_series(spy.index)

_, curves_h = sweep(panels=panels, cash_rate=rf)
blend = blend_curve(tranched_curve(curves_h, tuple(range(21))), spy).dropna()
b_ret = blend.pct_change().dropna()
rf_d = (rf.reindex(b_ret.index).ffill().fillna(0.0) + SPREAD) / 252.0
rf_plain = (rf.reindex(b_ret.index).ffill().fillna(0.0)) / 252.0


def lev_series(ret):
    """L_t series for the dynamic rule on a return stream (lookahead-free)."""
    sig = ret.rolling(VOL_LB).std() * math.sqrt(252)
    med = sig.expanding(WARM).median()
    L = pd.Series(L_MAX, index=ret.index, dtype=float)
    cur = L_MAX
    for i in range(len(ret)):
        if i % UPDATE == 0 and np.isfinite(sig.iloc[i]) and np.isfinite(med.iloc[i]) and sig.iloc[i] > 0:
            cur = float(np.clip(L_MAX * med.iloc[i] / sig.iloc[i], L_MIN, L_MAX))
        L.iloc[i] = cur
    return L


def apply_leverage(ret, L, fin_d):
    """Levered daily returns: L*r - (L-1)*financing - 5bps on |dL| at updates."""
    if np.isscalar(L):
        L = pd.Series(float(L), index=ret.index)
    dL = L.diff().abs().fillna(0.0)
    lr = L * ret - (L - 1.0) * fin_d - dL * 5e-4
    return (1.0 + lr).cumprod()


def stats(eq):
    r = eq.pct_change().dropna()
    ex = r - rf_plain.reindex(r.index).fillna(0.0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {"cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
            "exs": float(ex.mean() / ex.std() * math.sqrt(252)),
            "naive": float(r.mean() / r.std() * math.sqrt(252)),
            "dd": metrics.max_drawdown(eq),
            "terminal": float(eq.iloc[-1] / eq.iloc[0] * 10_000)}


const = apply_leverage(b_ret, 2.3, rf_d)
L_dyn = lev_series(b_ret)
dyn = apply_leverage(b_ret, L_dyn, rf_d)
sc, sd = stats(const), stats(dyn)

print("HONEST convention, 2006-26 tranched blend, external leverage overlays:")
for name, s in (("constant 2.3x", sc), ("dynamic L<=2.3", sd)):
    print(f"{name:15s} CAGR {s['cagr']*100:6.2f}%  exSharpe {s['exs']:.2f}  "
          f"maxDD {s['dd']*100:6.1f}%  $10k -> ${s['terminal']:>10,.0f}")
print(f"dynamic L: mean {L_dyn.mean():.2f}, %time at cap {(L_dyn >= L_MAX - 1e-9).mean()*100:.0f}%, "
      f"min {L_dyn.min():.2f}")

# tail annex: block bootstrap, both overlays on identical paths
rng = np.random.default_rng(BOOT_SEED)
vals = b_ret.to_numpy()
n = len(vals)
dd_c, dd_d = [], []
idx = b_ret.index
for _ in range(BOOT_N):
    starts = rng.integers(0, n - BOOT_BLOCK, size=n // BOOT_BLOCK + 1)
    path = np.concatenate([vals[s:s + BOOT_BLOCK] for s in starts])[:n]
    pr = pd.Series(path, index=idx)
    fin = rf_d.to_numpy()[:n]
    fin_s = pd.Series(fin, index=idx)
    dd_c.append(metrics.max_drawdown(apply_leverage(pr, 2.3, fin_s)))
    dd_d.append(metrics.max_drawdown(apply_leverage(pr, lev_series(pr), fin_s)))
dd_c, dd_d = np.array(dd_c), np.array(dd_d)
print(f"\nbootstrap tails (N={BOOT_N}, {BOOT_BLOCK}d blocks): "
      f"const maxDD p50 {np.median(dd_c)*100:.1f}% p95 {np.percentile(dd_c, 5)*100:.1f}%  |  "
      f"dynamic p50 {np.median(dd_d)*100:.1f}% p95 {np.percentile(dd_d, 5)*100:.1f}%")

p95_c, p95_d = np.percentile(dd_c, 5), np.percentile(dd_d, 5)
ok = (sd["exs"] >= sc["exs"] - 0.01
      and sd["dd"] >= sc["dd"] + 0.03
      and sd["terminal"] >= 0.80 * sc["terminal"]
      and p95_d >= p95_c + 0.05)
print(f"\nVERDICT vs pre-registered bar: "
      f"{'STAGE-1 PASS -> pre-register the OOS annex before the playbook flips' if ok else 'FAIL -> banked as a priced tail-management row'}")

# RESULTS (run 2026-07-15, unmodified from pre-registration 51a4d8b; honest convention):
#   constant 2.3x   CAGR 15.47%  exSharpe 0.76  maxDD -34.8%  $10k -> $178,174
#   dynamic L<=2.3  CAGR 13.44%  exSharpe 0.76  maxDD -31.0%  $10k -> $125,037
#   dynamic L: mean 2.00, at cap 45% of days, min 1.00
#   bootstrap (N=500, 21d blocks): const p50 -38.0% / p95 -54.6%; dyn p50 -34.8% / p95 -51.6%
#   VERDICT: FAIL (2 of 4 legs) — Sharpe held (0.76=0.76, pass) and historical DD
#   improved 3.8pts (pass), but TERMINAL kept only 70% of the pile (bar 80%) and the
#   p95 tail improved just 3.0pts (bar 5). Pre-run expectation (lean PASS) was WRONG —
#   fourth author-overrule of the summer.
#   WHY (the decision-relevant mechanism): the blend is ALREADY internally vol-managed
#   by the sleeve — its vol is pre-compressed (8.4% ann, worst month -6.9%) — so an
#   external Moreira-Muir overlay finds almost no vol-clustering signal left and just
#   averages L down (mean 2.0) without timing skill. THE DOMINATION: the ladder's own
#   constant 2.0x row ($133k, -30.8%, 0.76) beats dynamic<=2.3 ($125k, -31.0%, 0.76)
#   on the pile at the same DD/Sharpe. Vol-managing a vol-managed book double-charges.
#   VERDICT FOR THE PLAYBOOK: dial (e) CLOSED [EMPIRICAL] — tail management at the
#   margin era is a LADDER CHOICE (pick L on the constant ladder; every rung is priced
#   honestly), not an overlay. The remaining tail tools are the banked DBMF slice and
#   the paid-options studies at that era. Ledger: naive 0.86 -> TRIAL_SHARPES.
