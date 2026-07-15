# backtest/replica.py — CLEAN-ROOM replica of the production calculation.
#
#   .venv/bin/python -m backtest.replica
#
# An independent reimplementation of everything real money rides on, written from the
# documented SPEC (CURRENT_STATE.md + docstring behavior), compared against the
# production engine's output within pre-stated tolerances. The engine modules
# (engine_xs, trend_sleeve strategy classes, timing_luck) are IMPORTED ONLY in the
# compare section as the reference — never used by the replica itself. The replica
# deliberately uses a DIFFERENT internal representation (holdings tracked as VALUES,
# vectorized signal math) than the engine (shares, event loop), so shared-transcription
# errors are unlikely to line up.
#
# HONESTY CAVEAT (stated, not hidden): written by the same author/AI that has read the
# engine, in the same repo. This is an independent IMPLEMENTATION, not an independent
# IMPLEMENTER — it catches spec-vs-code divergence, off-by-one drift, and silent
# convention changes (most of the practical risk), but it cannot catch a shared
# misconception. A future third-party rewrite (e.g. on QuantConnect Lean with its own
# data) remains the stronger version; this is the free local rung of that ladder.
#
# THE SPEC REPLICATED (from CURRENT_STATE + the /picks runbook):
#   Sleeve: 6 ETFs (SPY/EFA/TLT/IEF/GLD/DBC). At each rebalance bar (every 21 trading
#   days at an offset, needing >= 252 bars of history): per-asset ensemble strength =
#   fraction of {21,63,252}-day total returns that are positive; assets with strength
#   > 0 get weight strength x inverse-vol (63-day daily-return std), normalized to
#   sum 1; portfolio vol from the 63-day covariance annualized x252; scale =
#   min(10% / pvol, 1.0) (long-only, never borrows). Decisions at the close fill at
#   the NEXT bar's open; 5bps proportional cost on traded notional; positive cash
#   earns ^IRX daily.
#   Blend: risk parity between SPY and the sleeve curve — full-sample inverse-vol of
#   daily returns sets a constant mix, compounded daily.
#   ssoB lock: 1/3 UPRO + 2/3 x the sleeve's current target weights; residual (if the
#   vol target leaves cash) parks in SGOV.
#
# PRE-STATED TOLERANCES (any breach = a FINDING to investigate, never to widen):
#   T1 LOCK WEIGHTS: reproduced within 5e-6 (the lock rounds to 6dp) — FROM THE LOCK'S
#      OWN signal_snapshot when present (locks >= Aug 2026). The July 2026 lock
#      predates snapshots and CANNOT be byte-verified from a re-fetch: the first
#      replica run (2026-07-15) proved adjusted-history re-basing flipped IEF's
#      knife-edge 21d vote (~+0.0x% at lock -> -0.10% two days later), moving ~15
#      weight points — a DATA-VINTAGE effect, not an engine bug (T2 passed exactly on
#      the same formula). Pre-snapshot locks get construction checks only (UPRO leg
#      exactly 1/3, ticker sets match, weights sum to 1) plus a drift report.
#   T2 SLEEVE CURVE (offset 0, full panel): daily-return correlation >= 0.999,
#      |terminal ratio - 1| <= 1%, |naive Sharpe delta| <= 0.02 vs run_xs.
#   T3 BLEND CURVE: same three bounds vs timing_luck.blend_curve.
# Exit 1 on any breach.
import json
import math
import os
import sys

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKS = (21, 63, 252)
VOL_LB = 63
TARGET_VOL = 0.10
EVERY = 21
COST_BPS = 5.0
MIN_HIST = max(LOOKS)


# ---------------------------------------------------------------- the replica
def replica_weights(closes, i):
    """Sleeve target weights at bar i, from the spec. Returns {ticker: weight} or None
    (not a rebalance-capable bar)."""
    if i < MIN_HIST:
        return None
    strength = {}
    for t in closes.columns:
        p0 = closes[t].iloc[i]
        if not (np.isfinite(p0) and p0 > 0):
            continue
        votes = []
        for lk in LOOKS:
            pk = closes[t].iloc[i - lk]
            if not (np.isfinite(pk) and pk > 0):
                votes = None
                break
            votes.append(1.0 if p0 / pk > 1.0 else 0.0)
        if votes is None:
            continue
        s = sum(votes) / len(votes)
        if s > 0:
            strength[t] = s
    if not strength:
        return {}
    window = closes.iloc[i - VOL_LB:i + 1].pct_change().iloc[1:]
    on = list(strength)
    sd = window[on].std()
    if not np.isfinite(sd).all() or (sd <= 0).any():
        return {}
    raw = pd.Series({t: strength[t] / sd[t] for t in on})
    raw = raw / raw.sum()
    cov = window[on].cov() * TRADING_DAYS
    pvol = float(np.sqrt(raw.values @ cov.values @ raw.values))
    scale = min(TARGET_VOL / pvol, 1.0) if pvol > 0 else 1.0
    return (raw * scale).to_dict()


def replica_sleeve_curve(closes, opens, rf_daily, offset=0):
    """Daily equity curve of the sleeve from the spec: value-based holdings, next-open
    fills, 5bps on traded notional, positive cash at rf. Normalized to 1.0 at start."""
    n = len(closes)
    hold = {}                                              # ticker -> $ value (at last close)
    cash = 1.0
    pending = None
    curve = np.empty(n)
    c = closes.to_numpy()
    o = opens.to_numpy()
    cols = list(closes.columns)
    for i in range(n):
        if i > 0:
            if pending is not None:                        # fill at today's open
                at_open = {}
                for t, v in hold.items():
                    j = cols.index(t)
                    ratio = o[i, j] / c[i - 1, j]
                    at_open[t] = v * (ratio if np.isfinite(ratio) else 1.0)
                pv = sum(at_open.values()) + cash
                tgt = {t: w * pv for t, w in pending.items()}
                traded = sum(abs(tgt.get(t, 0.0) - at_open.get(t, 0.0))
                             for t in set(tgt) | set(at_open))
                fee = traded * COST_BPS / 10_000.0
                cash = pv - sum(tgt.values()) - fee
                hold = {t: v for t, v in tgt.items() if v != 0.0}
                for t in list(hold):                       # open -> close leg of fill day
                    j = cols.index(t)
                    ratio = c[i, j] / o[i, j]
                    hold[t] *= ratio if np.isfinite(ratio) else 1.0
                pending = None
            else:                                          # close -> close
                for t in list(hold):
                    j = cols.index(t)
                    ratio = c[i, j] / c[i - 1, j]
                    hold[t] *= ratio if np.isfinite(ratio) else 1.0
        if cash > 0:
            cash *= 1.0 + rf_daily[i]
        if i % EVERY == offset:
            w = replica_weights(closes, i)
            if w is not None:
                pending = w
        curve[i] = sum(hold.values()) + cash
    return pd.Series(curve, index=closes.index)


def replica_blend(sleeve_curve, spy_close):
    """Constant-mix risk-parity blend from the spec: full-sample inverse-vol split."""
    df = pd.DataFrame({"s": spy_close, "t": sleeve_curve}).dropna().pct_change().dropna()
    iv_s, iv_t = 1.0 / df["s"].std(), 1.0 / df["t"].std()
    w = iv_s / (iv_s + iv_t)
    return (1.0 + w * df["s"] + (1.0 - w) * df["t"]).cumprod()


def naive_sharpe(eq):
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------- the comparison
def main():
    import warnings
    warnings.filterwarnings("ignore")
    from backtest.trend_sleeve import etf_panel
    from backtest.leverage_study import tbill_series

    panels = etf_panel()
    closes, opens = panels["Close"], panels["Open"]
    rf = tbill_series(closes.index)
    rf_daily = (rf.reindex(closes.index).ffill().fillna(0.0) / TRADING_DAYS).to_numpy()
    spy = closes["SPY"].dropna()
    findings = []

    # T1 — every sso_stack lock, independently recomputed
    picks_dir = os.path.join(os.path.dirname(__file__), "picks", "sso_stack")
    for fname in sorted(f for f in os.listdir(picks_dir) if f.endswith(".json")):
        rec = json.load(open(os.path.join(picks_dir, fname)))
        snap = rec.get("signal_snapshot")
        if snap:                                           # byte-verification from the
            cols = sorted(snap["closes"])                  # lock's own frozen inputs
            win = pd.DataFrame({t: snap["closes"][t]["window64"] for t in cols})
            tw = {}
            strength, inv_vol = {}, {}
            sd = win.pct_change().iloc[1:].std()
            for t in cols:
                p0 = win[t].iloc[-1]
                votes = [1.0 if p0 / snap["closes"][t]["looks"][str(lk)] > 1.0 else 0.0
                         for lk in LOOKS]
                s = sum(votes) / len(votes)
                if s > 0 and np.isfinite(sd[t]) and sd[t] > 0:
                    strength[t], inv_vol[t] = s, 1.0 / sd[t]
            raw = pd.Series({t: strength[t] * inv_vol[t] for t in strength})
            raw = raw / raw.sum()
            cov = win[list(raw.index)].pct_change().iloc[1:].cov() * TRADING_DAYS
            pvol = float(np.sqrt(raw.values @ cov.values @ raw.values))
            scale = min(TARGET_VOL / pvol, 1.0) if pvol > 0 else 1.0
            tw = (raw * scale).to_dict()
            basis = "snapshot"
        else:
            i_lock = closes.index.get_loc(pd.Timestamp(rec["data_asof"]))
            tw = replica_weights(closes, i_lock)
            basis = "re-fetch (pre-snapshot lock — data-vintage drift expected)"
        net = {"UPRO": 1.0 / 3.0}
        for t, w in tw.items():
            net[t] = net.get(t, 0.0) + (2.0 / 3.0) * w
        resid = 1.0 - sum(net.values())
        if resid > 0.005:
            net["SGOV"] = resid
        worst = max(abs(rec["picks"].get(t, 0.0) - net.get(t, 0.0))
                    for t in set(rec["picks"]) | set(net))
        upro_ok = abs(rec["picks"].get("UPRO", 0.0) - 1.0 / 3.0) < 5e-6
        sum_ok = abs(sum(rec["picks"].values()) - 1.0) < 1e-4
        if snap:
            ok = worst <= 5e-6 and upro_ok and sum_ok
            if not ok:
                findings.append(f"T1 {fname}: snapshot reproduction worst dev {worst:.2e}")
            print(f"  T1 {rec['data_asof']} [{basis}]: worst dev {worst:.2e}  "
                  f"{'OK' if ok else 'BREACH'}")
        else:
            ok = upro_ok and sum_ok and set(rec["picks"]) - {"SGOV"} == set(net) - {"SGOV"}
            if not ok:
                findings.append(f"T1 {fname}: construction check failed "
                                f"(UPRO {upro_ok}, sum {sum_ok})")
            print(f"  T1 {rec['data_asof']} [{basis}]: construction "
                  f"{'OK' if ok else 'BREACH'}; weight drift vs today's vintage "
                  f"{worst:.4f} (reported, not judged)")

    # T2 — the sleeve curve vs the production engine
    from backtest.engine_xs import run_xs
    from backtest.trend_sleeve import VolTargetTSMOM, ENSEMBLE_LOOKS
    from backtest import costs
    engine = run_xs(panels, VolTargetTSMOM(max_gross=1.0, looks=ENSEMBLE_LOOKS, offset=0),
                    cost=costs.proportional(COST_BPS), fill="next_open", cash_rate=rf)
    mine = replica_sleeve_curve(closes, opens, rf_daily, offset=0)
    both = pd.DataFrame({"e": engine / engine.iloc[0], "m": mine / mine.iloc[0]}).dropna()
    r = both.pct_change().dropna()
    corr = float(r["e"].corr(r["m"]))
    term = float(both["m"].iloc[-1] / both["e"].iloc[-1])
    dsh = abs(naive_sharpe(both["m"]) - naive_sharpe(both["e"]))
    t2_ok = corr >= 0.999 and abs(term - 1) <= 0.01 and dsh <= 0.02
    if not t2_ok:
        findings.append(f"T2 SLEEVE: corr {corr:.5f}, terminal ratio {term:.4f}, dSharpe {dsh:.3f}")
    print(f"  T2 sleeve curve: ret-corr {corr:.5f}, terminal ratio {term:.4f}, "
          f"|dSharpe| {dsh:.3f}  {'OK' if t2_ok else 'BREACH'}")

    # T3 — the blend construction vs timing_luck.blend_curve
    from backtest.timing_luck import blend_curve
    ref_blend = blend_curve(engine, spy)
    my_blend = replica_blend(mine, spy)
    bb = pd.DataFrame({"e": ref_blend / ref_blend.iloc[0],
                       "m": my_blend / my_blend.iloc[0]}).dropna()
    rb = bb.pct_change().dropna()
    corr_b = float(rb["e"].corr(rb["m"]))
    term_b = float(bb["m"].iloc[-1] / bb["e"].iloc[-1])
    dsh_b = abs(naive_sharpe(bb["m"]) - naive_sharpe(bb["e"]))
    t3_ok = corr_b >= 0.999 and abs(term_b - 1) <= 0.01 and dsh_b <= 0.02
    if not t3_ok:
        findings.append(f"T3 BLEND: corr {corr_b:.5f}, terminal ratio {term_b:.4f}, dSharpe {dsh_b:.3f}")
    print(f"  T3 blend curve:  ret-corr {corr_b:.5f}, terminal ratio {term_b:.4f}, "
          f"|dSharpe| {dsh_b:.3f}  {'OK' if t3_ok else 'BREACH'}")

    print(f"\nreplica: {'ALL CHECKS PASS — spec and engine agree' if not findings else 'BREACHES: ' + '; '.join(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
