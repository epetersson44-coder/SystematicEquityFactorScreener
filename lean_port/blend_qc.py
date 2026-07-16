# lean_port/blend_qc.py — THIRD-PARTY replication of the blend on QuantConnect LEAN.
#
# The final rung of the verification ladder (see backtest/replica.py's honesty caveat):
# a different ENGINE (LEAN's event loop, fills, fee models) on different DATA
# (QuantConnect's equity feed, not Yahoo). Paste this single file into a new algorithm
# in the QuantConnect web IDE and run a backtest 2006-07-01 -> 2026-07-01. Compare
# against the reference numbers in lean_port/README.md. Agreement within tolerance
# means the lab's entire result survives an engine AND vendor swap; a large breach is
# a finding to chase, never to shrug at.
#
# CONSTRUCTION (the spec, same as backtest/replica.py):
#   Sleeve: every 21 trading days, each of SPY/EFA/TLT/IEF/GLD/DBC gets an ensemble
#   strength = fraction of {21,63,252}-day returns that are positive; strength>0 assets
#   are weighted strength x inverse-vol (63d), normalized, then scaled by
#   min(10%/portfolio-vol, 1.0) — long-only, never borrows; the un-deployed remainder
#   stays in cash. Blend: expanding-window inverse-vol risk parity (min 252 obs)
#   between SPY and the sleeve's own return stream, 50/50 until warm — the
#   "implementable book" flavor of the lab's headline.
#
# KNOWN CONVENTION GAPS vs the reference (listed so nobody chases them as bugs):
#   - LEAN's default cash earns 0 -> the reference numbers below were computed with
#     cash at 0 too (NOT the lab's honest-^IRX headline).
#   - LEAN charges Interactive Brokers commissions by default vs the lab's 5/10bps
#     proportional model: small, systematic, expected to cost a few bps/yr.
#   - Rebalance day counting starts at algorithm start, not at the lab's offset-0
#     panel date: a timing-luck offset difference; the lab's 21-offset spread
#     [0.72, 0.82] naive bounds how much that can matter.
from AlgorithmImports import *
import numpy as np


class BlendReplica(QCAlgorithm):
    LOOKS = [21, 63, 252]
    VOL_LB = 63
    TARGET_VOL = 0.10
    EVERY = 21
    MIN_RP_OBS = 252

    def Initialize(self):
        self.SetStartDate(2006, 7, 1)
        self.SetEndDate(2026, 7, 1)
        self.SetCash(100000)
        self.tickers = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC"]
        self.syms = {t: self.AddEquity(t, Resolution.Daily).Symbol for t in self.tickers}
        self.SetBenchmark(self.syms["SPY"])
        self.day = 0
        self.sleeve_w = {}                      # current sleeve weights {ticker: w}
        self.prev_close = {}                    # for the sleeve's shadow return stream
        self.sleeve_rets = []                   # sleeve daily returns (expanding RP input)
        self.spy_rets = []
        self.SetWarmUp(self.LOOKS[-1] + 1, Resolution.Daily)

    def OnData(self, data):
        closes = {}
        for t, s in self.syms.items():
            bar = data.Bars.get(s) if data.Bars else None
            if bar is not None:
                closes[t] = float(bar.Close)
        if len(closes) < len(self.tickers):
            return

        # shadow-account the sleeve's own daily return (needed for expanding RP)
        if self.prev_close:
            sret = sum(w * (closes[t] / self.prev_close[t] - 1.0)
                       for t, w in self.sleeve_w.items() if t in self.prev_close)
            self.sleeve_rets.append(sret)
            self.spy_rets.append(closes["SPY"] / self.prev_close["SPY"] - 1.0)
        self.prev_close = dict(closes)

        if self.IsWarmingUp:
            return
        self.day += 1
        if self.day % self.EVERY != 1:          # rebalance every 21st trading day
            return

        hist = self.History(list(self.syms.values()), self.LOOKS[-1] + 1, Resolution.Daily)
        if hist.empty:
            return
        px = hist["close"].unstack(level=0)
        px.columns = [self.tickers[[str(self.syms[t].ID) for t in self.tickers].index(str(c))]
                      if str(c) in [str(self.syms[t].ID) for t in self.tickers] else str(c)
                      for c in px.columns]
        # LEAN column naming varies by version; fall back to matching by symbol value
        cols = {}
        for c in px.columns:
            for t in self.tickers:
                if t in str(c):
                    cols[c] = t
        px = px.rename(columns=cols)
        if not all(t in px.columns for t in self.tickers):
            return

        strength = {}
        for t in self.tickers:
            s = px[t].dropna()
            if len(s) < self.LOOKS[-1] + 1:
                continue
            votes = [1.0 if s.iloc[-1] / s.iloc[-1 - lk] > 1.0 else 0.0 for lk in self.LOOKS]
            v = sum(votes) / len(votes)
            if v > 0:
                strength[t] = v
        if strength:
            window = px[list(strength)].iloc[-(self.VOL_LB + 1):].pct_change().iloc[1:]
            sd = window.std()
            raw = {t: strength[t] / sd[t] for t in strength if sd[t] > 0}
            tot = sum(raw.values())
            raw = {t: w / tot for t, w in raw.items()}
            cov = window.cov() * 252
            wv = np.array([raw[t] for t in raw])
            cv = cov.loc[list(raw), list(raw)].values
            pvol = float(np.sqrt(wv @ cv @ wv))
            scale = min(self.TARGET_VOL / pvol, 1.0) if pvol > 0 else 1.0
            self.sleeve_w = {t: w * scale for t, w in raw.items()}
        else:
            self.sleeve_w = {}

        # expanding-window risk parity between SPY and the sleeve stream
        if len(self.sleeve_rets) >= self.MIN_RP_OBS:
            ss = float(np.std(self.spy_rets, ddof=1))
            ts = float(np.std(self.sleeve_rets, ddof=1))
            w_spy = (1 / ss) / (1 / ss + 1 / ts) if ss > 0 and ts > 0 else 0.5
        else:
            w_spy = 0.5

        total = {t: (1.0 - w_spy) * w for t, w in self.sleeve_w.items()}
        total["SPY"] = total.get("SPY", 0.0) + w_spy
        for t in self.tickers:
            self.SetHoldings(self.syms[t], total.get(t, 0.0))
