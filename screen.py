# screen.py — the scaled, filtered factor screen (Phase 3 / Path A increment 3).
#
# Turns the raw ~4,300-name SimFin ranking into a real SMALL-CAP VALUE screen by
# applying the THESIS FILTERS before ranking — Quantitative Value's "margin of
# safety first": (1) market-cap band, (2) exclude financials/REITs, (3) an Altman
# Z distress scrub. Then rank the survivors by the existing 5-factor composite.
#
# Run:  python screen.py

import pandas as pd

from fundamentals import get_fundamentals, _simfin_load, piotroski_fscore
from factors import calculate_factors, altman_z, beneish_m
from score import score

EXCLUDE_SECTORS = ("Financial Services", "Real Estate")   # banks + REITs (spec excludes)
MIN_CAP, MAX_CAP = 300e6, 5e9                             # $300M–$5B small-cap band
MIN_Z = 1.81                                              # Altman: below = distress, drop
MAX_M = -1.78                                             # Beneish: above = likely manipulator, drop


def simfin_universe():
    """All SimFin US tickers that have income + balance + price data."""
    d = _simfin_load()
    sets = [set(d[k].index.get_level_values(0)) for k in ("income", "balance", "prices")]
    return sorted(t for t in set.intersection(*sets) if isinstance(t, str))


def run_screen(source="simfin", tickers=None, min_cap=MIN_CAP, max_cap=MAX_CAP,
               exclude_sectors=EXCLUDE_SECTORS, min_z=MIN_Z, max_m=MAX_M, sector_neutral=True):
    """Filter the universe (band → ex-financials → distress → manipulators) then rank.
    sector_neutral=True (default) ranks each factor within its sector — the honest
    point-in-time backtest showed plain ranking took a big accidental sector bet (cheap
    energy/cyclicals) worth ~9%/yr; ranking within sector removes it (see factor_backtest)."""
    if tickers is None:
        tickers = simfin_universe() if source == "simfin" else None
    rows = []
    dropped = {"off_band": 0, "financial": 0, "distress": 0, "manipulator": 0}
    for t in tickers:
        try:
            f = get_fundamentals(t, source=source)
        except Exception:
            continue
        mc = f.get("market_cap")
        if mc is None or not (min_cap <= mc <= max_cap):
            dropped["off_band"] += 1
            continue
        if f.get("sector") in exclude_sectors:
            dropped["financial"] += 1
            continue
        z = altman_z(f)
        if z is not None and z < min_z:                  # only drop KNOWN-distressed
            dropped["distress"] += 1
            continue
        m = beneish_m(f)
        if m is not None and m > max_m:                  # only drop KNOWN-manipulators
            dropped["manipulator"] += 1
            continue
        rec = calculate_factors(f)
        rec["market_cap"] = mc
        rec["altman_z"] = z
        rec["beneish_m"] = m
        rec["sector"] = f.get("sector")                  # for sector-neutral ranking
        rec["fscore"] = piotroski_fscore(t)              # Piotroski F-Score (quality)
        rows.append(rec)
    df = pd.DataFrame(rows)
    ranked = score(df, sector_neutral=sector_neutral).dropna(subset=["composite"]) if not df.empty else df
    print(f"universe {len(tickers)} -> dropped {dropped['off_band']} off-band, "
          f"{dropped['financial']} financials/REITs, {dropped['distress']} distressed, "
          f"{dropped['manipulator']} manipulators -> {len(df)} screened, "
          f"{len(ranked)} with valid composite")
    return ranked


def run_short_screen(source="simfin", tickers=None, min_cap=MIN_CAP, max_cap=MAX_CAP,
                     exclude_sectors=EXCLUDE_SECTORS):
    """Rank the universe for SHORTING — the inverted screen (the dedicated short side).

    The long screen SCRUBS distress (Altman Z) and manipulation (Beneish M) to protect
    the long book. Those are exactly the best shorts, so the short screen does the
    opposite: it keeps them and turns them into positive short SIGNALS. The short_score
    blends three equal legs (mean of whatever's available per name):
      (1) inverted factor composite  — expensive + low quality scores high,
      (2) distress                   — LOW Altman Z scores high,
      (3) manipulation               — HIGH Beneish M scores high.
    Top of the ranking = most attractive shorts. The band + sector filters still apply
    (same investable small-cap, ex-financials universe) — only the distress/manipulation
    SCRUB is removed. Equal leg weights are a v1 choice; tune later if warranted."""
    if tickers is None:
        tickers = simfin_universe() if source == "simfin" else None
    rows = []
    dropped = {"off_band": 0, "financial": 0}
    for t in tickers:
        try:
            f = get_fundamentals(t, source=source)
        except Exception:
            continue
        mc = f.get("market_cap")
        if mc is None or not (min_cap <= mc <= max_cap):
            dropped["off_band"] += 1
            continue
        if f.get("sector") in exclude_sectors:
            dropped["financial"] += 1
            continue
        rec = calculate_factors(f)
        rec["market_cap"] = mc
        rec["altman_z"] = altman_z(f)
        rec["beneish_m"] = beneish_m(f)
        rec["sector"] = f.get("sector")                          # for sector-neutral ranking
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        print("short universe -> 0 names")
        return df

    ranked = score(df, sector_neutral=True).dropna(subset=["composite"])   # need a real factor read
    inv = 100.0 - ranked["composite"]                            # bad fundamentals -> high
    distress = ranked["altman_z"].rank(pct=True, ascending=False) * 100   # low Z -> high
    manip = ranked["beneish_m"].rank(pct=True, ascending=True) * 100      # high M -> high
    legs = pd.concat([inv.rename("inv_factor"), distress.rename("distress"),
                      manip.rename("manip")], axis=1)
    ranked["short_score"] = legs.mean(axis=1)                    # mean of available legs
    ranked["short_legs"] = legs.notna().sum(axis=1)
    ranked = ranked.sort_values("short_score", ascending=False)
    print(f"short universe {len(tickers)} -> dropped {dropped['off_band']} off-band, "
          f"{dropped['financial']} financials/REITs -> {len(ranked)} ranked "
          f"({int(distress.notna().sum())} w/ Altman-Z signal, "
          f"{int(manip.notna().sum())} w/ Beneish-M signal)")
    return ranked


if __name__ == "__main__":
    r = run_screen()
    show = r.head(20).copy()
    show["mkt_cap_$B"] = (show["market_cap"] / 1e9).round(2)
    show["altman_z"] = show["altman_z"].round(2)
    show["beneish_m"] = show["beneish_m"].round(2)
    print("\n=== TOP 20 — small-cap value screen (LONG) ===")
    print("($300M–$5B, ex-financials, Altman Z >= 1.81, Beneish M <= -1.78)")
    print(show[["ticker", "composite", "n_factors", "mkt_cap_$B", "altman_z", "beneish_m"]].to_string(index=False))

    s = run_short_screen()
    sshow = s.head(20).copy()
    sshow["mkt_cap_$B"] = (sshow["market_cap"] / 1e9).round(2)
    sshow["short_score"] = sshow["short_score"].round(1)
    sshow["altman_z"] = sshow["altman_z"].round(2)
    sshow["beneish_m"] = sshow["beneish_m"].round(2)
    print("\n=== TOP 20 — inverted SHORT screen ===")
    print("(expensive + low-quality + distress signal + manipulation signal; scrubs become signals)")
    print(sshow[["ticker", "short_score", "short_legs", "mkt_cap_$B", "altman_z", "beneish_m"]].to_string(index=False))
