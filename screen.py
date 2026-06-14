# screen.py — the scaled, filtered factor screen (Phase 3 / Path A increment 3).
#
# Turns the raw ~4,300-name SimFin ranking into a real SMALL-CAP VALUE screen by
# applying the THESIS FILTERS before ranking — Quantitative Value's "margin of
# safety first": (1) market-cap band, (2) exclude financials/REITs, (3) an Altman
# Z distress scrub. Then rank the survivors by the existing 5-factor composite.
#
# Run:  python screen.py

import pandas as pd

from fundamentals import get_fundamentals, _simfin_load
from factors import calculate_factors, altman_z
from score import score

EXCLUDE_SECTORS = ("Financial Services", "Real Estate")   # banks + REITs (spec excludes)
MIN_CAP, MAX_CAP = 300e6, 5e9                             # $300M–$5B small-cap band
MIN_Z = 1.81                                              # Altman: below = distress, drop


def simfin_universe():
    """All SimFin US tickers that have income + balance + price data."""
    d = _simfin_load()
    sets = [set(d[k].index.get_level_values(0)) for k in ("income", "balance", "prices")]
    return sorted(t for t in set.intersection(*sets) if isinstance(t, str))


def run_screen(source="simfin", tickers=None, min_cap=MIN_CAP, max_cap=MAX_CAP,
               exclude_sectors=EXCLUDE_SECTORS, min_z=MIN_Z):
    """Filter the universe (band → ex-financials → distress) then rank survivors."""
    if tickers is None:
        tickers = simfin_universe() if source == "simfin" else None
    rows = []
    dropped = {"off_band": 0, "financial": 0, "distress": 0}
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
        rec = calculate_factors(f)
        rec["market_cap"] = mc
        rec["altman_z"] = z
        rows.append(rec)
    df = pd.DataFrame(rows)
    ranked = score(df).dropna(subset=["composite"]) if not df.empty else df
    print(f"universe {len(tickers)} -> dropped {dropped['off_band']} off-band, "
          f"{dropped['financial']} financials/REITs, {dropped['distress']} distressed "
          f"-> {len(df)} screened, {len(ranked)} with valid composite")
    return ranked


if __name__ == "__main__":
    r = run_screen()
    show = r.head(20).copy()
    show["mkt_cap_$B"] = (show["market_cap"] / 1e9).round(2)
    show["altman_z"] = show["altman_z"].round(2)
    print("\n=== TOP 20 — small-cap value screen ($300M–$5B, ex-financials, Altman Z >= 1.81) ===")
    print(show[["ticker", "composite", "n_factors", "mkt_cap_$B", "altman_z"]].to_string(index=False))
