# edgar.py — survivorship-free, point-in-time fundamentals from raw SEC EDGAR filings.
#
# Escapes the SimFin data wall (free tier = 2020+ only) by pulling financial statement data
# straight from the SEC's XBRL "company facts" API — free, ~2009+, every public company,
# delisted names retained (survivorship-free), and every fact carries its FILING DATE so we
# can reconstruct exactly what was known on any past date (point-in-time, no look-ahead).
#
# The catch (the XBRL swamp): companies report the same line item under DIFFERENT us-gaap
# tags across years (Apple's revenue is "SalesRevenueNet" pre-2018, then "RevenueFrom
# ContractWithCustomer..."). So each canonical field maps to a PRIORITY LIST of tags; we
# take the primary where present and fall back to fill gaps.
#
# SEC rules: send a real User-Agent with contact, stay under 10 requests/sec. We cache every
# company's facts to disk so each is fetched once.
#
# Usage:  python -m edgar   (demo: AAPL fundamentals at several as-of dates)

import os
import json
import time

import requests

HEADERS = {"User-Agent": "Erik Petersson quant-lab epetersson44@gmail.com"}
EDGAR_DIR = os.path.expanduser("~/edgar_data")
FACTS_DIR = os.path.join(EDGAR_DIR, "facts")
_LAST = [0.0]                                            # rate-limit clock (last request time)

# canonical field -> priority list of us-gaap XBRL tags (primary first, then fallbacks)
FIELD_TAGS = {
    "revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit":        ["GrossProfit"],
    "net_income":          ["NetIncomeLoss"],
    "operating_income":    ["OperatingIncomeLoss"],
    "total_assets":        ["Assets"],
    "current_assets":      ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt":      ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cfo":                 ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "shares":              ["WeightedAverageNumberOfDilutedSharesOutstanding",
                            "WeightedAverageNumberOfSharesOutstandingBasic"],
}


def _get(url):
    """GET JSON from SEC with the required User-Agent, throttled to <10 req/sec."""
    wait = 0.12 - (time.time() - _LAST[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers=HEADERS, timeout=30)
    _LAST[0] = time.time()
    r.raise_for_status()
    return r.json()


def ticker_cik(refresh=False):
    """{TICKER: 10-digit CIK} from SEC's master list. Cached to disk."""
    path = os.path.join(EDGAR_DIR, "ticker_cik.json")
    if not refresh and os.path.exists(path):
        return json.load(open(path))
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    out = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
    os.makedirs(EDGAR_DIR, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def company_facts(cik, refresh=False):
    """The full XBRL company-facts JSON for a CIK. Cached to disk (fetched once). None if 404."""
    path = os.path.join(FACTS_DIR, f"CIK{cik}.json")
    if not refresh and os.path.exists(path):
        return json.load(open(path))
    try:
        facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    except requests.HTTPError:
        return None                                     # some CIKs have no XBRL facts
    os.makedirs(FACTS_DIR, exist_ok=True)
    json.dump(facts, open(path, "w"))
    return facts


def _annual(facts, field):
    """{fiscal_year_end_date: (value, filed_date)} for a canonical field — annual (10-K, FY)
    facts only, the EARLIEST filing per fiscal-year-end across ALL mapped tags. Earliest-
    across-tags (not primary-wins) is what point-in-time demands: a company that switches
    XBRL tags reports the old year under the new tag only in a LATER filing, so the original
    (earliest) filing — under whatever tag — is what was actually known at the time."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in FIELD_TAGS[field]:
        node = gaap.get(tag)
        if not node:
            continue
        for vals in node["units"].values():
            for x in vals:
                if x.get("form") not in ("10-K", "10-K/A") or x.get("fp") != "FY":
                    continue
                end, filed, val = x.get("end"), x.get("filed"), x.get("val")
                if not (end and filed and val is not None):
                    continue
                if end not in out or filed < out[end][1]:           # earliest filing wins
                    out[end] = (float(val), filed)
    return out


def fundamentals_asof(ticker, asof):
    """Point-in-time annual fundamentals KNOWN as of `asof` (current + prior fiscal year), from
    EDGAR. Returns a dict {field, field_prior, fy_end} or None if not covered / <2 years filed.
    Anchored on net income's fiscal-year-ends; each value taken only if filed on/before asof."""
    cik = ticker_cik().get(ticker.upper())
    if not cik:
        return None
    facts = company_facts(cik)
    if facts is None:
        return None
    asof = str(asof)
    series = {f: _annual(facts, f) for f in FIELD_TAGS}
    ends = sorted(e for e, (v, filed) in series["net_income"].items() if filed <= asof)
    if len(ends) < 2:
        return None
    cur_end, prior_end = ends[-1], ends[-2]

    def at(field, end):
        v = series[field].get(end)
        return v[0] if (v and v[1] <= asof) else None

    out = {"ticker": ticker.upper(), "fy_end": cur_end}
    for f in FIELD_TAGS:
        out[f] = at(f, cur_end)
        out[f + "_prior"] = at(f, prior_end)
    return out


if __name__ == "__main__":
    print("ticker->CIK entries:", len(ticker_cik()))
    for asof in ["2010-06-01", "2013-06-01", "2018-06-01", "2024-06-01"]:
        f = fundamentals_asof("AAPL", asof)
        if f:
            print(f"\nAAPL as of {asof}  (latest FY end {f['fy_end']}):")
            for k in ("revenue", "net_income", "total_assets", "cfo", "gross_profit", "shares"):
                v = f[k]
                print(f"  {k:16} {v:,.0f}" if v is not None else f"  {k:16} —")
