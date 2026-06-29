# edgar.py — survivorship-free, point-in-time fundamentals from raw SEC EDGAR filings.
#
# Escapes the SimFin data wall (free tier = 2020+ only) by pulling financial statement data
# straight from the SEC's XBRL "company facts" API — free, ~2009+, every public company,
# delisted names retained (survivorship-free), and every fact carries its FILING DATE so we
# can reconstruct exactly what was known on any past date (point-in-time, no look-ahead).
#
# Two layers:
#   1. EXTRACTION (this file's core) — pull annual line items as-of any date, earliest-filing
#      wins, period-length guarded. The hard, SEC-specific part.
#   2. FACTORS — edgar_fundamentals_asof() emits the project's CANONICAL schema, so the
#      existing source-agnostic factor recipes (factors.py: ev_ebit, roic, gm_stability,
#      net_debt_ebitda, altman_z) work UNCHANGED. edgar_fscore_asof() is the point-in-time
#      Piotroski F-Score, parallel to fundamentals.piotroski_fscore_asof (SimFin). This is
#      what makes EDGAR a drop-in survivorship-free fundamental SOURCE back to 2009.
#
# The catch (the XBRL swamp): companies report the same line item under DIFFERENT us-gaap
# tags across years (Apple's revenue is "SalesRevenueNet" pre-2018, then "RevenueFrom
# ContractWithCustomer..."). So each canonical field maps to a PRIORITY LIST of tags; we
# take the earliest filing per fiscal-year-end across ALL of them (see _annual). Per-name
# drift is normalized as far as is reliable: ProfitLoss as a net-income anchor fallback,
# goods+services revenue SUMMING for pre-ASC-606 retail/service filers, and EBIT reconstructed
# from pretax+interest for single-step income statements. This lifts small-cap factor coverage
# to ~10 in 11; a long tail of exotic tags remains (the rest).
#
# Coverage reality (MEASURED, not assumed): XBRL was phased in — large filers ~2009, all
# filers ~2011 — so usable history starts ~2007-08 for large-caps but ~2009-10 for most
# small-caps. Two consequences: (1) the practical backtest window is ~2010-present (~15yr,
# capturing the 2020 COVID crash + 2022 bear + 2015-16 + 2018-Q4 selloffs — three real
# drawdowns vs SimFin-free's zero), and (2) the 2008-09 GFC sits AT or BEFORE where small-cap
# data begins, so it can't be cleanly crash-tested for that universe. EDGAR roughly triples
# SimFin-free's window and adds real bears; it does NOT reach the GFC for small-caps.
#
# SEC rules: send a real User-Agent with contact, stay under 10 requests/sec. We cache every
# company's facts to disk so each is fetched once.
#
# Usage:  python -m edgar   (demo: AAPL canonical fundamentals + factors + F-Score, as-of dates)

import os
import json
import time
from datetime import date

import requests

HEADERS = {"User-Agent": "Erik Petersson quant-lab epetersson44@gmail.com"}
EDGAR_DIR = os.path.expanduser("~/edgar_data")
FACTS_DIR = os.path.join(EDGAR_DIR, "facts")
_LAST = [0.0]                                            # rate-limit clock (last request time)

# canonical field -> priority list of us-gaap XBRL tags (primary first, then fallbacks).
# Grouped by statement. Flows (income/cash-flow) are duration facts; instants (balance) carry
# no period start. _annual handles both and guards flow facts to the full annual period.
FIELD_TAGS = {
    # --- income statement (flows) ---
    "revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit":        ["GrossProfit"],
    "operating_income":    ["OperatingIncomeLoss"],
    "net_income":          ["NetIncomeLoss", "ProfitLoss"],   # ProfitLoss (incl. NCI) fills older years
                                                              # some filers tag only ProfitLoss (e.g. SKYW)
    "pretax_income":       ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_expense":         ["IncomeTaxExpenseBenefit"],
    "interest_expense":    ["InterestExpense", "InterestAndDebtExpense", "InterestExpenseDebt"],
    "dep_amort":           ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                            "DepreciationAndAmortization"],
    "shares":              ["WeightedAverageNumberOfDilutedSharesOutstanding",
                            "WeightedAverageNumberOfSharesOutstandingBasic"],
    # --- balance sheet (instants) ---
    "total_assets":        ["Assets"],
    "current_assets":      ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities":   ["Liabilities"],
    "equity":              ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings":   ["RetainedEarningsAccumulatedDeficit"],
    "cash":                ["CashAndCashEquivalentsAtCarryingValue",
                            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
                            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt":      ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt":     ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    # --- cash flow (flows) ---
    "cfo":                 ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex":               ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
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


_FACTS_CACHE = {}


def company_facts(cik, refresh=False):
    """The full XBRL company-facts JSON for a CIK. Cached to disk (fetched once) AND in memory
    (the multi-MB JSON is re-read across many as-of calls in a backtest). None if 404."""
    if not refresh and cik in _FACTS_CACHE:
        return _FACTS_CACHE[cik]
    path = os.path.join(FACTS_DIR, f"CIK{cik}.json")
    if not refresh and os.path.exists(path):
        facts = json.load(open(path))
        _FACTS_CACHE[cik] = facts
        return facts
    try:
        facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    except requests.HTTPError:
        return None                                     # some CIKs have no XBRL facts
    os.makedirs(FACTS_DIR, exist_ok=True)
    json.dump(facts, open(path, "w"))
    _FACTS_CACHE[cik] = facts
    return facts


def _facts_for(ticker):
    """Company-facts JSON for a ticker (via the SEC ticker->CIK map), or None if uncovered."""
    cik = ticker_cik().get(ticker.upper())
    return company_facts(cik) if cik else None


_SIC = {}


def company_sic(cik):
    """The 4-digit SIC industry code (str) for a CIK, from the SEC SUBMISSIONS endpoint
    (companyfacts has no SIC). Cached to disk in one small {cik: sic} map. '' if unknown."""
    path = os.path.join(EDGAR_DIR, "sic.json")
    if not _SIC and os.path.exists(path):
        _SIC.update(json.load(open(path)))
    if cik in _SIC:
        return _SIC[cik]
    try:
        sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        sic = str(sub.get("sic") or "")
    except requests.HTTPError:
        sic = ""
    _SIC[cik] = sic
    os.makedirs(EDGAR_DIR, exist_ok=True)
    json.dump(_SIC, open(path, "w"))
    return sic


def _sic_to_sector(sic):
    """Coarse SIC -> sector label. Names match the SimFin/yfinance sectors the screen already
    keys on, so screen.EXCLUDE_SECTORS ('Financial Services'/'Real Estate') filters correctly —
    that exclusion is the load-bearing part and is checked FIRST. The rest is granular enough
    to give a stable grouping for sector-neutral ranking, not a GICS-exact mapping. Unmapped /
    unknown SICs return None (dropped under sector-neutral rather than spuriously top-ranked)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    if 6500 <= s <= 6599 or s == 6798:                               # real estate, REITs
        return "Real Estate"
    if 6000 <= s <= 6799:                                            # banks, insurers, funds
        return "Financial Services"
    if 1300 <= s <= 1399 or 2900 <= s <= 2999 or 4922 <= s <= 4925:  # oil & gas, refining, gas
        return "Energy"
    if 4900 <= s <= 4991:                                            # utilities
        return "Utilities"
    if (3570 <= s <= 3579 or 3670 <= s <= 3679 or 3660 <= s <= 3669
            or 3820 <= s <= 3829 or 7370 <= s <= 7379):              # computers, semis, software
        return "Technology"
    if 2833 <= s <= 2836 or 3840 <= s <= 3851 or s == 3826 or 8000 <= s <= 8099:  # pharma, devices, health svc
        return "Health Care"
    if 2700 <= s <= 2799 or 4800 <= s <= 4899 or 7800 <= s <= 7841:  # publishing, telecom, media
        return "Communication Services"
    if (100 <= s <= 999 or 2000 <= s <= 2199 or 5400 <= s <= 5499    # agriculture, food/bev/tobacco, grocery
            or s == 5912):                                          # drug stores
        return "Consumer Defensive"
    if (1000 <= s <= 1299 or 1400 <= s <= 1499 or 2400 <= s <= 2499
            or 2600 <= s <= 2899 or 3300 <= s <= 3399):              # mining, paper, chemicals, metals
        return "Basic Materials"
    if (2300 <= s <= 2399 or 2500 <= s <= 2599 or 3000 <= s <= 3199 or 3710 <= s <= 3716
            or 5000 <= s <= 5999 or 7000 <= s <= 7299 or 7500 <= s <= 7699
            or 7900 <= s <= 7999 or 8100 <= s <= 8399):              # apparel, autos, retail, repair, leisure
        return "Consumer Cyclical"
    if (1500 <= s <= 1799 or 3400 <= s <= 3569 or 3580 <= s <= 3659 or 3680 <= s <= 3699
            or 3720 <= s <= 3799 or 4000 <= s <= 4799 or 7300 <= s <= 7369
            or 8700 <= s <= 8744):                                   # machinery, aero/defense, transport, svcs
        return "Industrials"
    return None


def _span_days(start, end):
    """Calendar days from start to end ('YYYY-MM-DD' strings), or None if unparseable."""
    try:
        y1, m1, d1 = (int(p) for p in start.split("-"))
        y2, m2, d2 = (int(p) for p in end.split("-"))
        return (date(y2, m2, d2) - date(y1, m1, d1)).days
    except Exception:
        return None


def _annual_tags(facts, tags):
    """{fiscal_year_end_date: (value, filed_date)} across a list of us-gaap tags — annual
    (10-K, FY) facts only, the EARLIEST filing per fiscal-year-end across ALL the tags. Two
    rules make this point-in-time correct:
      * earliest-across-tags (not primary-wins): a company that switches XBRL tags reports the
        old year under the new tag only in a LATER filing, so the original (earliest) filing —
        under whatever tag — is what was actually known at the time. When two tags share a
        filing date (same 10-K), the EARLIER-listed tag wins (so the primary tag is preferred).
      * period guard: flow facts (which carry a period `start`) are kept only if they span a
        full year (>=300d), so a quarter/stub period reported inside a 10-K can't masquerade as
        the annual figure. Instant facts (balance sheet) carry no start and pass through."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for tag in tags:
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
                start = x.get("start")
                if start is not None:                              # flow fact -> must be a full year
                    span = _span_days(start, end)
                    if span is not None and span < 300:
                        continue
                if end not in out or filed < out[end][1]:          # earliest filing wins
                    out[end] = (float(val), filed)
    return out


def _annual(facts, field):
    """{fiscal_year_end_date: (value, filed_date)} for a canonical field via its tag list.
    Revenue is special-cased: pre-ASC-606 (~2018) filers often report no single total-revenue
    tag, splitting it into goods + services lines that must be SUMMED (see _annual_revenue)."""
    if field == "revenue":
        return _annual_revenue(facts)
    return _annual_tags(facts, FIELD_TAGS[field])


def _annual_revenue(facts):
    """Split-aware annual revenue. A total-revenue tag (FIELD_TAGS['revenue']) wins where it
    exists; for any fiscal-year-end it doesn't cover, fall back to the SUM of the goods +
    services revenue lines (many retail/service small-caps tagged revenue only that way before
    ASC 606). Summing — not a naive tag fallback — is what keeps a true goods+services splitter
    from being undercounted. The summed year is 'known' only once BOTH parts are filed (the
    later filing date), preserving point-in-time."""
    total = _annual_tags(facts, FIELD_TAGS["revenue"])
    goods = _annual_tags(facts, ["SalesRevenueGoodsNet"])
    services = _annual_tags(facts, ["SalesRevenueServicesNet"])
    out = dict(total)
    for end in set(goods) | set(services):
        if end in out:                                            # a real total-revenue tag covers it
            continue
        parts = [p for p in (goods.get(end), services.get(end)) if p]
        out[end] = (sum(v for v, _ in parts), max(f for _, f in parts))
    return out


def _extract(facts, asof):
    """Point-in-time annual extraction: for every field, the current-FY and prior-FY value
    KNOWN as of `asof`, plus newest-first revenue/gross-profit histories. Anchored on net
    income's fiscal-year-ends (filed on/before asof). Returns None if <2 annual years filed."""
    asof = str(asof)
    series = {f: _annual(facts, f) for f in FIELD_TAGS}
    ends = sorted(e for e, (v, filed) in series["net_income"].items() if filed <= asof)
    if len(ends) < 2:
        return None
    cur_end, prior_end = ends[-1], ends[-2]

    def at(field, end):
        v = series[field].get(end)
        return v[0] if (v and v[1] <= asof) else None

    def hist(field):                                               # newest-first, point-in-time
        s = series[field]
        return [s[e][0] for e in reversed(ends) if e in s and s[e][1] <= asof]

    cur = {f: at(f, cur_end) for f in FIELD_TAGS}
    prior = {f: at(f, prior_end) for f in FIELD_TAGS}
    return {"fy_end": cur_end, "prior_fy_end": prior_end, "cur": cur, "prior": prior,
            "revenue_history": hist("revenue"), "gross_profit_history": hist("gross_profit")}


def fundamentals_asof(ticker, asof):
    """Raw point-in-time line items KNOWN as of `asof` (current + prior fiscal year), flattened
    to {field, field_prior, fy_end}. The low-level primitive; for factor work use
    edgar_fundamentals_asof (canonical) / edgar_fscore_asof. None if uncovered / <2 years."""
    facts = _facts_for(ticker)
    if facts is None:
        return None
    e = _extract(facts, asof)
    if e is None:
        return None
    out = {"ticker": ticker.upper(), "fy_end": e["fy_end"]}
    for f in FIELD_TAGS:
        out[f] = e["cur"][f]
        out[f + "_prior"] = e["prior"][f]
    return out


def edgar_fundamentals_asof(ticker, asof, price=None):
    """Point-in-time fundamentals as the project's CANONICAL schema, from EDGAR — a drop-in
    for fundamentals.simfin_fundamentals_asof. Source-agnostic factors.py recipes (ev_ebit,
    roic, gm_stability, net_debt_ebitda, altman_z) consume this directly. market_cap = `price`
    x diluted shares when a price is supplied (the backtest passes the as-of close), else None
    — which leaves the price-dependent factors (ev_ebit, price_fcf, altman_z) None until then.
    sector is mapped from the filer's SIC code (for the financials/REIT exclusion + sector-
    neutral ranking). Beneish inputs are partial (receivables/PPE/SGA not pulled yet) so
    beneish_m returns None for now. Returns None if uncovered / <2 annual years filed as of
    the date."""
    cik = ticker_cik().get(ticker.upper())
    if not cik:
        return None
    facts = company_facts(cik)
    if facts is None:
        return None
    e = _extract(facts, asof)
    if e is None:
        return None
    c, p = e["cur"], e["prior"]

    ebit = c["operating_income"]
    if ebit is None and c["pretax_income"] is not None and c["interest_expense"] is not None:
        ebit = c["pretax_income"] + c["interest_expense"]     # reconstruct EBIT for single-step
        #                                                       income statements (operating income untagged)
    da = c["dep_amort"]
    ebitda = (ebit + da) if (ebit is not None and da is not None) else None

    def total_debt(ltd, std):                              # missing-stays-missing (both None -> None)
        return None if (ltd is None and std is None) else (ltd or 0.0) + (std or 0.0)
    td = total_debt(c["long_term_debt"], c["short_term_debt"])
    td_p = total_debt(p["long_term_debt"], p["short_term_debt"])

    cash = c["cash"]
    tax, pretax = c["tax_expense"], c["pretax_income"]
    tax_rate = (tax / pretax) if (tax is not None and pretax) else None
    cfo, capex = c["cfo"], c["capex"]
    fcf = (cfo - capex) if (cfo is not None and capex is not None) else None   # capex = positive outflow
    shares = c["shares"]
    market_cap = (price * shares) if (price is not None and shares) else None
    ev = (market_cap + td - cash) if (market_cap is not None and td is not None and cash is not None) else None

    return {
        "ticker": ticker.upper(),
        "sector": _sic_to_sector(company_sic(cik)),
        "report_date": e["fy_end"],
        "market_cap": market_cap,
        "enterprise_value": ev,
        "total_debt": td,
        "cash": cash,
        "equity": c["equity"],
        "ebit": ebit,
        "ebitda": ebitda,
        "tax_rate": tax_rate,
        "free_cash_flow": fcf,
        "revenue": c["revenue"],
        "revenue_history": e["revenue_history"],
        "gross_profit_history": e["gross_profit_history"],
        "total_assets": c["total_assets"],
        "total_liabilities": c["total_liabilities"],
        "current_assets": c["current_assets"],
        "current_liabilities": c["current_liabilities"],
        "retained_earnings": c["retained_earnings"],
        # Beneish M-score inputs — partial from EDGAR (receivables/PPE/securities/SGA not pulled);
        # beneish_m needs all of them and returns None when any is missing, so the manipulation
        # scrub is simply inactive on the EDGAR source for now (a known increment-3 gap).
        "gross_profit": c["gross_profit"],
        "receivables": None, "ppe": None, "securities": None,
        "depreciation": da, "sga": None,
        "cfo": cfo,
        "income_continuing": c["net_income"],
        "prior": {
            "revenue": p["revenue"], "gross_profit": p["gross_profit"],
            "receivables": None, "current_assets": p["current_assets"],
            "ppe": None, "securities": None, "total_assets": p["total_assets"],
            "depreciation": p["dep_amort"], "sga": None,
            "current_liabilities": p["current_liabilities"], "total_debt": td_p,
        },
    }


def _fscore(c, p):
    """The 9-signal Piotroski F-Score (0-9) from current/prior EDGAR field dicts — the same
    recipe as fundamentals._piotroski, read off the EDGAR schema. None if a required field or
    denominator is missing (gross_profit is the usual gap: many filers don't tag it)."""
    ni_t, ni_p = c["net_income"], p["net_income"]
    ta_t, ta_p = c["total_assets"], p["total_assets"]
    cfo_t = c["cfo"]
    ca_t, ca_p = c["current_assets"], p["current_assets"]
    cl_t, cl_p = c["current_liabilities"], p["current_liabilities"]
    gp_t, gp_p = c["gross_profit"], p["gross_profit"]
    rev_t, rev_p = c["revenue"], p["revenue"]
    ltd_t = c["long_term_debt"] or 0.0
    ltd_p = p["long_term_debt"] or 0.0
    sh_t, sh_p = c["shares"], p["shares"]

    req = [ni_t, ni_p, ta_t, ta_p, cfo_t, ca_t, ca_p, cl_t, cl_p, gp_t, gp_p, rev_t, rev_p]
    if any(x is None for x in req) or not all([ta_t, ta_p, cl_t, cl_p, rev_t, rev_p]):
        return None

    roa_t, roa_p = ni_t / ta_t, ni_p / ta_p
    s = 0
    s += roa_t > 0                                         # 1  positive ROA
    s += cfo_t > 0                                         # 2  positive operating cash flow
    s += roa_t > roa_p                                     # 3  ROA improving
    s += cfo_t > ni_t                                      # 4  cash earnings > accounting (accruals)
    s += (ltd_t / ta_t) < (ltd_p / ta_p)                  # 5  leverage falling
    s += (ca_t / cl_t) > (ca_p / cl_p)                    # 6  current ratio rising
    s += bool(sh_t is not None and sh_p is not None and sh_t <= sh_p)   # 7  no share dilution
    s += (gp_t / rev_t) > (gp_p / rev_p)                  # 8  gross margin rising
    s += (rev_t / ta_t) > (rev_p / ta_p)                  # 9  asset turnover rising
    return int(s)


def edgar_fscore_asof(ticker, asof):
    """Piotroski F-Score (0-9) as of `asof` from EDGAR — point-in-time, survivorship-free.
    Parallel to fundamentals.piotroski_fscore_asof (SimFin). None if uncovered / incomplete."""
    facts = _facts_for(ticker)
    if facts is None:
        return None
    e = _extract(facts, asof)
    if e is None:
        return None
    return _fscore(e["cur"], e["prior"])


if __name__ == "__main__":
    from factors import calculate_factors                  # local import (factors imports fundamentals)

    print("ticker->CIK entries:", len(ticker_cik()))
    print("\nAAPL — point-in-time canonical fundamentals + factors + F-Score from EDGAR")
    print("(price-dependent factors ev_ebit/price_fcf/altman_z need an as-of price; the")
    print(" price-FREE quality block — roic, gm_stability, net_debt_ebitda, F-Score — is")
    print(" 60% of the composite weight and shown live below)")
    for asof in ["2010-06-01", "2013-06-01", "2018-06-01", "2024-06-01"]:
        f = edgar_fundamentals_asof("AAPL", asof)
        if not f:
            print(f"\nAAPL as of {asof}: not covered")
            continue
        fac = calculate_factors(f)
        fs = edgar_fscore_asof("AAPL", asof)
        print(f"\nAAPL as of {asof}  (latest FY end {f['report_date']}):")
        for k in ("revenue", "income_continuing", "total_assets", "equity", "cash", "total_debt",
                  "ebit", "free_cash_flow"):
            v = f[k]
            print(f"  {k:16} {v:,.0f}" if v is not None else f"  {k:16} —")

        def show(label, v, fmt):
            print(f"  {label:16} {format(v, fmt)}" if v is not None else f"  {label:16} —")
        show("roic", fac["roic"], ".3f")
        show("gm_stability", fac["gm_stability"], ".4f")
        show("net_debt_ebitda", fac["net_debt_ebitda"], ".2f")
        print(f"  {'F-Score':16} {fs}/9" if fs is not None else f"  {'F-Score':16} —")
