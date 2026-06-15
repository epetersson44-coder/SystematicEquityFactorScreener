# test_fundamentals.py — stress + contract tests for the translator (fundamentals.py).
#
# The translator is foundational: a quiet bug here poisons every factor and pick.
# These tests pin the contract (full schema, NaN-free, sane tax), the factor math
# (known answers), edge/missing handling, the yfinance adapter (mocked, offline),
# and the SimFin adapter (against the cached real data; skipped if no key).
#
# Run:  python test_fundamentals.py     (or pytest)

import numpy as np
import pandas as pd

import fundamentals as F
from factors import (calculate_factors, altman_z, beneish_m, get_ev_ebit, get_price_fcf,
                     get_roic, get_gm_stability, get_net_debt_ebitda)


def canon(**kw):
    d = F._blank("TST")
    d.update(kw)
    return d


# ----------------------------------------------- contract / normalize
def test_normalize_strips_nan_scalars():
    f = F._normalize(canon(ebit=float("nan"), market_cap=100.0))
    assert f["ebit"] is None and f["market_cap"] == 100.0


def test_normalize_strips_nan_from_lists():
    f = F._normalize(canon(revenue_history=[100.0, float("nan"), 90.0]))
    assert f["revenue_history"] == [100.0, 90.0]


def test_normalize_clamps_tax_high_and_low():
    assert F._normalize(canon(tax_rate=5.0))["tax_rate"] == F.TAX_RATE_MAX
    assert F._normalize(canon(tax_rate=-2.0))["tax_rate"] == 0.0


def test_normalize_fills_full_schema_from_minimal():
    f = F._normalize({"ticker": "X"})            # SimFin "not covered" shape
    F.validate_canonical(f)
    assert set(f) >= set(F.CANONICAL_KEYS)
    assert all(f[k] is None for k in F.SCALAR_KEYS)


def test_validate_catches_nan_scalar():
    bad = F._blank("X"); bad["ebit"] = float("nan")
    try:
        F.validate_canonical(bad)
    except AssertionError:
        return
    raise AssertionError("validate should catch a NaN scalar")


def test_validate_catches_missing_key():
    bad = F._blank("X"); del bad["ebit"]
    try:
        F.validate_canonical(bad)
    except AssertionError:
        return
    raise AssertionError("validate should catch a missing key")


# ----------------------------------------------- factor known-answers
def test_ev_ebit_known():
    assert abs(get_ev_ebit(canon(enterprise_value=1000.0, ebit=100.0)) - 10.0) < 1e-9


def test_price_fcf_known():
    assert abs(get_price_fcf(canon(market_cap=1000.0, free_cash_flow=100.0)) - 10.0) < 1e-9


def test_roic_known():
    # nopat = 100*(1-.25)=75 ; ic = 200+500-100 = 600 ; roic = .125
    f = canon(ebit=100.0, tax_rate=0.25, total_debt=200.0, equity=500.0, cash=100.0)
    assert abs(get_roic(f) - 0.125) < 1e-9


def test_net_debt_ebitda_known():
    assert abs(get_net_debt_ebitda(canon(total_debt=300.0, cash=100.0, ebitda=200.0)) - 1.0) < 1e-9


def test_gm_stability_skips_nan_and_matches_std():
    f = canon(revenue_history=[1000.0, 1000.0], gross_profit_history=[400.0, 300.0])
    assert abs(get_gm_stability(f) - np.std([0.4, 0.3], ddof=1)) < 1e-9


def test_score_direction_good_beats_bad():
    # GUARDRAIL: the composite is "higher = better" (sorted descending, top picked). A name
    # that's cheaper, higher-ROIC, more stable and lower-debt on EVERY factor must rank #1.
    # This catches the inverted-ranking bug (pre-2026-06-15 the screen bought the worst names).
    import pandas as pd
    from score import score
    df = pd.DataFrame([
        {"ticker": "GOOD", "ev_ebit": 5,  "price_fcf": 5,  "roic": 0.30, "gm_stability": 0.01, "net_debt_ebitda": 0.5},
        {"ticker": "MID",  "ev_ebit": 10, "price_fcf": 10, "roic": 0.15, "gm_stability": 0.05, "net_debt_ebitda": 2.0},
        {"ticker": "BAD",  "ev_ebit": 20, "price_fcf": 20, "roic": 0.05, "gm_stability": 0.10, "net_debt_ebitda": 4.0},
    ])
    ranked = score(df)
    assert list(ranked["ticker"]) == ["GOOD", "MID", "BAD"], "composite ranking is inverted"
    assert ranked.iloc[0]["composite"] > ranked.iloc[-1]["composite"]


def test_altman_z_known():
    f = canon(total_assets=1000.0, total_liabilities=400.0, current_assets=500.0,
              current_liabilities=200.0, retained_earnings=300.0, ebit=150.0,
              market_cap=2000.0, revenue=1200.0)
    # 1.2*.3 + 1.4*.3 + 3.3*.15 + 0.6*5 + 1.0*1.2 = 5.475
    assert abs(altman_z(f) - 5.475) < 1e-9


# ----------------------------------------------- Beneish M-score
def test_beneish_m_rederivation():
    cur = dict(revenue=1000.0, receivables=100.0, gross_profit=400.0, current_assets=500.0,
               ppe=300.0, securities=0.0, total_assets=1000.0, depreciation=50.0, sga=150.0,
               current_liabilities=200.0, total_debt=100.0, cfo=120.0, income_continuing=90.0)
    prior = dict(revenue=900.0, gross_profit=360.0, receivables=80.0, current_assets=450.0,
                 ppe=280.0, securities=0.0, total_assets=950.0, depreciation=45.0, sga=140.0,
                 current_liabilities=190.0, total_debt=95.0)
    f = canon(**cur); f["prior"] = prior
    dsri = (100 / 1000) / (80 / 900); gmi = (360 / 900) / (400 / 1000)
    aqi = (1 - 800 / 1000) / (1 - 730 / 950); sgi = 1000 / 900
    depi = (45 / 325) / (50 / 350); sgai = (150 / 1000) / (140 / 900)
    lvgi = (300 / 1000) / (285 / 950); tata = (90 - 120) / 1000
    expected = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    assert abs(beneish_m(f) - expected) < 1e-9


def test_beneish_none_without_prior():
    f = canon(revenue=1000.0, receivables=100.0, gross_profit=400.0, current_assets=500.0,
              ppe=300.0, total_assets=1000.0, depreciation=50.0, sga=150.0,
              current_liabilities=200.0, total_debt=100.0, cfo=120.0, income_continuing=90.0)
    assert beneish_m(f) is None                  # prior all None -> can't compute


def test_beneish_none_on_blank():
    assert beneish_m(F._blank("X")) is None


def test_normalize_cleans_prior_nan():
    f = canon()
    f["prior"] = {"revenue": float("nan"), "total_debt": 95.0}
    out = F._normalize(f)
    assert out["prior"]["revenue"] is None and out["prior"]["total_debt"] == 95.0
    F.validate_canonical(out)                    # prior dict still conforms


# ----------------------------------------------- edge / missing -> None, no crash
def test_all_factors_none_on_blank():
    f = F._blank("EMPTY")
    out = calculate_factors(f)
    assert all(out[k] is None for k in
               ("ev_ebit", "price_fcf", "roic", "gm_stability", "net_debt_ebitda"))
    assert altman_z(f) is None


def test_guard_zero_ebit():
    assert get_ev_ebit(canon(enterprise_value=1000.0, ebit=0.0)) is None


def test_guard_negative_fcf():
    assert get_price_fcf(canon(market_cap=1000.0, free_cash_flow=-50.0)) is None


def test_guard_negative_invested_capital():
    # ic = 0 + 50 - 200 = -150 -> None
    assert get_roic(canon(ebit=100.0, tax_rate=0.25, total_debt=0.0, equity=50.0, cash=200.0)) is None


def test_gm_stability_too_few_points():
    assert get_gm_stability(canon(revenue_history=[1000.0], gross_profit_history=[400.0])) is None


# ----------------------------------------------- yfinance adapter (mocked, offline)
def _fake_yf(ticker):
    dates = pd.to_datetime(["2025-12-31", "2024-12-31"])
    income = pd.DataFrame(
        {dates[0]: [1000, 400, 200, 250, 0.25], dates[1]: [900, 360, 180, 220, 0.25]},
        index=["Total Revenue", "Gross Profit", "EBIT", "EBITDA", "Tax Rate For Calcs"])
    balance = pd.DataFrame(
        {dates[0]: [float("nan"), 100, 500, 1000, 400, 500, 200, 300],
         dates[1]: [50, 90, 480, 950, 380, 480, 190, 290]},
        index=["Total Debt", "Cash And Cash Equivalents", "Stockholders Equity", "Total Assets",
               "Total Liabilities", "Total Current Assets", "Total Current Liabilities", "Retained Earnings"])
    cashflow = pd.DataFrame({dates[0]: [120], dates[1]: [110]}, index=["Free Cash Flow"])
    info = {"sector": "Technology", "marketCap": 5000, "enterpriseValue": 5300,
            "totalDebt": 250, "totalCash": 100}
    return {"ticker": ticker, "income": income, "balance": balance, "cashflow": cashflow, "info": info}


def test_yfinance_adapter_contract_and_nan_fallback():
    import fetch
    orig = getattr(fetch, "fetch_all", None)
    fetch.fetch_all = _fake_yf
    try:
        f = F.get_fundamentals("FAKE", source="yfinance")
        F.validate_canonical(f)                       # full schema, NaN-free
        assert f["ebit"] == 200.0 and f["revenue"] == 1000.0
        assert f["total_debt"] == 250.0               # balance Total Debt NaN -> info.totalDebt fallback
        assert f["sector"] == "Technology" and f["report_date"] == "2025-12-31"
        out = calculate_factors(f)
        assert out["ev_ebit"] is not None and out["roic"] is not None
        assert altman_z(f) is not None
    finally:
        if orig is not None:
            fetch.fetch_all = orig


# ----------------------------------------------- router
def test_router_rejects_unknown_source():
    try:
        F.get_fundamentals("X", source="bloomberg")
    except ValueError:
        return
    raise AssertionError("unknown source should raise ValueError")


# ----------------------------------------------- SimFin (cached real data; skip if no key)
def _simfin_ok():
    try:
        F._simfin_load()
        return True
    except Exception:
        return False


def test_simfin_contract_on_real_data():
    if not _simfin_ok():
        print("    (skipped — no SimFin key/cache)"); return
    for tk in ["AAPL", "IMAX", "ZUMZ"]:
        f = F.get_fundamentals(tk, source="simfin")
        F.validate_canonical(f)
        assert f["ticker"] == tk
    f = F.get_fundamentals("AAPL", source="simfin")
    assert f["market_cap"] and f["ebit"] and f["sector"] and f["report_date"]


def test_simfin_not_covered_returns_blank():
    if not _simfin_ok():
        print("    (skipped — no SimFin key/cache)"); return
    f = F.get_fundamentals("ZZZZ_NOT_A_TICKER", source="simfin")
    F.validate_canonical(f)
    assert all(f[k] is None for k in F.SCALAR_KEYS)


def test_simfin_asof_is_point_in_time():
    # the look-ahead gate: a later as-of date sees a NEWER vintage than an earlier one,
    # never the reverse — a statement is invisible until its Publish Date has passed.
    if not _simfin_ok():
        print("    (skipped — no SimFin key/cache)"); return
    early = F.get_fundamentals_asof("AAPL", "2022-06-01", price=150.0)
    late = F.get_fundamentals_asof("AAPL", "2024-06-01", price=150.0)
    F.validate_canonical(early); F.validate_canonical(late)
    assert early["report_date"] < late["report_date"]        # vintage advanced with time
    assert early["market_cap"] and early["market_cap"] > 0   # price-derived market cap present


def test_simfin_asof_blank_before_any_coverage():
    if not _simfin_ok():
        print("    (skipped — no SimFin key/cache)"); return
    f = F.get_fundamentals_asof("AAPL", "2000-01-01", price=100.0)   # nothing published by 2000
    F.validate_canonical(f)
    assert all(f[k] is None for k in F.SCALAR_KEYS)


if __name__ == "__main__":
    import sys
    tests = sorted((n, fn) for n, fn in globals().items()
                   if n.startswith("test_") and callable(fn))
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn(); passed += 1; print(f"  PASS  {name}")
        except Exception as e:                            # noqa: BLE001
            failed.append(name); print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
