# fetch.py — pulls financial data via yfinance

import yfinance as yf
from config import TICKERS

def fetch_all(ticker):
    print(f"Fetching {ticker}...")
    stock = yf.Ticker(ticker)

    income = stock.financials          # income statement (annual)
    balance = stock.balance_sheet      # balance sheet (annual)
    cashflow = stock.cashflow          # cash flow statement (annual)
    info = stock.info                  # key metrics (market cap, EV, etc.)

    return {
        "ticker": ticker,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "info": info
    }

if __name__ == "__main__":
    data = fetch_all("CALM")
    print("\n--- INCOME STATEMENT ---")
    print(data["income"])
    print("\n--- KEY INFO ---")
    print(data["info"].get("enterpriseValue"))
    print(data["info"].get("marketCap"))