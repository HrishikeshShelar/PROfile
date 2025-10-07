# Indian Stock Recommender (Streamlit)

A Streamlit app that lets you search Indian stocks (NSE/BSE), pulls live data using `yfinance`,
shows fundamentals (Market Cap, P/E, P/B, ROE, 52W High/Low, etc.), recent news headlines,
and gives a rule-based **Buy / Not Buy** suggestion with target or better entry price.

## Features
- Search with smart suggestions (company name or ticker). Example: `Infosys`, `INFY`, `INFY.NS`
- Live price & chart (auto-refresh)
- Fundamentals: Market Cap, P/E, P/B, ROE, Debt/Equity (best-effort), 52W High/Low, Dividend Yield
- 4–5 recent news bullets (from yfinance)
- Rule-based recommendation with reasons and **target price** or **suggested entry**

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Yahoo Finance symbols typically end with `.NS` for NSE and `.BO` for BSE. The app auto-detects.
- Some fundamentals can be missing on Yahoo; the app handles this gracefully.
- News availability varies per ticker.
- Suggestions are **not financial advice**; do your own research.
