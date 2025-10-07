import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf

from utils import (
    load_symbol_list, suggest_symbols, normalize_symbol,
    get_timeseries, safe_get_info,
    compute_rsi, compute_macd, rule_based_reco
)

# -------------------- Page & THEME --------------------
st.set_page_config(
    page_title="⚡ InsightXpert — Indian Stock Recommender (Rule-Based)",
    page_icon="⚡",
    layout="wide",
    menu_items={
        "Get help": "mailto:help@example.com",
        "About": "InsightXpert — educational research UI over Yahoo Finance. Not investment advice."
    }
)

# Global CSS (clean glass look)
st.markdown("""
<style>
:root {
  --border: rgba(255,255,255,0.12);
  --card-bg: rgba(17,17,27,0.5);
  --blur: saturate(180%) blur(8px);
}
html, body, [class*="css"] {
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system;
}
.ek-hero {
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(124,92,255,0.18), transparent 60%),
    radial-gradient(900px 500px at 90% -20%, rgba(0,214,143,0.14), transparent 60%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 18px 22px; margin-bottom: 12px;
  backdrop-filter: var(--blur);
}
.ek-subtle { font-size: 13px; opacity: .85; margin-top:.25rem; }
.ek-kpi {
  border: 1px solid var(--border); border-radius: 18px;
  padding: 14px 16px; position: relative; overflow: hidden;
  backdrop-filter: var(--blur);
}
.ek-kpi:before {
  content:""; position:absolute; inset:0;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,.06), transparent);
  transform: translateX(-100%); animation: shine 3s infinite;
}
@keyframes shine { 0%{transform:translateX(-100%);} 60%{transform:translateX(100%);} 100%{transform:translateX(100%);} }
.ek-badge {
  padding: 8px 14px; border-radius: 999px; border: 1px solid var(--border);
  font-weight: 700; display:inline-flex; align-items:center; gap:.5rem;
}
.ek-badge.buy { color: #0df2a0; }
.ek-badge.sell { color: #ff6285; }
.ek-chip {
  border: 1px solid var(--border); border-radius: 999px; padding: 6px 10px; display:inline-block; margin: 4px 6px 0 0;
  font-size: 12px;
}
.ek-footer { font-size:12px; opacity:.8; margin-top:8px;}
a, a:visited { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER (Logo in right corner) --------------------

st.markdown("""<div class="ek-hero">""", unsafe_allow_html=True) # Start the custom styled hero div

col_title, col_logo = st.columns([4, 1])

with col_title:
    # Title and Subtitle
    st.markdown('<h2 style="margin-top:2.5rem; margin-bottom:0;">⚡ UpsideXpert — by Hrishikesh Shelar</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ek-subtle">Rule-based signals • Transparent scoring • Clean visuals</p>', unsafe_allow_html=True)

with col_logo:
    # Logo
    st.image("insight.png", width=175) # This is the standard Streamlit way to display an image

st.markdown("""</div>""", unsafe_allow_html=True) # End the custom styled hero div


# -------------------- SIDEBAR (Scoring params only; NO ML) --------------------
with st.sidebar:
    st.header("⚙️ Scoring Settings")
    buy_cutoff = st.slider("BUY cutoff (score 0–5)", 0.0, 5.0, 3.0, 0.1,
                           help="Higher = stricter BUY decision.")
    min_avg_volume = st.number_input("Min avg volume (20d) for good liquidity", value=30000, step=5000)
    st.caption("Tip: raise the cutoff and liquidity bar for more conservative picks.")
    st.divider()
    st.caption("This is a fully rule-based engine using Momentum, Trend(EMA), Volume, MACD/RSI (no ML).")

# -------------------- Symbol helpers --------------------
@st.cache_data(show_spinner=False)
def _load_symbols():
    url = "https://raw.githubusercontent.com/HrishikeshShelar/PROfile/main/indian_symbols.csv"
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not load symbols from GitHub, using fallback. Error: {e}")
        return pd.DataFrame({
            "symbol": ["INFY.NS", "TCS.NS", "RELIANCE.NS", "TATAPOWER.NS", "TATAMOTORS.NS", "HDFCBANK.NS", "SBIN.NS"],
            "name":   ["Infosys", "Tata Consultancy Services", "Reliance Industries", "Tata Power", "Tata Motors", "HDFC Bank", "State Bank of India"]
        })



df_symbols = _load_symbols()

def fetch_last_price_and_ticker(candidates):
    """Robust price/ticker fetch: try fast_info, then history fallback."""
    for s in candidates:
        try:
            t = yf.Ticker(s)
            lp = None
            try:
                fi = getattr(t, "fast_info", {}) or {}
                lp = fi.get("last_price") or fi.get("lastPrice") or fi.get("lastClose")
            except Exception:
                lp = None
            if lp is None:
                hist = t.history(period="7d", interval="1d")
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    last_valid = hist["Close"].dropna()
                    if not last_valid.empty:
                        lp = float(last_valid.iloc[-1])
            if lp is not None and lp > 0:
                return s, float(lp), t
        except Exception:
            continue
    return None, None, None

# -------------------- Cached timeseries for speed --------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_timeseries(symbol: str, period="1y", interval="1d"):
    return get_timeseries(symbol, period=period, interval=interval)

# -------------------- Top ribbon: Search (Google-like suggestions) --------------------
colA, colB = st.columns([2, 1])
with colA:
    user_query = st.text_input(
        "🔎 Search company / ticker",
        value="",
        placeholder="Type e.g. tata → see TCS, Tata Power, Tata Motors"
    )

# Build suggestions as you type (dropdown)
suggestions_df = suggest_symbols(df_symbols, user_query, k=12) if user_query else df_symbols.head(12)
suggest_options = suggestions_df["symbol"].tolist()
suggest_labels = {row["symbol"]: f"{row['name']} · {row['symbol']}" for _, row in suggestions_df.iterrows()}

with colB:
    selected_from_dropdown = st.selectbox(
        "Suggestions",
        options=suggest_options if suggest_options else [""],
        format_func=lambda s: suggest_labels.get(s, s) if s else "—",
        index=0 if suggest_options else 0
    )

# Resolve chosen symbol
symbol = None
if selected_from_dropdown:
    symbol = selected_from_dropdown
elif user_query:
    symbol = normalize_symbol(user_query, df_symbols)

# -------------------- MAIN: per-symbol view --------------------
if symbol:
    base = symbol.replace(".NS", "").replace(".BO", "")
    candidates = [symbol] if (symbol.endswith(".NS") or symbol.endswith(".BO")) else [base + ".NS", base + ".BO"]
    valid_symbol, last_price, ticker = fetch_last_price_and_ticker(candidates)

    if valid_symbol is None:
        st.error("❌ Could not fetch live price. Check ticker or network.")
    else:
        with st.spinner("📡 Fetching data & computing signals..."):
            info = safe_get_info(ticker) if ticker else {}
            # cached history fetch (fast)
            df_d = cached_timeseries(valid_symbol, period="1y", interval="1d")
            # rule-based scoring (no ML)
            reco = rule_based_reco(
                price=last_price,
                info=info or {},
                df_hist=df_d,
                params={"buy_cutoff": buy_cutoff, "min_avg_volume": min_avg_volume},
            )

        # ---------- HEADER KPI ROW ----------
        k1, k2, k3, k4, k5 = st.columns([1.2, 1, 1, 1, 1])
        pe = info.get("trailingPE") if isinstance(info, dict) else None
        pb = info.get("priceToBook") if isinstance(info, dict) else None
        mc = info.get("market_cap") if isinstance(info, dict) else info.get("marketCap")

        k1.markdown(f'<div class="ek-kpi"><div style="font-size:13px;">Price</div>'
                    f'<div style="font-size:28px;font-weight:700;">₹ {last_price:,.2f}</div>'
                    f'<div class="ek-subtle">{valid_symbol}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="ek-kpi"><div style="font-size:13px;">Market Cap</div>'
                    f'<div style="font-size:22px;font-weight:700;">{f"₹ {mc/1e7:,.2f} Cr" if mc else "—"}</div></div>',
                    unsafe_allow_html=True)
        k3.markdown(f'<div class="ek-kpi"><div style="font-size:13px;">P/E</div>'
                    f'<div style="font-size:22px;font-weight:700;">{f"{pe:.2f}" if pe else "—"}</div></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="ek-kpi"><div style="font-size:13px;">P/B</div>'
                    f'<div style="font-size:22px;font-weight:700;">{f"{pb:.2f}" if pb else "—"}</div></div>',
                    unsafe_allow_html=True)
        k5.markdown(f'<div class="ek-kpi"><div style="font-size:13px;">Score (0–5)</div>'
                    f'<div style="font-size:26px;font-weight:800;">{reco["score"]:.2f}</div>'
                    f'<div class="ek-subtle">Cutoff {buy_cutoff:.1f}</div></div>',
                    unsafe_allow_html=True)

        # ---------- Verdict ----------
        st.markdown("### Verdict")
        badge_class = "buy" if reco["verdict"] == "BUY" else "sell"
        badge_text = "🟢 BUY" if reco["verdict"] == "BUY" else "🔴 NOT BUY"
        st.markdown(f'<span class="ek-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        st.info(reco["suggestion"])

        # ---------- NEW: Target / Better Entry (same KPI style) ----------
        if reco.get("target"):
            st.markdown(
                f'<div class="ek-kpi"><div style="font-size:13px;">🎯 Target Price</div>'
                f'<div style="font-size:26px;font-weight:800;">₹ {reco["target"]:,.2f}</div></div>',
                unsafe_allow_html=True
            )
        elif reco.get("better_entry"):
            st.markdown(
                f'<div class="ek-kpi"><div style="font-size:13px;">💡 Better Entry Price</div>'
                f'<div style="font-size:26px;font-weight:800;">₹ {reco["better_entry"]:,.2f}</div></div>',
                unsafe_allow_html=True
            )

        # ---------- Tabs (Summary + Charts only) ----------
        tab_info, tab_charts = st.tabs(["Summary", "Charts"])

        # -------- Summary Tab (TABLE) --------
        with tab_info:
            st.markdown("#### 📋 Stock Summary (detailed)")

            if df_d is not None and not df_d.empty:
                df_sum = df_d.copy()
                df_sum["Close"] = df_sum["Close"].astype(float)
                df_sum["Open"] = df_sum["Open"].astype(float)
                df_sum["High"] = df_sum["High"].astype(float)
                df_sum["Low"] = df_sum["Low"].astype(float)

                # SMAs for the table
                df_sum["SMA20"] = df_sum["Close"].rolling(window=20).mean()
                df_sum["SMA50"] = df_sum["Close"].rolling(window=50).mean()
                df_sum["SMA200"] = df_sum["Close"].rolling(window=200).mean()

                # RSI & MACD for the table (last values)
                rsi_series = compute_rsi(df_sum["Close"], period=14)
                macd, sig, hist = compute_macd(df_sum["Close"])

                latest = df_sum.iloc[-1]
                last_close = float(latest["Close"])
                sma20 = float(latest["SMA20"]) if not np.isnan(latest["SMA20"]) else None
                sma50 = float(latest["SMA50"]) if not np.isnan(latest["SMA50"]) else None
                sma200 = float(latest["SMA200"]) if not np.isnan(latest["SMA200"]) else None

                rsi_last = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None
                macd_last = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else None
                sig_last = float(sig.iloc[-1]) if not np.isnan(sig.iloc[-1]) else None

                avg20_vol = int(df_sum["Volume"].tail(20).mean()) if "Volume" in df_sum.columns and len(df_sum) >= 20 else None
                hi_52w = float(df_sum["High"].max())
                lo_52w = float(df_sum["Low"].min())

                metrics = reco.get("metrics", {})
                momentum_ratio = metrics.get("momentum_ratio")
                trend_ratio = metrics.get("ema50_ema200")
                volume_ratio = metrics.get("vol_ratio")

                verdict = reco["verdict"]
                liquidity = "OK" if reco["subscores"].get("liquidity", 0) >= 1.0 else "Thin"

                summary_rows = [{
                    "Company": next((row["name"] for _, row in df_symbols.iterrows() if row["symbol"] == valid_symbol), valid_symbol),
                    "Ticker": valid_symbol,
                    "Last Price": round(last_close, 2),
                    "52W High": round(hi_52w, 2),
                    "52W Low": round(lo_52w, 2),
                    "SMA 20": "—" if sma20 is None else round(sma20, 2),
                    "SMA 50": "—" if sma50 is None else round(sma50, 2),
                    "SMA 200": "—" if sma200 is None else round(sma200, 2),
                    "RSI (14)": "—" if rsi_last is None else round(rsi_last, 2),
                    "MACD": "—" if macd_last is None else round(macd_last, 3),
                    "Signal": "—" if sig_last is None else round(sig_last, 3),
                    "Avg Vol (20d)": "—" if avg20_vol is None else f"{avg20_vol:,}",
                    "Momentum Ratio (Close/SMA20)": "—" if momentum_ratio is None else round(momentum_ratio, 3),
                    "Trend Ratio (EMA50/EMA200)": "—" if trend_ratio is None else round(trend_ratio, 3),
                    "Volume Ratio (Last/Avg20)": "—" if volume_ratio is None else round(volume_ratio, 3),
                    "P/E": "—" if not isinstance(pe, (int, float)) else round(pe, 2),
                    "P/B": "—" if not isinstance(pb, (int, float)) else round(pb, 2),
                    "Market Cap": "—" if not isinstance(mc, (int, float)) else f"₹ {mc/1e7:,.2f} Cr",
                    "Liquidity": liquidity,
                    "Verdict": verdict
                }]

                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(f"Data period: {df_sum['Date'].iloc[0].date()} → {df_sum['Date'].iloc[-1].date()}  •  Fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            else:
                st.info("No data to summarize.")

            st.markdown("#### Top reasons")
            for r in reco["top_reasons"]:
                st.markdown(f"- {r}")

        # -------- Charts Tab --------
        with tab_charts:
            st.markdown("#### Price (1Y) — Candles, EMA(50/200), Volume + MACD & RSI")
            if df_d is not None and not df_d.empty:
                df_plot = df_d.copy()
                df_plot["Close"] = df_plot["Close"].astype(float)
                df_plot["Open"] = df_plot["Open"].astype(float)
                df_plot["High"] = df_plot["High"].astype(float)
                df_plot["Low"] = df_plot["Low"].astype(float)
                dates = df_plot["Date"]

                # EMAs & indicators
                closes = df_plot["Close"]
                ema50 = closes.ewm(span=50, adjust=False).mean()
                ema200 = closes.ewm(span=200, adjust=False).mean()

                # RSI and MACD
                df_plot["RSI"] = compute_rsi(closes, period=14)
                macd, signal, hist = compute_macd(closes, span_short=12, span_long=26, span_signal=9)
                df_plot["MACD"] = macd
                df_plot["MACD_signal"] = signal
                df_plot["MACD_hist"] = hist

                # Buy / Sell signals (from EMA50 cross)
                buy_signals = df_plot[(df_plot["Close"] > ema50) & (df_plot["Close"].shift(1) <= ema50.shift(1))]
                sell_signals = df_plot[(df_plot["Close"] < ema50) & (df_plot["Close"].shift(1) >= ema50.shift(1))]

                # Build subplot
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    row_heights=[0.6, 0.2, 0.2],
                                    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])

                # Candles
                fig.add_trace(go.Candlestick(
                    x=dates, open=df_plot["Open"], high=df_plot["High"],
                    low=df_plot["Low"], close=df_plot["Close"], name="Price"
                ), row=1, col=1)

                # EMAs — UPDATED COLORS (no dotted lines)
                fig.add_trace(go.Scatter(x=dates, y=ema50, mode="lines", name="EMA 50",
                                         line=dict(width=2, color="#FF8C00")), row=1, col=1)  # dark orange
                fig.add_trace(go.Scatter(x=dates, y=ema200, mode="lines", name="EMA 200",
                                         line=dict(width=2, color="#001F54")), row=1, col=1)  # dark navy

                # Markers (BUY red ▲, SELL green ▼)
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals["Date"], y=buy_signals["Close"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=14, color="red",
                                    line=dict(width=1.5, color="white"), opacity=0.95),
                        name="BUY Signal"
                    ), row=1, col=1)
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals["Date"], y=sell_signals["Close"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=14, color="green",
                                    line=dict(width=1.5, color="white"), opacity=0.95),
                        name="SELL Signal"
                    ), row=1, col=1)

                # Volume bar on secondary y
                if "Volume" in df_plot.columns:
                    fig.add_trace(go.Bar(x=dates, y=df_plot["Volume"], name="Volume", opacity=0.25),
                                  row=1, col=1, secondary_y=True)

                # MACD
                fig.add_trace(go.Bar(x=dates, y=df_plot["MACD_hist"], name="MACD Hist", opacity=0.6), row=2, col=1)
                fig.add_trace(go.Scatter(x=dates, y=df_plot["MACD"], mode="lines", name="MACD",
                                         line=dict(width=2.2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=dates, y=df_plot["MACD_signal"], mode="lines", name="Signal",
                                         line=dict(width=2)), row=2, col=1)

                # RSI (thicker, guides)
                fig.add_trace(go.Scatter(x=dates, y=df_plot["RSI"], mode="lines", name="RSI",
                                         line=dict(width=2.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", row=3, col=1,
                              annotation_text="Overbought (70)", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", row=3, col=1,
                              annotation_text="Oversold (30)", annotation_position="bottom left")

                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=840,
                    margin=dict(l=10, r=10, t=10, b=10),
                    template="plotly_dark",
                    hovermode="x unified"
                )
                fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="MACD", row=2, col=1)
                fig.update_yaxes(title_text="RSI", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily history available.")

# -------------------- Footer --------------------
st.caption("Educational only, not financial advice.")
