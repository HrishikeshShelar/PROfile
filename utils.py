import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Symbol helpers ----------
def load_symbol_list(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])

def suggest_symbols(df: pd.DataFrame, query: str, k: int = 12) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not query:
        return df.head(k)
    q = str(query).lower()
    mask = df["symbol"].str.lower().str.contains(q) | df["name"].str.lower().str.contains(q)
    out = df[mask].copy()
    # Rank: startswith gets higher rank than contains; then alphabetical
    out["__rank"] = (
        out["name"].str.lower().str.startswith(q).astype(int) * 2 +
        out["symbol"].str.lower().str.startswith(q).astype(int)
    )
    out = out.sort_values(["__rank", "name"], ascending=[False, True]).drop(columns="__rank")
    return out.head(k)

def normalize_symbol(query: str, df: pd.DataFrame) -> str:
    q = str(query).strip().upper()
    if df is not None and not df.empty and q in df["symbol"].values:
        return q
    if df is not None and not df.empty:
        match = df[df["name"].str.upper() == q]
        if not match.empty:
            return match.iloc[0]["symbol"]
    if not q.endswith(".NS") and not q.endswith(".BO"):
        return q + ".NS"
    return q

# ---------- Data fetch ----------
def get_timeseries(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    """
    Returns OHLCV dataframe with a 'Date' column.
    (Streamlit caches this from the app side.)
    """
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.reset_index()
            if "Datetime" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            if "Date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            return df
    except Exception:
        pass
    return pd.DataFrame()

def safe_get_info(ticker: yf.Ticker) -> dict:
    try:
        info = ticker.info
        if isinstance(info, dict):
            if "marketCap" in info and "market_cap" not in info:
                info["market_cap"] = info["marketCap"]
            return info
    except Exception:
        pass
    return {}

# ---------- Indicator helpers ----------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's method (SMA seed then EMA-like smoothing).
    Returns a series aligned to input.
    """
    s = pd.Series(series).astype(float)
    if s.isna().all() or len(s) < period + 1:
        return pd.Series([np.nan]*len(series), index=series.index)

    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # SMA seed
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, span_short: int = 12, span_long: int = 26, span_signal: int = 9):
    """
    MACD (EMA12-EMA26), Signal (EMA9), Histogram.
    """
    s = pd.Series(series).astype(float)
    if s.isna().all():
        empty = pd.Series([np.nan]*len(series), index=series.index)
        return empty, empty, empty
    ema_short = s.ewm(span=span_short, adjust=False).mean()
    ema_long = s.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ---------- Rule-based recommender (no ML) ----------
def _map_ratio_to_score(ratio, thresholds):
    try:
        r = float(ratio)
    except Exception:
        return 0.0
    for thr, s in thresholds:
        if r >= thr:
            return float(s)
    return 0.0

def rule_based_reco(price: float, info: dict, df_hist: pd.DataFrame, params: dict = None):
    """
    Returns dict with verdict, score, subscores, metrics, reasons, suggestion, target/better_entry.
    Uses: Momentum (close vs SMA20), Trend (EMA50 vs EMA200), Volume pickup, Fundamentals (PE/PB), Liquidity.
    """
    params = params or {}
    buy_cutoff = float(params.get("buy_cutoff", 3.0))
    min_avg_volume = int(params.get("min_avg_volume", 30000))

    reasons, raw_metrics, subscores = [], {}, {}
    weights = {"momentum": 1.2, "trend": 1.2, "volume": 1.0, "fundamentals": 1.0, "liquidity": 0.6}
    total_weight = sum(weights.values())

    if price is None or price <= 0:
        return {
            "verdict": "NOT BUY", "score": 0.0,
            "subscores": {k: 0.0 for k in weights},
            "weights": weights, "metrics": {},
            "reasons": ["Invalid price"], "top_reasons": ["Invalid / missing live price."],
            "suggestion": "Unable to score due to missing price.",
            "target": None, "better_entry": None
        }

    closes = pd.Series(dtype=float)
    vols = pd.Series(dtype=float)
    try:
        if isinstance(df_hist, pd.DataFrame) and not df_hist.empty and "Close" in df_hist.columns:
            closes = df_hist["Close"].astype(float).dropna()
            vols = df_hist["Volume"].astype(float).dropna() if "Volume" in df_hist.columns else pd.Series(dtype=float)
    except Exception:
        closes = pd.Series(dtype=float); vols = pd.Series(dtype=float)

    # Momentum (close vs SMA20)
    if len(closes) >= 21:
        sma20 = float(closes.tail(21).mean())
        last = float(closes.iloc[-1])
        ratio = last / sma20 if sma20 > 0 else 1.0
        raw_metrics["momentum_ratio"] = round(ratio, 4)
        subscores["momentum"] = _map_ratio_to_score(ratio, [(1.05, 1.0), (1.0, 0.8), (0.98, 0.5)])
        if ratio >= 1.05:
            reasons.append(f"Strong momentum: last close {last:.2f} > 20d SMA {sma20:.2f} (+{(ratio-1)*100:.2f}%).")
        elif ratio >= 1.0:
            reasons.append(f"Momentum positive: last close {last:.2f} ≥ 20d SMA {sma20:.2f}.")
        elif ratio >= 0.98:
            reasons.append(f"Weak momentum: close just below 20d SMA ({last:.2f} vs {sma20:.2f}).")
        else:
            reasons.append(f"Negative momentum: close below 20d SMA ({last:.2f} < {sma20:.2f}).")
    else:
        subscores["momentum"] = 0.0
        raw_metrics["momentum_ratio"] = None
        reasons.append("Not enough history (>=21 days) for momentum (SMA20) check.")

    # Trend (EMA50 vs EMA200)
    if len(closes) >= 50:
        ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1]) if len(closes) >= 200 else None
        raw_metrics["ema50"] = round(ema50, 4)
        raw_metrics["ema200"] = round(ema200, 4) if ema200 is not None else None
        if ema200 is not None:
            ratio = ema50 / ema200 if ema200 > 0 else 1.0
            raw_metrics["ema50_ema200"] = round(ratio, 4)
            subscores["trend"] = _map_ratio_to_score(ratio, [(1.0, 1.0), (0.98, 0.5)])
            if ratio > 1.0:
                reasons.append(f"Positive longer-term trend: EMA50 {ema50:.2f} > EMA200 {ema200:.2f}.")
            elif ratio >= 0.98:
                reasons.append("Trend neutral (EMA50 near EMA200).")
            else:
                reasons.append(f"Weak/negative trend: EMA50 {ema50:.2f} < EMA200 {ema200:.2f}.")
        else:
            ema50_series = closes.ewm(span=50, adjust=False).mean().iloc[-10:]
            if len(ema50_series) >= 6 and ema50_series.iloc[-1] > ema50_series.iloc[0]:
                subscores["trend"] = 0.6
                reasons.append("EMA50 rising (short-term trend positive).")
            else:
                subscores["trend"] = 0.0
                reasons.append("EMA50 not rising (insufficient longer-term trend data).")

    else:
        subscores["trend"] = 0.0
        raw_metrics["ema50"] = None
        raw_metrics["ema200"] = None
        reasons.append("Not enough history for EMA trend checks (need >=50 days).")

    # Volume pickup (last vs avg20)
    if len(vols) >= 20:
        last_vol = float(vols.iloc[-1])
        avg20 = float(vols.tail(20).mean())
        ratio = last_vol / (avg20 if avg20 > 0 else 1.0)
        raw_metrics["last_vol"] = int(last_vol)
        raw_metrics["avg20_vol"] = int(avg20)
        raw_metrics["vol_ratio"] = round(ratio, 3)
        if ratio > 2:
            subscores["volume"] = 1.0; reasons.append("Strong volume spike (last vol >2× 20d avg).")
        elif ratio > 1.4:
            subscores["volume"] = 0.6; reasons.append("Noticeable volume pickup (>1.4× avg).")
        elif ratio > 1.1:
            subscores["volume"] = 0.3; reasons.append("Mild volume pickup (>1.1× avg).")
        else:
            subscores["volume"] = 0.0; reasons.append("No recent volume pickup.")
    else:
        subscores["volume"] = 0.0
        raw_metrics["last_vol"] = None
        raw_metrics["avg20_vol"] = None
        raw_metrics["vol_ratio"] = None
        reasons.append("Insufficient volume history (need >=20 days).")

    # Fundamentals (PE/PB)
    pe = info.get("trailingPE") if isinstance(info, dict) else None
    pb = info.get("priceToBook") if isinstance(info, dict) else None
    raw_metrics["pe"] = round(pe, 3) if pe is not None else None
    raw_metrics["pb"] = round(pb, 3) if pb is not None else None

    pe_score = 0.0
    if pe is not None and pe > 0:
        if pe < 20: pe_score = 1.0
        elif pe < 40: pe_score = 0.6
        elif pe < 80: pe_score = 0.25
    pb_score = 0.0
    if pb is not None and pb > 0:
        if pb < 2: pb_score = 1.0
        elif pb < 5: pb_score = 0.6

    if (pe is None or pe <= 0) and (pb is None or pb <= 0):
        subscores["fundamentals"] = 0.0
        reasons.append("Fundamentals (PE/PB) not sufficiently available.")
    else:
        subscores["fundamentals"] = round(pe_score * 0.6 + pb_score * 0.4, 3)
        reasons.append(f"Fundamentals score from PE ({'N/A' if pe is None else round(pe,2)}) "
                       f"and PB ({'N/A' if pb is None else round(pb,2)}).")

    # Liquidity
    avg20_vol = raw_metrics.get("avg20_vol")
    if avg20_vol and avg20_vol > 0:
        subscores["liquidity"] = 1.0 if avg20_vol >= min_avg_volume else 0.3
        if subscores["liquidity"] == 1.0:
            reasons.append("Liquidity OK (20d average volume above threshold).")
        else:
            reasons.append("Liquidity low relative to threshold (thinly traded).")
    else:
        subscores["liquidity"] = 0.0
        reasons.append("Liquidity data missing.")

    # Final score
    weighted_sum = sum(weights[f] * float(subscores.get(f, 0.0)) for f in weights)
    score = float(weighted_sum) * (5.0 / total_weight)
    score = max(0.0, min(5.0, score))
    score_rounded = round(score, 2)

    # Top reasons
    top_reasons = []
    if score >= buy_cutoff:
        for r in reasons:
            if any(word in r.lower() for word in ["strong", "positive", "ok", "rising"]):
                top_reasons.append(r)
        if not top_reasons:
            top_reasons = reasons[:3]
    else:
        for r in reasons:
            if any(word in r.lower() for word in ["negative", "below", "insufficient", "low", "not"]):
                top_reasons.append(r)
        if not top_reasons:
            top_reasons = reasons[:3]
    top_reasons = top_reasons[:5]

    # Stops & action
    try:
        if len(closes) >= 21:
            sma20 = float(closes.tail(21).mean())
            stop_guess = round(min(price * 0.97, sma20 * 0.98), 2)
        else:
            stop_guess = round(price * 0.97, 2)
    except Exception:
        stop_guess = round(price * 0.97, 2)

    target, better_entry = None, None
    if score >= buy_cutoff:
        target = round(price * 1.06, 2)
        suggestion = (
            f"Score {score_rounded:.2f} ≥ cutoff {buy_cutoff:.2f}: GOOD candidate to consider buying. "
            f"Suggested target ≈ ₹ {target:.2f}. Suggested stop-loss ≈ ₹ {stop_guess:.2f}. "
            f"Monitor volume & news; size positions according to risk."
        )
    else:
        mom_ratio = raw_metrics.get("momentum_ratio")
        if isinstance(mom_ratio, float) and mom_ratio < 1.0:
            better_entry = round(price * 0.985, 2)
            suggestion = (
                f"Score {score_rounded:.2f} < cutoff {buy_cutoff:.2f}: NOT recommended right now. "
                f"Wait for a close above SMA20 or stronger momentum. "
                f"Better nibble entry ≈ ₹ {better_entry:.2f}."
            )
        else:
            better_entry = round(price * 0.97, 2)
            suggestion = (
                f"Score {score_rounded:.2f} < cutoff {buy_cutoff:.2f}: NOT recommended. "
                f"Consider lower entry near ₹ {better_entry:.2f} or wait for clearer trend/volume confirmation."
            )

    return {
        "verdict": "BUY" if score >= buy_cutoff else "NOT BUY",
        "score": float(score_rounded),
        "subscores": {k: float(round(float(v), 3)) for k, v in subscores.items()},
        "weights": weights,
        "metrics": raw_metrics,
        "reasons": reasons,
        "top_reasons": top_reasons,
        "suggestion": suggestion,
        "target": target,
        "better_entry": better_entry
    }
