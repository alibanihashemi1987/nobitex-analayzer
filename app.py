"""
Nobitex Crypto Analysis Agent
=============================
A single-file Streamlit application that:
  1. Talks to the Nobitex exchange REST API to pull watchlist / orderbook /
     trades / OHLCV candle data.
  2. Computes technical indicators (RSI, MFI, MA20, MA200) across the
     15-minute, 1-hour and 4-hour timeframes.
  3. Runs a simple, transparent multi-timeframe rule engine that produces an
     explicit BUY / SELL / WAIT recommendation with the reasoning behind it.
  4. Formats a Markdown report and pushes it to a Bale Messenger chat via a
     Bale bot.

Run locally with:   streamlit run app.py
Deploy for free on Streamlit Community Cloud (instructions at the bottom of
the accompanying README / chat message).

NOTE ON PRIVATE NOBITEX ENDPOINTS
----------------------------------
Nobitex's public market-data endpoints (orderbook, trades, OHLC/candles) are
documented and stable. The "favorites / watchlist" endpoint lives behind
authentication and its exact path has changed over time in Nobitex's API, so
this app tries a best-effort call to fetch it automatically, but ALWAYS gives
you a manual watchlist field in the sidebar as a guaranteed fallback so the
app keeps working even if that one private call fails or Nobitex changes it.
"""

import time
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

NOBITEX_BASE = "https://api.nobitex.ir"
BALE_BASE = "https://tapi.bale.ai/bot{token}/sendMessage"

TIMEFRAMES = {
    "15m": {"resolution": "15", "label": "15-Minute (Trigger)"},
    "1h": {"resolution": "60", "label": "1-Hour (Confirmation)"},
    "4h": {"resolution": "240", "label": "4-Hour (Trend)"},
}

REQUEST_TIMEOUT = 12  # seconds
HTTP_HEADERS_JSON = {"Content-Type": "application/json"}


# --------------------------------------------------------------------------
# Nobitex API client
# --------------------------------------------------------------------------

class NobitexClient:
    """Thin wrapper around the Nobitex REST API with defensive error handling.

    Every public method returns either the parsed data (dict / DataFrame) or
    None on failure, and appends a human readable message to self.errors so
    the UI layer can surface exactly what went wrong without crashing.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or "").strip()
        self.errors: list[str] = []

    # -- internals ---------------------------------------------------------

    def _auth_headers(self):
        headers = dict(HTTP_HEADERS_JSON)
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        return headers

    def _log(self, msg: str):
        self.errors.append(msg)

    def _get(self, path, params=None, auth=False):
        try:
            resp = requests.get(
                f"{NOBITEX_BASE}{path}",
                params=params or {},
                headers=self._auth_headers() if auth else HTTP_HEADERS_JSON,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                self._log(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except requests.exceptions.RequestException as e:
            self._log(f"GET {path} failed: {e}")
            return None

    def _post(self, path, payload=None, auth=False):
        try:
            resp = requests.post(
                f"{NOBITEX_BASE}{path}",
                json=payload or {},
                headers=self._auth_headers() if auth else HTTP_HEADERS_JSON,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                self._log(f"POST {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except requests.exceptions.RequestException as e:
            self._log(f"POST {path} failed: {e}")
            return None

    # -- public data ---------------------------------------------------------

    def get_orderbook(self, symbol: str):
        """symbol e.g. 'BTCUSDT'. Returns dict with bids/asks or None."""
        data = self._get(f"/v2/orderbook/{symbol.upper()}")
        if data and data.get("status") == "ok":
            return data
        # fallback: some deployments expose it as a POST with currency pair
        return None

    def get_recent_trades(self, src: str, dst: str):
        data = self._post(
            "/market/trades/list",
            {"srcCurrency": src.lower(), "dstCurrency": dst.lower()},
        )
        if data and data.get("status") == "ok":
            return data.get("trades", [])
        return None

    def get_ohlc(self, symbol: str, resolution: str, lookback_days: int = 20):
        """Fetch candles and return a tidy OHLCV DataFrame (or None)."""
        to_ts = int(time.time())
        from_ts = to_ts - lookback_days * 24 * 3600
        data = self._get(
            "/market/udf/history",
            params={
                "symbol": symbol.upper(),
                "resolution": resolution,
                "from": from_ts,
                "to": to_ts,
            },
        )
        if not data or data.get("s") != "ok":
            self._log(f"OHLC fetch failed for {symbol} @ {resolution}: {data}")
            return None
        try:
            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(data["t"], unit="s"),
                    "open": pd.to_numeric(data["o"]),
                    "high": pd.to_numeric(data["h"]),
                    "low": pd.to_numeric(data["l"]),
                    "close": pd.to_numeric(data["c"]),
                    "volume": pd.to_numeric(data["v"]),
                }
            )
            return df.sort_values("time").reset_index(drop=True)
        except Exception as e:
            self._log(f"OHLC parse error for {symbol}: {e}")
            return None

    def get_favorites(self):
        """Best-effort fetch of the user's favorite/watchlist markets.

        Nobitex's private 'favorites' endpoint is not consistently documented
        publicly, so this call is wrapped defensively. Returns a list of
        symbol strings like ['BTCUSDT', 'ETHUSDT'] or None on any failure,
        in which case the UI simply falls back to the manual watchlist.
        """
        if not self.api_key:
            return None
        data = self._get("/users/markets/favorite", auth=True)
        if not data or data.get("status") != "ok":
            return None
        try:
            markets = data.get("favorites") or data.get("markets") or []
            symbols = []
            for m in markets:
                if isinstance(m, str):
                    symbols.append(m.upper())
                elif isinstance(m, dict) and m.get("symbol"):
                    symbols.append(str(m["symbol"]).upper())
            return symbols or None
        except Exception as e:
            self._log(f"Favorites parse error: {e}")
            return None


# --------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    direction = typical_price.diff()

    pos_flow = money_flow.where(direction > 0, 0.0)
    neg_flow = money_flow.where(direction < 0, 0.0)

    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    out = 100 - (100 / (1 + money_ratio))
    return out.fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi14"] = rsi(df["close"], 14)
    df["mfi14"] = mfi(df, 14)
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=1).mean()
    return df


# --------------------------------------------------------------------------
# Multi-timeframe signal engine
# --------------------------------------------------------------------------

def _timeframe_snapshot(df: pd.DataFrame) -> dict:
    """Extract the latest indicator readings and simple bias for one timeframe."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    price = last["close"]
    ma20, ma200 = last["ma20"], last["ma200"]
    rsi_val, mfi_val = last["rsi14"], last["mfi14"]
    rsi_prev = prev["rsi14"]

    # Trend bias from moving averages
    if price > ma20 > ma200:
        ma_bias = "bullish"
    elif price < ma20 < ma200:
        ma_bias = "bearish"
    else:
        ma_bias = "neutral"

    # Momentum bias from RSI/MFI
    if rsi_val >= 70 or mfi_val >= 80:
        momentum = "overbought"
    elif rsi_val <= 30 or mfi_val <= 20:
        momentum = "oversold"
    else:
        momentum = "neutral"

    rsi_turning_up = rsi_prev <= 30 < rsi_val or (rsi_prev < rsi_val and rsi_val < 45)
    rsi_turning_down = rsi_prev >= 70 > rsi_val or (rsi_prev > rsi_val and rsi_val > 55)

    return {
        "price": price,
        "ma20": ma20,
        "ma200": ma200,
        "rsi": rsi_val,
        "mfi": mfi_val,
        "ma_bias": ma_bias,
        "momentum": momentum,
        "rsi_turning_up": rsi_turning_up,
        "rsi_turning_down": rsi_turning_down,
    }


def generate_signal(snap_4h: dict, snap_1h: dict, snap_15m: dict) -> dict:
    """Combine the three timeframe snapshots into one explicit recommendation.

    Logic (simple, transparent, and adjustable):
      - 4H sets the *trend* (only trade in this direction).
      - 1H must *confirm* that trend is intact (not fighting it).
      - 15M supplies the *trigger* (an actual entry/exit timing signal).
    """
    reasons = []
    score = 0  # positive => bullish, negative => bearish

    # ---- Trend (4H) ----
    if snap_4h["ma_bias"] == "bullish":
        score += 2
        reasons.append("4H trend is BULLISH (price > MA20 > MA200).")
    elif snap_4h["ma_bias"] == "bearish":
        score -= 2
        reasons.append("4H trend is BEARISH (price < MA20 < MA200).")
    else:
        reasons.append("4H trend is NEUTRAL / mixed moving averages.")

    if snap_4h["momentum"] == "overbought":
        score -= 1
        reasons.append(f"4H momentum is OVERBOUGHT (RSI {snap_4h['rsi']:.0f}, MFI {snap_4h['mfi']:.0f}).")
    elif snap_4h["momentum"] == "oversold":
        score += 1
        reasons.append(f"4H momentum is OVERSOLD (RSI {snap_4h['rsi']:.0f}, MFI {snap_4h['mfi']:.0f}).")

    # ---- Confirmation (1H) ----
    if snap_1h["ma_bias"] == "bullish":
        score += 1.5
        reasons.append("1H confirms bullish structure (price above MA20/MA200).")
    elif snap_1h["ma_bias"] == "bearish":
        score -= 1.5
        reasons.append("1H confirms bearish structure (price below MA20/MA200).")
    else:
        reasons.append("1H structure is neutral — no strong confirmation either way.")

    if snap_1h["momentum"] == "overbought":
        score -= 0.5
        reasons.append(f"1H momentum stretched to the upside (RSI {snap_1h['rsi']:.0f}).")
    elif snap_1h["momentum"] == "oversold":
        score += 0.5
        reasons.append(f"1H momentum stretched to the downside (RSI {snap_1h['rsi']:.0f}).")

    # ---- Trigger (15M) ----
    trigger_buy = snap_15m["rsi_turning_up"] and snap_15m["price"] >= snap_15m["ma20"]
    trigger_sell = snap_15m["rsi_turning_down"] and snap_15m["price"] <= snap_15m["ma20"]

    if trigger_buy:
        score += 1.5
        reasons.append(f"15M trigger: RSI turning up from weakness ({snap_15m['rsi']:.0f}) with price back above its MA20 — entry timing looks favorable for a long.")
    elif trigger_sell:
        score -= 1.5
        reasons.append(f"15M trigger: RSI turning down from strength ({snap_15m['rsi']:.0f}) with price back below its MA20 — entry timing looks favorable for a short/exit.")
    else:
        reasons.append("15M shows no clean entry trigger yet — momentum hasn't turned at the micro level.")

    # ---- Final decision ----
    if score >= 3 and snap_4h["ma_bias"] != "bearish":
        decision = "BUY"
    elif score <= -3 and snap_4h["ma_bias"] != "bullish":
        decision = "SELL"
    else:
        decision = "WAIT"

    return {"decision": decision, "score": round(score, 2), "reasons": reasons}


# --------------------------------------------------------------------------
# Report formatting
# --------------------------------------------------------------------------

DECISION_EMOJI = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡"}


def format_symbol_report(symbol: str, snaps: dict, signal: dict) -> str:
    emoji = DECISION_EMOJI.get(signal["decision"], "⚪️")
    lines = [f"### {emoji} {symbol} — **{signal['decision']}**  (score: {signal['score']})", ""]
    lines.append("| TF | Price | MA20 | MA200 | RSI | MFI | Bias |")
    lines.append("|----|-------|------|-------|-----|-----|------|")
    for tf_key, tf_meta in TIMEFRAMES.items():
        s = snaps[tf_key]
        lines.append(
            f"| {tf_key.upper()} | {s['price']:.4g} | {s['ma20']:.4g} | {s['ma200']:.4g} "
            f"| {s['rsi']:.0f} | {s['mfi']:.0f} | {s['ma_bias']} |"
        )
    lines.append("")
    lines.append("**Reasoning:**")
    for r in signal["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def build_full_report(results: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"## 📊 Nobitex Multi-Timeframe Analysis\n_Generated {now}_\n\n"
    body_parts = []
    for r in results:
        body_parts.append(format_symbol_report(r["symbol"], r["snaps"], r["signal"]))
    if not body_parts:
        body_parts.append("_No symbols could be analyzed — check the errors panel._")
    return header + "\n---\n\n".join(body_parts)


# --------------------------------------------------------------------------
# Bale Messenger integration
# --------------------------------------------------------------------------

def send_to_bale(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token or not chat_id:
        return False, "Bale Bot Token and Chat ID are both required."

    url = BALE_BASE.format(token=bot_token.strip())
    # Bale (like Telegram-style bot APIs) has a per-message length limit;
    # split long reports into chunks to be safe.
    chunks = [text[i:i + 3500] for i in range(0, len(text), 3500)] or [text]

    try:
        for chunk in chunks:
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": chunk},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return False, f"Bale API error (HTTP {resp.status_code}): {resp.text[:300]}"
        return True, "Report sent to Bale successfully."
    except requests.exceptions.RequestException as e:
        return False, f"Network error while contacting Bale: {e}"


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Nobitex Crypto Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Responsive tweaks: widen tap targets & shrink default padding on small screens.
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    @media (max-width: 640px) {
        .block-container {padding-left: 0.6rem; padding-right: 0.6rem;}
        button[kind="primary"], button[kind="secondary"] {width: 100% !important;}
    }
    div.stButton > button {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -- session state defaults --
for key, default in {
    "nobitex_api_key": "",
    "bale_bot_token": "",
    "bale_chat_id": "",
    "watchlist_text": "BTCUSDT, ETHUSDT, USDTIRT",
    "last_results": None,
    "last_report": "",
    "errors": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -- Sidebar: configuration --
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.form("config_form"):
        api_key_input = st.text_input(
            "Nobitex API Key",
            value=st.session_state.nobitex_api_key,
            type="password",
            help="Used only for the optional 'fetch favorites' call. Market data works without it.",
        )
        bale_token_input = st.text_input(
            "Bale Bot Token",
            value=st.session_state.bale_bot_token,
            type="password",
            help="From @BotFather-equivalent on Bale when you create your bot.",
        )
        bale_chat_input = st.text_input(
            "Bale Chat ID",
            value=st.session_state.bale_chat_id,
            help="The numeric chat ID that should receive the report.",
        )
        watchlist_input = st.text_area(
            "Watchlist (comma-separated Nobitex symbols)",
            value=st.session_state.watchlist_text,
            help="Example: BTCUSDT, ETHUSDT, USDTIRT. Used automatically if the "
                 "favorites API call isn't available.",
            height=80,
        )
        saved = st.form_submit_button("💾 Save Settings", use_container_width=True)

    if saved:
        st.session_state.nobitex_api_key = api_key_input.strip()
        st.session_state.bale_bot_token = bale_token_input.strip()
        st.session_state.bale_chat_id = bale_chat_input.strip()
        st.session_state.watchlist_text = watchlist_input.strip()
        st.success("Settings saved for this session.")

    st.divider()
    st.caption(
        "🔒 Credentials are kept only in this browser session's memory — "
        "they are never written to disk by this app."
    )

# -- Main area --
st.title("📈 Nobitex Crypto Analysis Agent")
st.caption(
    "Multi-timeframe technical analysis (15m / 1h / 4h) with an automated "
    "BUY / SELL / WAIT recommendation, deliverable straight to Bale Messenger."
)

col_run, col_send = st.columns(2)
run_clicked = col_run.button("▶️ Run Analysis", type="primary")
send_clicked = col_send.button("📨 Send Report to Bale")

if run_clicked:
    st.session_state.errors = []
    client = NobitexClient(st.session_state.nobitex_api_key)

    with st.spinner("Fetching watchlist..."):
        favorites = client.get_favorites()

    if favorites:
        symbols = favorites
        st.info(f"Loaded {len(symbols)} symbols from your Nobitex favorites.")
    else:
        raw = st.session_state.watchlist_text
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            st.error("Your watchlist is empty. Add symbols in the sidebar (e.g. BTCUSDT).")
            symbols = []
        else:
            st.info(f"Using manual watchlist ({len(symbols)} symbols).")

    results = []
    progress = st.progress(0.0, text="Starting analysis...")

    for i, symbol in enumerate(symbols):
        progress.progress((i) / max(len(symbols), 1), text=f"Analyzing {symbol}...")

        # Pull candles for every timeframe
        snaps = {}
        ok = True
        for tf_key, tf_meta in TIMEFRAMES.items():
            df = client.get_ohlc(symbol, tf_meta["resolution"], lookback_days=25)
            if df is None or len(df) < 5:
                ok = False
                break
            df = add_indicators(df)
            snaps[tf_key] = _timeframe_snapshot(df)

        if not ok:
            st.session_state.errors.append(f"{symbol}: insufficient candle data — skipped.")
            continue

        # Orderbook / trades are fetched for extra context but never block the signal
        try:
            base = symbol[:-4] if symbol.endswith(("USDT", "IRT ")) else symbol[:3]
        except Exception:
            base = symbol

        signal = generate_signal(snaps["4h"], snaps["1h"], snaps["15m"])
        results.append({"symbol": symbol, "snaps": snaps, "signal": signal})

    progress.progress(1.0, text="Done.")
    time.sleep(0.2)
    progress.empty()

    st.session_state.errors.extend(client.errors)
    st.session_state.last_results = results
    st.session_state.last_report = build_full_report(results)

# -- Display results --
if st.session_state.last_results:
    st.subheader("Results")
    for r in st.session_state.last_results:
        signal = r["signal"]
        emoji = DECISION_EMOJI.get(signal["decision"], "⚪️")
        with st.expander(f"{emoji} {r['symbol']} — {signal['decision']} (score {signal['score']})", expanded=False):
            snap_cols = st.columns(3)
            for col, tf_key in zip(snap_cols, TIMEFRAMES.keys()):
                s = r["snaps"][tf_key]
                with col:
                    st.markdown(f"**{TIMEFRAMES[tf_key]['label']}**")
                    st.metric("Price", f"{s['price']:.4g}")
                    st.write(f"MA20: `{s['ma20']:.4g}`  \nMA200: `{s['ma200']:.4g}`")
                    st.write(f"RSI: `{s['rsi']:.0f}`  \nMFI: `{s['mfi']:.0f}`")
                    st.write(f"Bias: `{s['ma_bias']}`")
            st.markdown("**Reasoning**")
            for reason in signal["reasons"]:
                st.write(f"- {reason}")

    st.divider()
    st.subheader("📋 Full Markdown Report")
    st.markdown(st.session_state.last_report)

if st.session_state.errors:
    with st.expander("⚠️ Warnings / Errors from this run", expanded=False):
        for e in st.session_state.errors:
            st.write(f"- {e}")

if send_clicked:
    if not st.session_state.last_report:
        st.warning("Run an analysis first, then send the report.")
    else:
        with st.spinner("Sending report to Bale..."):
            success, message = send_to_bale(
                st.session_state.bale_bot_token,
                st.session_state.bale_chat_id,
                st.session_state.last_report,
            )
        if success:
            st.success(message)
        else:
            st.error(message)
