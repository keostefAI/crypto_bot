"""
Crypto Alert Bot with Aggregated Summaries
-----------------------------------------

This script monitors a list of cryptocurrency trading pairs across several
timeframes and evaluates a set of technical indicators (moving averages,
RSI, MACD and volume) to detect potential bullish or bearish signals.  In
contrast to the original implementation that emitted a separate Telegram
message for every pair and every timeframe, this version aggregates the
signals into concise summaries.  It sends a single global summary that
captures the overall market tone (including Fear & Greed index and a tally
of bullish/bearish signals) and individual summaries for each asset that
describe the short‑term, medium‑term and long‑term trends.

If an OpenAI API key is supplied via the `OPENAI_API_KEY` environment
variable, the bot uses OpenAI's Chat Completions API to transform the raw
signal data into polished, human‑readable prose.  The OpenAI API can
perform a variety of text transformation tasks such as summarisation,
rewriting and improvement【435203994858066†L45-L51】.  A call to the
`chat.completions.create` endpoint with a prompt instructing the model to
"summarize the following text" will return a succinct summary【435203994858066†L185-L204】.
Without an API key, a simple fallback summarisation routine is used.

To run the bot you need to set the following environment variables:

* `TELEGRAM_TOKEN` – Bot token obtained from BotFather.
* `TELEGRAM_CHAT_ID` – Identifier of the chat (user, group or channel) that
  should receive the alerts.
* `OPENAI_API_KEY` – (optional) API key for OpenAI.  If omitted, the bot
  falls back to a basic summarisation strategy.

The script is designed for periodic execution (e.g. via a scheduler).

"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
import pandas as pd

try:
    # OpenAI is optional; import lazily.
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class PairConfig:
    """Configuration for a trading pair and its timeframes."""
    symbol: str
    timeframes: List[str]


# List of trading pairs to monitor.  Each entry defines the symbol and the
# intraday/daily/weekly timeframes of interest.  The HYPE pair has been
# replaced by ADA to avoid invalid symbol errors.
PAIRS: List[PairConfig] = [
    PairConfig(symbol="BTCUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="ETHUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="BNBUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="SOLUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="XRPUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="DOGEUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="AAVEUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="ADAUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="ATOMUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    PairConfig(symbol="LINKUSDT", timeframes=["1h", "4h", "1d", "1w"]),
]


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data from Binance.

    Args:
        symbol: Trading pair symbol (e.g. "BTCUSDT").
        interval: Candle interval (e.g. "1h", "1d").
        limit: Number of candles to retrieve.

    Returns:
        DataFrame with columns open_time, open, high, low, close, volume, or
        None if an error occurred.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(data, columns=columns)
        # Convert numeric columns to floats
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        return df
    except requests.RequestException as exc:
        logging.error(
            "Failed to fetch klines for %s %s: %s", symbol, interval, exc
        )
        return None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / period, min_periods=period).mean()
    loss = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series) -> pd.DataFrame:
    """Compute the MACD indicator along with its signal line and histogram."""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI and MACD columns to a DataFrame."""
    df = df.copy()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()
    df["rsi14"] = compute_rsi(df["close"], period=14)
    macd_df = compute_macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)
    return df


def evaluate_signals(df: pd.DataFrame) -> Dict[str, any]:
    """Evaluate technical indicators to classify bullish/bearish signals.

    Returns a dictionary containing:
        strength: number of validated indicators (0 means no signal)
        direction: "bullish", "bearish" or "neutral"
        indicators: list of indicator names that fired
    """
    indicators: List[str] = []
    last_close = df["close"].iloc[-1]
    ma50 = df["ma50"].iloc[-1]
    ma200 = df["ma200"].iloc[-1]
    rsi = df["rsi14"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]
    macd_hist = df["macd_hist"].iloc[-1]
    vol = df["volume"].iloc[-1]
    vol_mean = df["volume"].rolling(window=20).mean().iloc[-1]

    # Moving average crossovers
    if ma50 > ma200:
        indicators.append("ma50_above_ma200")
    elif ma50 < ma200:
        indicators.append("ma50_below_ma200")

    # Price relative to long MA
    if last_close > ma200:
        indicators.append("price_above_ma200")
    else:
        indicators.append("price_below_ma200")

    # RSI momentum
    if rsi > 55:
        indicators.append("rsi_bull")
    elif rsi < 45:
        indicators.append("rsi_bear")

    # MACD cross
    if macd > macd_signal:
        indicators.append("macd_bull")
    elif macd < macd_signal:
        indicators.append("macd_bear")

    # Histogram sign
    if macd_hist > 0:
        indicators.append("macd_hist_positive")
    elif macd_hist < 0:
        indicators.append("macd_hist_negative")

    # Volume confirmation
    if vol_mean > 0 and vol > vol_mean:
        indicators.append("volume_above_average")

    # Determine direction by majority of bull vs bear indicators
    bull_count = sum(
        1 for ind in indicators if "bull" in ind or "above" in ind or "positive" in ind
    )
    bear_count = sum(
        1 for ind in indicators if "bear" in ind or "below" in ind or "negative" in ind
    )
    if bull_count > bear_count:
        direction = "bullish"
    elif bear_count > bull_count:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength based on number of indicators (thresholds: >=3 strong, 2 medium, 1 weak)
    strength = len(indicators)
    return {"strength": strength, "direction": direction, "indicators": indicators}


def get_fear_greed_index() -> Optional[Dict[str, str]]:
    """Fetch the latest Crypto Fear & Greed index value and classification."""
    url = "https://api.alternative.me/fng/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"]:
            entry = data["data"][0]
            return {
                "value": entry.get("value"),
                "classification": entry.get("value_classification"),
            }
    except Exception as exc:
        logging.error("Failed to fetch Fear & Greed index: %s", exc)
    return None


def summarise_with_openai(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Use OpenAI's Chat Completions API to summarise text.

    Returns the assistant's message content or None if OpenAI is unavailable or
    not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None or not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model="gpt-4", messages=messages, temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logging.error("OpenAI summarisation failed: %s", exc)
        return None


def build_manual_global_summary(
    results: Dict[str, Dict[str, Dict[str, any]]], fear_greed: Optional[Dict[str, str]]
) -> str:
    """Fallback summary builder when OpenAI is unavailable.

    Creates a brief textual overview of the market by counting bullish and
    bearish timeframes and reporting the Fear & Greed index.  Also lists
    symbols with their dominant direction.
    """
    bullish = 0
    bearish = 0
    neutral = 0
    pair_overview = []
    for symbol, tf_signals in results.items():
        # Determine majority direction for the pair
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for tf_res in tf_signals.values():
            counts[tf_res["direction"]] += 1
        dominant = max(counts, key=counts.get)
        pair_overview.append(f"{symbol}: {dominant}")
        bullish += counts["bullish"]
        bearish += counts["bearish"]
        neutral += counts["neutral"]
    lines = []
    total_timeframes = bullish + bearish + neutral
    if total_timeframes > 0:
        lines.append(
            f"Global signals: {bullish} bullish, {bearish} bearish, {neutral} neutral across {total_timeframes} timeframes."
        )
    if fear_greed:
        lines.append(
            f"Fear & Greed index: {fear_greed.get('value')} ({fear_greed.get('classification')})."
        )
    if pair_overview:
        lines.append("Pair overview:")
        lines.extend(pair_overview)
    return "\n".join(lines)


def build_manual_pair_summary(symbol: str, tf_signals: Dict[str, Dict[str, any]]) -> str:
    """Fallback summary for a single asset.

    Summarises the short, medium and long‑term signals for the pair without
    external AI.
    """
    lines = [f"Summary for {symbol}:"]
    # Map timeframes to horizons
    horizon_map = {"1h": "short term", "4h": "short term", "1d": "medium term", "1w": "long term"}
    horizon_summary: Dict[str, List[str]] = {"short term": [], "medium term": [], "long term": []}
    for timeframe, sig in tf_signals.items():
        horizon = horizon_map.get(timeframe, timeframe)
        horizon_summary[horizon].append(sig["direction"])
    for horizon, directions in horizon_summary.items():
        if not directions:
            continue
        # Determine dominant direction for this horizon
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for d in directions:
            counts[d] += 1
        dominant = max(counts, key=counts.get)
        lines.append(f"- {horizon}: {dominant} ({len(directions)} timeframes)")
    return "\n".join(lines)


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    """Send a message via the Telegram Bot API synchronously.

    The python‑telegram‑bot library from version 20 uses asynchronous methods.
    To avoid awaiting coroutines in a synchronous script, this helper
    constructs a simple HTTP POST request to the Bot API.

    Args:
        token: Bot token provided by BotFather.
        chat_id: Identifier of the chat to send the message to.
        text: Message content.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logging.error("Failed to send Telegram message: %s", exc)


def summarise_results(
    results: Dict[str, Dict[str, Dict[str, any]]]
) -> (str, Dict[str, str]):
    """Create summaries for the market and each asset using OpenAI if available.

    Returns:
        global_summary: Single string summarising the overall market
        pair_summaries: Dict mapping symbol -> summary string
    """
    # Compute global stats for manual summarisation and to enrich AI prompts
    fear_greed = get_fear_greed_index()
    global_stats = {
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
    }
    for symbol, tf_signals in results.items():
        for tf_res in tf_signals.values():
            global_stats[f"{tf_res['direction']}_count"] += 1
    # Build JSON‑serialisable summary of signals
    summary_data = {
        "pairs": results,
        "global_stats": global_stats,
        "fear_greed": fear_greed,
    }
    # Try OpenAI summarisation first for the global view.  The pair summaries
    # will be constructed separately, so we ignore any per‑pair content
    # returned by the LLM.
    system_prompt = (
        "Tu es un trader crypto expérimenté et pédagogue. "
            "Ton rôle est d’analyser les signaux techniques et de marché fournis par l’agent, "
            "puis de donner une synthèse claire et actionnable à l’utilisateur. "
            "Ton objectif est d’aider l’utilisateur à maximiser ses profits tout en gérant les risques. "
            "Réponds toujours en français, de façon claire et structurée."
    )
    user_prompt = (
         "À partir des signaux donnés (indicateurs techniques, timeframes, intensité), fais :\n"
            "1. Un point synthétique sur la tendance du marché crypto :\n"
            "   - Court terme (H1/H4), moyen terme (D1), long terme (W1).\n"
            "   - Indique si une tendance haussière ou baissière est confirmée.\n"
            "   - Alerte si un retournement de tendance est constaté.\n"
            "   Exemple de sortie : \n"
            " Marché global : Le marché crypto montre une tendance haussière forte en court terme (H4), soutenue par un RSI global positif et une dominance BTC en hausse. \n"
            " À moyen terme (D1), on observe une consolidation, sans retournement confirmé. \n"
            " À long terme (W1), la tendance reste haussière mais le Fear & Greed élevé (>75) signale un risque de correction.\n"
            " Attention : possible retournement baissier si BTC casse les 62 000 $. \n\n"
            "2. Pour chaque paire analysée (BTC, ETH, et altcoins) :\n"
            "   - la tendance (court, moyen, long terme).\n"
            "   - Explique la pertinence des signaux détectés (fiables ou non).\n"
            "   - Précise le prix d’entrée, stop loss, take profit.\n"
            "   - Alerte si un retournement de tendance est constaté.\n"
            "   Exemple 1 de sortie par paire :\n"
            "   BTC/USDT :\n"
            "   Tendance haussière court et moyen terme, RSI et MACD alignés. \n"
            "   Plan de trade : achat entre 63 500 et 64 000 $, stop loss 62 800 $, take profit 66 500 $ puis 68 000 $.\n"
            "  Exemple 2 de sortie par paire :\n"
            "  SOL/USDT :\n"
            " Signal moyen, tendance court terme haussière mais volume faible. \n"
            " Plan : attendre confirmation avant entrée. \n\n"
            "Contraintes :\n"
            "- Structure la réponse en deux parties : Marché global puis pour chacune des valeurs (une sous-section par paire).\n"
            " Une partie pour le marché global, et une partie par valeur crypto analysée \n\n"
        + json.dumps(summary_data)
    )
    openai_result = summarise_with_openai(system_prompt, user_prompt)
    pair_summaries: Dict[str, str] = {}
    if openai_result:
        # If the AI returned a single text covering everything, we send it as the
        # global summary and don't create separate messages per pair.
        return openai_result, pair_summaries
    # Fallback: build manual summaries
    global_summary = build_manual_global_summary(results, fear_greed)
    for symbol, tf_signals in results.items():
        pair_summaries[symbol] = build_manual_pair_summary(symbol, tf_signals)
    return global_summary, pair_summaries


def run_bot() -> None:
    """Main entry point for running the crypto alert bot."""
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_token or not chat_id:
        logging.error("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables are missing.")
        return
    # We'll send messages via the raw Telegram HTTP API to avoid asynchronous
    # coroutines.  See `send_telegram_message()` above.
    results: Dict[str, Dict[str, Dict[str, any]]] = {}
    for pair in PAIRS:
        pair_results: Dict[str, Dict[str, any]] = {}
        for timeframe in pair.timeframes:
            df = fetch_klines(pair.symbol, timeframe)
            if df is None or len(df) < 200:
                continue
            df_ind = add_indicators(df)
            sig = evaluate_signals(df_ind)
            # Always capture the last close price for trade planning
            sig_with_price = dict(sig)
            sig_with_price["last_close"] = df_ind["close"].iloc[-1]
            # Only retain signals where at least one indicator fired
            if sig_with_price["strength"] > 0:
                pair_results[timeframe] = sig_with_price
        if pair_results:
            results[pair.symbol] = pair_results
    if not results:
        logging.info("No valid signals found across all pairs/timeframes.")
        return
    global_summary, pair_summaries = summarise_results(results)
    # Build custom summaries for each asset.  We ignore any per‑pair content
    # returned by the LLM and construct simplified messages instead.  Each
    # message lists the trend by timeframe and provides a basic trade plan
    # based on the current price and dominant trend.
    def summarise_pair_trade_plan(symbol: str, tf_signals: Dict[str, Dict[str, any]]) -> str:
        """Create a simplified summary and trade plan for a single trading pair.

        This helper determines the dominant trend on short (1h/4h), medium (1d)
        and long (1w) horizons and proposes a basic trade plan using the
        latest available close price.  The plan uses heuristics: for a
        predominantly bullish trend, it suggests a long entry range around
        the current price with stop loss and multiple take profit levels.
        For a bearish trend, it notes the opportunity on the short term and
        advises waiting for confirmation rather than opening a position.

        Args:
            symbol: The asset symbol (e.g. "BTCUSDT").
            tf_signals: Mapping from timeframe to signal data including
                direction and last_close.

        Returns:
            A human‑readable string summarising the signals and plan.
        """
        # Determine last close price (prefer 1h, else fall back through other
        # timeframes).  This price will be used to compute entry ranges and
        # profit/stop levels.
        last_close = None
        for tf in ["1h", "4h", "1d", "1w"]:
            if tf in tf_signals and "last_close" in tf_signals[tf]:
                last_close = tf_signals[tf]["last_close"]
                break
        # Build a string summarising the raw directions across all timeframes
        directions_parts: List[str] = []
        for tf in ["1h", "4h", "1d", "1w"]:
            if tf in tf_signals:
                dir_name = tf_signals[tf]["direction"].capitalize()
                directions_parts.append(f"{tf} : {dir_name}")
        directions_text = " - ".join(directions_parts)
        # Determine dominant directions for short, medium and long horizons
        horizon_map = {"1h": "short", "4h": "short", "1d": "medium", "1w": "long"}
        horizon_counts: Dict[str, Dict[str, int]] = {
            "short": {"Bullish": 0, "Bearish": 0, "Neutral": 0},
            "medium": {"Bullish": 0, "Bearish": 0, "Neutral": 0},
            "long": {"Bullish": 0, "Bearish": 0, "Neutral": 0},
        }
        for tf, data in tf_signals.items():
            horizon = horizon_map.get(tf, "short")
            direction = data["direction"].capitalize()
            if direction in horizon_counts[horizon]:
                horizon_counts[horizon][direction] += 1
        # Determine majority per horizon
        dominant_by_horizon: Dict[str, str] = {}
        for horizon, counts in horizon_counts.items():
            # Pick the direction with the highest count; if tie, pick Neutral
            dominant = max(counts, key=lambda k: (counts[k], k == "Neutral"))
            dominant_by_horizon[horizon] = dominant
        # Compose plan
        plan_lines: List[str] = []
        # Determine if there is any bullish direction across horizons
        any_bullish = any(val == "Bullish" for val in dominant_by_horizon.values())
        any_bearish = any(val == "Bearish" for val in dominant_by_horizon.values())
        # Example plan heuristics
        if last_close and any_bullish:
            # Compute entry range ±0.5% of last close
            entry_min = last_close * 0.995
            entry_max = last_close * 1.005
            stop_loss = last_close * 0.97
            tp1 = last_close * 1.03
            tp2 = last_close * 1.05
            # Format numbers with dynamic precision based on price magnitude
            def fmt(x: float) -> str:
                if x >= 1000:
                    return f"{x:,.0f}".replace(",", " ")
                elif x >= 100:
                    return f"{x:,.2f}".replace(",", " ")
                elif x >= 1:
                    return f"{x:,.3f}".replace(",", " ")
                else:
                    return f"{x:,.5f}".replace(",", " ")
            plan_lines.append(
                f"Plan : achat entre {fmt(entry_min)} et {fmt(entry_max)} $, stop loss {fmt(stop_loss)} $, take profit {fmt(tp1)} $ puis {fmt(tp2)} $."
            )
        elif any_bearish:
            # Bearish trend; note opportunity on short term
            plan_lines.append("Plan : attendre confirmation ou envisager une position courte si vous êtes expérimenté.")
        else:
            # Neutral or mixed; advise caution
            plan_lines.append("Plan : rester à l’écart ou attendre un signal plus clair avant d’entrer.")
        # Determine short‑term opportunity statement
        opp_lines: List[str] = []
        short_dir = dominant_by_horizon.get("short")
        if short_dir == "Bullish":
            opp_lines.append("Opportunité haussière court terme")
        elif short_dir == "Bearish":
            opp_lines.append("Opportunité baissière court terme")
        elif short_dir == "Neutral":
            opp_lines.append("Pas d’opportunité claire court terme")
        # Build final summary
        summary_lines = [f"{symbol} :", directions_text]
        if opp_lines:
            summary_lines.append(opp_lines[0])
        if plan_lines:
            summary_lines.extend(plan_lines)
        return "\n".join(summary_lines)

    # Send messages using the helper
    try:
        # Global summary always goes first
        send_telegram_message(telegram_token, chat_id, global_summary)
        # For each asset, build a simplified summary and send a separate message
        for symbol, tf_signals in results.items():
            summary_text = summarise_pair_trade_plan(symbol, tf_signals)
            send_telegram_message(telegram_token, chat_id, summary_text)
    except Exception as exc:
        logging.error("Failed to send Telegram message: %s", exc)


if __name__ == "__main__":
    run_bot()
