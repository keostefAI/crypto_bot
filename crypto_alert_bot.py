"""
Crypto Market Alert Bot
=======================

This module implements a Telegram bot that monitors cryptocurrency markets
and sends alerts when certain technical and sentiment‑based conditions are met.

The bot follows the specification provided by the user. It pulls OHLCV price
data from Binance’s public REST API, computes a suite of technical indicators
(moving averages, RSI, MACD, and volume averages), integrates sentiment data
from the Alternative.me Fear & Greed index, and optionally fetches global
market metrics from CoinMarketCap. When multiple indicators align to form a
clear bullish or bearish setup, the bot notifies a configured Telegram chat
with a concise summary of the signal and its strength.

Key features
------------

* Fetches historical price and volume data for a configurable list of trading
  pairs (e.g. ``BTCUSDT``, ``ETHUSDT``) across several timeframes (1h, 4h,
  daily, weekly).
* Computes technical indicators without relying on external TA libraries:
  - 50‑ and 200‑period simple moving averages (MA50/MA200).
  - Relative Strength Index (RSI, 14‑period) using a smoothed calculation.
  - Moving Average Convergence Divergence (MACD) with standard (12,26,9)
    parameters and histogram.
  - 20‑period average volume for volume confirmation.
* Evaluates pre‑defined rules for bullish and bearish setups. Signals are
  classified as **weak**, **medium** or **strong** depending on how many
  conditions are satisfied.
* Pulls the latest crypto Fear & Greed index from Alternative.me’s public
  endpoint ``/fng/``. The API returns a single number (0 – 100) where 0
  denotes extreme fear and 100 represents extreme greed【239967436881371†L64-L79】.
  A fear reading below 25 can flag a contrarian accumulation signal while
  readings above 75 can warn of potential tops.
* Optionally uses the CoinMarketCap API to retrieve global market metrics
  such as total market capitalisation and BTC dominance. Integrating with
  CoinMarketCap requires an API key (see
  https://coinmarketcap.com/api/documentation/v1/). The public tier of
  CoinGecko, a popular alternative, allows roughly 30 calls per minute【675826768962597†L120-L123】.
* Persists previously detected signals in a lightweight SQLite database to
  prevent duplicate notifications.

Before running the bot you need to:

1. Obtain a Telegram bot token by talking to ``@BotFather`` on Telegram.
2. Create a chat (private or group) where the bot will post alerts and note
   down the chat ID. You can obtain the chat ID by sending a message to
   ``https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`` after the bot has
   been added to the chat.
3. Optionally sign up for a CoinMarketCap API key if you wish to include
   global market data. Without this key the bot will still operate, but
   ``fetch_global_metrics`` will be skipped.

To install dependencies, run:

```
pip install requests python‑telegram‑bot==20.1 pandas
```

If you plan to extend the bot to execute trades or interface with other
exchanges, consider using the ``ccxt`` library, which provides a unified
interface to over 100 cryptocurrency exchanges【860877663968419†L18-L23】.  However,
for this monitoring‑only bot the direct REST calls to Binance are sufficient.
"""

import asyncio
import dataclasses
import datetime as dt
import logging
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from telegram import Bot


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class PairConfig:
    """Configuration for a single trading pair and its checked timeframes."""

    symbol: str
    timeframes: List[str]  # e.g. ["1h", "4h", "1d", "1w"]


@dataclasses.dataclass
class BotConfig:
    """Overall configuration for the alert bot."""

    telegram_token: str
    telegram_chat_id: str
    pairs: List[PairConfig]
    cmc_api_key: Optional[str] = None
    check_interval_seconds: int = 300  # how often to scan markets (default 5min)


# -----------------------------------------------------------------------------
# Helper functions for technical indicators
# -----------------------------------------------------------------------------

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Compute a simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) over the given period.

    The RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions. This implementation uses the simple
    moving average of gains and losses. See
    https://en.wikipedia.org/wiki/Relative_strength_index for details.
    """
    delta = series.diff()
    # separate positive and negative deltas
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # calculate rolling means of gains and losses
    roll_up = up.rolling(window=period, min_periods=period).mean()
    roll_down = down.rolling(window=period, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, signal line and histogram.

    * MACD line = EMA(fast) − EMA(slow)
    * Signal line = EMA of MACD line
    * Histogram = MACD line − Signal line
    """
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# -----------------------------------------------------------------------------
# Data retrieval
# -----------------------------------------------------------------------------

BINANCE_BASE_URL = "https://api.binance.com/api/v3"


def fetch_binance_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV data from Binance for a given symbol and interval.

    Binance provides public candlestick (kline) data without requiring an API key.
    This function converts the returned list into a DataFrame with appropriate
    datatypes and column names. Each row corresponds to one candlestick.

    :param symbol: Trading pair symbol (e.g. "BTCUSDT").
    :param interval: Candlestick interval (1m, 5m, 1h, 1d, 1w, etc.).
    :param limit: Number of candlesticks to fetch (max 1000 for Binance).
    :return: DataFrame with columns ["open_time", "open", "high", "low", "close",
             "volume", "close_time", "quote_asset_volume", "trades", ...].
    """
    url = f"{BINANCE_BASE_URL}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logging.error("Failed to fetch klines for %s %s: %s", symbol, interval, exc)
        raise
    data = resp.json()
    # Build DataFrame
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"])
    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base_volume",
                "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = df["trades"].astype(int)
    return df


def fetch_fear_greed_index(limit: int = 1) -> Optional[Dict[str, str]]:
    """Fetch the crypto Fear & Greed index from alternative.me.

    The API returns a JSON object with a list of data points. By default only
    the most recent value is returned. If the request fails, ``None`` is
    returned.

    According to Alternative.me’s documentation, the API endpoint is
    ``https://api.alternative.me/fng/`` and supports optional parameters
    ``limit``, ``format`` and ``date_format``【239967436881371†L208-L248】.  The index is
    interpreted as follows: values near 0 represent extreme fear, values near
    100 represent extreme greed【239967436881371†L64-L79】.
    """
    try:
        url = "https://api.alternative.me/fng/"
        params = {"limit": limit, "format": "json"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"):
            return None
        return data["data"][0]  # return the latest index entry
    except Exception as exc:
        logging.warning("Failed to fetch Fear & Greed index: %s", exc)
        return None


def fetch_global_metrics(cmc_api_key: str) -> Optional[Dict]:
    """Fetch global crypto metrics using the CoinMarketCap API.

    CoinMarketCap exposes ``/v1/global-metrics/quotes/latest`` to return
    aggregated market data such as total market cap and BTC dominance
    【256771951794996†L3631-L3638】.  An API key is required and should be passed via
    the ``X-CMC_PRO_API_KEY`` header. On success, this function returns the
    JSON response; otherwise ``None`` is returned. Users on the free tier must
    observe CoinMarketCap’s rate limits – consult their documentation for
    details (public tiers often allow a few hundred calls per day).
    """
    try:
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": cmc_api_key, "Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logging.warning("Failed to fetch global metrics: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Signal detection logic
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class Signal:
    """Represents a detected trading signal."""

    pair: str
    timeframe: str
    direction: str  # "bullish" or "bearish"
    strength: str   # "weak", "medium", "strong"
    reasons: List[str]
    timestamp: dt.datetime


def evaluate_signals(df: pd.DataFrame,
                      pair: str,
                      timeframe: str,
                      fear_greed: Optional[Dict[str, str]] = None) -> Optional[Signal]:
    """Evaluate the latest candle in ``df`` and generate a signal if conditions align.

    This function implements the rule set described in the specification. It
    compares the most recent values of technical indicators and counts how
    many bullish or bearish conditions are met. Based on the count, it assigns
    a strength level (weak/medium/strong). If no meaningful alignment is
    detected, the function returns ``None``.
    """
    if df.empty or len(df) < 200:
        # Not enough data to compute MA200
        return None
    # Use the last two rows to evaluate crossovers
    last = df.iloc[-1]
    prev = df.iloc[-2]
    reasons = []
    direction = None
    # Compute indicator conditions
    # Moving average crossovers
    ma50_cross_up = prev["ma50"] <= prev["ma200"] and last["ma50"] > last["ma200"]
    ma50_cross_down = prev["ma50"] >= prev["ma200"] and last["ma50"] < last["ma200"]
    price_above_ma200 = last["close"] > last["ma200"]
    price_below_ma200 = last["close"] < last["ma200"]
    # RSI signals
    rsi_gt_55 = last["rsi"] > 55
    rsi_lt_45 = last["rsi"] < 45
    rsi_cross_50_up = prev["rsi"] <= 50 and last["rsi"] > 50
    rsi_cross_50_down = prev["rsi"] >= 50 and last["rsi"] < 50
    # MACD signals
    macd_bullish = last["macd_line"] > last["signal_line"]
    macd_bearish = last["macd_line"] < last["signal_line"]
    # Volume confirmation
    volume_high = last["volume"] > last["vol_mean20"]
    volume_low = last["volume"] <= last["vol_mean20"]
    # Evaluate bullish configuration
    bullish_conditions = []
    bearish_conditions = []
    # Short‑term bullish: RSI>55 + MACD bullish + high volume
    if rsi_gt_55:
        bullish_conditions.append("RSI>55 (momentum confirmed)")
    if macd_bullish:
        bullish_conditions.append("MACD crossover bullish")
    if volume_high:
        bullish_conditions.append("Volume above 20‑period average")
    # Long‑term bullish: MA50 cross up or RSI cross above 50
    if ma50_cross_up:
        bullish_conditions.append("MA50 crossed above MA200 (Golden Cross)")
    if rsi_cross_50_up:
        bullish_conditions.append("RSI crossed above 50 (long‑term momentum)")
    if price_above_ma200:
        bullish_conditions.append("Price above MA200 (long‑term uptrend)")
    # Evaluate bearish configuration
    if rsi_lt_45:
        bearish_conditions.append("RSI<45 (momentum bearish)")
    if macd_bearish:
        bearish_conditions.append("MACD crossover bearish")
    if volume_high:
        bearish_conditions.append("Volume above 20‑period average")
    if ma50_cross_down:
        bearish_conditions.append("MA50 crossed below MA200 (Death Cross)")
    if rsi_cross_50_down:
        bearish_conditions.append("RSI crossed below 50 (long‑term bearish)")
    if price_below_ma200:
        bearish_conditions.append("Price below MA200 (long‑term downtrend)")
    # Contrarian signal based on Fear & Greed index
    if fear_greed is not None:
        try:
            idx_value = float(fear_greed.get("value"))
            classification = fear_greed.get("value_classification", "").lower()
            # Extreme fear may indicate accumulation opportunity
            if idx_value < 25:
                bullish_conditions.append(
                    f"Fear & Greed extremely low ({idx_value}) – potential accumulation"
                )
            # Extreme greed may warn of pullback
            if idx_value > 75:
                bearish_conditions.append(
                    f"Fear & Greed extremely high ({idx_value}) – potential top"
                )
        except Exception:
            pass
    # Determine direction and strength
    num_bull = len(bullish_conditions)
    num_bear = len(bearish_conditions)
    if num_bull == 0 and num_bear == 0:
        return None
    if num_bull > num_bear:
        direction = "bullish"
        reasons = bullish_conditions
        strength_count = num_bull
    elif num_bear > num_bull:
        direction = "bearish"
        reasons = bearish_conditions
        strength_count = num_bear
    else:
        # tie – ignore to avoid conflicting signals
        return None
    # Determine strength label
    if strength_count >= 3:
        strength = "strong"
    elif strength_count == 2:
        strength = "medium"
    else:
        strength = "weak"
    return Signal(
        pair=pair,
        timeframe=timeframe,
        direction=direction,
        strength=strength,
        reasons=reasons,
        timestamp=last["close_time"],
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to a candlestick DataFrame.

    This function computes MA50, MA200, RSI(14), MACD and 20‑period volume mean
    for the provided DataFrame. It returns the same DataFrame with added
    columns. Any rows lacking enough history for the calculations will
    contain NaNs, which the signal evaluator will filter out by requiring
    at least 200 rows before evaluating.
    """
    df = df.copy()
    # Moving averages
    df["ma50"] = compute_sma(df["close"], 50)
    df["ma200"] = compute_sma(df["close"], 200)
    # RSI
    df["rsi"] = compute_rsi(df["close"], 14)
    # MACD
    macd_line, signal_line, histogram = compute_macd(df["close"])
    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["macd_hist"] = histogram
    # Volume mean
    df["vol_mean20"] = df["volume"].rolling(window=20, min_periods=20).mean()
    return df


# -----------------------------------------------------------------------------
# Database to track sent signals
# -----------------------------------------------------------------------------

class SignalDatabase:
    """SQLite wrapper to store and check previously sent signals."""

    def __init__(self, path: str = "signals.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                timeframe TEXT,
                direction TEXT,
                strength TEXT,
                timestamp INTEGER,
                sent_at INTEGER
            )
            """
        )
        self.conn.commit()

    def has_signal(self, pair: str, timeframe: str, direction: str, strength: str,
                   timestamp: dt.datetime) -> bool:
        """Check whether a signal has already been sent for the given candle.

        This prevents duplicate alerts if the bot is restarted or scans multiple
        times before a new candle closes. We index by pair, timeframe,
        direction, strength and the candle's closing timestamp.
        """
        ts_int = int(timestamp.timestamp())
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM signals
            WHERE pair=? AND timeframe=? AND direction=? AND strength=? AND timestamp=?
            LIMIT 1
            """,
            (pair, timeframe, direction, strength, ts_int),
        )
        return cur.fetchone() is not None

    def store_signal(self, signal: Signal) -> None:
        ts_int = int(signal.timestamp.timestamp())
        sent_int = int(time.time())
        self.conn.execute(
            """
            INSERT INTO signals (pair, timeframe, direction, strength, timestamp, sent_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (signal.pair, signal.timeframe, signal.direction,
             signal.strength, ts_int, sent_int),
        )
        self.conn.commit()


# -----------------------------------------------------------------------------
# Main bot class
# -----------------------------------------------------------------------------

class CryptoAlertBot:
    """Encapsulates the alerting logic and Telegram integration."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.bot = Bot(token=config.telegram_token)
        self.db = SignalDatabase()
        # Session for HTTP calls
        self.session = requests.Session()

    async def scan_once(self) -> None:
        """Perform a single scan across all configured pairs/timeframes."""
        # Fetch fear & greed index once per scan
        fear_data = fetch_fear_greed_index()
        # Optionally fetch global metrics (not used directly in rules but could
        # be included in the alert message)
        global_metrics = None
        if self.config.cmc_api_key:
            global_metrics = fetch_global_metrics(self.config.cmc_api_key)
        for pair_cfg in self.config.pairs:
            for tf in pair_cfg.timeframes:
                try:
                    df_raw = fetch_binance_klines(pair_cfg.symbol, tf)
                except Exception as exc:
                    logging.error("Error fetching data for %s %s: %s", pair_cfg.symbol, tf, exc)
                    continue
                df = prepare_dataframe(df_raw)
                signal = evaluate_signals(df, pair_cfg.symbol, tf, fear_data)
                if signal is None:
                    continue
                # Avoid duplicate alerts
                if self.db.has_signal(signal.pair, signal.timeframe,
                                      signal.direction, signal.strength,
                                      signal.timestamp):
                    continue
                # Compose alert message
                msg = self.compose_message(signal, fear_data, global_metrics)
                # Send via Telegram
                await self.send_message(msg)
                # Store in DB
                self.db.store_signal(signal)

    async def send_message(self, text: str) -> None:
        """Send a message to the configured Telegram chat."""
        try:
            await self.bot.send_message(chat_id=self.config.telegram_chat_id, text=text)
        except Exception as exc:
            logging.error("Failed to send Telegram message: %s", exc)

    def compose_message(self, signal: Signal,
                         fear_data: Optional[Dict[str, str]],
                         global_metrics: Optional[Dict]) -> str:
        """Create a human‑readable message summarising the signal."""
        emoji = "🚀" if signal.direction == "bullish" else "🔻"
        lines = [
            f"{emoji} *{signal.direction.capitalize()} {signal.strength.capitalize()} Signal*",
            f"Pair: {signal.pair}",
            f"Timeframe: {signal.timeframe}",
            f"Close time: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "Reasons:"
        ]
        for reason in signal.reasons:
            lines.append(f"• {reason}")
        # Include Fear & Greed index reading
        if fear_data:
            idx_value = fear_data.get("value")
            classification = fear_data.get("value_classification")
            lines.append(f"Fear & Greed Index: {idx_value} ({classification})")
        # Include global metrics if available
        if global_metrics and "data" in global_metrics:
            gm = global_metrics["data"]
            total_cap = gm.get("quote", {}).get("USD", {}).get("total_market_cap")
            btc_dom = gm.get("btc_dominance")
            if total_cap:
                cap_str = f"${total_cap/1e12:.2f}T" if total_cap > 1e12 else f"${total_cap/1e9:.2f}B"
                lines.append(f"Global Market Cap: {cap_str}")
            if btc_dom:
                lines.append(f"BTC Dominance: {btc_dom:.2f}%")
        return "\n".join(lines)

    async def run(self) -> None:
        """Run the bot indefinitely, scanning at fixed intervals."""
        logging.info("Starting CryptoAlertBot...")
        while True:
            start_time = time.time()
            try:
                await self.scan_once()
            except Exception as exc:
                logging.exception("Unexpected error during scan: %s", exc)
            # Sleep until next iteration
            elapsed = time.time() - start_time
            sleep_for = max(0, self.config.check_interval_seconds - elapsed)
            await asyncio.sleep(sleep_for)


# -----------------------------------------------------------------------------
# Entrypoint helper
# -----------------------------------------------------------------------------

def load_default_config() -> BotConfig:
    """Create a default configuration skeleton.

    This helper reads sensitive values from environment variables if they are
    set. To avoid accidentally committing secrets to version control, the
    Telegram bot token, chat ID and optional CoinMarketCap API key can be
    provided through the variables ``TELEGRAM_TOKEN``, ``TELEGRAM_CHAT_ID``
    and ``CMC_API_KEY`` respectively. If these variables are absent, the
    placeholders ``"YOUR_TELEGRAM_BOT_TOKEN"`` and ``"YOUR_TELEGRAM_CHAT_ID"``
    remain and an exception will be raised at runtime.  Adjust the list of
    trading pairs and timeframes as needed.
    """
    import os

    pairs = [
        PairConfig(symbol="BTCUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="ETHUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="BNBUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="SOLUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="XRPUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="DOGEUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="AAVEUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="HYPEUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="ATOMUSDT", timeframes=["1h", "4h", "1d", "1w"]),
        PairConfig(symbol="LINKUSDT", timeframes=["1h", "4h", "1d", "1w"]),
    ]
    telegram_token = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
    cmc_api_key = os.environ.get("CMC_API_KEY")
    return BotConfig(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        pairs=pairs,
        cmc_api_key=cmc_api_key,
        check_interval_seconds=3600,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    config = load_default_config()
    # Prompt the user to fill in the token if it still contains placeholder
    if config.telegram_token.startswith("YOUR_"):
        raise RuntimeError(
            "Please set your Telegram bot token and chat ID in the configuration. "
            "Call load_default_config() and override the fields accordingly."
        )
    bot = CryptoAlertBot(config)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")