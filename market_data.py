"""
market_data.py
===============
Builds 5-min and 15-min OHLCV candles for the NIFTY spot index, either
from live ticks (paper/live mode) or from historical bars (backtest mode).

Rules enforced here (per your "never repaint / never use future candles"
requirement):
  - A candle is only exposed to strategy.py once it has CLOSED.
  - The in-progress candle is tracked separately and never handed to
    the signal engine.
  - VWAP resets at the start of each session day and only uses bars
    up to and including the last CLOSED candle.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config import CONFIG
from logger import get_logger
from utils import Candle

_log = get_logger("market_data")


def _floor_to_bucket(ts: datetime, minutes: int) -> datetime:
    discard = timedelta(
        minutes=ts.minute % minutes,
        seconds=ts.second,
        microseconds=ts.microsecond,
    )
    return ts - discard


@dataclass
class _SeriesState:
    closed: deque = field(default_factory=lambda: deque(maxlen=500))
    current: Optional[Candle] = None
    current_bucket: Optional[datetime] = None
    # running VWAP accumulators for the trading day
    vwap_cum_pv: float = 0.0
    vwap_cum_vol: float = 0.0
    vwap_day: Optional[datetime] = None


class CandleStore:
    """Maintains 5-min and 15-min candle series for one or more instrument tokens."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # state[token][timeframe_minutes] -> _SeriesState
        self._state: dict[int, dict[int, _SeriesState]] = defaultdict(
            lambda: {
                CONFIG.session.primary_timeframe_min: _SeriesState(),
                CONFIG.session.higher_timeframe_min: _SeriesState(),
            }
        )
        self._on_candle_close_callbacks: list = []

    def on_candle_close(self, callback) -> None:
        """callback(token, timeframe_min, Candle) fired whenever a candle finalizes."""
        self._on_candle_close_callbacks.append(callback)

    # ------------------------------------------------------------
    # Live tick ingestion
    # ------------------------------------------------------------

    def on_tick(self, token: int, price: float, volume_traded: int, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now()
        with self._lock:
            for tf_min, state in self._state[token].items():
                self._update_series(token, tf_min, state, price, volume_traded, ts)

    def _update_series(self, token: int, tf_min: int, state: _SeriesState,
                        price: float, volume_traded: int, ts: datetime) -> None:
        bucket = _floor_to_bucket(ts, tf_min)
        day = ts.date()

        # reset VWAP accumulators on a new day
        if state.vwap_day != day:
            state.vwap_cum_pv = 0.0
            state.vwap_cum_vol = 0.0
            state.vwap_day = day

        if state.current is None or bucket != state.current_bucket:
            # finalize previous candle (only if it existed and is fully formed)
            if state.current is not None:
                state.closed.append(state.current)
                for cb in self._on_candle_close_callbacks:
                    try:
                        cb(token, tf_min, state.current)
                    except Exception:
                        _log.exception("on_candle_close callback failed")
            state.current = Candle(ts=bucket, open=price, high=price, low=price, close=price, volume=0)
            state.current_bucket = bucket

        c = state.current
        c.high = max(c.high, price)
        c.low = min(c.low, price)
        c.close = price
        c.volume += max(volume_traded, 0)

        # VWAP uses traded volume * price accumulated intraday
        state.vwap_cum_pv += price * max(volume_traded, 1)
        state.vwap_cum_vol += max(volume_traded, 1)

    # ------------------------------------------------------------
    # Historical warm-up / backtest loading
    # ------------------------------------------------------------

    def load_historical(self, token: int, tf_min: int, bars: list[dict]) -> None:
        """
        bars: list of dicts as returned by KiteAPI.historical_data(), i.e.
              {"date": datetime, "open":.., "high":.., "low":.., "close":.., "volume":..}
        All bars loaded here are treated as CLOSED candles.
        """
        with self._lock:
            state = self._state[token][tf_min]
            day = None
            for b in bars:
                ts = b["date"]
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                if day != ts.date():
                    day = ts.date()
                    state.vwap_cum_pv = 0.0
                    state.vwap_cum_vol = 0.0
                    state.vwap_day = day
                candle = Candle(ts=ts, open=b["open"], high=b["high"], low=b["low"],
                                 close=b["close"], volume=b.get("volume", 0))
                state.closed.append(candle)
                state.vwap_cum_pv += candle.close * max(candle.volume, 1)
                state.vwap_cum_vol += max(candle.volume, 1)
            _log.info("Loaded %d historical %d-min bars for token %s", len(bars), tf_min, token)

    # ------------------------------------------------------------
    # Read access (strategy.py only ever sees CLOSED candles)
    # ------------------------------------------------------------

    def get_closed_candles(self, token: int, tf_min: int, n: Optional[int] = None) -> list[Candle]:
        with self._lock:
            state = self._state[token][tf_min]
            data = list(state.closed)
        return data[-n:] if n else data

    def get_current_vwap(self, token: int, tf_min: int) -> Optional[float]:
        with self._lock:
            state = self._state[token][tf_min]
            if state.vwap_cum_vol == 0:
                return None
            return state.vwap_cum_pv / state.vwap_cum_vol

    def get_last_price(self, token: int, tf_min: int) -> Optional[float]:
        with self._lock:
            state = self._state[token][tf_min]
            if state.current is not None:
                return state.current.close
            if state.closed:
                return state.closed[-1].close
            return None