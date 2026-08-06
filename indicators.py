"""
indicators.py
=============
Pure functions operating on lists of closed Candle objects. No I/O,
no state - everything here is deterministic and testable, and shared
identically between live/paper and backtest so results never diverge
between modes.

All EMA/ADX/ATR values here are computed using ONLY candles passed in,
so callers must never pass an in-progress candle (market_data.py
enforces that boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils import Candle


def ema_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


def latest_ema(closes: list[float], period: int) -> Optional[float]:
    series = ema_series(closes, period)
    return series[-1] if series else None


def true_range(prev_close: float, high: float, low: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(candles: list[Candle], period: int) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs = [
        true_range(candles[i - 1].close, candles[i].high, candles[i].low)
        for i in range(1, len(candles))
    ]
    if len(trs) < period:
        return None
    # Wilder's smoothing
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def adx(candles: list[Candle], period: int) -> Optional[float]:
    if len(candles) < period * 2:
        return None

    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(candles)):
        up_move = candles[i].high - candles[i - 1].high
        down_move = candles[i - 1].low - candles[i].low
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        trs.append(true_range(candles[i - 1].close, candles[i].high, candles[i].low))

    def wilder_smooth(values: list[float], period: int) -> list[float]:
        smoothed = [sum(values[:period])]
        for v in values[period:]:
            smoothed.append(smoothed[-1] - (smoothed[-1] / period) + v)
        return smoothed

    if len(trs) < period:
        return None

    smoothed_tr = wilder_smooth(trs, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)

    dx_values = []
    for tr_s, pdm_s, mdm_s in zip(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm):
        if tr_s == 0:
            dx_values.append(0.0)
            continue
        plus_di = 100 * (pdm_s / tr_s)
        minus_di = 100 * (mdm_s / tr_s)
        di_sum = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / di_sum) if di_sum != 0 else 0.0
        dx_values.append(dx)

    if len(dx_values) < period:
        return None
    return sum(dx_values[-period:]) / period


@dataclass
class CandleShape:
    body: float
    range_: float
    body_pct: float
    is_bullish: bool
    is_bearish: bool


def candle_shape(c: Candle) -> CandleShape:
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-9)
    return CandleShape(
        body=body,
        range_=rng,
        body_pct=body / rng,
        is_bullish=c.close > c.open,
        is_bearish=c.close < c.open,
    )


def is_strong_bull_candle(c: Candle, min_body_pct: float) -> bool:
    s = candle_shape(c)
    return s.is_bullish and s.body_pct >= min_body_pct


def is_strong_bear_candle(c: Candle, min_body_pct: float) -> bool:
    s = candle_shape(c)
    return s.is_bearish and s.body_pct >= min_body_pct


def is_bullish_engulfing(prev: Candle, cur: Candle) -> bool:
    return (
        cur.close > cur.open
        and prev.close < prev.open
        and cur.close >= prev.open
        and cur.open <= prev.close
    )


def is_bearish_engulfing(prev: Candle, cur: Candle) -> bool:
    return (
        cur.close < cur.open
        and prev.close > prev.open
        and cur.close <= prev.open
        and cur.open >= prev.close
    )


def is_hammer(c: Candle) -> bool:
    s = candle_shape(c)
    lower_wick = min(c.open, c.close) - c.low
    upper_wick = c.high - max(c.open, c.close)
    return lower_wick >= 2 * s.body and upper_wick <= s.body * 0.5 and s.body_pct > 0


def is_shooting_star(c: Candle) -> bool:
    s = candle_shape(c)
    upper_wick = c.high - max(c.open, c.close)
    lower_wick = min(c.open, c.close) - c.low
    return upper_wick >= 2 * s.body and lower_wick <= s.body * 0.5 and s.body_pct > 0


def volume_confirms(cur: Candle, prev: Candle) -> bool:
    return cur.volume > prev.volume


def ema_is_flat(closes: list[float], period: int, lookback: int, min_move_points: float) -> bool:
    """True if EMA(period) has moved less than min_move_points over the lookback window."""
    series = ema_series(closes, period)
    if len(series) < lookback + 1:
        return True  # not enough data -> treat as flat/unsafe
    move = abs(series[-1] - series[-1 - lookback])
    return move < min_move_points


def vwap_cross_count(candles: list[Candle], vwap_values: list[float], lookback: int) -> int:
    """Counts how many times close crossed VWAP over the last `lookback` candles."""
    n = min(lookback, len(candles) - 1, len(vwap_values) - 1)
    if n <= 0:
        return 0
    crosses = 0
    for i in range(-n, 0):
        prev_above = candles[i - 1].close > vwap_values[i - 1]
        cur_above = candles[i].close > vwap_values[i]
        if prev_above != cur_above:
            crosses += 1
    return crosses


def swing_high(candles: list[Candle], lookback: int) -> Optional[float]:
    if not candles:
        return None
    window = candles[-lookback:] if lookback else candles
    return max(c.high for c in window)


def swing_low(candles: list[Candle], lookback: int) -> Optional[float]:
    if not candles:
        return None
    window = candles[-lookback:] if lookback else candles
    return min(c.low for c in window)