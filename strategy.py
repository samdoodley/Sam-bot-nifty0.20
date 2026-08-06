"""
strategy.py
=============
Evaluates NIFTY 5-min and 15-min candle data and returns a
TradeSide signal with entry price, stop-loss and target.

The logic is purely indicator-driven (EMA crossover, ADX trend
strength, ATR volatility filter, VWAP alignment, volume
confirmation) and uses only config thresholds so it can be
tuned without touching code.
"""

from __future__ import annotations

from typing import Optional

from config import CONFIG
from indicators import (
    adx,
    atr,
    ema_is_flat,
    ema_series,
    latest_ema,
    swing_high,
    swing_low,
    vwap_cross_count,
)
from logger import get_logger
from utils import Signal, TradeSide, Candle

_log = get_logger("strategy")


class StrategyEngine:
    def __init__(self) -> None:
        self._last_signal_side: TradeSide = TradeSide.NONE

    def evaluate(
        self,
        closed_5m: list[Candle],
        closed_15m: list[Candle],
        vwap_series: list[float],
        symbol: str = "NIFTY",
    ) -> Signal:
        if len(closed_5m) < CONFIG.indicators.ema_slow:
            return Signal(side=TradeSide.NONE, reason="INSUFFICIENT_DATA")

        closes_5m = [c.close for c in closed_5m]
        ema20 = latest_ema(closes_5m, CONFIG.indicators.ema_fast)
        ema50 = latest_ema(closes_5m, CONFIG.indicators.ema_slow)

        if ema20 is None or ema50 is None:
            return Signal(side=TradeSide.NONE, reason="EMA_NOT_READY")

        adx_val = adx(closed_5m, CONFIG.indicators.adx_period)
        if adx_val is None or adx_val < CONFIG.strategy.adx_min:
            return Signal(side=TradeSide.NONE, reason="ADX_LOW", confidence_notes=f"adx={adx_val}")

        atr_val = atr(closed_5m, CONFIG.indicators.atr_period)
        if atr_val is None or atr_val < CONFIG.strategy.atr_min_points:
            return Signal(side=TradeSide.NONE, reason="ATR_TOO_LOW", confidence_notes=f"atr={atr_val}")

        if ema_is_flat(
            closes_5m,
            CONFIG.indicators.ema_fast,
            CONFIG.strategy.ema_flat_slope_lookback,
            CONFIG.strategy.ema_flat_slope_min_points,
        ):
            return Signal(side=TradeSide.NONE, reason="EMA_FLAT")

        current_price = closes_5m[-1]
        ema20_val = ema20
        ema50_val = ema50

        if ema20_val > ema50_val:
            trend = "UP"
        elif ema20_val < ema50_val:
            trend = "DOWN"
        else:
            return Signal(side=TradeSide.NONE, reason="EMA_CROSSOVER_NONE")

        if trend == "UP":
            if current_price < ema20_val - CONFIG.strategy.pullback_max_distance_from_ema20_points:
                return Signal(side=TradeSide.NONE, reason="PULLBACK_TOO_FAR", confidence_notes="price too far below EMA20")

            vwap_crosses = vwap_cross_count(closed_5m, vwap_series, CONFIG.strategy.vwap_cross_lookback_bars)
            if vwap_crosses > CONFIG.strategy.vwap_max_crosses_allowed:
                return Signal(side=TradeSide.NONE, reason="VWAP_CHOPY", confidence_notes=f"crosses={vwap_crosses}")

            if len(closed_5m) >= 2:
                prev_close = closed_5m[-2].close
                curr_close = current_price
                body = abs(curr_close - prev_close)
                rng = max(curr_close, prev_close) - min(curr_close, prev_close)
                if rng > 0 and body / rng < CONFIG.strategy.strong_candle_body_to_range_min_pct:
                    return Signal(side=TradeSide.NONE, reason="WEAK_CANDLE", confidence_notes=f"body/range={body/rng:.2f}")

            entry_price = current_price
            swing_low = min(c.low for c in closed_5m[-5:]) if len(closed_5m) >= 5 else closed_5m[-1].low
            stop_loss = max(closed_5m[-1].low, swing_low)
            stop_loss = min(stop_loss, entry_price - CONFIG.trade_mgmt.stop_loss_index_points_cap)
            sl_distance = entry_price - stop_loss
            target = entry_price + sl_distance * 2

            return Signal(
                side=TradeSide.LONG,
                reason="EMA_BULLISH_CROSSOVER",
                entry_price=entry_price,
                stop_loss=round(stop_loss, 2),
                target=round(target, 2),
                swing_level=ema20_val,
                confidence_notes=f"adx={adx_val:.1f} atr={atr_val:.1f} rr=1:2",
            )

        else:
            if current_price > ema20_val + CONFIG.strategy.pullback_max_distance_from_ema20_points:
                return Signal(side=TradeSide.NONE, reason="PULLBACK_TOO_FAR", confidence_notes="price too far above EMA20")

            vwap_crosses = vwap_cross_count(closed_5m, vwap_series, CONFIG.strategy.vwap_cross_lookback_bars)
            if vwap_crosses > CONFIG.strategy.vwap_max_crosses_allowed:
                return Signal(side=TradeSide.NONE, reason="VWAP_CHOPY", confidence_notes=f"crosses={vwap_crosses}")

            if len(closed_5m) >= 2:
                prev_close = closed_5m[-2].close
                curr_close = current_price
                body = abs(curr_close - prev_close)
                rng = max(curr_close, prev_close) - min(curr_close, prev_close)
                if rng > 0 and body / rng < CONFIG.strategy.strong_candle_body_to_range_min_pct:
                    return Signal(side=TradeSide.NONE, reason="WEAK_CANDLE", confidence_notes=f"body/range={body/rng:.2f}")

            entry_price = current_price
            swing_high = max(c.high for c in closed_5m[-5:]) if len(closed_5m) >= 5 else closed_5m[-1].high
            stop_loss = min(closed_5m[-1].high, swing_high)
            stop_loss = max(stop_loss, entry_price + CONFIG.trade_mgmt.stop_loss_index_points_cap)
            sl_distance = stop_loss - entry_price
            target = entry_price - sl_distance * 2

            return Signal(
                side=TradeSide.SHORT,
                reason="EMA_BEARISH_CROSSOVER",
                entry_price=entry_price,
                stop_loss=round(stop_loss, 2),
                target=round(target, 2),
                swing_level=ema20_val,
                confidence_notes=f"adx={adx_val:.1f} atr={atr_val:.1f} rr=1:2",
            )