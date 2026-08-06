"""
order_manager.py
==================
Manages live order placement, SL tracking, position monitoring
and force-square-off. Communicates with the kite_api REST layer
for all broker interactions and calls the registered on_exit
callback whenever a position is closed (by SL, target, or manual
square-off).
"""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from config import CONFIG
from kite_api import KiteAPI
from logger import get_logger, log_decision
from trade_journal import record_trade
from utils import OptionContract, Position, TradeSide, TRAIL_INITIAL_RATIO, TRAIL_FINAL_RATIO, TRAIL_TRIGGER_POINTS

_log = get_logger("order_manager")


@dataclass
class ExitResult:
    pnl: float = 0.0
    reason: str = ""
    exit_price: float = 0.0


class OrderManager:
    def __init__(self, kite: KiteAPI, get_ltp: Callable[[str], Optional[float]]) -> None:
        self.kite = kite
        self._get_ltp = get_ltp
        self._on_exit_callback: Optional[Callable[[str, ExitResult], None]] = None
        self._positions: dict[str, Position] = {}
        self._sl_order_ids: dict[str, str] = {}
        self._entry_order_ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def on_exit(self, callback: Callable[[str, ExitResult], None]) -> None:
        self._on_exit_callback = callback

    def open_position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def get_open_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def enter_position(
        self,
        contract: OptionContract,
        side: TradeSide,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        initial_sl: float,
    ) -> None:
        option_ltp = self._get_ltp(contract.tradingsymbol) or 0.0
        if option_ltp <= 0:
            _log.warning("Cannot enter %s — LTP is 0 or unavailable", contract.tradingsymbol)
            return

        transaction_type = self.kite.TRANSACTION_TYPE_BUY if side == TradeSide.LONG else self.kite.TRANSACTION_TYPE_SELL

        try:
            order_id = self.kite.place_order(
                tradingsymbol=contract.tradingsymbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                product=CONFIG.capital.product_type,
                order_type="MARKET",
            )
            self._entry_order_ids[contract.tradingsymbol] = order_id
            _log.info("Entry order placed: %s qty=%d order_id=%s", contract.tradingsymbol, quantity, order_id)
        except Exception:
            _log.exception("Failed to place entry order for %s", contract.tradingsymbol)
            return

        sl_offset = CONFIG.trade_mgmt.sl_limit_offset_points
        if side == TradeSide.LONG:
            sl_trigger = stop_loss
            sl_limit = sl_trigger + sl_offset
        else:
            sl_trigger = stop_loss
            sl_limit = sl_trigger - sl_offset

        try:
            sl_order_id = self.kite.place_order(
                tradingsymbol=contract.tradingsymbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL if side == TradeSide.LONG else self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=CONFIG.capital.product_type,
                order_type="SL",
                price=round(sl_limit, 2),
                trigger_price=round(sl_trigger, 2),
            )
            self._sl_order_ids[contract.tradingsymbol] = sl_order_id
            _log.info("SL order placed: %s trigger=%.2f limit=%.2f order_id=%s", contract.tradingsymbol, sl_trigger, sl_limit, sl_order_id)
        except Exception:
            _log.exception("Failed to place SL order for %s", contract.tradingsymbol)

        position = Position(
            contract=contract,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            entry_time=datetime.now(),
            initial_sl=initial_sl,
        )

        with self._lock:
            self._positions[contract.tradingsymbol] = position

        entry_time_iso = position.entry_time.isoformat()
        record_trade(
            event="ENTRY",
            symbol=contract.tradingsymbol,
            side=side.value,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            pnl=0.0,
            reason="ENTRY_PLACED",
            strike=contract.strike,
            option_type=contract.option_type,
            expiry=contract.expiry,
            lot_size=contract.lot_size,
            entry_time=entry_time_iso,
            exit_time="",
        )

        log_decision(
            "ENTRY_PLACED",
            symbol=contract.tradingsymbol,
            side=side.value,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
        )

    def monitor_positions(self) -> None:
        with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            self._check_position_exit(symbol)

    def _check_position_exit(self, symbol: str) -> None:
        position = None
        with self._lock:
            position = self._positions.get(symbol)
        if position is None:
            return

        ltp = self._get_ltp(symbol)
        if ltp is None:
            return

        should_exit = False
        reason = ""
        exit_price = ltp

        # Trailing SL: when profit >= 2 points, start trailing
        profit = (
            ltp - position.entry_price
            if position.side == TradeSide.LONG
            else position.entry_price - ltp
        )
        if profit >= 2.0 and position.initial_sl > 0:
            progress = min(
                (profit - 2.0)
                / max(1, position.target - position.entry_price - 2.0),
                1.0,
            )
            trail_ratio = TRAIL_INITIAL_RATIO - progress * (TRAIL_INITIAL_RATIO - TRAIL_FINAL_RATIO)
            trail_sl = (
                position.entry_price + position.initial_sl * trail_ratio / 8.0
                if position.side == TradeSide.LONG
                else position.entry_price - position.initial_sl * trail_ratio / 8.0
            )
            if position.side == TradeSide.LONG:
                position.stop_loss = max(position.stop_loss, trail_sl)
            else:
                position.stop_loss = min(position.stop_loss, trail_sl)

        if position.side == TradeSide.LONG:
            if ltp >= position.target:
                should_exit = True
                reason = "TARGET"
                exit_price = position.target
            elif ltp <= position.stop_loss:
                should_exit = True
                reason = "STOP_LOSS"
                exit_price = position.stop_loss
        else:
            if ltp <= position.target:
                should_exit = True
                reason = "TARGET"
                exit_price = position.target
            elif ltp >= position.stop_loss:
                should_exit = True
                reason = "STOP_LOSS"
                exit_price = position.stop_loss

        if should_exit:
            self._close_position(symbol, position, exit_price, reason)

    def _close_position(self, symbol: str, position: Position, exit_price: float, reason: str) -> None:
        try:
            transaction_type = (
                self.kite.TRANSACTION_TYPE_SELL if position.side == TradeSide.LONG else self.kite.TRANSACTION_TYPE_BUY
            )
            self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="MARKET",
            )
            _log.info("Exit order placed: %s reason=%s price=%.2f", symbol, reason, exit_price)
        except Exception:
            _log.exception("Failed to place exit order for %s", symbol)
            return

        pnl = self._compute_pnl(position, exit_price)
        result = ExitResult(pnl=pnl, reason=reason, exit_price=exit_price)

        with self._lock:
            self._positions.pop(symbol, None)
            self._sl_order_ids.pop(symbol, None)
            self._entry_order_ids.pop(symbol, None)

        log_decision(
            "EXIT",
            symbol=symbol,
            reason=reason,
            exit_price=exit_price,
            pnl=round(pnl, 2),
        )

        record_trade(
            event="EXIT",
            symbol=symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=round(pnl, 2),
            reason=reason,
            strike=position.contract.strike,
            option_type=position.contract.option_type,
            expiry=position.contract.expiry,
            lot_size=position.contract.lot_size,
            entry_time=position.entry_time.isoformat(),
            exit_time=datetime.now().isoformat(),
        )

        if self._on_exit_callback:
            try:
                self._on_exit_callback(symbol, result)
            except Exception:
                _log.exception("on_exit callback failed for %s", symbol)

    def _compute_pnl(self, position: Position, exit_price: float) -> float:
        if position.side == TradeSide.LONG:
            return (exit_price - position.entry_price) * position.quantity
        return (position.entry_price - exit_price) * position.quantity

    def force_square_off_all(self) -> None:
        with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            position = None
            with self._lock:
                position = self._positions.get(symbol)
            if position is None:
                continue

            try:
                transaction_type = (
                    self.kite.TRANSACTION_TYPE_SELL if position.side == TradeSide.LONG else self.kite.TRANSACTION_TYPE_BUY
                )
                self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=CONFIG.instrument.option_exchange,
                    transaction_type=transaction_type,
                    quantity=position.quantity,
                    product=CONFIG.capital.product_type,
                    order_type="MARKET",
                )
                _log.info("Force square-off: %s", symbol)
            except Exception:
                _log.exception("Failed to force square-off %s", symbol)
                continue

            pnl = self._compute_pnl(position, position.entry_price)
            result = ExitResult(pnl=pnl, reason="FORCE_SQUARE_OFF", exit_price=position.entry_price)

            with self._lock:
                self._positions.pop(symbol, None)
                self._sl_order_ids.pop(symbol, None)
                self._entry_order_ids.pop(symbol, None)

            log_decision(
                "FORCE_SQUARE_OFF",
                symbol=symbol,
                pnl=round(pnl, 2),
            )

            record_trade(
                event="FORCE_SQUARE_OFF",
                symbol=symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=position.entry_price,
                quantity=position.quantity,
                pnl=round(pnl, 2),
                reason="FORCE_SQUARE_OFF",
                strike=position.contract.strike,
                option_type=position.contract.option_type,
                expiry=position.contract.expiry,
                lot_size=position.contract.lot_size,
                entry_time=position.entry_time.isoformat(),
                exit_time=datetime.now().isoformat(),
            )

            if self._on_exit_callback:
                try:
                    self._on_exit_callback(symbol, result)
                except Exception:
                    _log.exception("on_exit callback failed for %s", symbol)

    def cancel_open_orders(self) -> None:
        with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            sl_order_id = self._sl_order_ids.get(symbol)
            if sl_order_id:
                try:
                    self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
                    _log.info("Cancelled SL order for %s", symbol)
                except Exception:
                    _log.exception("Failed to cancel SL order for %s", symbol)