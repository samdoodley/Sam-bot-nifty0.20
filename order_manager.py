"""
order_manager.py
=================
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
from utils import OptionContract, Position, PositionState, TradeSide, TRAIL_INITIAL_RATIO, TRAIL_FINAL_RATIO, TRAIL_TRIGGER_POINTS

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
        self._exit_order_ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def on_exit(self, callback: Callable[[str, ExitResult], None]) -> None:
        self._on_exit_callback = callback

    def open_position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def get_open_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    # ------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------

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

        transaction_type = "BUY" if side == TradeSide.LONG else "SELL"

        position = Position(
            contract=contract,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            entry_time=datetime.now(),
            initial_sl=initial_sl,
            state=PositionState.ENTRY_REQUESTED,
        )

        try:
            _log.info(
                "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=LIMIT price=%.2f trigger_price=None product=%s exchange=%s caller=enter_position",
                contract.tradingsymbol, transaction_type, quantity, round(entry_price, 2), CONFIG.capital.product_type, CONFIG.instrument.option_exchange,
            )
            order_id = self.kite.place_order(
                tradingsymbol=contract.tradingsymbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                product=CONFIG.capital.product_type,
                order_type="LIMIT",
                price=round(entry_price, 2),
            )
            position.entry_order_id = order_id
            position.state = PositionState.ENTRY_PENDING
            _log.info("ENTRY_ORDER_SUBMITTED | symbol=%s order_id=%s", contract.tradingsymbol, order_id)
        except Exception:
            _log.exception("Failed to place entry order for %s", contract.tradingsymbol)
            return

        with self._lock:
            self._positions[contract.tradingsymbol] = position
            self._entry_order_ids[contract.tradingsymbol] = position.entry_order_id

        entry_time_iso = position.entry_time.isoformat()
        record_trade(
            event="ENTRY_REQUESTED",
            symbol=contract.tradingsymbol,
            side=side.value,
            entry_price=entry_price,
            exit_price=0.0,
            quantity=quantity,
            pnl=0.0,
            reason="ENTRY_REQUESTED",
            strike=contract.strike,
            option_type=contract.option_type,
            expiry=contract.expiry,
            lot_size=contract.lot_size,
            entry_time=entry_time_iso,
            exit_time="",
        )

        log_decision(
            "ENTRY_REQUESTED",
            symbol=contract.tradingsymbol,
            side=side.value,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
        )

    # ------------------------------------------------------------
    # MONITOR LOOP
    # ------------------------------------------------------------

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

        if position.state in (PositionState.ENTRY_REQUESTED, PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED):
            self._check_entry_fill(symbol, position)
            return

        if position.state == PositionState.EXIT_PENDING:
            self._poll_exit_order_status(symbol, position)
            return

        if position.state == PositionState.CLOSED:
            return

        if not self._can_send_exit(position):
            _log.debug("Cannot send exit for %s: state=%s", symbol, position.state.value)
            return

        self._check_sl_escalation_watchdog(symbol, position)
        if position.state == PositionState.EXIT_PENDING:
            self._poll_exit_order_status(symbol, position)
            return

        self._reconcile_position(position)

        if position.state == PositionState.CLOSED:
            _log.info("Position %s reconciled as CLOSED by broker", symbol)
            self._finalize_closed_position(symbol, position)
            return

        if position.state == PositionState.EXIT_PENDING:
            self._poll_exit_order_status(symbol, position)
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

        new_trigger = round(position.stop_loss, 2)
        if position.sl_order_id and new_trigger != position.last_sl_trigger:
            self._modify_sl_order(symbol, position)

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

    def _check_entry_fill(self, symbol: str, position: Position) -> None:
        if position.state == PositionState.ENTRY_REQUESTED:
            _log.info("RECONCILIATION_PENDING | symbol=%s | reason=ENTRY_NOT_CONFIRMED", symbol)
            return

        if position.state == PositionState.ENTRY_FILLED:
            if not position.sl_order_id:
                self._place_sl_order(symbol, position)
                if position.sl_order_id:
                    with self._lock:
                        position.state = PositionState.OPEN
                        position.sl_placed_at = datetime.now().isoformat()
                else:
                    _log.warning("Entry filled but SL placement failed for %s — remaining in ENTRY_FILLED", symbol)
            else:
                with self._lock:
                    position.state = PositionState.OPEN
                    position.sl_placed_at = datetime.now().isoformat()
            return

        entry_order_id = position.entry_order_id
        if not entry_order_id:
            _log.warning("Entry order ID missing for %s — marking CLOSED", symbol)
            with self._lock:
                position.state = PositionState.CLOSED
            return

        try:
            status = self._get_order_status(entry_order_id)
            if status is None:
                status = "UNKNOWN"
            _log.debug("ENTRY_STATUS | order_id=%s | status=%s", entry_order_id, status)

            if status == "COMPLETE" or status == "FILLED":
                history = self._get_order_history(entry_order_id)
                filled_qty, avg_price = self._parse_order_history(history)

                with self._lock:
                    position.entry_filled_quantity = filled_qty
                    position.entry_avg_price = avg_price if avg_price > 0 else position.entry_price
                    position.entry_filled_at = datetime.now().isoformat()
                    position.state = PositionState.ENTRY_FILLED
                    position.was_ever_filled = True

                _log.info("ENTRY_FILLED | symbol=%s | qty=%d | average_price=%.2f", symbol, filled_qty, position.entry_avg_price)

                broker_qty = self._get_broker_position_quantity(symbol)
                _log.info("POSITION_CONFIRMED | symbol=%s | broker_qty=%d", symbol, broker_qty)

                self._place_sl_order(symbol, position)

                with self._lock:
                    position.state = PositionState.OPEN
                    position.sl_placed_at = datetime.now().isoformat()

                _log.info("POSITION_OPEN | symbol=%s | qty=%d", symbol, position.quantity)

                entry_time_iso = position.entry_time.isoformat()
                record_trade(
                    event="ENTRY",
                    symbol=symbol,
                    side=position.side.value,
                    entry_price=position.entry_avg_price,
                    exit_price=0.0,
                    quantity=position.quantity,
                    pnl=0.0,
                    reason="ENTRY_FILLED",
                    strike=position.contract.strike,
                    option_type=position.contract.option_type,
                    expiry=position.contract.expiry,
                    lot_size=position.contract.lot_size,
                    entry_time=entry_time_iso,
                    exit_time="",
                )

                log_decision(
                    "ENTRY_FILLED",
                    symbol=symbol,
                    side=position.side.value,
                    quantity=position.quantity,
                    entry_price=position.entry_avg_price,
                    stop_loss=position.stop_loss,
                    target=position.target,
                )
                return

            if status == "CANCELLED" or status == "REJECTED":
                _log.warning("Entry order %s for %s is %s", entry_order_id, symbol, status)
                with self._lock:
                    position.state = PositionState.ENTRY_FAILED
                    position.was_ever_filled = False
                self._finalize_failed_entry(symbol, position, status)
                return

            _log.debug("Entry order %s for %s still pending: %s", entry_order_id, symbol, status)
        except Exception:
            _log.exception("Error checking entry order %s for %s", entry_order_id, symbol)

    def _place_sl_order(self, symbol: str, position: Position) -> None:
        sl_offset = CONFIG.trade_mgmt.sl_limit_offset_points
        if position.side == TradeSide.LONG:
            sl_trigger = position.stop_loss
            sl_limit = sl_trigger + sl_offset
        else:
            sl_trigger = position.stop_loss
            sl_limit = sl_trigger - sl_offset

        try:
            _log.info(
                "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=SL price=%.2f trigger_price=%.2f product=%s exchange=%s caller=_place_sl_order",
                symbol, "SELL" if position.side == TradeSide.LONG else "BUY", position.quantity, sl_limit, sl_trigger, CONFIG.capital.product_type, CONFIG.instrument.option_exchange,
            )
            sl_order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type="SELL" if position.side == TradeSide.LONG else "BUY",
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="SL",
                price=round(sl_limit, 2),
                trigger_price=round(sl_trigger, 2),
            )
            with self._lock:
                position.sl_order_id = sl_order_id
                position.last_sl_trigger = round(sl_trigger, 2)
                self._sl_order_ids[symbol] = sl_order_id
            _log.info("SL_ORDER_SUBMITTED | symbol=%s | order_id=%s | trigger=%.2f | limit=%.2f", symbol, sl_order_id, sl_trigger, sl_limit)
        except Exception:
            _log.exception("Failed to place SL order for %s", symbol)

    def _modify_sl_order(self, symbol: str, position: Position) -> None:
        if not position.sl_order_id:
            return

        sl_status = self._get_order_status(position.sl_order_id)
        if sl_status in ("COMPLETE", "FILLED", "CANCELLED", "REJECTED"):
            _log.debug("SL order %s for %s has status %s — skipping modification", position.sl_order_id, symbol, sl_status)
            return

        sl_offset = CONFIG.trade_mgmt.sl_limit_offset_points
        new_trigger = round(position.stop_loss, 2)
        new_limit = round(new_trigger + sl_offset, 2) if position.side == TradeSide.LONG else round(new_trigger - sl_offset, 2)

        try:
            _log.info(
                "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=SL price=%.2f trigger_price=%.2f product=%s exchange=%s caller=_modify_sl_order",
                symbol, "SELL" if position.side == TradeSide.LONG else "BUY", position.quantity, new_limit, new_trigger, CONFIG.capital.product_type, CONFIG.instrument.option_exchange,
            )
            self.kite.modify_order(
                variety=CONFIG.kite.variety_regular,
                order_id=position.sl_order_id,
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type="SELL" if position.side == TradeSide.LONG else "BUY",
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="SL",
                price=new_limit,
                trigger_price=new_trigger,
            )
            with self._lock:
                position.last_sl_trigger = new_trigger
            _log.info("SL_MODIFIED | symbol=%s | order_id=%s | new_trigger=%.2f | new_limit=%.2f", symbol, position.sl_order_id, new_trigger, new_limit)
        except Exception:
            _log.exception("Failed to modify SL order for %s", symbol)

    # ------------------------------------------------------------
    # CLOSE POSITION — SAFE (NON-BLOCKING)
    # ------------------------------------------------------------

    def _close_position(self, symbol: str, position: Position, exit_price: float, reason: str) -> None:
        with self._lock:
            if position.state == PositionState.CLOSED:
                _log.info("Duplicate exit prevented for %s: already CLOSED", symbol)
                return
            if position.state == PositionState.EXIT_PENDING:
                _log.info("Duplicate exit prevented for %s: already EXIT_PENDING", symbol)
                return
            position.state = PositionState.EXIT_PENDING

        try:
            sl_order_id = position.sl_order_id
            if sl_order_id:
                _log.info("SL cancellation requested for %s order_id=%s", symbol, sl_order_id)
                try:
                    self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
                except Exception:
                    _log.exception("Failed to cancel SL order for %s", symbol)

                sl_status = self._get_order_status(sl_order_id)
                if sl_status is None:
                    sl_status = "UNKNOWN"
                _log.info("SL order status after cancel request for %s: %s", symbol, sl_status)

                if sl_status == "COMPLETE" or sl_status == "FILLED":
                    _log.info("SL already filled for %s — reconciling position, not submitting target exit", symbol)
                    with self._lock:
                        position.state = PositionState.CLOSED
                        position.was_ever_filled = True
                    self._finalize_closed_position(symbol, position)
                    return

                if sl_status == "CANCELLED":
                    _log.info("SL cancelled for %s — proceeding with %s exit", symbol, reason)
                else:
                    _log.info("SL status=%s for %s — reconciling before exit", symbol, sl_status)
                    self._reconcile_position(position)
                    if position.state == PositionState.CLOSED:
                        self._finalize_closed_position(symbol, position)
                        return
                    broker_qty = self._get_broker_position_quantity(symbol)
                    if broker_qty == 0:
                        _log.info("Broker position quantity is 0 for %s — marking CLOSED", symbol)
                        self._mark_position_closed(symbol, position, 0.0, "BROKER_QTY_ZERO")
                        return
                    if position.state == PositionState.EXIT_PENDING and position.exit_order_id:
                        _log.info("Exit already pending for %s order_id=%s", symbol, position.exit_order_id)
                        return

            transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
            limit_buffer = 0.01
            limit_price = round(
                exit_price * (1 - limit_buffer) if position.side == TradeSide.LONG else exit_price * (1 + limit_buffer),
                2,
            )
            _log.info(
                "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=LIMIT price=%.2f trigger_price=None product=%s exchange=%s caller=_close_position reason=%s",
                symbol, transaction_type, position.quantity, limit_price, CONFIG.capital.product_type, CONFIG.instrument.option_exchange, reason,
            )
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="LIMIT",
                price=limit_price,
            )
            _log.info("Exit order placed: %s reason=%s limit=%.2f order_id=%s", symbol, reason, limit_price, order_id)

            with self._lock:
                position.exit_order_id = order_id
                position.exit_reason = reason
                position.exit_requested_quantity = position.quantity
                position.exit_requested_at = datetime.now().isoformat()
                position.state = PositionState.EXIT_PENDING
                self._exit_order_ids[symbol] = order_id

            _log.info("Exit order initiated for %s order_id=%s — monitoring in next cycle", symbol, order_id)
        except Exception:
            _log.exception("Failed to place exit order for %s", symbol)
            with self._lock:
                if position.state == PositionState.EXIT_PENDING:
                    position.state = PositionState.OPEN

    def _poll_exit_order_status(self, symbol: str, position: Position) -> None:
        order_id = position.exit_order_id
        if not order_id:
            with self._lock:
                position.state = PositionState.OPEN
            return

        timeout = CONFIG.order.fill_confirmation_timeout_sec
        if position.exit_requested_at:
            try:
                requested = datetime.fromisoformat(position.exit_requested_at)
                elapsed = (datetime.now() - requested).total_seconds()
                if elapsed > timeout:
                    _log.warning("Exit order %s for %s timed out after %.1fs", order_id, symbol, elapsed)
                    self._reconcile_position(position)
                    with self._lock:
                        if position.state == PositionState.CLOSED:
                            return
                        position.state = PositionState.OPEN
                        position.exit_order_id = None
                        position.exit_reason = ""
                        position.exit_requested_quantity = 0
                        position.exit_filled_quantity = 0
                        position.exit_remaining_quantity = 0
                        position.exit_avg_price = 0.0
                        position.exit_requested_at = None
                        position.exit_filled_at = None
                    return
            except Exception:
                pass

        try:
            status = self._get_order_status(order_id)
            if status is None:
                status = "UNKNOWN"
            _log.debug("Exit order %s status for %s: %s", order_id, symbol, status)

            if status == "COMPLETE" or status == "FILLED":
                history = self._get_order_history(order_id)
                filled_qty, avg_price = self._parse_order_history(history)
                self._reconcile_position(position)
                with self._lock:
                    position.exit_filled_quantity = filled_qty
                    position.exit_avg_price = avg_price
                    position.exit_filled_at = datetime.now().isoformat()
                self._mark_position_closed(symbol, position, avg_price, position.exit_reason)
                return

            if status == "CANCELLED" or status == "REJECTED":
                _log.warning("Exit order %s for %s is %s", order_id, symbol, status)
                self._reconcile_position(position)
                with self._lock:
                    position.state = PositionState.OPEN
                    position.exit_order_id = None
                    position.exit_reason = ""
                    position.exit_requested_quantity = 0
                    position.exit_filled_quantity = 0
                    position.exit_remaining_quantity = 0
                    position.exit_avg_price = 0.0
                    position.exit_requested_at = None
                    position.exit_filled_at = None
                return

            if status == "OPEN" or status == "TRIGGER PENDING":
                filled_qty = self._get_filled_quantity_from_history(order_id)
                with self._lock:
                    position.exit_filled_quantity = filled_qty
                    position.exit_remaining_quantity = max(0, position.exit_requested_quantity - filled_qty)
                _log.debug("Exit order %s for %s still pending: filled=%d remaining=%d", order_id, symbol, filled_qty, position.exit_remaining_quantity)
                return

            _log.warning("Exit order %s for %s has unknown status=%s", order_id, symbol, status)
            self._reconcile_position(position)
            with self._lock:
                if position.state == PositionState.CLOSED:
                    return
                position.exit_remaining_quantity = max(0, position.exit_requested_quantity - position.exit_filled_quantity)
            return
        except Exception:
            _log.exception("Error polling exit order %s for %s", order_id, symbol)

    # ------------------------------------------------------------
    # FINALIZE CLOSED POSITION
    # ------------------------------------------------------------

    def _finalize_closed_position(self, symbol: str, position: Position) -> None:
        exit_price = position.exit_avg_price if position.exit_avg_price > 0 else position.entry_price
        pnl = self._compute_pnl(position, exit_price)
        result = ExitResult(pnl=pnl, reason=position.exit_reason or "CLOSED", exit_price=exit_price)

        with self._lock:
            self._positions.pop(symbol, None)
            self._sl_order_ids.pop(symbol, None)
            self._entry_order_ids.pop(symbol, None)
            self._exit_order_ids.pop(symbol, None)

        log_decision(
            "EXIT",
            symbol=symbol,
            reason=position.exit_reason or "CLOSED",
            exit_price=round(exit_price, 2),
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
            reason=position.exit_reason or "CLOSED",
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

    def _finalize_failed_entry(self, symbol: str, position: Position, reason: str) -> None:
        with self._lock:
            self._positions.pop(symbol, None)
            self._sl_order_ids.pop(symbol, None)
            self._entry_order_ids.pop(symbol, None)
            self._exit_order_ids.pop(symbol, None)

        log_decision(
            "ENTRY_FAILED",
            symbol=symbol,
            reason=reason,
            exit_price=0.0,
            pnl=0.0,
        )

        record_trade(
            event="ENTRY_FAILED",
            symbol=symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=0.0,
            quantity=position.quantity,
            pnl=0.0,
            reason=reason,
            strike=position.contract.strike,
            option_type=position.contract.option_type,
            expiry=position.contract.expiry,
            lot_size=position.contract.lot_size,
            entry_time=position.entry_time.isoformat(),
            exit_time="",
        )

        if self._on_exit_callback:
            try:
                self._on_exit_callback(symbol, ExitResult(pnl=0.0, reason=reason, exit_price=0.0))
            except Exception:
                _log.exception("on_exit callback failed for %s", symbol)

    def _mark_position_closed(self, symbol: str, position: Position, exit_price: float, reason: str) -> None:
        with self._lock:
            position.state = PositionState.CLOSED
            position.exit_reason = reason
            if exit_price > 0:
                position.exit_avg_price = exit_price
            position.exit_filled_at = datetime.now().isoformat()
        _log.info("Position %s marked CLOSED reason=%s exit_price=%.2f", symbol, reason, exit_price)
        self._finalize_closed_position(symbol, position)

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------

    def _compute_pnl(self, position: Position, exit_price: float) -> float:
        if position.side == TradeSide.LONG:
            return (exit_price - position.entry_price) * position.quantity
        return (position.entry_price - exit_price) * position.quantity

    def _get_order_status(self, order_id: str) -> Optional[str]:
        try:
            history = self.kite.order_history(order_id)
            if not history:
                return "UNKNOWN"
            latest = history[-1]
            return latest.get("status", "UNKNOWN")
        except Exception:
            _log.exception("Failed to fetch order history for %s", order_id)
            return None

    def _get_order_history(self, order_id: str) -> list[dict]:
        try:
            return self.kite.order_history(order_id) or []
        except Exception:
            _log.exception("Failed to fetch order history for %s", order_id)
            return []

    def _get_filled_quantity_from_history(self, order_id: str) -> int:
        try:
            history = self.kite.order_history(order_id) or []
            filled = 0
            for h in history:
                filled = max(filled, int(h.get("filled_quantity", 0) or 0))
            return filled
        except Exception:
            _log.exception("Failed to fetch filled quantity for %s", order_id)
            return 0

    def _parse_order_history(self, history: list[dict]) -> tuple[int, float]:
        filled_qty = 0
        avg_price = 0.0
        for h in history:
            filled_qty = max(filled_qty, int(h.get("filled_quantity", 0) or 0))
            avg_price = float(h.get("average_price", 0.0) or 0.0)
        return filled_qty, avg_price

    def _get_broker_position_quantity(self, symbol: str) -> int:
        try:
            positions = self.kite.positions()
            net = positions.get("net", [])
            for p in net:
                if p.get("tradingsymbol") == symbol:
                    return int(p.get("quantity", 0) or 0)
            return 0
        except Exception:
            _log.exception("Failed to fetch broker positions for %s", symbol)
            return -1

    def _get_broker_open_orders(self, symbol: str) -> list[dict]:
        try:
            orders = self.kite.orders() or []
            return [o for o in orders if o.get("tradingsymbol") == symbol and o.get("status") not in ("COMPLETE", "CANCELLED", "REJECTED")]
        except Exception:
            _log.exception("Failed to fetch broker orders for %s", symbol)
            return []

    def _reconcile_position(self, position: Position) -> None:
        symbol = position.contract.tradingsymbol
        broker_qty = self._get_broker_position_quantity(symbol)
        open_orders = self._get_broker_open_orders(symbol)

        sl_filled = False
        exit_filled = False
        exit_pending = False

        for o in open_orders:
            oid = o.get("order_id")
            if oid == position.sl_order_id:
                if o.get("status") in ("COMPLETE", "FILLED"):
                    sl_filled = True
            if oid == position.exit_order_id:
                if o.get("status") in ("OPEN", "TRIGGER PENDING"):
                    exit_pending = True
                elif o.get("status") in ("COMPLETE", "FILLED"):
                    exit_filled = True

        if position.state in (PositionState.ENTRY_REQUESTED, PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED):
            if broker_qty > 0:
                with self._lock:
                    position.state = PositionState.OPEN
                    position.was_ever_filled = True
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s broker_qty=%d -> OPEN", symbol, broker_qty)
            elif sl_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                    position.was_ever_filled = True
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s SL filled during entry -> CLOSED", symbol)
            elif exit_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                    position.was_ever_filled = True
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s exit filled during entry -> CLOSED", symbol)
            else:
                _log.debug("Reconciliation: %s state=%s broker_qty=%d -> keeping state", symbol, position.state.value, broker_qty)
            return

        if position.state == PositionState.EXIT_PENDING:
            if exit_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s exit order filled -> CLOSED", symbol)
            elif sl_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s SL filled -> CLOSED", symbol)
            elif exit_pending:
                _log.debug("Reconciliation: %s exit pending -> EXIT_PENDING", symbol)
            else:
                with self._lock:
                    position.state = PositionState.OPEN
                position.last_reconciled_at = datetime.now().isoformat()
                _log.debug("Reconciliation: %s no pending exit orders -> OPEN", symbol)
            return

        if position.state == PositionState.CLOSED:
            return

        if position.state == PositionState.OPEN:
            if sl_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s SL filled -> CLOSED", symbol)
                return

            if exit_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s exit order filled -> CLOSED", symbol)
                return

            if exit_pending:
                with self._lock:
                    position.state = PositionState.EXIT_PENDING
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s exit pending -> EXIT_PENDING", symbol)
                return

            if broker_qty == 0 and position.was_ever_filled:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s broker_qty=0, was_ever_filled=True -> CLOSED", symbol)
                return

            if broker_qty > 0:
                with self._lock:
                    position.state = PositionState.OPEN
                position.last_reconciled_at = datetime.now().isoformat()
                _log.debug("Reconciliation: %s broker_qty=%d -> OPEN", symbol, broker_qty)
                return

            _log.debug("Reconciliation: %s ambiguous state broker_qty=%d was_ever_filled=%s -> keeping OPEN", symbol, broker_qty, position.was_ever_filled)
            return

    def _can_send_exit(self, position: Position) -> bool:
        if position.state == PositionState.CLOSED:
            return False
        if position.state == PositionState.EXIT_PENDING:
            if position.exit_order_id:
                return False
        if position.state in (PositionState.ENTRY_REQUESTED, PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED):
            return False
        return True

    def _check_sl_escalation_watchdog(self, symbol: str, position: Position) -> None:
        if position.state != PositionState.OPEN:
            return

        sl_order_id = position.sl_order_id
        if not sl_order_id or not position.sl_placed_at:
            return

        try:
            placed = datetime.fromisoformat(position.sl_placed_at)
        except Exception:
            return

        elapsed = (datetime.now() - placed).total_seconds()
        watchdog_sec = CONFIG.trade_mgmt.sl_escalation_watchdog_sec
        if elapsed < watchdog_sec:
            return

        if not self._can_send_exit(position):
            return

        sl_status = self._get_order_status(sl_order_id)
        if sl_status is None:
            sl_status = "UNKNOWN"

        _log.info("SL escalation watchdog triggered for %s elapsed=%.1fs status=%s", symbol, elapsed, sl_status)

        if sl_status == "COMPLETE" or sl_status == "FILLED":
            _log.info("SL watchdog: SL already filled for %s — reconciling", symbol)
            self._reconcile_position(position)
            if position.state == PositionState.CLOSED:
                self._finalize_closed_position(symbol, position)
            return

        if sl_status == "CANCELLED":
            _log.info("SL watchdog: SL already cancelled for %s — no action needed", symbol)
            return

        _log.info("SL watchdog: cancelling stale SL for %s order_id=%s", symbol, sl_order_id)
        try:
            self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
        except Exception:
            _log.exception("SL watchdog: failed to cancel SL for %s", symbol)

        sl_status_after = self._get_order_status(sl_order_id)
        if sl_status_after is None:
            sl_status_after = "UNKNOWN"
        _log.info("SL watchdog: SL status after cancel for %s: %s", symbol, sl_status_after)

        if sl_status_after == "COMPLETE" or sl_status_after == "FILLED":
            _log.info("SL watchdog: SL filled during cancel for %s — reconciling", symbol)
            self._reconcile_position(position)
            if position.state == PositionState.CLOSED:
                self._finalize_closed_position(symbol, position)
            return

        if sl_status_after == "CANCELLED":
            _log.info("SL watchdog: SL cancelled for %s — firing market exit", symbol)
            self._fire_market_exit(symbol, position, "SL_ESCALATION")
            return

        _log.warning("SL watchdog: unknown SL status=%s for %s — reconciling", sl_status_after, symbol)
        self._reconcile_position(position)
        if position.state == PositionState.CLOSED:
            self._finalize_closed_position(symbol, position)
            return
        broker_qty = self._get_broker_position_quantity(symbol)
        if broker_qty == 0:
            self._mark_position_closed(symbol, position, 0.0, "BROKER_QTY_ZERO")
            return
        if self._can_send_exit(position):
            self._fire_market_exit(symbol, position, "SL_ESCALATION")

    def _fire_market_exit(self, symbol: str, position: Position, reason: str) -> None:
        transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
        _log.info(
            "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=MARKET price=None trigger_price=None product=%s exchange=%s caller=_fire_market_exit reason=%s",
            symbol, transaction_type, position.quantity, CONFIG.capital.product_type, CONFIG.instrument.option_exchange, reason,
        )
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="MARKET",
            )
        except Exception:
            _log.exception("Failed to place market exit for %s", symbol)
            return

        _log.info("Market exit order placed: %s reason=%s order_id=%s", symbol, reason, order_id)

        with self._lock:
            position.exit_order_id = order_id
            position.exit_reason = reason
            position.exit_requested_quantity = position.quantity
            position.exit_requested_at = datetime.now().isoformat()
            position.state = PositionState.EXIT_PENDING
            self._exit_order_ids[symbol] = order_id

        _log.info("Market exit initiated for %s order_id=%s — monitoring in next cycle", symbol, order_id)

    # ------------------------------------------------------------
    # FORCE SQUARE-OFF — SAFE
    # ------------------------------------------------------------

    def force_square_off_all(self) -> None:
        with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            position = None
            with self._lock:
                position = self._positions.get(symbol)
            if position is None:
                continue

            if not self._can_send_exit(position):
                _log.info("Skipping force square-off for %s: cannot send exit", symbol)
                continue

            try:
                ltp = self._get_ltp(symbol)
                if ltp is None or ltp <= 0:
                    _log.warning("Cannot force square-off %s — LTP unavailable", symbol)
                    continue

                self._reconcile_position(position)
                if position.state == PositionState.CLOSED:
                    _log.info("Force square-off skipped for %s: already CLOSED", symbol)
                    self._finalize_closed_position(symbol, position)
                    continue

                if position.state == PositionState.EXIT_PENDING and position.exit_order_id:
                    _log.info("Force square-off skipped for %s: exit pending order_id=%s", symbol, position.exit_order_id)
                    continue

                sl_order_id = position.sl_order_id
                if sl_order_id:
                    try:
                        self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
                        _log.info("SL cancelled for force square-off %s", symbol)
                    except Exception:
                        _log.exception("Failed to cancel SL for force square-off %s", symbol)

                transaction_type = (
                    "SELL" if position.side == TradeSide.LONG else "BUY"
                )
                _log.info(
                    "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=MARKET price=%.2f trigger_price=None product=%s exchange=%s caller=force_square_off_all",
                    symbol, transaction_type, position.quantity, ltp, CONFIG.capital.product_type, CONFIG.instrument.option_exchange,
                )
                order_id = self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=CONFIG.instrument.option_exchange,
                    transaction_type=transaction_type,
                    quantity=position.quantity,
                    product=CONFIG.capital.product_type,
                    order_type="MARKET",
                    price=ltp,
                )
                _log.info("Force square-off: %s", symbol)

                with self._lock:
                    position.state = PositionState.EXIT_PENDING
                    position.exit_order_id = order_id
                    position.exit_reason = "FORCE_SQUARE_OFF"
                    position.exit_requested_quantity = position.quantity
                    position.exit_requested_at = datetime.now().isoformat()
                    self._exit_order_ids[symbol] = order_id
            except Exception:
                _log.exception("Failed to force square-off %s", symbol)
                continue

            pnl = self._compute_pnl(position, position.entry_price)
            result = ExitResult(pnl=pnl, reason="FORCE_SQUARE_OFF", exit_price=position.entry_price)

            with self._lock:
                self._positions.pop(symbol, None)
                self._sl_order_ids.pop(symbol, None)
                self._entry_order_ids.pop(symbol, None)
                self._exit_order_ids.pop(symbol, None)

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

    # ------------------------------------------------------------
    # CANCEL OPEN ORDERS
    # ------------------------------------------------------------

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
