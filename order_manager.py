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

from config import CONFIG, TradingMode
from kite_api import KiteAPI
from logger import get_logger, log_decision, log_structured
from trade_journal import record_trade
from utils import OptionContract, Position, PositionState, TradeSide, TRAIL_TRIGGER_POINTS

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
            return sum(1 for p in self._positions.values() if p.state != PositionState.CLOSED)

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
            position.entry_requested_at = datetime.now().isoformat()
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

        if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING):
            self._poll_exit_order_status(symbol, position)
            return

        if position.state == PositionState.CLOSED:
            return

        if position.state == PositionState.EXIT_FAILED:
            _log.warning("Position %s is in EXIT_FAILED state - manual intervention may be required", symbol)
            return

        if not self._can_send_exit(position):
            _log.debug("Cannot send exit for %s: state=%s", symbol, position.state.value)
            return

        self._check_sl_escalation_watchdog(symbol, position)
        if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING):
            self._poll_exit_order_status(symbol, position)
            return

        self._reconcile_position(position)

        if position.state == PositionState.CLOSED:
            _log.info("Position %s reconciled as CLOSED by broker", symbol)
            self._finalize_closed_position(symbol, position)
            return

        if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING):
            self._poll_exit_order_status(symbol, position)
            return

        ltp = self._get_ltp(symbol)
        if ltp is None:
            return

        if CONFIG.mode == TradingMode.PAPER:
            self.kite.process_paper_orders(symbol, ltp)

        should_exit = False
        reason = ""
        exit_price = ltp

        # Trailing SL: when profit >= 2 points, start trailing
        profit = (
            ltp - position.entry_price
            if position.side == TradeSide.LONG
            else position.entry_price - ltp
        )

        if profit >= TRAIL_TRIGGER_POINTS and position.initial_sl > 0 and CONFIG.trade_mgmt.use_sl_m_after_trail:
            new_sl_trigger = None
            if position.side == TradeSide.LONG:
                if profit >= CONFIG.trade_mgmt.trail_step_points + TRAIL_TRIGGER_POINTS:
                    new_sl_trigger = max(position.entry_price + CONFIG.trade_mgmt.trail_lock_points, ltp - CONFIG.trade_mgmt.trail_step_points)
                    position.stop_loss = new_sl_trigger
                elif profit >= TRAIL_TRIGGER_POINTS:
                    new_sl_trigger = position.entry_price
                    position.stop_loss = new_sl_trigger
            else:
                if profit >= CONFIG.trade_mgmt.trail_step_points + TRAIL_TRIGGER_POINTS:
                    new_sl_trigger = min(position.entry_price - CONFIG.trade_mgmt.trail_lock_points, ltp + CONFIG.trade_mgmt.trail_step_points)
                    position.stop_loss = new_sl_trigger
                elif profit >= TRAIL_TRIGGER_POINTS:
                    new_sl_trigger = position.entry_price
                    position.stop_loss = new_sl_trigger

            if new_sl_trigger is not None and position.sl_order_id and new_sl_trigger != position.last_sl_trigger:
                self._cancel_sl_order(symbol, position)
                self._place_sl_m_order(symbol, position, new_sl_trigger)
        elif profit >= TRAIL_TRIGGER_POINTS and position.initial_sl > 0:
            if position.side == TradeSide.LONG:
                if profit >= TRAIL_TRIGGER_POINTS + 1.0:
                    position.stop_loss = max(position.stop_loss, position.entry_price + 2.0)
                else:
                    position.stop_loss = max(position.stop_loss, position.entry_price)
            else:
                if profit >= TRAIL_TRIGGER_POINTS + 1.0:
                    position.stop_loss = min(position.stop_loss, position.entry_price - 2.0)
                else:
                    position.stop_loss = min(position.stop_loss, position.entry_price)

            new_trigger = round(position.stop_loss, 2)
            if position.sl_order_id and new_trigger != position.last_sl_trigger:
                self._modify_sl_order(symbol, position)

        target_distance = abs(position.target - position.entry_price)
        if target_distance > 0 and profit >= CONFIG.trade_mgmt.profit_trail_pct * target_distance:
            if CONFIG.trade_mgmt.use_sl_m_after_trail:
                self._cancel_sl_order(symbol, position)
                self._place_sl_m_order(symbol, position, position.target)
            else:
                self._close_position(symbol, position, ltp, "PROFIT_TRAIL")
            return

        if position.side == TradeSide.LONG:
            if ltp >= position.target:
                should_exit = True
                reason = "TARGET"
                exit_price = position.target
            elif ltp <= position.stop_loss:
                self._cancel_sl_and_market_exit(symbol, position, "STOP_LOSS_TOUCH")
                return
        else:
            if ltp <= position.target:
                should_exit = True
                reason = "TARGET"
                exit_price = position.target
            elif ltp >= position.stop_loss:
                self._cancel_sl_and_market_exit(symbol, position, "STOP_LOSS_TOUCH")
                return

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

        if position.state == PositionState.ENTRY_CANCELLED_TIMEOUT:
            _log.debug("Entry for %s already cancelled due to timeout", symbol)
            return

        entry_order_id = position.entry_order_id
        if not entry_order_id:
            _log.warning("Entry order ID missing for %s — marking CLOSED", symbol)
            with self._lock:
                position.state = PositionState.CLOSED
            return

        if position.entry_requested_at:
            try:
                requested = datetime.fromisoformat(position.entry_requested_at)
                elapsed = (datetime.now() - requested).total_seconds()
                if elapsed > CONFIG.order.entry_fill_timeout_sec:
                    _log.warning("Entry order %s for %s timed out after %.1fs", entry_order_id, symbol, elapsed)
                    try:
                        self.kite.cancel_order(CONFIG.kite.variety_regular, entry_order_id)
                        _log.info("Entry order cancelled for timeout: %s", entry_order_id)
                    except Exception:
                        _log.exception("Failed to cancel timed-out entry order for %s", symbol)

                    self._reconcile_position(position)
                    broker_qty = self._get_broker_position_quantity(symbol)
                    with self._lock:
                        if broker_qty == 0 and not position.was_ever_filled:
                            position.state = PositionState.ENTRY_CANCELLED_TIMEOUT
                            position.entry_order_id = None
                            position.entry_filled_quantity = 0
                            position.entry_avg_price = 0.0
                            _log.info("ENTRY_CANCELLED_TIMEOUT | symbol=%s | no position opened", symbol)
                        elif broker_qty > 0 or position.was_ever_filled:
                            position.state = PositionState.OPEN
                            position.was_ever_filled = True
                            _log.info("Entry timeout but position exists for %s — keeping OPEN", symbol)
                        else:
                            position.state = PositionState.ENTRY_FAILED
                            position.entry_order_id = None
                    return
            except Exception:
                pass

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

            if status == "PARTIALLY_FILLED":
                history = self._get_order_history(entry_order_id)
                filled_qty, avg_price = self._parse_order_history(history)
                with self._lock:
                    position.entry_filled_quantity = filled_qty
                    position.entry_avg_price = avg_price if avg_price > 0 else position.entry_price
                    position.entry_filled_at = datetime.now().isoformat()
                    position.was_ever_filled = True
                _log.info("ENTRY_PARTIAL | symbol=%s | filled=%d | remaining=%d", symbol, filled_qty, position.quantity - filled_qty)
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

        order_type = CONFIG.trade_mgmt.sl_order_type

        if order_type == "SL-M":
            self._cancel_sl_order(symbol, position)
            self._place_sl_m_order(symbol, position, new_trigger)
            return

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

    def _place_sl_m_order(self, symbol: str, position: Position, trigger_price: float) -> None:
        transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
        try:
            _log.info(
                "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=SL-M trigger_price=%.2f product=%s exchange=%s caller=_place_sl_m_order",
                symbol, transaction_type, position.quantity, round(trigger_price, 2), CONFIG.capital.product_type, CONFIG.instrument.option_exchange,
            )
            sl_order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="SL-M",
                trigger_price=round(trigger_price, 2),
            )
            with self._lock:
                position.sl_order_id = sl_order_id
                position.last_sl_trigger = round(trigger_price, 2)
                self._sl_order_ids[symbol] = sl_order_id
            _log.info("SL_M_ORDER_SUBMITTED | symbol=%s | order_id=%s | trigger=%.2f", symbol, sl_order_id, round(trigger_price, 2))
        except Exception:
            _log.exception("Failed to place SL-M order for %s", symbol)

    def _cancel_sl_order(self, symbol: str, position: Position) -> None:
        sl_order_id = position.sl_order_id
        if not sl_order_id:
            return
        try:
            self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
            _log.info("SL_CANCELLED | symbol=%s | order_id=%s", symbol, sl_order_id)
        except Exception:
            _log.exception("Failed to cancel SL order for %s", symbol)
        with self._lock:
            position.sl_order_id = None
            position.last_sl_trigger = 0.0
            self._sl_order_ids.pop(symbol, None)

    # ------------------------------------------------------------
    # CLOSE POSITION — SAFE (NON-BLOCKING)
    # ------------------------------------------------------------

    def _close_position(self, symbol: str, position: Position, exit_price: float, reason: str) -> None:
        with self._lock:
            if position.state == PositionState.CLOSED:
                _log.info("Duplicate exit prevented for %s: already CLOSED", symbol)
                return
            if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING):
                _log.info("Duplicate exit prevented for %s: already %s", symbol, position.state.value)
                return
            position.state = PositionState.EXIT_LIMIT_PLACED

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
                    recovered, sl_price = self._try_recover_sl_fill_price(symbol, position)
                    with self._lock:
                        if recovered:
                            position.exit_avg_price = sl_price
                            position.exit_reason = "SL_FILLED"
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
                    if broker_qty == 0 and not self._is_paper_mode():
                        _log.info("Broker position quantity is 0 for %s — marking CLOSED", symbol)
                        self._mark_position_closed(symbol, position, 0.0, "BROKER_QTY_ZERO")
                        return
                    if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING) and position.exit_order_id:
                        _log.info("Exit already pending for %s order_id=%s", symbol, position.exit_order_id)
                        return

            transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
            buffer_pct = CONFIG.trade_mgmt.exit_limit_buffer_pct
            limit_price = round(
                exit_price * (1 - buffer_pct) if position.side == TradeSide.LONG else exit_price * (1 + buffer_pct),
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
                position.exit_limit_order_id = order_id
                position.exit_reason = reason
                position.exit_requested_quantity = position.quantity
                position.exit_requested_at = datetime.now().isoformat()
                position.exit_filled_quantity = 0
                position.exit_remaining_quantity = position.quantity
                position.state = PositionState.EXIT_LIMIT_PLACED
                self._exit_order_ids[symbol] = order_id

            log_structured(
                "EXIT_LIMIT_PLACED",
                symbol=symbol,
                side=position.side.value,
                position_quantity=position.quantity,
                exit_reason=reason,
                strategy_exit_price=round(exit_price, 2),
                limit_price=limit_price,
                buffer_pct=buffer_pct,
                limit_order_id=order_id,
                filled_quantity=0,
                remaining_quantity=position.quantity,
                timeout_seconds=CONFIG.order.exit_limit_timeout_sec,
                fallback_used=False,
                final_exit_status="PENDING",
            )
        except Exception:
            _log.exception("Failed to place exit order for %s", symbol)
            with self._lock:
                if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_PENDING):
                    position.state = PositionState.OPEN
                    position.exit_order_id = None
                    position.exit_limit_order_id = None
                    position.exit_reason = ""
                    position.exit_requested_quantity = 0
                    position.exit_filled_quantity = 0
                    position.exit_remaining_quantity = 0
                    position.exit_avg_price = 0.0
                    position.exit_requested_at = None
                    position.exit_filled_at = None

    def _poll_exit_order_status(self, symbol: str, position: Position) -> None:
        order_id = position.exit_order_id
        if not order_id:
            with self._lock:
                position.state = PositionState.OPEN
            return

        if position.exit_requested_at:
            try:
                requested = datetime.fromisoformat(position.exit_requested_at)
                elapsed = (datetime.now() - requested).total_seconds()
                exit_timeout = CONFIG.order.exit_limit_timeout_sec if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL) else CONFIG.order.fill_confirmation_timeout_sec
                if elapsed > exit_timeout:
                    _log.warning("Exit order %s for %s timed out after %.1fs", order_id, symbol, elapsed)
                    self._reconcile_position(position)
                    with self._lock:
                        if position.state == PositionState.CLOSED:
                            return
                        remaining_qty = position.exit_remaining_quantity
                        position.state = PositionState.OPEN
                        position.exit_order_id = None
                        position.exit_limit_order_id = None
                        position.exit_reason = ""
                        position.exit_requested_quantity = 0
                        position.exit_filled_quantity = 0
                        position.exit_remaining_quantity = 0
                        position.exit_avg_price = 0.0
                        position.exit_requested_at = None
                        position.exit_filled_at = None
                        position.exit_market_fallback_used = False
                        position.exit_fallback_quantity = 0
                    if remaining_qty > 0:
                        self._execute_market_fallback(symbol, position, position.exit_reason or "LIMIT_TIMEOUT", quantity=remaining_qty)
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
                    if position.state == PositionState.CLOSED:
                        return
                    position.state = PositionState.OPEN
                    position.exit_order_id = None
                    position.exit_limit_order_id = None
                    position.exit_reason = ""
                    position.exit_requested_quantity = 0
                    position.exit_filled_quantity = 0
                    position.exit_remaining_quantity = 0
                    position.exit_avg_price = 0.0
                    position.exit_requested_at = None
                    position.exit_filled_at = None
                    position.exit_market_fallback_used = False
                    position.exit_fallback_quantity = 0
                return

            if status == "OPEN" or status == "TRIGGER PENDING":
                filled_qty = self._get_filled_quantity_from_history(order_id)
                with self._lock:
                    position.exit_filled_quantity = filled_qty
                    position.exit_remaining_quantity = max(0, position.exit_requested_quantity - filled_qty)
                    if filled_qty > 0 and filled_qty < position.exit_requested_quantity and position.state == PositionState.EXIT_LIMIT_PLACED:
                        position.state = PositionState.EXIT_LIMIT_PARTIAL
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

    def _try_recover_sl_fill_price(self, symbol: str, position: Position, sl_order_id: Optional[str] = None) -> tuple[bool, float]:
        if not sl_order_id:
            sl_order_id = position.sl_order_id
        if not sl_order_id:
            return False, 0.0
        try:
            history = self.kite.order_history(sl_order_id) or []
            filled_qty, avg_price = self._parse_order_history(history)
            if filled_qty > 0 and avg_price > 0:
                _log.info("SL fill price recovered for %s: qty=%d avg_price=%.2f", symbol, filled_qty, avg_price)
                return True, avg_price
        except Exception:
            _log.exception("Failed to recover SL fill price for %s", symbol)
        return False, 0.0

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
            if oid == position.exit_order_id or oid == position.exit_limit_order_id:
                if o.get("status") in ("OPEN", "TRIGGER PENDING"):
                    exit_pending = True
                elif o.get("status") in ("COMPLETE", "FILLED"):
                    exit_filled = True

        if position.state in (PositionState.ENTRY_REQUESTED, PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED):
            if broker_qty > 0 or position.was_ever_filled:
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

        if position.state in (PositionState.ENTRY_CANCELLED_TIMEOUT, PositionState.ENTRY_FAILED):
            if broker_qty == 0:
                with self._lock:
                    position.state = PositionState.CLOSED
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s state=%s broker_qty=0 -> CLOSED", symbol, position.state.value)
            return

        if position.state in (PositionState.EXIT_PENDING, PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK):
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
                _log.debug("Reconciliation: %s exit pending -> %s", symbol, position.state.value)
            else:
                if broker_qty == 0 and position.was_ever_filled and not open_orders:
                    recovered, sl_price = self._try_recover_sl_fill_price(symbol, position)
                    with self._lock:
                        if recovered:
                            position.exit_avg_price = sl_price
                            position.exit_reason = "SL_FILLED"
                        position.state = PositionState.CLOSED
                    position.last_reconciled_at = datetime.now().isoformat()
                    _log.info("Reconciliation: %s broker_qty=0, was_ever_filled=True, no open orders -> CLOSED", symbol)
                else:
                    _log.debug("Reconciliation: %s broker_qty=%d was_ever_filled=%s -> keeping %s", symbol, broker_qty, position.was_ever_filled, position.state.value)
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
                    if position.state == PositionState.OPEN:
                        position.state = PositionState.EXIT_PENDING
                position.last_reconciled_at = datetime.now().isoformat()
                _log.info("Reconciliation: %s exit pending -> %s", symbol, position.state.value)
                return

            if broker_qty == 0 and position.was_ever_filled:
                if not open_orders:
                    recovered, sl_price = self._try_recover_sl_fill_price(symbol, position)
                    with self._lock:
                        if recovered:
                            position.exit_avg_price = sl_price
                            position.exit_reason = "SL_FILLED"
                        position.state = PositionState.CLOSED
                    position.last_reconciled_at = datetime.now().isoformat()
                    _log.info("Reconciliation: %s broker_qty=0, was_ever_filled=True, no open orders -> CLOSED", symbol)
                else:
                    _log.debug("Reconciliation: %s broker_qty=0 but open orders exist -> keeping OPEN", symbol)
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
        if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING):
            if position.exit_order_id:
                return False
        if position.state in (PositionState.ENTRY_REQUESTED, PositionState.ENTRY_PENDING, PositionState.ENTRY_FILLED, PositionState.ENTRY_CANCELLED_TIMEOUT, PositionState.ENTRY_FAILED):
            return False
        return True

    def _check_sl_escalation_watchdog(self, symbol: str, position: Position) -> None:
        if position.state != PositionState.OPEN:
            return

        sl_order_id = position.sl_order_id
        if not sl_order_id:
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_MISSING", symbol)
            self._repair_sl_order(symbol, position, "SL_MISSING")
            return

        try:
            sl_status = self._get_order_status(sl_order_id)
        except Exception:
            _log.exception("SL_WATCHDOG_CHECK | symbol=%s | order_id=%s | failed to fetch status", symbol, sl_order_id)
            return

        if sl_status is None:
            sl_status = "UNKNOWN"

        ltp = self._get_ltp(symbol) if self._get_ltp else None
        _log.info(
            "SL_WATCHDOG_CHECK | symbol=%s | position_qty=%d | sl_order_id=%s | sl_status=%s | sl_trigger=%.2f | current_price=%.2f",
            symbol, position.quantity, sl_order_id, sl_status, position.stop_loss, ltp or 0.0,
        )

        if sl_status in ("COMPLETE", "FILLED"):
            _log.info("SL_WATCHDOG_OK | symbol=%s | SL filled -> reconcile", symbol)
            self._reconcile_position(position)
            if position.state == PositionState.CLOSED:
                self._finalize_closed_position(symbol, position)
            return

        if sl_status in ("REJECTED",):
            _log.warning("SL_WATCHDOG_FAILURE | symbol=%s | order_id=%s | status=REJECTED", symbol, sl_order_id)
            self._attempt_sl_recovery(symbol, position, "SL_REJECTED")
            return

        if sl_status in ("UNKNOWN", "API_ERROR"):
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | order_id=%s | status=%s", symbol, sl_order_id, sl_status)
            self._attempt_sl_recovery(symbol, position, f"SL_UNKNOWN_{sl_status}")
            return

        if sl_status in ("CANCELLED",):
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | order_id=%s | status=CANCELLED (unexpected)", symbol, sl_order_id)
            self._repair_sl_order(symbol, position, "SL_CANCELLED")
            return

        # Active protective SL: TRIGGER PENDING / OPEN / ACCEPTED / etc.
        # A pending SL is NORMAL. Verify it is valid, then keep the position open.
        sl_details = self._get_order_details(sl_order_id)
        if sl_details:
            if not self._validate_sl_order(symbol, position, sl_details):
                return
            self._guard_trailing_sl(symbol, position, sl_details)

        self._reconcile_duplicate_sl(symbol, position, sl_order_id)

        _log.info("SL_WATCHDOG_OK | symbol=%s | order_id=%s | status=%s", symbol, sl_order_id, sl_status)

    def _get_order_details(self, order_id: str) -> Optional[dict]:
        try:
            history = self.kite.order_history(order_id) or []
            if not history:
                return None
            return dict(history[-1])
        except Exception:
            _log.exception("Failed to fetch order details for %s", order_id)
            return None

    def _validate_sl_order(self, symbol: str, position: Position, details: dict) -> bool:
        """Validate the protective SL order. Returns True if OK (or not enough
        data to judge). Returns False if a repair was attempted."""
        expected_txn = "SELL" if position.side == TradeSide.LONG else "BUY"

        if details.get("tradingsymbol") and details.get("tradingsymbol") != symbol:
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_WRONG_SYMBOL actual=%s", symbol, details.get("tradingsymbol"))
            self._repair_sl_order(symbol, position, "SL_WRONG_SYMBOL")
            return False

        if details.get("transaction_type") and details.get("transaction_type") != expected_txn:
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_WRONG_TXN actual=%s", symbol, details.get("transaction_type"))
            self._repair_sl_order(symbol, position, "SL_WRONG_TXN")
            return False

        qty = details.get("quantity")
        if qty is not None and int(qty) != int(position.quantity):
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_QTY_MISMATCH sl_qty=%s pos_qty=%d", symbol, qty, position.quantity)
            self._repair_sl_order(symbol, position, "SL_QTY_MISMATCH")
            return False

        trig = details.get("trigger_price")
        if trig is not None and float(trig) <= 0:
            _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_INVALID_TRIGGER", symbol)
            self._repair_sl_order(symbol, position, "SL_INVALID_TRIGGER")
            return False

        return True

    def _guard_trailing_sl(self, symbol: str, position: Position, details: dict) -> None:
        trig = details.get("trigger_price")
        if trig is None:
            return
        try:
            order_trigger = float(trig)
        except (TypeError, ValueError):
            return
        if position.side == TradeSide.LONG:
            if order_trigger < position.stop_loss - 1e-6:
                _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_TRAILING_BEHIND order_trigger=%.2f current_sl=%.2f", symbol, order_trigger, position.stop_loss)
                self._modify_sl_order(symbol, position)
        else:
            if order_trigger > position.stop_loss + 1e-6:
                _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=SL_TRAILING_BEHIND order_trigger=%.2f current_sl=%.2f", symbol, order_trigger, position.stop_loss)
                self._modify_sl_order(symbol, position)

    def _reconcile_duplicate_sl(self, symbol: str, position: Position, current_sl_id: str) -> None:
        try:
            orders = self.kite.orders() or []
        except Exception:
            return
        sl_txn = "SELL" if position.side == TradeSide.LONG else "BUY"
        active = ("OPEN", "TRIGGER PENDING", "AMO MODIFIED", "ACCEPTED", "OPEN PENDING")
        for o in orders:
            if o.get("tradingsymbol") != symbol:
                continue
            oid = o.get("order_id")
            if oid == current_sl_id or not oid:
                continue
            if o.get("order_type") == "SL" and o.get("transaction_type") == sl_txn and o.get("status") in active:
                try:
                    self.kite.cancel_order(CONFIG.kite.variety_regular, oid)
                    _log.warning("SL_WATCHDOG_DUPLICATE_REMOVED | symbol=%s | duplicate_sl_order_id=%s", symbol, oid)
                except Exception:
                    _log.exception("SL_WATCHDOG_DUPLICATE_REMOVED | failed to cancel duplicate %s", oid)

    def _repair_sl_order(self, symbol: str, position: Position, reason: str) -> None:
        if position.state == PositionState.CLOSED:
            return
        if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING) or position.exit_order_id:
            _log.info("SL_WATCHDOG_REPAIR_SKIPPED | symbol=%s | reason=EXIT_IN_PROGRESS", symbol)
            return

        _log.warning("SL_WATCHDOG_REPAIR_REQUIRED | symbol=%s | reason=%s", symbol, reason)
        old_sl_id = position.sl_order_id
        try:
            if old_sl_id:
                try:
                    self.kite.cancel_order(CONFIG.kite.variety_regular, old_sl_id)
                    _log.info("SL_WATCHDOG_REPAIRING | symbol=%s | cancelled old SL order_id=%s", symbol, old_sl_id)
                except Exception:
                    _log.exception("SL_WATCHDOG_REPAIRING | failed to cancel old SL %s", old_sl_id)

            _log.info("SL_WATCHDOG_REPAIRING | symbol=%s | placing protective SL", symbol)
            self._place_sl_order(symbol, position)

            if position.sl_order_id and position.sl_order_id != old_sl_id:
                _log.info("SL_WATCHDOG_REPAIRED | symbol=%s | new_sl_order_id=%s", symbol, position.sl_order_id)
            else:
                _log.warning("SL_WATCHDOG_REPAIR_FAILED | symbol=%s | falling back to emergency exit", symbol)
                self._attempt_sl_recovery(symbol, position, f"SL_REPAIR_FAILED_{reason}")
        except Exception:
            _log.exception("SL_WATCHDOG_REPAIR_FAILED | symbol=%s", symbol)
            self._attempt_sl_recovery(symbol, position, f"SL_REPAIR_FAILED_{reason}")

    def _attempt_sl_recovery(self, symbol: str, position: Position, reason: str) -> None:
        if not self._can_send_exit(position):
            _log.warning("SL recovery skipped for %s: cannot send exit", symbol)
            return

        try:
            broker_qty = self._get_broker_position_quantity(symbol)
        except Exception:
            _log.exception("SL recovery: failed to fetch broker position for %s", symbol)
            broker_qty = position.quantity

        if broker_qty == 0 and not self._is_paper_mode():
            _log.info("SL recovery: broker position is 0 for %s — no action needed", symbol)
            with self._lock:
                position.state = PositionState.CLOSED
            self._finalize_closed_position(symbol, position)
            return

        _log.warning("SL_RECOVERY_ATTEMPTED | symbol=%s | reason=%s | action=market_exit", symbol, reason)
        self._fire_market_exit(symbol, position, f"SL_RECOVERY_{reason}")

    def _fire_market_exit(self, symbol: str, position: Position, reason: str) -> None:
        transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
        fill_price = 0.0
        if CONFIG.mode == TradingMode.PAPER:
            ltp = self._get_ltp(symbol) if self._get_ltp else None
            fill_price = ltp if ltp and ltp > 0 else position.entry_price
        _log.info(
            "ORDER_REQUEST | symbol=%s transaction_type=%s quantity=%d order_type=MARKET price=%.2f trigger_price=None product=%s exchange=%s caller=_fire_market_exit reason=%s",
            symbol, transaction_type, position.quantity, fill_price, CONFIG.capital.product_type, CONFIG.instrument.option_exchange, reason,
        )
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=CONFIG.instrument.option_exchange,
                transaction_type=transaction_type,
                quantity=position.quantity,
                product=CONFIG.capital.product_type,
                order_type="MARKET",
                price=fill_price,
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
            position.state = PositionState.EXIT_MARKET_FALLBACK
            position.exit_market_fallback_used = True
            position.exit_fallback_quantity = position.quantity
            self._exit_order_ids[symbol] = order_id

        log_structured(
            "EXIT_MARKET_FALLBACK",
            symbol=symbol,
            side=position.side.value,
            position_quantity=position.quantity,
            exit_reason=reason,
            strategy_exit_price=round(position.entry_price, 2),
            limit_price=None,
            buffer_pct=CONFIG.trade_mgmt.exit_limit_buffer_pct,
            limit_order_id=position.exit_limit_order_id,
            filled_quantity=0,
            remaining_quantity=position.quantity,
            timeout_seconds=CONFIG.order.exit_limit_timeout_sec,
            fallback_used=True,
            final_exit_status="PENDING",
        )

    def _cancel_sl_and_market_exit(self, symbol: str, position: Position, reason: str) -> None:
        if not self._can_send_exit(position):
            _log.warning("Cannot cancel SL and market exit for %s: state=%s", symbol, position.state.value)
            return

        sl_order_id = position.sl_order_id

        if sl_order_id:
            try:
                self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
                _log.info("SL cancellation requested for %s order_id=%s", symbol, sl_order_id)
            except Exception:
                _log.exception("Failed to cancel SL order for %s", symbol)

            with self._lock:
                position.sl_order_id = None
                position.last_sl_trigger = 0.0

        self._reconcile_position(position)
        if position.state == PositionState.CLOSED:
            _log.info("Position %s already closed after SL cancellation", symbol)
            recovered, sl_price = self._try_recover_sl_fill_price(symbol, position, sl_order_id=sl_order_id)
            if recovered:
                with self._lock:
                    position.exit_avg_price = sl_price
                    position.exit_reason = "SL_FILLED"
            self._finalize_closed_position(symbol, position)
            return

        if not self._can_send_exit(position):
            _log.info("Cannot send market exit for %s: state=%s", symbol, position.state.value)
            return

        self._fire_market_exit(symbol, position, reason)

    def _execute_market_fallback(self, symbol: str, position: Position, reason: str, quantity: int | None = None) -> None:
        remaining = quantity if quantity is not None else position.exit_remaining_quantity
        if remaining <= 0:
            self._reconcile_position(position)
            if position.state == PositionState.CLOSED:
                self._finalize_closed_position(symbol, position)
            return

        if not self._can_send_exit(position):
            _log.warning("Cannot send MARKET fallback for %s: cannot send exit", symbol)
            return

        transaction_type = "SELL" if position.side == TradeSide.LONG else "BUY"
        order_kwargs = dict(
            tradingsymbol=symbol,
            exchange=CONFIG.instrument.option_exchange,
            transaction_type=transaction_type,
            quantity=remaining,
            product=CONFIG.capital.product_type,
            order_type="MARKET",
        )
        if CONFIG.mode == TradingMode.PAPER:
            ltp = self._get_ltp(symbol) if self._get_ltp else None
            if ltp and ltp > 0:
                order_kwargs["price"] = round(ltp, 2)
        try:
            order_id = self.kite.place_order(**order_kwargs)
            with self._lock:
                position.exit_order_id = order_id
                position.exit_reason = reason
                position.exit_requested_quantity = remaining
                position.exit_requested_at = datetime.now().isoformat()
                position.exit_filled_quantity = 0
                position.exit_remaining_quantity = remaining
                position.state = PositionState.EXIT_MARKET_FALLBACK
                position.exit_market_fallback_used = True
                position.exit_fallback_quantity = remaining
                self._exit_order_ids[symbol] = order_id

            log_structured(
                "EXIT_MARKET_FALLBACK",
                symbol=symbol,
                side=position.side.value,
                position_quantity=position.quantity,
                exit_reason=reason,
                strategy_exit_price=round(position.exit_avg_price, 2) if position.exit_avg_price > 0 else round(position.entry_price, 2),
                limit_price=None,
                buffer_pct=CONFIG.trade_mgmt.exit_limit_buffer_pct,
                limit_order_id=position.exit_limit_order_id,
                filled_quantity=0,
                remaining_quantity=remaining,
                timeout_seconds=CONFIG.order.exit_limit_timeout_sec,
                fallback_used=True,
                final_exit_status="PENDING",
            )
        except Exception:
            _log.exception("Failed to place MARKET fallback for %s", symbol)
            with self._lock:
                if position.state == PositionState.EXIT_MARKET_FALLBACK:
                    position.state = PositionState.OPEN
                    position.exit_order_id = None
                    position.exit_limit_order_id = None
                    position.exit_reason = ""
                    position.exit_requested_quantity = 0
                    position.exit_filled_quantity = 0
                    position.exit_remaining_quantity = 0
                    position.exit_avg_price = 0.0
                    position.exit_requested_at = None
                    position.exit_filled_at = None
                    position.exit_market_fallback_used = False
                    position.exit_fallback_quantity = 0

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

                if position.state in (PositionState.EXIT_LIMIT_PLACED, PositionState.EXIT_LIMIT_PARTIAL, PositionState.EXIT_MARKET_FALLBACK, PositionState.EXIT_PENDING) and position.exit_order_id:
                    _log.info("Force square-off skipped for %s: exit pending order_id=%s", symbol, position.exit_order_id)
                    continue

                sl_order_id = position.sl_order_id
                if sl_order_id:
                    try:
                        self.kite.cancel_order(CONFIG.kite.variety_regular, sl_order_id)
                        _log.info("SL cancelled for force square-off %s", symbol)
                    except Exception:
                        _log.exception("Failed to cancel SL for force square-off %s", symbol)

                self._fire_market_exit(symbol, position, "FORCE_SQUARE_OFF")
                _log.info("Force square-off initiated for %s", symbol)
            except Exception:
                _log.exception("Failed to force square-off %s", symbol)
                continue

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

    # ------------------------------------------------------------
    # SESSION PERSISTENCE
    # ------------------------------------------------------------

    def export_session_state(self) -> dict:
        with self._lock:
            positions = []
            for symbol, pos in self._positions.items():
                positions.append({
                    "symbol": symbol,
                    "contract_tradingsymbol": pos.contract.tradingsymbol,
                    "contract_strike": pos.contract.strike,
                    "contract_option_type": pos.contract.option_type,
                    "contract_expiry": pos.contract.expiry,
                    "contract_instrument_token": pos.contract.instrument_token,
                    "contract_lot_size": pos.contract.lot_size,
                    "side": pos.side.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "stop_loss": pos.stop_loss,
                    "target": pos.target,
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                    "initial_sl": pos.initial_sl,
                    "state": pos.state.value,
                    "was_ever_filled": pos.was_ever_filled,
                    "entry_order_id": pos.entry_order_id,
                    "sl_order_id": pos.sl_order_id,
                    "exit_order_id": pos.exit_order_id,
                    "entry_avg_price": pos.entry_avg_price,
                    "entry_filled_quantity": pos.entry_filled_quantity,
                    "sl_placed_at": pos.sl_placed_at,
                })
            return {
                "positions": positions,
                "sl_order_ids": dict(self._sl_order_ids),
                "entry_order_ids": dict(self._entry_order_ids),
                "exit_order_ids": dict(self._exit_order_ids),
            }

    def import_session_state(self, data: dict) -> None:
        from utils import OptionContract, PositionState, TradeSide
        with self._lock:
            self._positions.clear()
            self._sl_order_ids.clear()
            self._entry_order_ids.clear()
            self._exit_order_ids.clear()
            for p in data.get("positions", []):
                contract = OptionContract(
                    tradingsymbol=p["contract_tradingsymbol"],
                    strike=p["contract_strike"],
                    option_type=p["contract_option_type"],
                    expiry=p["contract_expiry"],
                    instrument_token=p["contract_instrument_token"],
                    lot_size=p["contract_lot_size"],
                )
                entry_time = datetime.fromisoformat(p["entry_time"]) if p.get("entry_time") else datetime.now()
                state = PositionState(p.get("state", "OPEN"))
                if state == PositionState.CLOSED:
                    continue
                position = Position(
                    contract=contract,
                    side=TradeSide(p["side"]),
                    entry_price=p["entry_price"],
                    quantity=p["quantity"],
                    stop_loss=p["stop_loss"],
                    target=p["target"],
                    entry_time=entry_time,
                    initial_sl=p.get("initial_sl", 0.0),
                    state=state,
                    was_ever_filled=p.get("was_ever_filled", False),
                    entry_order_id=p.get("entry_order_id"),
                    sl_order_id=p.get("sl_order_id"),
                    exit_order_id=p.get("exit_order_id"),
                    entry_avg_price=p.get("entry_avg_price", 0.0),
                    entry_filled_quantity=p.get("entry_filled_quantity", 0),
                    sl_placed_at=p.get("sl_placed_at"),
                )
                self._positions[p["symbol"]] = position
            self._sl_order_ids.update(data.get("sl_order_ids", {}))
            self._entry_order_ids.update(data.get("entry_order_ids", {}))
            self._exit_order_ids.update(data.get("exit_order_ids", {}))
        _log.info("Imported %d positions from session state", len(self._positions))
