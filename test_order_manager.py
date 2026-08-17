"""
Tests for order_manager.py safety fixes.
Covers TC01-TC10 from the order-management fix spec plus
SL escalation watchdog and MARKET entry tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import unittest

from config import CONFIG
from kite_api import KiteAPI
from order_manager import OrderManager
from risk_manager import RiskManager
from utils import OptionContract, Position, PositionState, TradeSide


def _make_contract(symbol: str = "TESTCE") -> OptionContract:
    return OptionContract(
        tradingsymbol=symbol,
        strike=100,
        option_type="CE",
        expiry="2025-12-31",
        instrument_token=12345,
        lot_size=50,
    )


def _make_position(symbol: str = "TESTCE", state: PositionState = PositionState.OPEN, sl_placed_at: str | None = None) -> Position:
    return Position(
        contract=_make_contract(symbol),
        side=TradeSide.LONG,
        entry_price=100.0,
        quantity=50,
        stop_loss=90.0,
        target=120.0,
        entry_time=datetime.now(),
        initial_sl=10.0,
        state=state,
        sl_order_id="sl123",
        entry_order_id="entry123",
        sl_placed_at=sl_placed_at,
        was_ever_filled=True,
    )


class TestOrderManagerSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.kite = MagicMock(spec=KiteAPI)
        self.get_ltp = MagicMock(return_value=110.0)
        self.mgr = OrderManager(self.kite, self.get_ltp)
        self.mgr.on_exit(lambda s, r: None)

    def test_tc01_target_hits_first_sl_cancelled_exit_filled(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        def order_history_side_effect(order_id):
            if order_id == "sl123":
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "CANCELLED", "filled_quantity": 0, "average_price": 0.0},
                ]
            if order_id and order_id.startswith("exit"):
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "COMPLETE", "filled_quantity": 50, "average_price": 120.0},
                ]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": []}

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)
        self.assertIsNotNone(position.exit_order_id)

        self.mgr._poll_exit_order_status(position.contract.tradingsymbol, position)
        self.assertEqual(position.state, PositionState.CLOSED)
        self.kite.cancel_order.assert_called_once_with(CONFIG.kite.variety_regular, "sl123")
        self.kite.place_order.assert_called_once()

    def test_tc02_target_hits_but_sl_already_filled(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
            {"status": "COMPLETE", "filled_quantity": 50, "average_price": 90.0},
        ]
        self.kite.positions.return_value = {"net": []}

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.kite.place_order.assert_not_called()
        self.assertEqual(position.state, PositionState.CLOSED)

    def test_tc03_sl_hits_first_next_cycle_no_duplicate_exit(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 0}]}
        self.kite.orders.return_value = []

        self.mgr._reconcile_position(position)
        self.assertEqual(position.state, PositionState.CLOSED)

        self.mgr._check_position_exit(position.contract.tradingsymbol)
        self.kite.place_order.assert_not_called()

    def test_tc04_sl_cancel_races_with_sl_execution(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
            {"status": "COMPLETE", "filled_quantity": 50, "average_price": 90.0},
        ]
        self.kite.positions.return_value = {"net": []}

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.kite.place_order.assert_not_called()
        self.assertEqual(position.state, PositionState.CLOSED)

    def test_tc05_exit_pending_next_cycle_no_duplicate(self):
        position = _make_position(state=PositionState.EXIT_PENDING)
        position.exit_order_id = "exit999"
        position.exit_requested_at = datetime.now().isoformat()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = [
            {"order_id": "exit999", "tradingsymbol": "TESTCE", "status": "OPEN"},
        ]

        self.mgr._check_position_exit(position.contract.tradingsymbol)
        self.kite.place_order.assert_not_called()

    def test_tc06_exit_partially_filled_remaining_tracked(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        def order_history_side_effect(order_id):
            if order_id == "sl123":
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                ]
            if order_id and order_id.startswith("exit"):
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "OPEN", "filled_quantity": 20, "average_price": 120.0},
                ]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 30}]}
        self.kite.orders.return_value = [
            {"order_id": "exit456", "tradingsymbol": "TESTCE", "status": "OPEN"},
        ]

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)

        self.mgr._poll_exit_order_status(position.contract.tradingsymbol, position)
        self.assertEqual(position.exit_filled_quantity, 20)
        self.assertEqual(position.exit_remaining_quantity, 30)
        self.assertEqual(position.state, PositionState.EXIT_PENDING)

    def test_tc07_exit_rejected_safe_handling(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        def order_history_side_effect(order_id):
            if order_id == "sl123":
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "REJECTED", "filled_quantity": 0, "average_price": 0.0},
                ]
            if order_id and order_id.startswith("exit"):
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "REJECTED", "filled_quantity": 0, "average_price": 0.0},
                ]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)

        self.mgr._poll_exit_order_status(position.contract.tradingsymbol, position)
        self.assertEqual(position.state, PositionState.OPEN)
        self.assertIsNone(position.exit_order_id)

    def test_tc08_api_timeout_does_not_assume_failed(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        call_count = [0]

        def order_history_side_effect(order_id):
            call_count[0] += 1
            if order_id == "sl123":
                if call_count[0] == 1:
                    return [
                        {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                        {"status": "CANCELLED", "filled_quantity": 0, "average_price": 0.0},
                    ]
                raise Exception("timeout")
            if order_id and order_id.startswith("exit"):
                raise Exception("timeout")
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}

        def orders_side_effect():
            return [
                {"order_id": position.exit_order_id, "tradingsymbol": "TESTCE", "status": "OPEN"},
            ] if position.exit_order_id else []

        self.kite.orders.side_effect = orders_side_effect

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)
        self.assertIsNotNone(position.exit_order_id)

        self.mgr._poll_exit_order_status(position.contract.tradingsymbol, position)
        self.assertEqual(position.state, PositionState.EXIT_PENDING)
        self.assertIsNotNone(position.exit_order_id)

    def test_tc09_bot_restart_reconcile_correctly(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 0}]}
        self.kite.orders.return_value = []

        self.mgr._reconcile_position(position)
        self.assertEqual(position.state, PositionState.CLOSED)

    def test_tc10_target_exit_success_broker_qty_zero(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        def order_history_side_effect(order_id):
            if order_id == "sl123":
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "CANCELLED", "filled_quantity": 0, "average_price": 0.0},
                ]
            if order_id and order_id.startswith("exit"):
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "COMPLETE", "filled_quantity": 50, "average_price": 120.0},
                ]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": []}

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)
        self.assertIsNotNone(position.exit_order_id)

        self.mgr._poll_exit_order_status(position.contract.tradingsymbol, position)
        self.assertEqual(position.state, PositionState.CLOSED)
        self.kite.cancel_order.assert_called_once_with(CONFIG.kite.variety_regular, "sl123")
        self.kite.place_order.assert_called_once()


class TestOrderManagerIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        self.kite = MagicMock(spec=KiteAPI)
        self.get_ltp = MagicMock(return_value=110.0)
        self.mgr = OrderManager(self.kite, self.get_ltp)
        self.mgr.on_exit(lambda s, r: None)

    def test_duplicate_close_call_safe(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        def order_history_side_effect(order_id):
            if order_id == "sl123":
                return [
                    {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
                    {"status": "COMPLETE", "filled_quantity": 50, "average_price": 90.0},
                ]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": []}

        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.assertEqual(position.state, PositionState.CLOSED)

        self.kite.reset_mock()
        self.mgr._close_position(position.contract.tradingsymbol, position, 120.0, "TARGET")
        self.kite.place_order.assert_not_called()

    def test_force_square_off_safety(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 0}]}
        self.kite.orders.return_value = []

        self.mgr.force_square_off_all()
        self.kite.place_order.assert_not_called()
        self.assertEqual(position.state, PositionState.CLOSED)


class TestSLWatchdog(unittest.TestCase):
    def setUp(self) -> None:
        self.kite = MagicMock(spec=KiteAPI)
        self.get_ltp = MagicMock(return_value=110.0)
        self.mgr = OrderManager(self.kite, self.get_ltp)
        self.mgr.on_exit(lambda s, r: None)

    def test_watchdog_fires_when_sl_stale(self):
        old_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        position = _make_position(sl_placed_at=old_time)
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.kite.cancel_order.assert_called_once_with(CONFIG.kite.variety_regular, "sl123")
        self.kite.place_order.assert_called_once()
        call_kwargs = self.kite.place_order.call_args
        self.assertEqual(call_kwargs.kwargs["order_type"], "MARKET")
        self.assertEqual(position.state, PositionState.EXIT_PENDING)

    def test_watchdog_no_fire_when_sl_fresh(self):
        position = _make_position(sl_placed_at=datetime.now().isoformat())
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.kite.cancel_order.assert_not_called()
        self.kite.place_order.assert_not_called()

    def test_watchdog_no_fire_when_sl_already_filled(self):
        old_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        position = _make_position(sl_placed_at=old_time)
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
            {"status": "COMPLETE", "filled_quantity": 50, "average_price": 90.0},
        ]
        self.kite.positions.return_value = {"net": []}

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.kite.cancel_order.assert_not_called()
        self.kite.place_order.assert_not_called()
        self.assertEqual(position.state, PositionState.CLOSED)

    def test_watchdog_race_sl_fills_during_cancel(self):
        old_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        position = _make_position(sl_placed_at=old_time)
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        call_count = [0]

        def order_history_side_effect(order_id):
            call_count[0] += 1
            if order_id == "sl123":
                if call_count[0] == 1:
                    return [{"status": "OPEN", "filled_quantity": 0, "average_price": 0.0}]
                return [{"status": "COMPLETE", "filled_quantity": 50, "average_price": 90.0}]
            return []

        self.kite.order_history.side_effect = order_history_side_effect
        self.kite.positions.return_value = {"net": []}

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.kite.cancel_order.assert_called_once_with(CONFIG.kite.variety_regular, "sl123")
        self.kite.place_order.assert_not_called()
        self.assertEqual(position.state, PositionState.CLOSED)


class TestMarketEntry(unittest.TestCase):
    def test_entry_uses_market_order(self):
        kite = MagicMock(spec=KiteAPI)
        kite.TRANSACTION_TYPE_BUY = "BUY"
        kite.TRANSACTION_TYPE_SELL = "SELL"
        get_ltp = MagicMock(return_value=100.0)
        mgr = OrderManager(kite, get_ltp)
        mgr.on_exit(lambda s, r: None)

        contract = _make_contract()
        kite.place_order.return_value = "entry456"

        mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        call_kwargs = kite.place_order.call_args_list[0]
        self.assertEqual(call_kwargs.kwargs["order_type"], "LIMIT")
        self.assertEqual(call_kwargs.kwargs["price"], 100.0)


class TestEntryStateFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.kite = MagicMock(spec=KiteAPI)
        self.get_ltp = MagicMock(return_value=100.0)
        self.mgr = OrderManager(self.kite, self.get_ltp)
        self.mgr.on_exit(lambda s, r: None)

    def test_entry_remains_pending_when_order_open(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": []}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        position = self.mgr._positions["TESTCE"]
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)

        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)
        self.assertIsNone(position.sl_order_id)

    def test_entry_becomes_open_when_filled(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
            {"status": "COMPLETE", "filled_quantity": 50, "average_price": 100.5},
        ]
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        position = self.mgr._positions["TESTCE"]
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)

        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.OPEN)
        self.assertTrue(position.was_ever_filled)
        self.assertEqual(position.entry_avg_price, 100.5)
        self.assertIsNotNone(position.sl_order_id)

    def test_entry_rejected_marks_failed_no_loss(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "REJECTED", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": []}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        position = self.mgr._positions["TESTCE"]
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)

        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.ENTRY_FAILED)
        self.assertFalse(position.was_ever_filled)
        self.assertNotIn("TESTCE", self.mgr._positions)

    def test_entry_cancelled_marks_failed_no_loss(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "CANCELLED", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": []}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        position = self.mgr._positions["TESTCE"]
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)

        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.ENTRY_FAILED)
        self.assertFalse(position.was_ever_filled)

    def test_broker_qty_zero_while_pending_does_not_close(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 0}]}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        position = self.mgr._positions["TESTCE"]
        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)
        self.assertFalse(position.was_ever_filled)

    def test_confirmed_open_position_broker_qty_zero_after_exit_closes(self):
        position = _make_position(state=PositionState.OPEN, sl_placed_at=datetime.now().isoformat())
        position.exit_order_id = "exit999"
        position.exit_requested_at = datetime.now().isoformat()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 0}]}
        self.kite.orders.return_value = [
            {"order_id": "exit999", "tradingsymbol": "TESTCE", "status": "COMPLETE"},
        ]

        self.mgr._reconcile_position(position)
        self.assertEqual(position.state, PositionState.CLOSED)
        self.assertTrue(position.was_ever_filled)

    def test_duplicate_entry_blocked_while_pending(self):
        contract = _make_contract()
        self.kite.place_order.return_value = "entry123"
        self.kite.order_history.return_value = [
            {"status": "OPEN", "filled_quantity": 0, "average_price": 0.0},
        ]
        self.kite.positions.return_value = {"net": []}
        self.kite.orders.return_value = []

        self.mgr.enter_position(
            contract=contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )

        self.assertEqual(self.mgr.open_position_count(), 1)
        self.assertEqual(self.mgr._positions["TESTCE"].state, PositionState.ENTRY_PENDING)

        second_contract = _make_contract(symbol="TESTCE2")
        self.mgr.enter_position(
            contract=second_contract,
            side=TradeSide.LONG,
            quantity=50,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            initial_sl=10.0,
        )
        self.assertEqual(self.mgr.open_position_count(), 2)
        self.assertIn("TESTCE2", self.mgr._positions)
        self.assertEqual(self.mgr._positions["TESTCE2"].state, PositionState.ENTRY_PENDING)

    def test_trailing_sl_modifies_broker_order_when_profit_exceeds_threshold(self):
        position = _make_position(state=PositionState.OPEN, sl_placed_at=datetime.now().isoformat())
        position.last_sl_trigger = 90.0
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []
        self.get_ltp.return_value = 110.0

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.assertTrue(position.stop_loss > 90.0)
        self.assertEqual(position.last_sl_trigger, round(position.stop_loss, 2))
        self.kite.modify_order.assert_called_once()
        call_kwargs = self.kite.modify_order.call_args
        self.assertEqual(call_kwargs.kwargs["order_id"], "sl123")
        self.assertEqual(call_kwargs.kwargs["trigger_price"], round(position.stop_loss, 2))
        self.get_ltp.return_value = 100.0

    def test_trailing_sl_does_not_modify_when_profit_below_threshold(self):
        position = _make_position(state=PositionState.OPEN, sl_placed_at=datetime.now().isoformat())
        position.last_sl_trigger = 90.0
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []

        original_ltp = self.mgr._get_ltp
        self.mgr._get_ltp = lambda s: 101.0

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.assertEqual(position.stop_loss, 90.0)
        self.assertEqual(position.last_sl_trigger, 90.0)
        self.kite.modify_order.assert_not_called()

        self.mgr._get_ltp = original_ltp

    def test_trailing_sl_does_not_modify_when_already_at_new_trigger(self):
        position = _make_position(state=PositionState.OPEN, sl_placed_at=datetime.now().isoformat())
        position.last_sl_trigger = 108.33
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position

        self.kite.positions.return_value = {"net": [{"tradingsymbol": "TESTCE", "quantity": 50}]}
        self.kite.orders.return_value = []
        self.get_ltp.return_value = 110.0

        self.mgr._check_position_exit(position.contract.tradingsymbol)

        self.kite.modify_order.assert_not_called()
        self.get_ltp.return_value = 100.0


class TestPaperModeGuard(unittest.TestCase):
    def test_paper_mode_blocks_order_placement(self):
        kite = KiteAPI.__new__(KiteAPI)
        kite._paper_mode = True
        kite._paper_orders = {}
        kite._paper_positions = {}
        kite.kite = MagicMock()
        kite.kite.VARIETY_REGULAR = "regular"
        
        result = kite.place_order(
            tradingsymbol="TESTCE",
            exchange="NFO",
            transaction_type="BUY",
            quantity=50,
            product="MIS",
            order_type="LIMIT",
            price=100.0,
        )
        self.assertTrue(result.startswith("PAPER_"))
        self.assertIn(result, kite._paper_orders)
        kite.kite.place_order.assert_not_called()

    def test_live_mode_allows_order_placement(self):
        kite = KiteAPI.__new__(KiteAPI)
        kite._paper_mode = False
        kite._paper_orders = {}
        kite._paper_positions = {}
        kite.kite = MagicMock()
        kite.kite.VARIETY_REGULAR = "regular"
        kite.kite.place_order.return_value = "order123"
        
        result = kite.place_order(
            tradingsymbol="TESTCE",
            exchange="NFO",
            transaction_type="BUY",
            quantity=50,
            product="MIS",
            order_type="LIMIT",
            price=100.0,
        )
        self.assertEqual(result, "order123")
        kite.kite.place_order.assert_called_once()

    def test_paper_mode_simulates_modify_order(self):
        kite = KiteAPI.__new__(KiteAPI)
        kite._paper_mode = True
        kite._paper_orders = {}
        kite._paper_positions = {}
        kite.kite = MagicMock()
        kite.kite.VARIETY_REGULAR = "regular"

        order_id = kite.place_order(
            tradingsymbol="TESTCE",
            exchange="NFO",
            transaction_type="BUY",
            quantity=50,
            product="MIS",
            order_type="LIMIT",
            price=100.0,
        )
        self.assertIn(order_id, kite._paper_orders)
        self.assertEqual(kite._paper_orders[order_id]["status"], "COMPLETE")

        kite.modify_order(
            variety="regular",
            order_id=order_id,
            trigger_price=105.0,
            price=106.0,
        )

        self.assertEqual(kite._paper_orders[order_id]["trigger_price"], 105.0)
        self.assertEqual(kite._paper_orders[order_id]["price"], 106.0)
        kite.kite.modify_order.assert_not_called()


class TestLossCounting(unittest.TestCase):
    def test_zero_pnl_does_not_increase_consecutive_losses(self):
        rm = RiskManager(starting_equity=100000.0)
        rm.record_trade_result(0.0)
        self.assertEqual(rm.stats.trades_taken, 1)
        self.assertEqual(rm.stats.consecutive_losses, 0)
        self.assertEqual(rm.stats.losses, 0)
        self.assertEqual(rm.stats.wins, 0)

    def test_negative_pnl_increases_consecutive_losses(self):
        rm = RiskManager(starting_equity=100000.0)
        rm.record_trade_result(-1000.0)
        self.assertEqual(rm.stats.trades_taken, 1)
        self.assertEqual(rm.stats.consecutive_losses, 1)
        self.assertEqual(rm.stats.losses, 1)

    def test_positive_pnl_resets_consecutive_losses(self):
        rm = RiskManager(starting_equity=100000.0)
        rm.record_trade_result(-1000.0)
        rm.record_trade_result(500.0)
        self.assertEqual(rm.stats.trades_taken, 2)
        self.assertEqual(rm.stats.consecutive_losses, 0)
        self.assertEqual(rm.stats.losses, 1)
        self.assertEqual(rm.stats.wins, 1)


if __name__ == "__main__":
    unittest.main()
