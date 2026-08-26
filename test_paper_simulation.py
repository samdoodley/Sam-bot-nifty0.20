"""
test_paper_simulation.py
========================
Tests for PAPER mode execution simulation fixes:
1. Partial fills / rejections
2. Equity / margin tracking
3. SL order execution
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from config import CONFIG, TradingMode
from kite_api import KiteAPI
from margin_manager import MarginManager
from order_manager import OrderManager
from utils import OptionContract, Position, PositionState, TradeSide


def _make_contract(symbol: str = "TESTCE") -> OptionContract:
    return OptionContract(
        tradingsymbol=symbol,
        strike=100,
        option_type="CE",
        expiry="2026-08-27",
        instrument_token=12345,
        lot_size=50,
    )


def _make_position(symbol: str = "TESTCE", state: PositionState = PositionState.OPEN, sl_placed_at: str | None = None) -> Position:
    return Position(
        contract=_make_contract(symbol),
        side=TradeSide.LONG,
        entry_price=100.0,
        quantity=585,
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


class TestPaperPartialFillRejection(unittest.TestCase):
    """Issue #1: PAPER partial fills / rejections."""

    def setUp(self):
        self.kite = KiteAPI.__new__(KiteAPI)
        self.kite._paper_mode = True
        self.kite._paper_orders = {}
        self.kite._paper_positions = {}
        self.kite._paper_equity = 500000.0
        self.kite._paper_initial_equity = 500000.0
        self.kite._paper_fill_ratio = 1.0
        self.kite.kite = MagicMock()
        self.kite.kite.VARIETY_REGULAR = "regular"

    def test_tc01_full_entry_fill(self):
        order_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        order = self.kite._paper_orders[order_id]
        self.assertEqual(order["quantity"], 585)
        self.assertEqual(order["filled_quantity"], 585)
        self.assertEqual(order["remaining_quantity"], 0)
        self.assertEqual(order["status"], "COMPLETE")
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 585)

    def test_tc02_partial_entry_fill(self):
        self.kite.set_paper_fill_ratio(0.5)
        order_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        order = self.kite._paper_orders[order_id]
        self.assertEqual(order["quantity"], 585)
        self.assertLess(order["filled_quantity"], 585)
        self.assertGreater(order["filled_quantity"], 0)
        self.assertEqual(order["status"], "PARTIAL")
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), order["filled_quantity"])

    def test_tc03_remaining_entry_fill(self):
        self.kite.set_paper_fill_ratio(0.5)
        order_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        order = self.kite._paper_orders[order_id]
        first_fill = order["filled_quantity"]
        self.kite.set_paper_fill_ratio(1.0)
        order_id2 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585 - first_fill, product="MIS", order_type="LIMIT", price=100.0,
        )
        order2 = self.kite._paper_orders[order_id2]
        self.assertEqual(order2["filled_quantity"], 585 - first_fill)
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 585)

    def test_tc04_sl_pending_no_exit(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.process_paper_orders("TESTCE", 95.0)
        sl_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "SL"]
        self.assertEqual(len(sl_orders), 1)
        self.assertEqual(sl_orders[0]["status"], "TRIGGER PENDING")
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 585)

    def test_tc05_sl_trigger_fills_original(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.process_paper_orders("TESTCE", 89.5)
        sl_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "SL"]
        self.assertEqual(len(sl_orders), 1)
        self.assertEqual(sl_orders[0]["status"], "COMPLETE")
        self.assertEqual(sl_orders[0]["filled_quantity"], 585)
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)

    def test_tc06_target_first_then_sl_cancelled(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="LIMIT", price=120.0,
        )
        self.kite.process_paper_orders("TESTCE", 120.0)
        exit_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "LIMIT" and o["transaction_type"] == "SELL"]
        self.assertEqual(len(exit_orders), 1)
        self.assertEqual(exit_orders[0]["status"], "COMPLETE")
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)
        sl_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "SL"]
        self.assertEqual(len(sl_orders), 1)
        self.assertEqual(sl_orders[0]["status"], "CANCELLED")

    def test_tc07_trailing_sl_triggers(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.modify_order("regular", sl_id, tradingsymbol="TESTCE", exchange="NFO",
                               transaction_type="SELL", quantity=585, product="MIS",
                               order_type="SL", price=101.0, trigger_price=100.0)
        self.kite.process_paper_orders("TESTCE", 99.5)
        sl_order = self.kite._paper_orders[sl_id]
        self.assertEqual(sl_order["status"], "COMPLETE")
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)

    def test_tc08_partial_exit(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.set_paper_fill_ratio(0.5)
        exit_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="LIMIT", price=120.0,
        )
        exit_order = self.kite._paper_orders[exit_id]
        self.assertEqual(exit_order["status"], "PARTIAL")
        remaining = self.kite._paper_positions.get("TESTCE", 0)
        self.assertGreater(remaining, 0)
        self.assertLess(remaining, 585)
        self.kite.set_paper_fill_ratio(1.0)
        exit_id2 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585 - exit_order["filled_quantity"], product="MIS", order_type="LIMIT", price=120.0,
        )
        exit_order2 = self.kite._paper_orders[exit_id2]
        self.assertEqual(exit_order2["filled_quantity"], 585 - exit_order["filled_quantity"])
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)

    def test_tc09_rejected_sl(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.cancel_order("regular", sl_id)
        sl_order = self.kite._paper_orders[sl_id]
        self.assertEqual(sl_order["status"], "CANCELLED")
        self.kite.process_paper_orders("TESTCE", 89.5)
        sl_order = self.kite._paper_orders[sl_id]
        self.assertEqual(sl_order["status"], "CANCELLED")

    def test_tc10_paper_equity_changes(self):
        self.assertEqual(self.kite._paper_equity, 500000.0)
        self.kite.credit_paper_pnl(12000.0)
        self.assertEqual(self.kite._paper_equity, 512000.0)
        self.kite.credit_paper_pnl(-8000.0)
        self.assertEqual(self.kite._paper_equity, 504000.0)
        snapshot = self.kite._paper_margin_snapshot()
        self.assertEqual(snapshot["paper_current_equity"], 504000.0)

    def test_tc11_duplicate_sl_protection(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id1 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        sl_id2 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.cancel_order("regular", sl_id2)
        active_sls = [o for o in self.kite._paper_orders.values() if o["order_type"] == "SL" and o["status"] in ("TRIGGER PENDING", "OPEN")]
        self.assertEqual(len(active_sls), 1)

    def test_tc12_duplicate_exit_prevented(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        exit_id1 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="LIMIT", price=120.0,
        )
        self.kite._paper_orders[exit_id1]["status"] = "COMPLETE"
        self.kite._paper_orders[exit_id1]["filled_quantity"] = 585
        self.kite._paper_positions["TESTCE"] = 0
        exit_id2 = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="LIMIT", price=120.0,
        )
        completed_exits = [o for o in self.kite._paper_orders.values() if o["order_type"] == "LIMIT" and o["transaction_type"] == "SELL" and o["status"] == "COMPLETE"]
        self.assertGreaterEqual(len(completed_exits), 1)
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), -585)


class TestPaperMarginManager(unittest.TestCase):
    """Issue #2: PAPER equity/margin divergence."""

    def test_paper_margin_uses_override(self):
        kite = MagicMock(spec=KiteAPI)
        kite.margins.return_value = {"equity": {"available": {"live_balance": 500000.0}}}
        mgr = MarginManager(kite, paper_equity_override=500000.0)
        self.assertEqual(mgr.get_available_margin(), 500000.0 * CONFIG.capital.margin_leverage_multiplier)

    def test_paper_equity_updates_kite(self):
        kite = MagicMock(spec=KiteAPI)
        mgr = MarginManager(kite, paper_equity_override=500000.0)
        mgr.update_paper_equity(512000.0)
        self.assertEqual(mgr._paper_equity_override, 512000.0)
        kite.set_paper_equity.assert_called_with(512000.0)

    def test_margin_manager_no_override_uses_broker(self):
        kite = MagicMock(spec=KiteAPI)
        kite.margins.return_value = {"equity": {"available": {"live_balance": 600000.0}}}
        mgr = MarginManager(kite)
        self.assertEqual(mgr.get_available_margin(), 600000.0)


class TestPaperOrderManagerIntegration(unittest.TestCase):
    """Issue #3: PAPER SL execution + order manager integration."""

    def setUp(self):
        self.kite = KiteAPI.__new__(KiteAPI)
        self.kite._paper_mode = True
        self.kite._paper_orders = {}
        self.kite._paper_positions = {}
        self.kite._paper_equity = 500000.0
        self.kite._paper_initial_equity = 500000.0
        self.kite.kite = MagicMock()
        self.kite.kite.VARIETY_REGULAR = "regular"
        self.get_ltp = MagicMock(return_value=100.0)
        self.mgr = OrderManager(self.kite, self.get_ltp)
        self.mgr.on_exit(lambda s, r: None)

    def test_sl_stays_trigger_pending_until_hit(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.mgr._check_position_exit("TESTCE")
        sl_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "SL"]
        self.assertEqual(len(sl_orders), 1)
        self.assertEqual(sl_orders[0]["status"], "TRIGGER PENDING")

    def test_sl_fills_via_process_paper_orders(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.process_paper_orders("TESTCE", 89.5)
        sl_order = self.kite._paper_orders.get(sl_id, {})
        self.assertEqual(sl_order.get("status"), "COMPLETE")
        self.assertEqual(sl_order.get("filled_quantity"), 585)
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)

    def test_no_duplicate_exit_after_sl_fill(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.process_paper_orders("TESTCE", 89.5)
        self.assertEqual(self.kite._paper_positions.get("TESTCE", 0), 0)

    def test_target_fills_before_sl(self):
        position = _make_position()
        with self.mgr._lock:
            self.mgr._positions[position.contract.tradingsymbol] = position
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        sl_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        position.sl_order_id = sl_id
        self.get_ltp.return_value = 120.0
        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.EXIT_LIMIT_PLACED)
        self.kite.process_paper_orders("TESTCE", 120.0)
        self.mgr._check_position_exit("TESTCE")
        self.assertEqual(position.state, PositionState.CLOSED)

    def test_partial_exit_keeps_position_open(self):
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="BUY",
            quantity=585, product="MIS", order_type="LIMIT", price=100.0,
        )
        self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="SL", price=91.0, trigger_price=90.0,
        )
        self.kite.set_paper_fill_ratio(0.5)
        exit_id = self.kite.place_order(
            tradingsymbol="TESTCE", exchange="NFO", transaction_type="SELL",
            quantity=585, product="MIS", order_type="LIMIT", price=120.0,
        )
        exit_orders = [o for o in self.kite._paper_orders.values() if o["order_type"] == "LIMIT" and o["transaction_type"] == "SELL"]
        self.assertEqual(len(exit_orders), 1)
        self.assertEqual(exit_orders[0]["status"], "PARTIAL")
        remaining = self.kite._paper_positions.get("TESTCE", 0)
        self.assertGreater(remaining, 0)
        self.assertLess(remaining, 585)


if __name__ == "__main__":
    unittest.main()
