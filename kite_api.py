"""
kite_api.py
===========
Thin wrapper around KiteConnect + KiteTicker.

- Browser-based authentication using the official Zerodha Kite Connect flow:
  1. Open login URL in browser
  2. User logs in and is redirected to the configured redirect URL
  3. Extract request_token from the redirect URL
  4. Exchange request_token for access_token via kite.generate_session()
  5. Cache the access_token in tokens.json for reuse
- Every REST call goes through retry_with_backoff.
- WebSocket uses the subscribe-on-connect queuing pattern (fixes the
  race condition you hit in WickFill's execution.py: symbols are only
  ever subscribed from inside on_connect, never from outside it).
"""

from __future__ import annotations

import json
import os
import threading
import time as _time
import webbrowser
from datetime import datetime, date
from typing import Callable, Optional
from urllib.parse import urlparse, parse_qs

from kiteconnect import KiteConnect, KiteTicker

from config import CONFIG
from logger import get_logger
from utils import retry_with_backoff

_log = get_logger("kite_api")


class KiteAuthError(RuntimeError):
    pass


class KiteAPI:
    def __init__(self) -> None:
        self.kite = KiteConnect(api_key=CONFIG.kite.api_key, timeout=CONFIG.kite.request_timeout_sec)
        self._ticker: Optional[KiteTicker] = None
        self._subscribed_tokens: set[int] = set()
        self._tick_callback: Optional[Callable] = None
        self._ws_lock = threading.Lock()
        self._ws_connected = threading.Event()
        self._paper_mode = CONFIG.mode.value == "PAPER"
        self._paper_orders: dict[str, dict] = {}
        self._paper_positions: dict[str, int] = {}
        self._paper_equity: float = 0.0
        self._paper_initial_equity: float = 0.0
        self._paper_fill_ratio: float = 1.0

    # ------------------------------------------------------------
    # AUTH
    # ------------------------------------------------------------

    def login(self) -> None:
        """
        Loads a cached access token if it's still valid; otherwise
        performs a full browser-based Kite login and caches the new token.
        """
        _log.info("MODE=%s | DATA=Real Kite Live | ORDERS=%s", CONFIG.mode.value, "SIMULATED (Paper)" if self._paper_mode else "REAL (Broker)")
        cached = self._load_cached_token()
        if cached:
            self.kite.set_access_token(cached)
            try:
                self.kite.profile()
                _log.info("Using Cached Token")
                _log.info("Authentication Successful")
                return
            except Exception:
                _log.info("Cached Token Expired - Requesting New Login")

        request_token = self._get_request_token()
        access_token = self._generate_session(request_token)
        self._save_cached_token(access_token)
        self.kite.set_access_token(access_token)
        _log.info("Authentication Successful")

    def _get_request_token(self) -> str:
        """
        Opens the Zerodha login URL in a browser, then prompts the user
        to paste the full redirect URL or the request_token extracted from it.
        """
        login_url = f"https://kite.zerodha.com/connect/login?api_key={CONFIG.kite.api_key}&v=3"
        _log.info("Opening Zerodha Login URL")
        opened = False
        try:
            opened = webbrowser.open(login_url)
        except Exception:
            pass
        if not opened:
            print(f"\n>>> Open this URL in your browser and log in:\n    {login_url}\n")
        else:
            print(f"\n>>> Browser opened. If it didn't open, paste this URL manually:\n    {login_url}\n")

        _log.info("Waiting for Request Token")
        _log.info("After login, paste the full redirect URL or just the request_token below.")

        try:
            user_input = input("Paste redirect URL or request_token: ").strip()
        except (EOFError, KeyboardInterrupt):
            _log.error("Authentication Failed - Keyboard interrupt")
            raise KiteAuthError("Authentication cancelled by user")

        if not user_input:
            raise KiteAuthError("No input provided")

        request_token = self._extract_request_token(user_input)
        if not request_token:
            raise KiteAuthError("Could not extract request_token from input")

        _log.info("Request Token Received")
        return request_token

    def _extract_request_token(self, url_or_token: str) -> Optional[str]:
        if "request_token=" in url_or_token:
            try:
                parsed = urlparse(url_or_token)
                params = parse_qs(parsed.query)
                token = params.get("request_token", [None])[0]
                if token:
                    return token
            except Exception:
                pass
        # Assume the input is just the raw request_token
        stripped = url_or_token.strip()
        if stripped:
            return stripped
        return None

    def _generate_session(self, request_token: str) -> str:
        """
        Exchanges a request_token for an access_token using
        kite.generate_session().
        """
        _log.info("Generating Session")
        try:
            data = self.kite.generate_session(request_token, api_secret=CONFIG.kite.api_secret)
        except Exception as e:
            _log.error("Authentication Failed - Invalid request token or API secret")
            raise KiteAuthError(f"Failed to generate session: {e}") from e

        access_token = data.get("access_token")
        if not access_token:
            raise KiteAuthError("No access_token returned from generate_session")

        _log.info("Access Token Generated")
        return access_token

    def _tokens_path(self) -> os.PathLike:
        return CONFIG.kite.tokens_path

    def _load_cached_token(self) -> Optional[str]:
        path = self._tokens_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            token = data.get("access_token")
            if not token:
                return None
            return token
        except Exception:
            return None

    def _save_cached_token(self, token: str) -> None:
        path = self._tokens_path()
        data = {
            "access_token": token,
            "login_date": date.today().isoformat(),
            "generated_at": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))
        _log.info("Token Saved")

    def _simulate_fill(self, order: dict) -> tuple[int, float]:
        order_type = order.get("order_type", "LIMIT")
        requested = order.get("quantity", 0)
        price = order.get("price", 0.0)
        trigger = order.get("trigger_price", 0.0)
        if requested <= 0:
            return 0, 0.0
        if order_type == "MARKET":
            return requested, price if price > 0 else 0.0
        if order_type == "SL-M":
            return requested, trigger if trigger > 0 else price
        if order_type == "SL":
            return requested, price if price > 0 else 0.0
        fill_ratio = getattr(self, "_paper_fill_ratio", 1.0)
        if fill_ratio >= 1.0:
            return requested, price
        fill_qty = max(1, int(requested * fill_ratio))
        return fill_qty, price

    def _apply_fill_to_position(self, symbol: str, transaction_type: str, qty: int) -> None:
        current = self._paper_positions.get(symbol, 0)
        if transaction_type == "BUY":
            self._paper_positions[symbol] = current + qty
        else:
            self._paper_positions[symbol] = current - qty

    def process_paper_orders(self, symbol: str, ltp: float) -> None:
        if not self._paper_mode:
            return
        for order_id, order in list(self._paper_orders.items()):
            if order.get("tradingsymbol") != symbol:
                continue
            status = order.get("status", "")
            if status in ("CANCELLED", "REJECTED"):
                continue
            order_type = order.get("order_type", "")
            if order_type not in ("LIMIT", "MARKET", "SL", "SL-M"):
                continue
            txn = order.get("transaction_type", "BUY")
            filled_qty = order.get("filled_quantity", 0)
            if filled_qty > 0 and status == "COMPLETE":
                current_qty = self._paper_positions.get(symbol, 0)
                if txn == "SELL" and current_qty >= 0:
                    self._cancel_opposite_protective_orders(symbol, "SELL")
                elif txn == "BUY" and current_qty <= 0:
                    self._cancel_opposite_protective_orders(symbol, "BUY")

        for order_id, order in list(self._paper_orders.items()):
            if order.get("tradingsymbol") != symbol:
                continue
            status = order.get("status", "")
            if status in ("COMPLETE", "FILLED", "CANCELLED", "REJECTED"):
                continue
            order_type = order.get("order_type", "")
            trigger = order.get("trigger_price", 0.0)
            limit = order.get("price", 0.0)
            requested = order.get("quantity", 0)
            remaining = order.get("remaining_quantity", requested)
            txn = order.get("transaction_type", "BUY")

            if order_type == "SL" and status == "TRIGGER PENDING":
                if (txn == "SELL" and ltp <= trigger) or (txn == "BUY" and ltp >= trigger):
                    fill_qty, fill_price = self._simulate_fill(order)
                    fill_qty = min(fill_qty, remaining)
                    if fill_qty > 0:
                        order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                        order["remaining_quantity"] = remaining - fill_qty
                        order["average_price"] = fill_price
                        order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                        self._apply_fill_to_position(symbol, txn, fill_qty)
                        _log.info("PAPER SL filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)
                        self._cancel_opposite_protective_orders(symbol, txn)
            elif order_type == "SL-M" and status == "TRIGGER PENDING":
                if (txn == "SELL" and ltp <= trigger) or (txn == "BUY" and ltp >= trigger):
                    fill_qty, fill_price = self._simulate_fill(order)
                    fill_qty = min(fill_qty, remaining)
                    if fill_qty > 0:
                        order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                        order["remaining_quantity"] = remaining - fill_qty
                        order["average_price"] = fill_price
                        order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                        self._apply_fill_to_position(symbol, txn, fill_qty)
                        _log.info("PAPER SL-M filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)
                        self._cancel_opposite_protective_orders(symbol, txn)
            elif order_type == "LIMIT" and status in ("OPEN", "PARTIAL"):
                current_qty = self._paper_positions.get(symbol, 0)
                if txn == "SELL" and current_qty > 0 and ltp <= limit:
                    fill_qty, fill_price = self._simulate_fill(order)
                    fill_qty = min(fill_qty, remaining, current_qty)
                    if fill_qty > 0:
                        order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                        order["remaining_quantity"] = remaining - fill_qty
                        order["average_price"] = fill_price
                        order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                        self._apply_fill_to_position(symbol, txn, fill_qty)
                        _log.info("PAPER LIMIT exit filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)
                        self._cancel_opposite_protective_orders(symbol, txn)
                elif txn == "BUY" and current_qty < 0 and ltp >= limit:
                    fill_qty, fill_price = self._simulate_fill(order)
                    fill_qty = min(fill_qty, remaining, abs(current_qty))
                    if fill_qty > 0:
                        order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                        order["remaining_quantity"] = remaining - fill_qty
                        order["average_price"] = fill_price
                        order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                        self._apply_fill_to_position(symbol, txn, fill_qty)
                        _log.info("PAPER LIMIT exit filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)
                        self._cancel_opposite_protective_orders(symbol, txn)
            elif order_type == "MARKET" and status in ("OPEN", "PARTIAL"):
                fill_qty, fill_price = self._simulate_fill(order)
                fill_qty = min(fill_qty, remaining)
                if fill_qty > 0:
                    order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                    order["remaining_quantity"] = remaining - fill_qty
                    order["average_price"] = fill_price
                    order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                    self._apply_fill_to_position(symbol, txn, fill_qty)
                    _log.info("PAPER MARKET exit filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)
                    self._cancel_opposite_protective_orders(symbol, txn)

    def _cancel_opposite_protective_orders(self, symbol: str, filled_txn: str) -> None:
        opposite = "BUY" if filled_txn == "SELL" else "SELL"
        for order_id, order in list(self._paper_orders.items()):
            if order.get("tradingsymbol") != symbol:
                continue
            if order.get("transaction_type") == opposite and order.get("order_type") == "SL" and order.get("status") == "TRIGGER PENDING":
                order["status"] = "CANCELLED"
                _log.info("PAPER SL cancelled after exit fill: symbol=%s order_id=%s", symbol, order_id)

    def fill_sl_order(self, symbol: str, fill_price: float) -> None:
        if not self._paper_mode:
            return
        for order_id, order in list(self._paper_orders.items()):
            if order.get("tradingsymbol") != symbol:
                continue
            if order.get("order_type") != "SL":
                continue
            status = order.get("status", "")
            if status not in ("TRIGGER PENDING", "OPEN"):
                continue
            requested = order.get("quantity", 0)
            remaining = order.get("remaining_quantity", requested)
            txn = order.get("transaction_type", "BUY")
            fill_qty, _ = self._simulate_fill(order)
            fill_qty = min(fill_qty, remaining)
            if fill_qty > 0:
                order["filled_quantity"] = order.get("filled_quantity", 0) + fill_qty
                order["remaining_quantity"] = remaining - fill_qty
                order["average_price"] = fill_price
                order["status"] = "COMPLETE" if order["remaining_quantity"] == 0 else "PARTIAL"
                self._apply_fill_to_position(symbol, txn, fill_qty)
                _log.info("PAPER SL force-filled: symbol=%s order_id=%s filled=%d remaining=%d price=%.2f", symbol, order_id, fill_qty, remaining - fill_qty, fill_price)

    def _paper_margin_snapshot(self) -> dict:
        return {
            "paper_initial_equity": self._paper_initial_equity,
            "paper_current_equity": self._paper_equity,
            "paper_available_margin": self._paper_equity,
        }

    def set_paper_equity(self, equity: float) -> None:
        self._paper_equity = float(equity)
        if self._paper_initial_equity == 0.0:
            self._paper_initial_equity = self._paper_equity

    def set_paper_fill_ratio(self, ratio: float) -> None:
        self._paper_fill_ratio = max(0.0, min(1.0, float(ratio)))

    def credit_paper_pnl(self, pnl: float) -> None:
        self._paper_equity += pnl

    # ------------------------------------------------------------
    # REST wrappers (retry-protected)
    # ------------------------------------------------------------

    @retry_with_backoff()
    def margins(self) -> dict:
        if self._paper_mode:
            return {
                "equity": {
                    "available": {
                        "live_balance": self._paper_equity,
                    }
                },
                "paper_equity": self._paper_equity,
                "paper_initial_equity": self._paper_initial_equity,
            }
        return self.kite.margins()

    @retry_with_backoff()
    def instruments(self, exchange: Optional[str] = None) -> list[dict]:
        return self.kite.instruments(exchange) if exchange else self.kite.instruments()

    @retry_with_backoff()
    def ltp(self, instruments: list[str]) -> dict:
        return self.kite.ltp(instruments)

    @retry_with_backoff()
    def quote(self, instruments: list[str]) -> dict:
        return self.kite.quote(instruments)

    @retry_with_backoff()
    def historical_data(self, instrument_token: int, from_dt: datetime, to_dt: datetime,
                         interval: str, continuous: bool = False) -> list[dict]:
        return self.kite.historical_data(instrument_token, from_dt, to_dt, interval, continuous)

    @retry_with_backoff(exceptions=(Exception,))
    def place_order(self, **kwargs) -> str:
        if self._paper_mode:
            order_id = "PAPER_" + str(abs(hash(str(kwargs) + str(datetime.now()))))
            tradingsymbol = kwargs.get("tradingsymbol", "UNKNOWN")
            requested_qty = int(kwargs.get("quantity", 0))
            transaction_type = kwargs.get("transaction_type", "BUY")
            order_type = kwargs.get("order_type", "LIMIT")
            price = float(kwargs.get("price", 0.0) or 0.0)
            trigger_price = float(kwargs.get("trigger_price", 0.0) or 0.0)

            order = {
                "order_id": order_id,
                "tradingsymbol": tradingsymbol,
                "quantity": requested_qty,
                "filled_quantity": 0,
                "remaining_quantity": requested_qty,
                "average_price": 0.0,
                "status": "OPEN",
                "transaction_type": transaction_type,
                "order_type": order_type,
                "price": price,
                "trigger_price": trigger_price,
                "placed_at": datetime.now().isoformat(),
            }
            self._paper_orders[order_id] = order

            if order_type == "SL":
                order["status"] = "TRIGGER PENDING"
                order["filled_quantity"] = 0
                order["remaining_quantity"] = requested_qty
            elif order_type == "SL-M":
                order["status"] = "TRIGGER PENDING"
                order["filled_quantity"] = 0
                order["remaining_quantity"] = requested_qty
            elif order_type == "MARKET":
                fill_qty, fill_price = self._simulate_fill(order)
                if fill_qty > 0:
                    order["filled_quantity"] = fill_qty
                    order["remaining_quantity"] = requested_qty - fill_qty
                    order["average_price"] = fill_price
                    order["status"] = "COMPLETE" if fill_qty >= requested_qty else "PARTIAL"
                    self._apply_fill_to_position(tradingsymbol, transaction_type, fill_qty)
            else:
                fill_qty, fill_price = self._simulate_fill(order)
                if fill_qty > 0:
                    order["filled_quantity"] = fill_qty
                    order["remaining_quantity"] = requested_qty - fill_qty
                    order["average_price"] = fill_price
                    order["status"] = "COMPLETE" if fill_qty >= requested_qty else "PARTIAL"
                    self._apply_fill_to_position(tradingsymbol, transaction_type, fill_qty)

            _log.info("PAPER mode: simulated order placed. order_id=%s type=%s symbol=%s requested=%d filled=%d remaining=%d price=%.2f status=%s",
                      order_id, order_type, tradingsymbol, requested_qty, order["filled_quantity"], order["remaining_quantity"], price, order["status"])
            return order_id
        return self.kite.place_order(variety=kwargs.pop("variety", self.kite.VARIETY_REGULAR), **kwargs)

    @retry_with_backoff()
    def modify_order(self, variety: str, order_id: str, **kwargs) -> str:
        if self._paper_mode:
            if order_id in self._paper_orders:
                order = self._paper_orders[order_id]
                for key, value in kwargs.items():
                    if key in ("price", "trigger_price"):
                        order[key] = float(value) if value is not None else order.get(key, 0.0)
                    else:
                        order[key] = value
                _log.info("PAPER mode: order modification simulated. order_id=%s new_trigger=%s new_limit=%s",
                          order_id, order.get("trigger_price"), order.get("price"))
            return order_id
        return self.kite.modify_order(variety=variety, order_id=order_id, **kwargs)

    @retry_with_backoff()
    def cancel_order(self, variety: str, order_id: str) -> str:
        if self._paper_mode:
            if order_id in self._paper_orders:
                self._paper_orders[order_id]["status"] = "CANCELLED"
            _log.info("PAPER mode: order cancellation simulated. order_id=%s", order_id)
            return order_id
        return self.kite.cancel_order(variety=variety, order_id=order_id)

    @retry_with_backoff()
    def orders(self) -> list[dict]:
        if self._paper_mode:
            return list(self._paper_orders.values())
        return self.kite.orders()

    @retry_with_backoff()
    def order_history(self, order_id: str) -> list[dict]:
        if self._paper_mode:
            order = self._paper_orders.get(order_id)
            if not order:
                return []
            return [dict(order)]
        return self.kite.order_history(order_id)

    @retry_with_backoff()
    def positions(self) -> dict:
        if self._paper_mode:
            net = []
            for symbol, qty in self._paper_positions.items():
                net.append({"tradingsymbol": symbol, "quantity": qty})
            return {"net": net}
        return self.kite.positions()

    # ------------------------------------------------------------
    # WEBSOCKET (subscribe-on-connect pattern - fixes WickFill's
    # race condition where subscriptions fired before the socket
    # was actually open)
    # ------------------------------------------------------------

    def start_ticker(self, tokens: list[int], on_tick: Callable[[list[dict]], None]) -> None:
        self._tick_callback = on_tick
        self._subscribed_tokens = set(tokens)

        self._ticker = KiteTicker(CONFIG.kite.api_key, self.kite.access_token)
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_ticks = self._on_ticks

        self._ticker.connect(threaded=True)

    def _on_connect(self, ws, response) -> None:
        with self._ws_lock:
            tokens = list(self._subscribed_tokens)
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            _log.info("WebSocket connected - subscribed to %d tokens.", len(tokens))
        self._ws_connected.set()

    def _on_close(self, ws, code, reason) -> None:
        self._ws_connected.clear()
        _log.warning("WebSocket closed: %s %s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        _log.error("WebSocket error: %s %s", code, reason)

    def _on_reconnect(self, ws, attempt_count) -> None:
        _log.warning("WebSocket reconnecting (attempt %d)...", attempt_count)

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        if self._tick_callback:
            self._tick_callback(ticks)

    def add_subscription(self, token: int) -> None:
        with self._ws_lock:
            self._subscribed_tokens.add(token)
        if self._ticker and self._ws_connected.is_set():
            self._ticker.subscribe([token])
            self._ticker.set_mode(self._ticker.MODE_FULL, [token])

    def remove_subscription(self, token: int) -> None:
        with self._ws_lock:
            self._subscribed_tokens.discard(token)
        if self._ticker and self._ws_connected.is_set():
            self._ticker.unsubscribe([token])

    def wait_until_connected(self, timeout: float = 15.0) -> bool:
        return self._ws_connected.wait(timeout)

    def stop_ticker(self) -> None:
        if self._ticker:
            self._ticker.close()