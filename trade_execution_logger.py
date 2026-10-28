"""
trade_execution_logger.py
=========================
Logs every trade lifecycle event to mode-specific JSONL files:

    trade_execution_paper.jsonl   (paper mode events)
    trade_execution_live.jsonl    (live mode events)
    trade_errors.jsonl            (trade-related errors, both modes)

Events tracked:
    ORDER_REQUESTED   - LIMIT entry order placed
    ORDER_FILLED      - Entry order filled by broker
    ORDER_CANCELLED   - Entry order cancelled (timeout / manual)
    SL_INITIATED      - SL order placed after entry fill
    SL_FILLED         - SL order triggered and filled (exit)
    SL_CANCELLED      - SL order cancelled (trailing / exit / repair)
    SL_MODIFIED       - SL trigger updated by trailing logic
    POSITION_CLOSED   - Position fully closed (exit reason + PnL)
    TRADE_ERROR       - Any error during order/SL/exit flow

Each record includes: ts, event, mode, symbol, side, qty, prices, order_ids, reason, pnl.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONFIG

_lock = threading.Lock()

_test_mode = "pytest" in sys.modules or os.environ.get("NIFTY_BOT_TEST_MODE") == "1"


def _get_log_dir() -> Path:
    if _test_mode:
        test_log_dir = Path.home() / ".nifty_bot" / "test_logs"
        test_log_dir.mkdir(parents=True, exist_ok=True)
        return test_log_dir
    return CONFIG.logging.log_dir


def _mode_suffix() -> str:
    """Return 'paper' or 'live' based on current CONFIG.mode."""
    return "paper" if CONFIG.mode.value == "PAPER" else "live"


def _write_record(filename: str, record: dict[str, Any]) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    record["mode"] = CONFIG.mode.value
    path = _get_log_dir() / filename
    line = json.dumps(record, default=str)
    with _lock:
        with open(path, "a") as f:
            f.write(line + "\n")


def log_order_requested(
    symbol: str,
    side: str,
    quantity: int,
    entry_price: float,
    order_id: str | None,
    order_type: str,
    reason: str = "",
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "ORDER_REQUESTED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "order_id": order_id,
            "order_type": order_type,
            "reason": reason,
            **extra,
        },
    )


def log_order_filled(
    symbol: str,
    side: str,
    quantity: int,
    filled_quantity: int,
    avg_price: float,
    order_id: str | None,
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "ORDER_FILLED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "filled_quantity": filled_quantity,
            "avg_price": avg_price,
            "order_id": order_id,
            **extra,
        },
    )


def order_cancelled(
    symbol: str,
    side: str,
    quantity: int,
    order_id: str | None,
    reason: str = "",
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "ORDER_CANCELLED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_id": order_id,
            "reason": reason,
            **extra,
        },
    )


def log_sl_initiated(
    symbol: str,
    side: str,
    quantity: int,
    sl_trigger: float,
    sl_limit: float | None,
    sl_order_id: str | None,
    order_type: str = "SL",
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "SL_INITIATED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "sl_trigger": sl_trigger,
            "sl_limit": sl_limit,
            "sl_order_id": sl_order_id,
            "order_type": order_type,
            **extra,
        },
    )


def log_sl_filled(
    symbol: str,
    side: str,
    quantity: int,
    sl_trigger: float,
    fill_price: float,
    sl_order_id: str | None,
    pnl: float,
    reason: str = "SL_FILLED",
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "SL_FILLED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "sl_trigger": sl_trigger,
            "fill_price": fill_price,
            "sl_order_id": sl_order_id,
            "pnl": pnl,
            "reason": reason,
            **extra,
        },
    )


def log_sl_cancelled(
    symbol: str,
    side: str,
    quantity: int,
    sl_order_id: str | None,
    reason: str = "",
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "SL_CANCELLED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "sl_order_id": sl_order_id,
            "reason": reason,
            **extra,
        },
    )


def log_sl_modified(
    symbol: str,
    side: str,
    quantity: int,
    old_trigger: float,
    new_trigger: float,
    sl_order_id: str | None,
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "SL_MODIFIED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "old_trigger": old_trigger,
            "new_trigger": new_trigger,
            "sl_order_id": sl_order_id,
            **extra,
        },
    )


def log_position_closed(
    symbol: str,
    side: str,
    quantity: int,
    entry_price: float,
    exit_price: float,
    pnl: float,
    reason: str = "",
    sl_order_id: str | None = None,
    exit_order_id: str | None = None,
    **extra: Any,
) -> None:
    _write_record(
        f"trade_execution_{_mode_suffix()}.jsonl",
        {
            "event": "POSITION_CLOSED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "sl_order_id": sl_order_id,
            "exit_order_id": exit_order_id,
            **extra,
        },
    )


def log_trade_error(
    symbol: str,
    side: str,
    quantity: int,
    error: str,
    stage: str = "",
    order_id: str | None = None,
    **extra: Any,
) -> None:
    _write_record(
        "trade_errors.jsonl",
        {
            "event": "TRADE_ERROR",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "error": error,
            "stage": stage,
            "order_id": order_id,
            **extra,
        },
    )


def read_trade_execution(n: int = 200, mode: str | None = None) -> list[dict]:
    """Read recent trade execution records. If mode given, filter by paper/live."""
    filename = f"trade_execution_{mode}.jsonl" if mode else f"trade_execution_{_mode_suffix()}.jsonl"
    path = _get_log_dir() / filename
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()[-n:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_trade_errors(n: int = 200) -> list[dict]:
    path = _get_log_dir() / "trade_errors.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()[-n:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
