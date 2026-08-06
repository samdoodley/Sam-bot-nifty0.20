"""
utils.py
========
Small shared helpers used across every module: retry/backoff decorator,
market-time-window checks, safe rounding to lot size, and a couple of
common dataclasses (Candle, Signal) that get passed between modules.
"""

from __future__ import annotations

import functools
import time as _time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from enum import Enum
from typing import Callable, Optional, TypeVar

from config import CONFIG
from logger import get_logger

_log = get_logger("utils")

T = TypeVar("T")


# ------------------------------------------------------------------
# Retry / backoff
# ------------------------------------------------------------------

def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exceptions: tuple = (Exception,),
):
    """Decorator: retries a function on exception with exponential backoff."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            _max = max_retries if max_retries is not None else CONFIG.kite.max_retries
            _base = base_delay if base_delay is not None else CONFIG.kite.retry_backoff_base_sec
            _cap = max_delay if max_delay is not None else CONFIG.kite.retry_backoff_max_sec

            last_exc = None
            for attempt in range(_max + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt == _max:
                        break
                    delay = min(_base * (2 ** attempt), _cap)
                    _log.warning(
                        "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        fn.__name__, attempt + 1, _max, exc, delay,
                    )
                    _time.sleep(delay)
            _log.error("%s failed permanently after %d attempts: %s", fn.__name__, _max, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ------------------------------------------------------------------
# Time window helpers
# ------------------------------------------------------------------

def now_ist_time() -> dtime:
    # Assumes host clock is IST, or NTP-synced to IST. For VPS deployments
    # outside IST, set TZ=Asia/Kolkata at the OS level.
    return datetime.now().time()


def is_market_open(t: Optional[dtime] = None) -> bool:
    t = t or now_ist_time()
    return CONFIG.session.market_open <= t <= CONFIG.session.market_close


def is_warm_up_done(t: Optional[dtime] = None) -> bool:
    t = t or now_ist_time()
    return t >= CONFIG.session.warm_up_until


def is_within_entry_window(t: Optional[dtime] = None) -> bool:
    t = t or now_ist_time()
    return CONFIG.session.setup_scan_start <= t <= CONFIG.session.last_entry_time


def is_force_square_off_time(t: Optional[dtime] = None) -> bool:
    t = t or now_ist_time()
    return t >= CONFIG.session.force_square_off_time


# ------------------------------------------------------------------
# Rounding / lot math
# ------------------------------------------------------------------

def round_down_to_lot(quantity: int, lot_size: int) -> int:
    if lot_size <= 0:
        return 0
    lots = quantity // lot_size
    return lots * lot_size


def round_to_strike_step(price: float, step: int) -> int:
    return int(round(price / step) * step)


# ------------------------------------------------------------------
# Shared dataclasses
# ------------------------------------------------------------------

TRAIL_TRIGGER_POINTS = 2.0
TRAIL_INITIAL_RATIO = 8.0
TRAIL_FINAL_RATIO = 5.0

@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    side: TradeSide
    reason: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    swing_level: float = 0.0
    confidence_notes: str = ""


@dataclass
class OptionContract:
    tradingsymbol: str
    strike: int
    option_type: str          # "CE" / "PE"
    expiry: str                # ISO date string
    instrument_token: int
    lot_size: int


@dataclass
class Position:
    contract: OptionContract
    side: TradeSide
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    entry_time: datetime
    initial_sl: float = 0.0
    breakeven_moved: bool = False
    highest_favorable_price: float = 0.0
    sl_order_id: Optional[str] = None
    entry_order_id: Optional[str] = None