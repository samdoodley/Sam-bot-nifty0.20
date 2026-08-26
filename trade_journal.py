"""
trade_journal.py
===============
Append-only trade journal for paper/live sessions. Every completed
trade (exit by target, SL, or force square-off) is written as a
single JSON line to trades.jsonl for post-session review.
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


def record_trade(**fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    path = _get_log_dir() / "trades.jsonl"
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_recent_trades(n: int = 200) -> list[dict]:
    path = _get_log_dir() / "trades.jsonl"
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
