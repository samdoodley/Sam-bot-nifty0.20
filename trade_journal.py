"""
trade_journal.py
================
Append-only trade journal for paper/live sessions. Every completed
trade (exit by target, SL, or force square-off) is written as a
single JSON line to trades.jsonl for post-session review.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONFIG

_lock = threading.Lock()


def record_trade(**fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    path = CONFIG.logging.log_dir / "trades.jsonl"
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_recent_trades(n: int = 200) -> list[dict]:
    path = CONFIG.logging.log_dir / "trades.jsonl"
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
