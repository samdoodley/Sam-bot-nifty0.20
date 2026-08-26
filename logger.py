"""
logger.py
=========
Two logging channels:

1. Standard rotating logger (console + file) for operational logs
   (connects, errors, order acks, etc.)
2. A structured JSONL "decision log" - one line per strategy decision,
   used by the dashboard and post-session review. Every reject reason
   ("ADX Low", "EMA Flat", ...) goes here, per your spec.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONFIG

_lock = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(CONFIG.logging.log_level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if CONFIG.logging.log_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if CONFIG.logging.log_to_file:
        CONFIG.logging.log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            CONFIG.logging.log_dir / f"{name}.log",
            maxBytes=CONFIG.logging.log_rotate_max_bytes,
            backupCount=CONFIG.logging.log_rotate_backups,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_decision(event: str, **fields: Any) -> None:
    """
    Append one structured decision record to decisions.jsonl.

    Example:
        log_decision("REJECTED", reason="ADX_LOW", adx=18.3, symbol="NIFTY25200CE")
        log_decision("ENTRY_EXECUTED", side="LONG", symbol=..., qty=50, price=142.5)
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    path = CONFIG.logging.log_dir / CONFIG.logging.decision_log_filename
    line = json.dumps(record, default=str)
    with _lock:
        with open(path, "a") as f:
            f.write(line + "\n")

    # also mirror to the operational logger so it's visible live in console
    op_logger = get_logger("decisions")
    op_logger.info("%s | %s", event, {k: v for k, v in fields.items()})


def read_recent_decisions(n: int = 200) -> list[dict]:
    path = CONFIG.logging.log_dir / CONFIG.logging.decision_log_filename
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


def log_structured(event: str, **fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "mode": CONFIG.mode.value,
        **fields,
    }
    path = CONFIG.logging.log_dir / CONFIG.logging.decision_log_filename
    line = json.dumps(record, default=str)
    with _lock:
        with open(path, "a") as f:
            f.write(line + "\n")
    op_logger = get_logger("decisions")
    op_logger.info("%s | %s", event, {k: v for k, v in fields.items()})


def send_telegram_alert(event: str, message: str) -> None:
    if not CONFIG.alerting.enabled:
        return
    text = f"[NIFTY Bot] {event}: {message}"
    url = f"https://api.telegram.org/bot{CONFIG.alerting.telegram_bot_token}/sendMessage"
    payload = json.dumps({"chat_id": CONFIG.alerting.telegram_chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                _log.warning("Telegram alert failed: HTTP %d", resp.status)
    except Exception as exc:
        _log.warning("Telegram alert failed: %s", exc)