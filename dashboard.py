"""
dashboard.py
============
Minimal dependency-free dashboard (stdlib http.server) - serves:
  GET /            -> terminal-style HTML page (auto-refreshes)
  GET /api/state   -> current bot state as JSON

State is fed in by main.py via `dashboard.update_state(...)` from the
main loop; the HTTP handler just reads the latest snapshot under a lock.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from config import CONFIG
from logger import get_logger, read_recent_decisions
from trade_journal import read_recent_trades

_log = get_logger("dashboard")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "nifty_spot": None, "atm_strike": None, "option_premium": None,
    "trend": None, "ema20": None, "ema50": None, "vwap": None,
    "adx": None, "atr": None, "signal": None,
    "position": None, "entry_price": None, "current_pnl": None,
    "daily_pnl": None, "available_margin": None, "margin_used": None,
    "trades_today": None, "win_rate": None, "mode": CONFIG.mode.value,
}


def update_state(**fields: Any) -> None:
    with _state_lock:
        _state.update(fields)


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NIFTY Bot Dashboard</title>
<style>
body{background:#0b0f0b;color:#37f26b;font-family:'Courier New',monospace;padding:20px;}
h1{color:#37f26b;border-bottom:1px solid #37f26b;padding-bottom:8px;}
table{width:100%;border-collapse:collapse;margin-top:10px;}
td{padding:6px 10px;border-bottom:1px solid #123;}
td.label{color:#7fdc9a;width:220px;}
.decisions{margin-top:24px;max-height:400px;overflow-y:auto;font-size:12px;white-space:pre-wrap;}
</style></head>
<body>
<h1>NIFTY Weekly Options Bot [MODE: <span id="mode">-</span>]</h1>
<table id="state"></table>
<h2>Recent Decisions</h2>
<div class="decisions" id="decisions"></div>
<h2>Recent Trades</h2>
<div class="decisions" id="trades"></div>
<script>
async function refresh() {
  const res = await fetch('/api/state');
  const data = await res.json();
  document.getElementById('mode').textContent = data.mode || '-';
  const t = document.getElementById('state');
  t.innerHTML = '';
  for (const [k, v] of Object.entries(data)) {
    const row = document.createElement('tr');
    row.innerHTML = `<td class="label">${k}</td><td>${v === null || v === undefined ? '-' : v}</td>`;
    t.appendChild(row);
  }
  const dres = await fetch('/api/decisions');
  const decisions = await dres.json();
  document.getElementById('decisions').textContent =
    decisions.map(d => JSON.stringify(d)).join('\\n');
  const tres = await fetch('/api/trades');
  const trades = await tres.json();
  document.getElementById('trades').textContent =
    trades.map(d => JSON.stringify(d)).join('\\n');
}
refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence default stderr logging
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            self._send_json(get_state())
        elif self.path.startswith("/api/decisions"):
            self._send_json(read_recent_decisions(100))
        elif self.path.startswith("/api/trades"):
            self._send_json(read_recent_trades(100))
        else:
            page = _PAGE.replace("REFRESH_MS", str(int(CONFIG.dashboard.refresh_interval_sec * 1000)))
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _send_json(self, payload) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_dashboard() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((CONFIG.dashboard.host, CONFIG.dashboard.port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _log.info("Dashboard running at http://%s:%d", CONFIG.dashboard.host, CONFIG.dashboard.port)
    return server