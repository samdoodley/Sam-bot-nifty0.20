"""
trade_tracking_dashboard.py
===========================
Separate dashboard for trade execution tracking.
Serves on port 8767 (configurable via CONFIG.trade_tracking_dashboard).
Reads from trade_execution_paper.jsonl / trade_execution_live.jsonl
and trade_errors.jsonl - no strategy/logic dependencies.
Shows every trade lifecycle event: ORDER_REQUESTED, ORDER_FILLED,
ORDER_CANCELLED, SL_INITIATED, SL_FILLED, SL_CANCELLED, SL_MODIFIED,
POSITION_CLOSED, TRADE_ERROR.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from config import CONFIG
from logger import get_logger
from trade_execution_logger import _get_log_dir, _mode_suffix

_log = get_logger("trade_tracking_dashboard")

_STATE_LOCK = threading.Lock()
_SERVER: ThreadingHTTPServer | None = None


def _read_jsonl(filename: str, limit: int = 500) -> list[dict]:
    path = _get_log_dir() / filename
    if not path.exists():
        return []
    with open(path) as f:
        lines = f.readlines()[-limit:]
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


def _get_trade_data() -> dict[str, Any]:
    mode = CONFIG.mode.value.lower()
    exec_file = f"trade_execution_{mode}.jsonl"
    events = _read_jsonl(exec_file, limit=500)
    errors = _read_jsonl("trade_errors.jsonl", limit=100)

    # Compute summary
    total_entries = sum(1 for e in events if e.get("event") == "ORDER_REQUESTED")
    total_fills = sum(1 for e in events if e.get("event") == "ORDER_FILLED")
    total_sl_initiated = sum(1 for e in events if e.get("event") == "SL_INITIATED")
    total_sl_filled = sum(1 for e in events if e.get("event") == "SL_FILLED")
    total_positions_closed = sum(1 for e in events if e.get("event") == "POSITION_CLOSED")
    total_errors = len(errors)

    total_pnl = 0.0
    wins = 0
    losses = 0
    for e in events:
        if e.get("event") == "POSITION_CLOSED":
            pnl = float(e.get("pnl", 0))
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    win_rate = (wins / total_positions_closed * 100) if total_positions_closed > 0 else 0

    # Build HTML rows for events (newest first)
    rows = []
    for e in reversed(events[-200:]):
        ts = e.get("ts", "")
        event = e.get("event", "")
        symbol = e.get("symbol", "-")
        side = e.get("side", "-")
        qty = e.get("quantity", "-")
        price = e.get("entry_price", e.get("sl_trigger", e.get("fill_price", e.get("avg_price", "-"))))
        pnl = e.get("pnl", "-")
        reason = e.get("reason", "")
        order_id = e.get("order_id", e.get("sl_order_id", e.get("exit_order_id", "-")))

        pnl_color = "#37f26b" if isinstance(pnl, (int, float)) and pnl > 0 else ("#ff4444" if isinstance(pnl, (int, float)) and pnl < 0 else "#777")
        event_color = {
            "ORDER_REQUESTED": "#7fdc9a",
            "ORDER_FILLED": "#37f26b",
            "ORDER_CANCELLED": "#ffaa00",
            "SL_INITIATED": "#7fdc9a",
            "SL_FILLED": "#ff4444",
            "SL_CANCELLED": "#ffaa00",
            "SL_MODIFIED": "#aaffff",
            "POSITION_CLOSED": "#37f26b",
        }.get(event, "#ffffff")

        rows.append(f"""
        <tr>
            <td>{ts}</td>
            <td style="color:{event_color};font-weight:bold;">{event}</td>
            <td>{symbol}</td>
            <td>{side}</td>
            <td>{qty}</td>
            <td>{price}</td>
            <td style="color:{pnl_color};">{pnl if isinstance(pnl, (int, float)) else '-'}</td>
            <td>{reason}</td>
            <td>{order_id}</td>
        </tr>""")

    # Build HTML rows for errors (newest first)
    error_rows = []
    for e in reversed(errors[-100:]):
        ts = e.get("ts", "")
        symbol = e.get("symbol", "-")
        error = e.get("error", "")
        stage = e.get("stage", "")
        order_id = e.get("order_id", "-")
        error_rows.append(f"""
        <tr style="background:#1a0a0a;">
            <td>{ts}</td>
            <td style="color:#ff4444;">{symbol}</td>
            <td>{stage}</td>
            <td>{error}</td>
            <td>{order_id}</td>
        </tr>""")

    table_rows = "".join(rows) if rows else '<tr><td colspan="9" style="text-align:center;padding:40px;color:#777;">No trade events recorded yet</td></tr>'
    error_table_rows = "".join(error_rows) if error_rows else '<tr><td colspan="5" style="text-align:center;padding:40px;color:#777;">No errors recorded yet</td></tr>'

    total_pnl_color = "#37f26b" if total_pnl >= 0 else "#ff4444"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIFTY Trade Execution Tracker</title>
<style>
body{{background:#0b0f0b;color:#37f26b;font-family:'Courier New',monospace;padding:20px;margin:0;}}
h1{{color:#37f26b;border-bottom:1px solid #37f26b;padding-bottom:8px;font-size:1.4rem;}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px;}}
th,td{{padding:5px 8px;border-bottom:1px solid #123;text-align:left;}}
th{{color:#7fdc9a;background:#0d120d;position:sticky;top:0;}}
tr:hover{{background:#122;}}
.refresh-btn{{background:#37f26b;color:#0b0f0b;border:none;padding:10px 20px;font-family:inherit;font-weight:bold;margin:10px 0;cursor:pointer;}}
.refresh-btn:active{{background:#2ad95b;}}
.summary{{background:#0d120d;padding:12px;border:1px solid #123;margin-bottom:16px;border-radius:4px;}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;}}
.summary-item{{text-align:center;}}
.summary-label{{color:#7fdc9a;font-size:11px;}}
.summary-value{{font-size:18px;font-weight:bold;}}
.positive{{color:#37f26b;}}
.negative{{color:#ff4444;}}
.section-title{{color:#7fdc9a;font-size:13px;margin-top:24px;margin-bottom:8px;font-weight:bold;}}
@media (max-width:600px){{table,th,td{{font-size:10px;padding:3px 4px;}}}}
</style></head><body>
<h1>📊 NIFTY Trade Execution Tracker</h1>
<div class="summary">
    <div class="summary-grid">
        <div class="summary-item"><div class="summary-label">Mode</div><div class="summary-value">{CONFIG.mode.value}</div></div>
        <div class="summary-item"><div class="summary-label">Entries</div><div class="summary-value">{total_entries}</div></div>
        <div class="summary-item"><div class="summary-label">Fills</div><div class="summary-value">{total_fills}</div></div>
        <div class="summary-item"><div class="summary-label">SL Initiated</div><div class="summary-value">{total_sl_initiated}</div></div>
        <div class="summary-item"><div class="summary-label">SL Filled</div><div class="summary-value">{total_sl_filled}</div></div>
        <div class="summary-item"><div class="summary-label">Positions Closed</div><div class="summary-value">{total_positions_closed}</div></div>
        <div class="summary-item"><div class="summary-label">Wins</div><div class="summary-value positive">{wins}</div></div>
        <div class="summary-item"><div class="summary-label">Losses</div><div class="summary-value negative">{losses}</div></div>
        <div class="summary-item"><div class="summary-label">Win Rate</div><div class="summary-value">{win_rate:.1f}%</div></div>
        <div class="summary-item"><div class="summary-label">Net P&L</div><div class="summary-value" style="color:{total_pnl_color};">₹{total_pnl:,.2f}</div></div>
        <div class="summary-item"><div class="summary-label">Errors</div><div class="summary-value negative">{total_errors}</div></div>
    </div>
</div>
<button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>

<div class="section-title">Trade Execution Events (latest 200)</div>
<table>
<thead>
<tr>
    <th>Time</th><th>Event</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>P&L</th><th>Reason</th><th>Order ID</th>
</tr>
</thead>
<tbody>{table_rows}</tbody>
</table>

<div class="section-title">Trade Errors (latest 100)</div>
<table>
<thead>
<tr>
    <th>Time</th><th>Symbol</th><th>Stage</th><th>Error</th><th>Order ID</th>
</tr>
</thead>
<tbody>{error_table_rows}</tbody>
</table>
<script>
// Auto-refresh every 15 seconds
setInterval(function() {{ location.reload(); }}, 15000);
</script>
</body></html>"""
    return html


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/trades"):
            mode = CONFIG.mode.value.lower()
            exec_file = f"trade_execution_{mode}.jsonl"
            events = _read_jsonl(exec_file, limit=500)
            errors = _read_jsonl("trade_errors.jsonl", limit=100)
            body = json.dumps({"events": events, "errors": errors, "mode": CONFIG.mode.value}, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            data = _get_trade_data()
            html = data  # _get_trade_data returns html string
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True


def start_trade_tracking_dashboard() -> ThreadingHTTPServer:
    global _SERVER
    host = CONFIG.trade_tracking_dashboard.host
    port = CONFIG.trade_tracking_dashboard.port
    _SERVER = _Server((host, port), _Handler)
    thread = threading.Thread(target=_SERVER.serve_forever, daemon=True)
    thread.start()
    _log.info("Trade tracking dashboard running at http://%s:%d", host, port)
    return _SERVER


def stop_trade_tracking_dashboard():
    global _SERVER
    if _SERVER:
        _SERVER.shutdown()
        _SERVER = None


def get_trade_tracking_dashboard_url() -> str:
    host = CONFIG.trade_tracking_dashboard.host
    port = CONFIG.trade_tracking_dashboard.port
    if host in ("127.0.0.1", "localhost"):
        return f"http://localhost:{port}"
    return f"http://{host}:{port}"


if __name__ == "__main__":
    start_trade_tracking_dashboard()
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_trade_tracking_dashboard()
