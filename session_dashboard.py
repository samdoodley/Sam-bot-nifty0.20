"""
session_dashboard.py
====================
Separate dashboard for session-wise trade analytics.
Serves on a different port than main dashboard.
Reads from trade_journal.jsonl - no strategy/logic dependencies.
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
from trade_journal import _get_log_dir

_log = get_logger("session_dashboard")

_STATE_LOCK = threading.Lock()
_SERVER: ThreadingHTTPServer | None = None


def _parse_trades() -> list[dict]:
    """Parse all trades from journal, grouped by date and session."""
    path = _get_log_dir() / "trades.jsonl"
    if not path.exists():
        return []

    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("event") == "EXIT" and "pnl" in t:
                    ts_str = t.get("ts") or t.get("exit_time")
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except Exception:
                            continue
                        t["_dt"] = dt
                        t["_date"] = dt.date().isoformat()
                        h = dt.hour
                        m = dt.minute
                        if (h, m) < (13, 50):
                            t["_session"] = 1
                        elif (h, m) < (14, 20):
                            t["_session"] = 2
                        elif (h, m) < (15, 0):
                            t["_session"] = 3
                        else:
                            t["_session"] = 4
                        trades.append(t)
            except json.JSONDecodeError:
                continue
    return trades


def _compute_session_stats(trades: list[dict]) -> dict[str, Any]:
    """Compute per-date, per-session stats."""
    by_date: dict[str, dict[int, dict]] = {}

    for t in trades:
        date = t["_date"]
        session = t["_session"]
        pnl = float(t.get("pnl", 0))

        if date not in by_date:
            by_date[date] = {}
        if session not in by_date[date]:
            by_date[date][session] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "win_pnl": 0.0,
                "loss_pnl": 0.0,
            }

        s = by_date[date][session]
        s["trades"] += 1
        s["total_pnl"] += pnl
        if pnl > 0:
            s["wins"] += 1
            s["win_pnl"] += pnl
        else:
            s["losses"] += 1
            s["loss_pnl"] += pnl

    return by_date


def _generate_html(stats: dict[str, Any]) -> str:
    """Generate complete HTML page with session-wise report (server-side rendered)."""
    # Compute totals
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    rows = []
    for date in sorted(stats.keys(), reverse=True):
        sessions = stats[date]
        date_total_pnl = sum(s["total_pnl"] for s in sessions.values())
        date_total_trades = sum(s["trades"] for s in sessions.values())
        date_wins = sum(s["wins"] for s in sessions.values())
        date_losses = sum(s["losses"] for s in sessions.values())

        total_trades += date_total_trades
        total_wins += date_wins
        total_losses += date_losses
        total_pnl += date_total_pnl

        pnl_color = "#37f26b" if date_total_pnl >= 0 else "#ff4444"
        rows.append(f"""
        <tr style="background:#111;">
            <td colspan="9" style="padding:8px;font-weight:bold;color:#37f26b;border-top:2px solid #37f26b;">
                {date} | Total Trades: {date_total_trades} | Wins: {date_wins} | Losses: {date_losses} | Net P&L: <span style="color:{pnl_color};">₹{date_total_pnl:,.2f}</span>
            </td>
        </tr>""")

        for session in sorted(sessions.keys()):
            s = sessions[session]
            win_rate = (s["wins"] / s["trades"] * 100) if s["trades"] > 0 else 0
            total_pnl_color = "#37f26b" if s["total_pnl"] >= 0 else "#ff4444"
            rows.append(f"""
            <tr>
                <td></td>
                <td>Session {session}</td>
                <td>{s["trades"]}</td>
                <td>{s["wins"]}</td>
                <td>{s["losses"]}</td>
                <td>{win_rate:.1f}%</td>
                <td style="color:#37f26b;">₹{s["win_pnl"]:,.2f}</td>
                <td style="color:#ff4444;">₹{s["loss_pnl"]:,.2f}</td>
                <td style="color:{total_pnl_color};font-weight:bold;">₹{s["total_pnl"]:,.2f}</td>
            </tr>""")

    if not rows:
        table_rows = '<tr><td colspan="9" style="text-align:center;padding:40px;color:#777;">No trades recorded yet</td></tr>'
    else:
        table_rows = "".join(rows)

    # Summary values
    win_rate_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl_color = "#37f26b" if total_pnl >= 0 else "#ff4444"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NIFTY Session Analytics</title>
<style>
body{{background:#0b0f0b;color:#37f26b;font-family:'Courier New',monospace;padding:20px;margin:0;}}
h1{{color:#37f26b;border-bottom:1px solid #37f26b;padding-bottom:8px;font-size:1.4rem;}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;}}
th,td{{padding:6px 8px;border-bottom:1px solid #123;text-align:right;}}
th{{color:#7fdc9a;background:#0d120d;position:sticky;top:0;}}
td:first-child,th:first-child{{text-align:left;}}
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
@media (max-width:600px){{table,th,td{{font-size:11px;padding:4px 6px;}}}}
</style></head><body>
<h1>📊 NIFTY Session-Wise Analytics</h1>
<div class="summary">
    <div class="summary-grid">
        <div class="summary-item"><div class="summary-label">Total Trades</div><div class="summary-value">{total_trades}</div></div>
        <div class="summary-item"><div class="summary-label">Wins</div><div class="summary-value positive">{total_wins}</div></div>
        <div class="summary-item"><div class="summary-label">Losses</div><div class="summary-value negative">{total_losses}</div></div>
        <div class="summary-item"><div class="summary-label">Win Rate</div><div class="summary-value">{win_rate_pct:.1f}%</div></div>
        <div class="summary-item"><div class="summary-label">Net P&L</div><div class="summary-value" style="color:{total_pnl_color};">₹{total_pnl:,.2f}</div></div>
    </div>
</div>
<button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
<table>
<thead>
<tr>
    <th>Date</th><th>Session</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win%</th>
    <th>Win P&L</th><th>Loss P&L</th><th>Net P&L</th>
</tr>
</thead>
<tbody>{table_rows}</tbody>
</table>
<script>
// Auto-refresh every 30 seconds
setInterval(function() {{ location.reload(); }}, 30000);
</script>
</body></html>"""
    return html


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/stats"):
            trades = _parse_trades()
            stats = _compute_session_stats(trades)
            self._send_json(stats)
        else:
            trades = _parse_trades()
            stats = _compute_session_stats(trades)
            html = _generate_html(stats)
            self._send_html(html)

    def _send_json(self, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True


def start_session_dashboard() -> ThreadingHTTPServer:
    global _SERVER
    host = CONFIG.dashboard.host
    port = CONFIG.dashboard.port + 1  # different port from main dashboard
    _SERVER = _Server((host, port), _Handler)
    thread = threading.Thread(target=_SERVER.serve_forever, daemon=True)
    thread.start()
    _log.info("Session dashboard running at http://%s:%d", host, port)
    return _SERVER


def stop_session_dashboard():
    global _SERVER
    if _SERVER:
        _SERVER.shutdown()
        _SERVER = None


def get_session_dashboard_url() -> str:
    host = CONFIG.dashboard.host
    port = CONFIG.dashboard.port + 1
    if host in ("127.0.0.1", "localhost"):
        return f"http://localhost:{port}"
    return f"http://{host}:{port}"


if __name__ == "__main__":
    start_session_dashboard()
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_session_dashboard()