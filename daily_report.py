"""
daily_report.py
===============
Generates persistent day-by-day trade summaries from trades.jsonl.

Outputs:
  ~/.nifty_bot/logs/daily_report.csv      - cumulative CSV with all days
  ~/.nifty_bot/logs/daily_reports/YYYY-MM-DD.json  - one JSON file per day

Usage:
    python daily_report.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config import CONFIG
from trade_journal import read_recent_trades

_DAY_CSV = CONFIG.logging.log_dir / "daily_report.csv"
_DAY_DIR = CONFIG.logging.log_dir / "daily_reports"


def _summarize_trades(trades: list[dict]) -> dict:
    trades_count = len(trades)
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    losses = trades_count - wins
    net_pnl = round(sum(t.get("pnl", 0.0) or 0.0 for t in trades), 2)
    win_rate = round(wins / trades_count * 100, 2) if trades_count else 0.0
    gross_profit = round(sum(t.get("pnl", 0.0) for t in trades if (t.get("pnl") or 0) > 0), 2)
    gross_loss = round(abs(sum(t.get("pnl", 0.0) for t in trades if (t.get("pnl") or 0) <= 0)), 2)

    # individual trade details
    trade_details = []
    for t in trades:
        trade_details.append({
            "event": t.get("event"),
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "quantity": t.get("quantity"),
            "pnl": t.get("pnl"),
            "reason": t.get("reason"),
            "strike": t.get("strike"),
            "option_type": t.get("option_type"),
            "expiry": t.get("expiry"),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
        })

    return {
        "date": trades[0].get("exit_time", "")[:10] if trades else "unknown",
        "trades_count": trades_count,
        "wins": wins,
        "losses": losses,
        "net_pnl": net_pnl,
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "trades": trade_details,
    }


def build_daily_report(max_trades: int = 5000) -> dict[str, dict]:
    trades = read_recent_trades(max_trades)

    by_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        exit_time = t.get("exit_time") or t.get("ts") or ""
        date_str = exit_time[:10] if len(exit_time) >= 10 else "unknown"
        by_date[date_str].append(t)

    report: dict[str, dict] = {}
    for date_str, day_trades in sorted(by_date.items()):
        report[date_str] = _summarize_trades(day_trades)

    return report


def cleanup_old_reports(retention_days: int = 30) -> tuple[list[str], list[str]]:
    _DAY_DIR.mkdir(parents=True, exist_ok=True)
    _DAY_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_dates = sorted([p.stem for p in _DAY_DIR.glob("*.json") if p.is_file()])
    cutoff_count = max(0, len(all_dates) - retention_days)
    to_delete = all_dates[:cutoff_count]

    deleted_json = []
    for date_str in to_delete:
        day_file = _DAY_DIR / f"{date_str}.json"
        if day_file.exists():
            day_file.unlink()
            deleted_json.append(date_str)

    if to_delete:
        rows = []
        if _DAY_CSV.exists():
            with open(_DAY_CSV, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("date", "") not in to_delete:
                        rows.append(row)
        with open(_DAY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "trades_count", "wins", "losses", "net_pnl", "win_rate_pct", "gross_profit", "gross_loss"])
            writer.writeheader()
            writer.writerows(rows)
        deleted_csv = to_delete
    else:
        deleted_csv = []

    return deleted_json, deleted_csv


def save_daily_report() -> tuple[Path, Path]:
    report = build_daily_report()

    _DAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_DAY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "trades_count", "wins", "losses", "net_pnl", "win_rate_pct", "gross_profit", "gross_loss"])
        for day, summary in sorted(report.items()):
            writer.writerow([
                summary["date"],
                summary["trades_count"],
                summary["wins"],
                summary["losses"],
                summary["net_pnl"],
                summary["win_rate_pct"],
                summary["gross_profit"],
                summary["gross_loss"],
            ])

    _DAY_DIR.mkdir(parents=True, exist_ok=True)
    for day, summary in report.items():
        day_file = _DAY_DIR / f"{day}.json"
        day_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    deleted_json, deleted_csv = cleanup_old_reports(retention_days=30)
    if deleted_json:
        print(f"Cleaned up {len(deleted_json)} old daily reports (kept last 30 days)")

    return _DAY_CSV, _DAY_DIR


def main() -> None:
    csv_path, json_dir = save_daily_report()
    print(f"Daily CSV saved to: {csv_path}")
    print(f"Daily JSON files saved to: {json_dir}")
    print()

    report = build_daily_report()
    for day, summary in sorted(report.items()):
        print(f"{day}: {summary['trades_count']} trades, {summary['wins']}W/{summary['losses']}L, "
              f"PnL Rs.{summary['net_pnl']:,.2f}, WinRate {summary['win_rate_pct']}%")


if __name__ == "__main__":
    main()
