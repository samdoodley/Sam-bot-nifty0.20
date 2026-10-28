"""
session_validation.py
=====================
Shows every trading session from the trade journal and validates each
session against the session rules defined in CONFIG.strategy:

    Session boundaries (IST):
        S1 = 12:30 - 13:50   S2 = 13:50 - 14:20
        S3 = 14:20 - 15:00   S4 = 15:00 - 15:25 (force square-off)

    Checks per session (per date):
        - time boundary: entry time falls inside the assigned session window
        - trade cap: trades <= session_trades_max[session]
        - profit target: no trades logged after the session profit target was hit

Usage:
    python session_validation.py               # print report to stdout
    python session_validation.py --json        # print machine-readable JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import CONFIG
from trade_journal import _get_log_dir


SESSION_WINDOWS: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    1: ((12, 30), (13, 50)),
    2: ((13, 50), (14, 20)),
    3: ((14, 20), (15, 0)),
    4: ((15, 0), (15, 25)),
}


def _session_of(h: int, m: int) -> int:
    """Return session number (1-4) for an IST hour/minute, 0 if pre-session."""
    if (h, m) < (12, 30):
        return 0
    if (h, m) < (13, 50):
        return 1
    if (h, m) < (14, 20):
        return 2
    if (h, m) < (15, 0):
        return 3
    return 4


def _in_window(h: int, m: int, session: int) -> bool:
    start, end = SESSION_WINDOWS[session]
    return start <= (h, m) < end


def _parse_trades() -> list[dict]:
    path = _get_log_dir() / "trades.jsonl"
    if not path.exists():
        return []
    trades: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("event") != "EXIT" or "pnl" not in t:
                continue
            ts_str = t.get("entry_time") or t.get("ts")
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except Exception:
                continue
            # entry_time is logged as naive IST; fall back to ts (UTC -> IST)
            if dt.tzinfo is not None:
                from zoneinfo import ZoneInfo

                dt = dt.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
            t["_dt"] = dt
            t["_date"] = dt.date().isoformat()
            t["_session"] = _session_of(dt.hour, dt.minute)
            trades.append(t)
    return trades


def validate_session(
    session: int, trades: list[dict], cfg: Any
) -> dict[str, Any]:
    caps: dict[int, int] = dict(cfg.session_trades_max)
    targets: dict[int, float] = dict(cfg.session_profit_target)

    issues: list[str] = []
    total_pnl = 0.0
    wins = losses = 0
    target_hit_at: int | None = None
    cumulative = 0.0

    for idx, t in enumerate(trades):
        pnl = float(t.get("pnl", 0))
        total_pnl += pnl
        cumulative += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        # time boundary check (session 0 = pre-session, always flagged)
        if session == 0:
            issues.append(f"trade {idx+1} entry {t['_dt'].time()} before session start (pre-session)")
        elif not _in_window(t["_dt"].hour, t["_dt"].minute, session):
            issues.append(
                f"trade {idx+1} entry {t['_dt'].time()} outside session {session} window"
            )

        # profit target check: no trades after target reached
        if target_hit_at is None and targets.get(session, 0) > 0 and cumulative >= targets[session]:
            target_hit_at = idx + 1
        elif target_hit_at is not None:
            issues.append(
                f"trade {idx+1} logged after profit target hit at trade {target_hit_at}"
            )

    cap = caps.get(session)
    cap_ok = cap is None or len(trades) <= cap
    if not cap_ok:
        issues.append(f"{len(trades)} trades exceeds session cap of {cap}")

    return {
        "session": session,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round((wins / len(trades) * 100) if trades else 0, 1),
        "cap": cap,
        "cap_ok": cap_ok,
        "target": targets.get(session, 0),
        "target_hit_at": target_hit_at,
        "issues": issues,
        "valid": not issues,
    }


def build_report() -> list[dict[str, Any]]:
    trades = _parse_trades()
    cfg = CONFIG.strategy
    by_date: dict[str, dict[int, list[dict]]] = {}
    for t in trades:
        by_date.setdefault(t["_date"], {}).setdefault(t["_session"], []).append(t)

    report: list[dict[str, Any]] = []
    for date in sorted(by_date):
        sessions = by_date[date]
        date_entries: list[dict[str, Any]] = []
        for session in sorted(sessions):
            session_trades = sessions[session]
            validation = validate_session(session, session_trades, cfg)
            validation["date"] = date
            date_entries.append(validation)
        report.append({"date": date, "sessions": date_entries})
    return report


def _print_text(report: list[dict[str, Any]]) -> None:
    for day in report:
        print(f"\n=== {day['date']} ===")
        for s in day["sessions"]:
            status = "VALID" if s["valid"] else "INVALID"
            print(
                f"  Session {s['session']}: {status} | "
                f"trades={s['trades']} wins={s['wins']} losses={s['losses']} "
                f"pnl=₹{s['total_pnl']:,.2f} win%={s['win_rate']:.1f} | "
                f"cap={s['cap']} ({'ok' if s['cap_ok'] else 'EXCEEDED'}) | "
                f"target=₹{s['target']:.0f}"
            )
            for issue in s["issues"]:
                print(f"    - {issue}")


def _print_json(report: list[dict[str, Any]]) -> None:
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show and validate every trading session.")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        _print_json(report)
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
