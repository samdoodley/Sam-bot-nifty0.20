"""
generate_report.py
==================
Generates a comprehensive 1-year backtest report with daily, weekly,
and monthly breakdowns using the NIFTY strategy with trailing SL
and 3x leverage. Produces a PDF with trade counts, PnL, win rates,
and trade details.

Usage:
    NIFTY_BOT_MODE=PAPER KITE_API_KEY=<key> KITE_API_SECRET=<secret> python generate_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib.colors import HexColor

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from kite_api import KiteAPI
from strategy import StrategyEngine
from risk_manager import RiskManager
from utils import Candle, TradeSide

REPORT_DIR = Path(__file__).parent / "reports"
LEVERAGE = 1
CAPITAL_PER_TRADE = 3_00_000.0
STARTING_EQUITY = CAPITAL_PER_TRADE * LEVERAGE
TRAIL_TRIGGER_POINTS = 2.0
MAX_TRADES_PER_DAY = 10


@dataclass
class TradeRecord:
    date: str
    entry_time: str
    exit_time: str
    side: str
    entry: float
    exit: float
    index_points_pnl: float
    rupee_pnl: float
    reason: str


@dataclass
class DailyResult:
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trades_list: list[TradeRecord] = field(default_factory=list)


@dataclass
class WeeklyResult:
    week: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0


@dataclass
class MonthlyResult:
    month: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades_detail: list[dict] = field(default_factory=list)


def download_historical_data(kite: KiteAPI, token: int, from_dt: datetime, to_dt: datetime, interval: str) -> list[dict]:
    bars = kite.historical_data(token, from_dt, to_dt, interval)
    return bars


def bars_to_candles(bars: list[dict]) -> list[Candle]:
    candles: list[Candle] = []
    for b in bars:
        ts = b["date"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        candles.append(Candle(
            ts=ts,
            open=float(b["open"]),
            high=float(b["high"]),
            low=float(b["low"]),
            close=float(b["close"]),
            volume=int(b.get("volume", 0)),
        ))
    candles.sort(key=lambda c: c.ts)
    return candles


def compute_vwap_series(candles: list[Candle]) -> list[float]:
    vwap_vals = []
    cum_pv, cum_vol, day = 0.0, 0.0, None
    for c in candles:
        if day != c.ts.date():
            day = c.ts.date()
            cum_pv, cum_vol = 0.0, 0.0
        vol = max(c.volume, 1)
        cum_pv += c.close * vol
        cum_vol += vol
        vwap_vals.append(cum_pv / cum_vol)
    return vwap_vals


def run_daily_backtest(kite: KiteAPI, spot_token: int, target_date: date) -> DailyResult:
    """Run backtest for a single day and return results."""
    date_str = target_date.strftime("%Y-%m-%d")
    result = DailyResult(date=date_str)

    start_dt = datetime.combine(target_date, dtime(9, 15))
    end_dt = datetime.combine(target_date, dtime(15, 30))

    try:
        bars_5m = download_historical_data(kite, spot_token, start_dt, end_dt, "5minute")
    except Exception:
        return result

    if not bars_5m:
        return result

    candles_5m = bars_to_candles(bars_5m)
    vwap_series = compute_vwap_series(candles_5m)

    try:
        bars_15m = download_historical_data(kite, spot_token, start_dt, end_dt, "15minute")
    except Exception:
        bars_15m = []

    candles_15m = bars_to_candles(bars_15m) if bars_15m else []

    engine = StrategyEngine()
    equity = STARTING_EQUITY
    trades_today = 0
    wins_today = 0
    losses_today = 0
    pnl_today = 0.0

    open_trade = None

    for i in range(len(candles_5m)):
        cur_time = candles_5m[i].ts.time()
        window_5m = candles_5m[: i + 1]

        if open_trade is not None:
            px = candles_5m[i].close
            hit_target = (
                px >= open_trade["target"]
                if open_trade["side"] == TradeSide.LONG
                else px <= open_trade["target"]
            )
            hit_sl = (
                px <= open_trade["sl"]
                if open_trade["side"] == TradeSide.LONG
                else px >= open_trade["sl"]
            )
            force_exit = cur_time >= dtime(15, 20)

            # Trailing SL update
            if not hit_target and not hit_sl and not force_exit:
                profit = (
                    px - open_trade["entry"]
                    if open_trade["side"] == TradeSide.LONG
                    else open_trade["entry"] - px
                )
                if profit >= TRAIL_TRIGGER_POINTS:
                    if open_trade["side"] == TradeSide.LONG:
                        if profit >= TRAIL_TRIGGER_POINTS + 1.0:
                            open_trade["sl"] = max(open_trade["sl"], open_trade["entry"] + 2.0)
                        else:
                            open_trade["sl"] = max(open_trade["sl"], open_trade["entry"])
                    else:
                        if profit >= TRAIL_TRIGGER_POINTS + 1.0:
                            open_trade["sl"] = min(open_trade["sl"], open_trade["entry"] - 2.0)
                        else:
                            open_trade["sl"] = min(open_trade["sl"], open_trade["entry"])

            if hit_target or hit_sl or force_exit:
                exit_price = (
                    open_trade["target"] if hit_target
                    else (open_trade["sl"] if hit_sl else px)
                )
                reason = "TARGET" if hit_target else ("SL" if hit_sl else "FORCE_SQUARE_OFF")
                index_points = (
                    exit_price - open_trade["entry"]
                    if open_trade["side"] == TradeSide.LONG
                    else open_trade["entry"] - exit_price
                )
                delta_approx = 0.45
                lot_size = 75
                rupee_pnl = index_points * delta_approx * lot_size * LEVERAGE - 40.0

                result.trades_list.append(TradeRecord(
                    date=date_str,
                    entry_time=open_trade["entry_time"].strftime("%H:%M"),
                    exit_time=candles_5m[i].ts.strftime("%H:%M"),
                    side=open_trade["side"].value,
                    entry=round(open_trade["entry"], 2),
                    exit=round(exit_price, 2),
                    index_points_pnl=round(index_points, 2),
                    rupee_pnl=round(rupee_pnl, 2),
                    reason=reason,
                ))

                pnl_today += rupee_pnl
                trades_today += 1
                if rupee_pnl > 0:
                    wins_today += 1
                else:
                    losses_today += 1

                open_trade = None

                if trades_today >= MAX_TRADES_PER_DAY:
                    break
            continue

        # Look for new entries
        if not (dtime(9, 15) <= cur_time <= dtime(15, 20)):
            continue
        if trades_today >= MAX_TRADES_PER_DAY:
            break
        if len(window_5m) < 15:
            continue

        bucket = candles_5m[i].ts.replace(
            minute=(candles_5m[i].ts.minute // 15) * 15, second=0, microsecond=0,
        )
        visible_15m = [c for c in candles_15m if c.ts < bucket]
        vwap_slice = vwap_series[: i + 1]

        signal = engine.evaluate(window_5m, visible_15m, vwap_slice, symbol="NIFTY")
        if signal.side == TradeSide.NONE:
            continue

        slippage = 0.5
        entry_px = signal.entry_price + (slippage if signal.side == TradeSide.LONG else -slippage)
        initial_sl = abs(signal.stop_loss - signal.entry_price)
        open_trade = {
            "side": signal.side,
            "entry": entry_px,
            "sl": signal.stop_loss,
            "target": signal.target,
            "initial_sl": initial_sl,
            "entry_time": candles_5m[i].ts,
        }

    result.trades = trades_today
    result.wins = wins_today
    result.losses = losses_today
    result.total_pnl = pnl_today
    result.win_rate = (wins_today / trades_today * 100) if trades_today > 0 else 0.0

    return result


def run_monthly_backtest(kite: KiteAPI, spot_token: int, year: int, month: int) -> MonthlyResult:
    month_str = f"{year}-{month:02d}"
    result = MonthlyResult(month=month_str)

    start_date = datetime(year, month, 1).date()
    if month == 12:
        end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)

    # Collect daily results
    current_date = start_date
    all_trades: list[TradeRecord] = []

    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        daily = run_daily_backtest(kite, spot_token, current_date)
        all_trades.extend(daily.trades_list)
        current_date += timedelta(days=1)

    result.trades = len(all_trades)
    result.wins = sum(1 for t in all_trades if t.rupee_pnl > 0)
    result.losses = sum(1 for t in all_trades if t.rupee_pnl <= 0)
    result.total_pnl = sum(t.rupee_pnl for t in all_trades)
    result.win_rate = (result.wins / result.trades * 100) if result.trades > 0 else 0.0

    if result.wins > 0:
        result.avg_win = sum(t.rupee_pnl for t in all_trades if t.rupee_pnl > 0) / result.wins
    if result.losses > 0:
        result.avg_loss = sum(t.rupee_pnl for t in all_trades if t.rupee_pnl <= 0) / result.losses

    # Equity curve
    equity = STARTING_EQUITY
    equity_curve = [equity]
    for t in all_trades:
        equity += t.rupee_pnl
        equity_curve.append(equity)
    result.equity_curve = equity_curve

    # Max drawdown
    peak = STARTING_EQUITY
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)
    result.max_drawdown = round(max_dd, 2)

    # Store trades_detail for PDF
    for t in all_trades:
        result.trades_detail.append({
            "entry_time": f"{t.date} {t.entry_time}",
            "exit_time": f"{t.date} {t.exit_time}",
            "side": t.side,
            "entry": t.entry,
            "exit": t.exit,
            "index_points_pnl": t.index_points_pnl,
            "rupee_pnl": t.rupee_pnl,
            "reason": t.reason,
        })

    return result


def generate_pdf_report(monthly_results: list[MonthlyResult], daily_results: list[DailyResult], output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=4,
        textColor=HexColor("#1a1a2e"),
    )
    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=4,
        textColor=HexColor("#16213e"),
    )
    h3_style = ParagraphStyle(
        "CustomH3",
        parent=styles["Heading3"],
        fontSize=11,
        spaceBefore=6,
        spaceAfter=3,
        textColor=HexColor("#0f3460"),
    )
    cell_style = ParagraphStyle(
        "CellStyle",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
    )
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    green_style = ParagraphStyle(
        "GreenCell",
        parent=cell_style,
        textColor=HexColor("#006400"),
    )
    red_style = ParagraphStyle(
        "RedCell",
        parent=cell_style,
        textColor=HexColor("#8B0000"),
    )
    bold_cell = ParagraphStyle(
        "BoldCell",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    elements = []

    # Title
    elements.append(Paragraph("NIFTY Weekly Options Bot — 1-Year Backtest Report", title_style))
    elements.append(Paragraph(
        f"Capital: Rs.{CAPITAL_PER_TRADE:,.0f}/trade (3x Leverage = Rs.{STARTING_EQUITY:,.0f}) "
        f"| Max {MAX_TRADES_PER_DAY} trades/day "
        f"| SL: 1:2 RR + Step Trailing SL ({TRAIL_TRIGGER_POINTS}pts trigger, entry+B/E at 3pts)",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 4 * mm))

    # === MONTHLY SUMMARY ===
    elements.append(Paragraph("Monthly Summary", h2_style))

    monthly_header = [
        Paragraph("<b>Month</b>", header_style),
        Paragraph("<b>Trades</b>", header_style),
        Paragraph("<b>Wins</b>", header_style),
        Paragraph("<b>Losses</b>", header_style),
        Paragraph("<b>Win Rate</b>", header_style),
        Paragraph("<b>Total PnL (Rs.)</b>", header_style),
        Paragraph("<b>Avg Win</b>", header_style),
        Paragraph("<b>Avg Loss</b>", header_style),
    ]
    monthly_data = [monthly_header]

    for res in monthly_results:
        pnl_style = green_style if res.total_pnl > 0 else red_style
        monthly_data.append([
            Paragraph(res.month, cell_style),
            Paragraph(str(res.trades), cell_style),
            Paragraph(str(res.wins), green_style),
            Paragraph(str(res.losses), red_style),
            Paragraph(f"{res.win_rate:.1f}%", cell_style),
            Paragraph(f"{res.total_pnl:,.2f}", pnl_style),
            Paragraph(f"{res.avg_win:,.2f}", green_style if res.avg_win > 0 else cell_style),
            Paragraph(f"{res.avg_loss:,.2f}", red_style if res.avg_loss < 0 else cell_style),
        ])

    # Totals row
    total_trades = sum(r.trades for r in monthly_results)
    total_wins = sum(r.wins for r in monthly_results)
    total_losses = sum(r.losses for r in monthly_results)
    total_pnl = sum(r.total_pnl for r in monthly_results)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_win = sum(r.avg_win for r in monthly_results if r.avg_win > 0) / max(1, sum(1 for r in monthly_results if r.avg_win > 0))
    avg_loss = sum(r.avg_loss for r in monthly_results if r.avg_loss < 0) / max(1, sum(1 for r in monthly_results if r.avg_loss < 0))

    monthly_data.append([
        Paragraph("<b>TOTAL</b>", bold_cell),
        Paragraph(f"<b>{total_trades}</b>", bold_cell),
        Paragraph(f"<b>{total_wins}</b>", bold_cell),
        Paragraph(f"<b>{total_losses}</b>", bold_cell),
        Paragraph(f"<b>{overall_wr:.1f}%</b>", bold_cell),
        Paragraph(f"<b>{total_pnl:,.2f}</b>", bold_cell),
        Paragraph(f"<b>{avg_win:,.2f}</b>", bold_cell),
        Paragraph(f"<b>{avg_loss:,.2f}</b>", bold_cell),
    ])

    monthly_table = Table(monthly_data, colWidths=[22*mm, 14*mm, 12*mm, 12*mm, 14*mm, 24*mm, 20*mm, 20*mm])
    monthly_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [HexColor("#f8f8f8"), colors.white]),
        ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e8e8e8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(monthly_table)
    elements.append(Spacer(1, 6 * mm))

    # === WEEKLY SUMMARY ===
    elements.append(Paragraph("Weekly Summary", h2_style))

    # Group daily results by ISO week
    weekly_map: dict[str, WeeklyResult] = {}
    for dr in daily_results:
        if dr.trades == 0:
            continue
        dt = datetime.strptime(dr.date, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        if week_key not in weekly_map:
            weekly_map[week_key] = WeeklyResult(week=week_key)
        wr = weekly_map[week_key]
        wr.trades += dr.trades
        wr.wins += dr.wins
        wr.losses += dr.losses
        wr.total_pnl += dr.total_pnl

    weekly_header = [
        Paragraph("<b>Week</b>", header_style),
        Paragraph("<b>Trades</b>", header_style),
        Paragraph("<b>Wins</b>", header_style),
        Paragraph("<b>Losses</b>", header_style),
        Paragraph("<b>Win Rate</b>", header_style),
        Paragraph("<b>Total PnL (Rs.)</b>", header_style),
    ]
    weekly_data = [weekly_header]

    for wk in sorted(weekly_map.keys()):
        wr = weekly_map[wk]
        wr.win_rate = (wr.wins / wr.trades * 100) if wr.trades > 0 else 0.0
        pnl_style = green_style if wr.total_pnl > 0 else red_style
        weekly_data.append([
            Paragraph(wr.week, cell_style),
            Paragraph(str(wr.trades), cell_style),
            Paragraph(str(wr.wins), green_style),
            Paragraph(str(wr.losses), red_style),
            Paragraph(f"{wr.win_rate:.1f}%", cell_style),
            Paragraph(f"{wr.total_pnl:,.2f}", pnl_style),
        ])

    weekly_table = Table(weekly_data, colWidths=[22*mm, 16*mm, 12*mm, 12*mm, 16*mm, 28*mm])
    weekly_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(weekly_table)
    elements.append(Spacer(1, 6 * mm))

    # === DAILY DETAIL (last 30 days) ===
    elements.append(Paragraph("Daily Detail (Last 30 Trading Days)", h2_style))

    recent_daily = [dr for dr in daily_results if dr.trades > 0][-30:]
    daily_header = [
        Paragraph("<b>Date</b>", header_style),
        Paragraph("<b>Trades</b>", header_style),
        Paragraph("<b>Wins</b>", header_style),
        Paragraph("<b>Losses</b>", header_style),
        Paragraph("<b>Win Rate</b>", header_style),
        Paragraph("<b>PnL (Rs.)</b>", header_style),
    ]
    daily_data = [daily_header]
    for dr in recent_daily:
        pnl_style = green_style if dr.total_pnl > 0 else red_style
        daily_data.append([
            Paragraph(dr.date, cell_style),
            Paragraph(str(dr.trades), cell_style),
            Paragraph(str(dr.wins), green_style),
            Paragraph(str(dr.losses), red_style),
            Paragraph(f"{dr.win_rate:.1f}%", cell_style),
            Paragraph(f"{dr.total_pnl:,.2f}", pnl_style),
        ])

    daily_table = Table(daily_data, colWidths=[24*mm, 14*mm, 12*mm, 12*mm, 16*mm, 28*mm])
    daily_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(daily_table)
    elements.append(Spacer(1, 6 * mm))

    # === MONTHLY TRADE DETAILS ===
    elements.append(Paragraph("Monthly Trade Details", h2_style))

    for res in monthly_results:
        if res.trades == 0:
            continue
        elements.append(Paragraph(f"<b>{res.month}</b> — {res.trades} trades, PnL: Rs.{res.total_pnl:,.2f}, Win Rate: {res.win_rate:.1f}%", cell_style))

        trade_header = [
            Paragraph("<b>Date</b>", header_style),
            Paragraph("<b>Entry</b>", header_style),
            Paragraph("<b>Exit</b>", header_style),
            Paragraph("<b>Side</b>", header_style),
            Paragraph("<b>Entry Px</b>", header_style),
            Paragraph("<b>Exit Px</b>", header_style),
            Paragraph("<b>PnL (Rs)</b>", header_style),
            Paragraph("<b>Reason</b>", header_style),
        ]
        trade_data = [trade_header]
        for t in res.trades_detail:
            pnl_style = green_style if t["rupee_pnl"] > 0 else red_style
            trade_data.append([
                Paragraph(t["entry_time"][:10], cell_style),
                Paragraph(t["entry_time"][11:], cell_style),
                Paragraph(t["exit_time"][11:], cell_style),
                Paragraph(t["side"], cell_style),
                Paragraph(f"{t['entry']:.2f}", cell_style),
                Paragraph(f"{t['exit']:.2f}", cell_style),
                Paragraph(f"{t['rupee_pnl']:,.2f}", pnl_style),
                Paragraph(t["reason"], cell_style),
            ])

        trade_table = Table(trade_data, colWidths=[16*mm, 12*mm, 12*mm, 10*mm, 16*mm, 16*mm, 18*mm, 16*mm])
        trade_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        elements.append(trade_table)
        elements.append(Spacer(1, 3 * mm))

    doc.build(elements)
    print(f"PDF report saved to: {output_path}")


def main() -> None:
    print("=" * 60)
    print("NIFTY Bot — 1-Year Comprehensive Backtest Report")
    print(f"Capital per Trade: Rs.{CAPITAL_PER_TRADE:,.0f}")
    print(f"Leverage: {LEVERAGE}x (Total Equity: Rs.{STARTING_EQUITY:,.0f})")
    print(f"Max Trades/Day: {MAX_TRADES_PER_DAY}")
    print(f"SL: 1:2 RR | Trailing SL: {TRAIL_TRIGGER_POINTS}pts trigger, entry+B/E at 3pts")
    print("=" * 60)

    kite = KiteAPI()
    kite.login()

    spot_token = kite.instruments("NSE")
    spot_token_val = None
    for inst in spot_token:
        if inst.get("tradingsymbol") == "NIFTY 50" and inst.get("segment") == "INDICES":
            spot_token_val = int(inst["instrument_token"])
            break

    if spot_token_val is None:
        print("ERROR: Could not find NIFTY 50 spot token.")
        sys.exit(1)

    print(f"NIFTY 50 Spot Token: {spot_token_val}")

    # Generate months for last 1 year
    end_month = datetime.now().month
    end_year = datetime.now().year
    months: list[tuple[int, int]] = []

    for y in range(end_year - 1, end_year + 1):
        start_m = 1 if y > end_year - 1 else 1
        end_m = 12 if y < end_year else end_month
        for m in range(start_m, end_m + 1):
            if y == end_year and m > end_month:
                continue
            months.append((y, m))

    print(f"Running backtest for {len(months)} months...")
    monthly_results: list[MonthlyResult] = []
    all_daily_results: list[DailyResult] = []

    for year, month in months:
        month_str = f"{year}-{month:02d}"
        print(f"Processing {month_str}...", end=" ")

        # Run daily backtests for this month
        start_date = datetime(year, month, 1).date()
        if month == 12:
            end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)

        current_date = start_date
        month_trades = 0
        month_wins = 0
        month_losses = 0
        month_pnl = 0.0
        month_trades_detail = []

        while current_date <= end_date:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            daily = run_daily_backtest(kite, spot_token_val, current_date)
            all_daily_results.append(daily)
            month_trades += daily.trades
            month_wins += daily.wins
            month_losses += daily.losses
            month_pnl += daily.total_pnl
            month_trades_detail.extend(daily.trades_list)
            current_date += timedelta(days=1)

        result = MonthlyResult(month=month_str)
        result.trades = month_trades
        result.wins = month_wins
        result.losses = month_losses
        result.total_pnl = month_pnl
        result.win_rate = (month_wins / month_trades * 100) if month_trades > 0 else 0.0

        if month_wins > 0:
            result.avg_win = sum(t.rupee_pnl for t in month_trades_detail if t.rupee_pnl > 0) / month_wins
        if month_losses > 0:
            result.avg_loss = sum(t.rupee_pnl for t in month_trades_detail if t.rupee_pnl <= 0) / month_losses

        # Equity curve
        equity = STARTING_EQUITY
        equity_curve = [equity]
        for t in month_trades_detail:
            equity += t.rupee_pnl
            equity_curve.append(equity)
        result.equity_curve = equity_curve

        # Max drawdown
        peak = STARTING_EQUITY
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak else 0
            max_dd = max(max_dd, dd)
        result.max_drawdown = round(max_dd, 2)

        # Store trades_detail
        for t in month_trades_detail:
            result.trades_detail.append({
                "entry_time": f"{t.date} {t.entry_time}",
                "exit_time": f"{t.date} {t.exit_time}",
                "side": t.side,
                "entry": t.entry,
                "exit": t.exit,
                "index_points_pnl": t.index_points_pnl,
                "rupee_pnl": t.rupee_pnl,
                "reason": t.reason,
            })

        monthly_results.append(result)
        print(f"trades={month_trades} pnl=Rs.{month_pnl:,.2f} wr={result.win_rate:.1f}%")

    # Generate PDF
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"backtest_report_1yr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    generate_pdf_report(monthly_results, all_daily_results, output_path)

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_trades = sum(r.trades for r in monthly_results)
    total_wins = sum(r.wins for r in monthly_results)
    total_losses = sum(r.losses for r in monthly_results)
    total_pnl = sum(r.total_pnl for r in monthly_results)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"Total Months: {len(monthly_results)}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Wins: {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print(f"Total PnL: Rs.{total_pnl:,.2f}")
    print(f"Starting Equity: Rs.{STARTING_EQUITY:,.0f}")
    print(f"Ending Equity: Rs.{STARTING_EQUITY + total_pnl:,.2f}")


if __name__ == "__main__":
    main()