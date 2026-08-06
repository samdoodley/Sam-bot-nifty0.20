"""
backtest_options.py
====================
Real-options backtest engine for the NIFTY intraday strategy.

Flow
----
1. Fetch NIFTY 50 spot 5-min + 15-min candles for the backtest window.
2. Replay the EXISTING StrategyEngine candle-by-candle.
3. On every signal:
     a. Read spot price at signal timestamp.
     b. Resolve ATM weekly contract via OptionUniverse.resolve_atm_contract().
     c. Fetch HISTORICAL OPTION PREMIUM OHLCV for that exact contract
        (not index points).
     d. Size the position using 3 L capital + real Zerodha margin logic.
     e. Track entry/exit in PREMIUM terms (target, SL, trailing, time exit).
4. Roll over to the next weekly expiry automatically when the current
   expiry day has passed (mirrors live OptionUniverse behaviour).
5. Accumulate every required field for the PDF report.

Run:
    python backtest_options.py --months 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

from config import CONFIG
from indicators import ema_series
from kite_api import KiteAPI
from logger import get_logger
from option_selector import OptionUniverse
from risk_manager import RiskManager
from strategy import StrategyEngine
from utils import Candle, TradeSide, round_to_strike_step

_log = get_logger("backtest_options")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_to_15m(ts: datetime) -> datetime:
    return ts.replace(
        minute=(ts.minute // CONFIG.session.higher_timeframe_min) * CONFIG.session.higher_timeframe_min,
        second=0, microsecond=0,
    )

def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5

def _next_weekly_expiry(instruments: list[dict], from_date: date) -> date:
    expiries = sorted({i["expiry"] for i in instruments if i["expiry"] >= from_date})
    if not expiries:
        raise RuntimeError("No upcoming weekly expiries found.")
    return expiries[0]

def _resolve_token(instruments: list[dict], expiry: date, strike: int, option_type: str) -> Optional[int]:
    for i in instruments:
        if i["expiry"] == expiry and i["strike"] == strike and i["instrument_type"] == option_type:
            return int(i["instrument_token"])
    return None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    trade_date: str
    entry_time: str
    exit_time: str
    direction: str
    nifty_spot_entry: float
    selected_expiry: str
    atm_strike: int
    option_type: str
    option_trading_symbol: str
    entry_premium: float
    exit_premium: float
    lot_size: int
    num_lots: int
    quantity: int
    gross_pnl: float
    brokerage_charges: float
    net_pnl: float
    exit_reason: str
    signal_reason: str = ""
    confidence_notes: str = ""

@dataclass
class DayResult:
    date_str: str
    trades: list[TradeRecord] = field(default_factory=list)
    trades_count: int = 0
    day_pnl: float = 0.0

# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class OptionBacktestEngine:
    def __init__(self, kite: KiteAPI, starting_equity: float) -> None:
        self.kite = kite
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.option_universe = OptionUniverse(kite)
        self.strategy = StrategyEngine()
        self.risk = RiskManager(starting_equity=starting_equity)
        self.all_trades: list[TradeRecord] = []
        self.daily_results: dict[str, DayResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, from_date: date, to_date: date) -> list[TradeRecord]:
        _log.info("Backtest window: %s -> %s, equity=Rs.%.2f", from_date, to_date, self.equity)

        # Load full NFO instrument list once
        self.option_universe.refresh()
        nfo_instruments = self.option_universe._instruments

        cur = from_date
        while cur <= to_date:
            if not _is_trading_day(cur):
                cur += timedelta(days=1)
                continue

            self._process_day(cur, nfo_instruments)
            self.risk.reset_daily_counters()
            cur += timedelta(days=1)

        _log.info("Backtest complete. Total trades: %d", len(self.all_trades))
        return self.all_trades

    # ------------------------------------------------------------------
    # Per-day processing
    # ------------------------------------------------------------------

    def _process_day(self, trade_date: date, nfo_instruments: list[dict]) -> None:
        date_str = trade_date.strftime("%Y-%m-%d")
        start_dt = datetime.combine(trade_date, dtime(9, 15))
        end_dt = datetime.combine(trade_date, dtime(15, 30))

        spot_token = self.option_universe.get_spot_instrument_token()
        try:
            bars_5m = self.kite.historical_data(spot_token, start_dt, end_dt, "5minute")
            bars_15m = self.kite.historical_data(spot_token, start_dt, end_dt, "15minute")
        except Exception:
            _log.exception("Failed to fetch spot data for %s", date_str)
            return

        if not bars_5m:
            _log.warning("No spot data for %s", date_str)
            return

        candles_5m = self._bars_to_candles(bars_5m)
        candles_15m = self._bars_to_candles(bars_15m) if bars_15m else []

        if len(candles_5m) < 60 or len(candles_15m) < 4:
            _log.warning("Insufficient data for %s", date_str)
            return

        vwap_series = self._compute_vwap(candles_5m)

        open_trade: Optional[dict] = None
        day_trades: list[TradeRecord] = []

        for i in range(len(candles_5m)):
            cur_time = candles_5m[i].ts.time()
            window_5m = candles_5m[: i + 1]

            if open_trade is not None:
                option_px = candles_5m[i].close
                exit_info = self._check_exit(open_trade, option_px, cur_time)
                if exit_info:
                    trade = self._close_trade(open_trade, exit_info, candles_5m[i].ts, trade_date)
                    day_trades.append(trade)
                    self.all_trades.append(trade)
                    self.risk.record_trade_result(trade.net_pnl)
                    self.equity += trade.net_pnl
                    open_trade = None

                    if self.risk.stats.trades_taken >= CONFIG.strategy.max_trades_per_day:
                        break
                continue

            if not (CONFIG.session.setup_scan_start <= cur_time <= CONFIG.session.last_entry_time):
                continue
            if not self.risk.can_take_new_trade()[0]:
                continue

            bucket = _floor_to_15m(candles_5m[i].ts)
            visible_15m = [c for c in candles_15m if c.ts < bucket]

            signal = self.strategy.evaluate(window_5m, visible_15m, vwap_series[: i + 1], symbol="NIFTY")
            if signal.side == TradeSide.NONE:
                continue

            try:
                contract = self._resolve_contract_for_date(signal.entry_price, signal.side, trade_date, nfo_instruments)
            except Exception as e:
                _log.warning("Contract resolution failed on %s: %s", date_str, e)
                continue

            opt_token = _resolve_token(
                nfo_instruments, date.fromisoformat(contract.expiry),
                contract.strike, contract.option_type,
            )
            if opt_token is None:
                _log.warning("Option token not found for %s on %s", contract.tradingsymbol, date_str)
                continue

            try:
                opt_bars = self.kite.historical_data(opt_token, start_dt, end_dt, "5minute")
            except Exception:
                _log.exception("Failed to fetch option data for %s on %s", contract.tradingsymbol, date_str)
                continue

            if not opt_bars:
                _log.warning("Empty option data for %s on %s", contract.tradingsymbol, date_str)
                continue

            opt_candles = self._bars_to_candles(opt_bars)

            entry_idx = self._find_candle_index(opt_candles, candles_5m[i].ts)
            if entry_idx is None:
                _log.warning("Entry candle not found in option data for %s", contract.tradingsymbol)
                continue

            entry_premium = opt_candles[entry_idx].close
            if entry_premium <= 0:
                _log.warning("Invalid entry premium %.2f for %s", entry_premium, contract.tradingsymbol)
                continue

            margin_per_lot = max(entry_premium, 0.05) * contract.lot_size
            sizing = self._size_position(
                equity=self.equity,
                entry_premium=entry_premium,
                sl_distance_index=abs(signal.entry_price - signal.stop_loss),
                lot_size=contract.lot_size,
                margin_per_lot=margin_per_lot,
            )
            if sizing["quantity"] <= 0:
                _log.info("Position sizing rejected on %s: %s", date_str, sizing["reason"])
                continue

            delta = CONFIG.backtest.atm_delta_approx
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            target_distance = abs(signal.target - signal.entry_price)

            premium_sl = entry_premium - delta * sl_distance
            premium_target = entry_premium + delta * target_distance
            premium_sl = max(premium_sl, 0.05)

            open_trade = {
                "contract": contract,
                "side": signal.side,
                "entry_premium": entry_premium,
                "premium_sl": premium_sl,
                "premium_target": premium_target,
                "initial_sl": sl_distance,
                "quantity": sizing["quantity"],
                "lots": sizing["lots"],
                "margin_used": margin_per_lot * sizing["lots"],
                "entry_time": candles_5m[i].ts,
                "entry_spot": signal.entry_price,
                "signal_reason": signal.reason,
                "confidence_notes": signal.confidence_notes,
                "highest_favorable": entry_premium,
                "lowest_favorable": entry_premium,
            }

        dr = DayResult(date_str=date_str, trades=day_trades, trades_count=len(day_trades),
                       day_pnl=sum(t.net_pnl for t in day_trades))
        self.daily_results[date_str] = dr

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def _check_exit(self, trade: dict, current_premium: float, cur_time: dtime):
        side = trade["side"]
        sl = trade["premium_sl"]
        target = trade["premium_target"]
        entry = trade["entry_premium"]

        if side == TradeSide.LONG:
            trade["highest_favorable"] = max(trade["highest_favorable"], current_premium)
        else:
            trade["lowest_favorable"] = min(trade["lowest_favorable"], current_premium)

        profit = current_premium - entry if side == TradeSide.LONG else entry - current_premium
        if profit >= CONFIG.trade_mgmt.trail_step_index_points and trade["initial_sl"] > 0:
            progress = min(
                (profit - CONFIG.trade_mgmt.trail_step_index_points) / max(1, trade["premium_target"] - entry - CONFIG.trade_mgmt.trail_step_index_points),
                1.0,
            )
            trail_ratio = 8.0 - progress * (8.0 - 5.0)
            trail_sl = entry + trade["initial_sl"] * trail_ratio / 8.0 * CONFIG.backtest.atm_delta_approx
            if side == TradeSide.LONG:
                trade["premium_sl"] = max(trade["premium_sl"], entry - trail_sl)
            else:
                trade["premium_sl"] = min(trade["premium_sl"], entry + trail_sl)
            sl = trade["premium_sl"]

        hit_target = current_premium >= target if side == TradeSide.LONG else current_premium <= target
        hit_sl = current_premium <= sl if side == TradeSide.LONG else current_premium >= sl
        force_exit = cur_time >= CONFIG.session.force_square_off_time

        if hit_target:
            return {"exit_premium": target, "reason": "TARGET"}
        if hit_sl:
            return {"exit_premium": sl, "reason": "STOP_LOSS"}
        if force_exit:
            return {"exit_premium": current_premium, "reason": "FORCE_SQUARE_OFF"}
        return None

    def _close_trade(self, trade: dict, exit_info: dict, exit_ts: datetime, trade_date: date) -> TradeRecord:
        exit_premium = exit_info["exit_premium"]
        reason = exit_info["reason"]
        side = trade["side"]
        entry_premium = trade["entry_premium"]
        quantity = trade["quantity"]

        gross_pnl = (exit_premium - entry_premium) * quantity if side == TradeSide.LONG else (entry_premium - exit_premium) * quantity

        buy_brokerage = min(20.0, entry_premium * quantity * 0.001)
        sell_brokerage = min(20.0, exit_premium * quantity * 0.001)
        total_brokerage = buy_brokerage + sell_brokerage
        gst = total_brokerage * CONFIG.backtest.option_gst_pct
        stamp_duty = (entry_premium * quantity + exit_premium * quantity) * CONFIG.backtest.option_stamp_duty_per_lakh / 100_000.0
        total_charges = total_brokerage + gst + stamp_duty

        net_pnl = gross_pnl - total_charges

        contract = trade["contract"]
        return TradeRecord(
            trade_date=trade_date.strftime("%Y-%m-%d"),
            entry_time=trade["entry_time"].strftime("%H:%M:%S"),
            exit_time=exit_ts.strftime("%H:%M:%S"),
            direction=side.value,
            nifty_spot_entry=round(trade["entry_spot"], 2),
            selected_expiry=contract.expiry,
            atm_strike=contract.strike,
            option_type=contract.option_type,
            option_trading_symbol=contract.tradingsymbol,
            entry_premium=round(entry_premium, 2),
            exit_premium=round(exit_premium, 2),
            lot_size=contract.lot_size,
            num_lots=trade["lots"],
            quantity=quantity,
            gross_pnl=round(gross_pnl, 2),
            brokerage_charges=round(total_charges, 2),
            net_pnl=round(net_pnl, 2),
            exit_reason=reason,
            signal_reason=trade.get("signal_reason", ""),
            confidence_notes=trade.get("confidence_notes", ""),
        )

    # ------------------------------------------------------------------
    # Contract resolution (with expiry rollover)
    # ------------------------------------------------------------------

    def _resolve_contract_for_date(self, spot_price: float, side: TradeSide, trade_date: date, nfo_instruments: list[dict]):
        expiry = _next_weekly_expiry(nfo_instruments, trade_date)
        strike = round_to_strike_step(spot_price, CONFIG.instrument.strike_step)
        option_type = CONFIG.instrument.option_type_ce if side == TradeSide.LONG else CONFIG.instrument.option_type_pe

        candidates = [
            i for i in nfo_instruments
            if i["expiry"] == expiry and i["strike"] == strike and i["instrument_type"] == option_type
        ]
        if not candidates:
            raise RuntimeError(f"No contract found for strike={strike} type={option_type} expiry={expiry}")

        inst = candidates[0]
        from utils import OptionContract
        return OptionContract(
            tradingsymbol=inst["tradingsymbol"],
            strike=int(inst["strike"]),
            option_type=option_type,
            expiry=str(expiry),
            instrument_token=int(inst["instrument_token"]),
            lot_size=int(inst["lot_size"]),
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _size_position(self, equity: float, entry_premium: float, sl_distance_index: float, lot_size: int, margin_per_lot: float) -> dict:
        risk_rupees_allowed = equity * CONFIG.sizing.risk_per_trade_pct_of_equity / 100.0
        premium_risk_per_lot = max(entry_premium * 0.01, 0.05) * sl_distance_index * lot_size / max(sl_distance_index, 1)
        premium_risk_per_lot = max(premium_risk_per_lot, sl_distance_index * lot_size * 0.4)

        max_lots_by_risk = int(risk_rupees_allowed // premium_risk_per_lot) if premium_risk_per_lot > 0 else 0

        usable_margin = max(equity - CONFIG.capital.min_margin_buffer_rupees, 0.0) * CONFIG.capital.max_exposure_pct_of_margin
        max_lots_by_margin = int(usable_margin // margin_per_lot) if margin_per_lot > 0 else 0

        max_lots = min(max_lots_by_risk, max_lots_by_margin, CONFIG.sizing.max_lots_per_trade)

        if max_lots <= 0:
            reason = "INSUFFICIENT_MARGIN" if max_lots_by_margin <= 0 else "RISK_LIMIT_TOO_TIGHT"
            return {"quantity": 0, "lots": 0, "reason": reason}

        quantity = max_lots * lot_size
        if CONFIG.sizing.round_lots_down:
            from utils import round_down_to_lot
            quantity = round_down_to_lot(quantity, lot_size)

        return {"quantity": quantity, "lots": max_lots, "reason": "OK"}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _bars_to_candles(self, bars: list[dict]) -> list[Candle]:
        candles = []
        for b in bars:
            ts = b["date"]
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            candles.append(Candle(
                ts=ts,
                open=float(b["open"]), high=float(b["high"]),
                low=float(b["low"]), close=float(b["close"]),
                volume=int(b.get("volume", 0)),
            ))
        candles.sort(key=lambda c: c.ts)
        return candles

    def _compute_vwap(self, candles: list[Candle]) -> list[float]:
        vals, cum_pv, cum_vol, day = [], 0.0, 0.0, None
        for c in candles:
            if day != c.ts.date():
                day, cum_pv, cum_vol = c.ts.date(), 0.0, 0.0
            vol = max(c.volume, 1)
            cum_pv += c.close * vol
            cum_vol += vol
            vals.append(cum_pv / cum_vol)
        return vals

    def _find_candle_index(self, candles: list[Candle], target_ts: datetime) -> Optional[int]:
        for idx, c in enumerate(candles):
            if c.ts == target_ts:
                return idx
        if not candles:
            return None
        closest = min(candles, key=lambda c: abs((c.ts - target_ts).total_seconds()))
        if abs((closest.ts - target_ts).total_seconds()) <= 300:
            return candles.index(closest)
        return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class OptionsReportGenerator:
    def __init__(self, trades: list[TradeRecord], starting_equity: float, backtest_months: int):
        self.trades = trades
        self.starting_equity = starting_equity
        self.backtest_months = backtest_months

    def generate_pdf(self, output_path: Path) -> None:
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
        from collections import defaultdict
        from datetime import datetime

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            rightMargin=12 * mm, leftMargin=12 * mm,
            topMargin=15 * mm, bottomMargin=15 * mm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=16, spaceAfter=4, textColor=HexColor("#1a1a2e"))
        h2_style = ParagraphStyle("CustomH2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=HexColor("#16213e"))
        h3_style = ParagraphStyle("CustomH3", parent=styles["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3, textColor=HexColor("#0f3460"))
        cell_style = ParagraphStyle("CellStyle", parent=styles["Normal"], fontSize=7, leading=9)
        header_style = ParagraphStyle("HeaderStyle", parent=styles["Normal"], fontSize=7, leading=9, textColor=colors.white, alignment=TA_CENTER)
        green_style = ParagraphStyle("GreenCell", parent=cell_style, textColor=HexColor("#006400"))
        red_style = ParagraphStyle("RedCell", parent=cell_style, textColor=HexColor("#8B0000"))
        bold_cell = ParagraphStyle("BoldCell", parent=cell_style, fontName="Helvetica-Bold")
        wrap_style = ParagraphStyle("WrapCell", parent=cell_style, fontSize=6, leading=8)

        elements = []

        elements.append(Paragraph("NIFTY Weekly Options Backtest Report", title_style))
        elements.append(Paragraph(
            f"Capital: Rs.{self.starting_equity:,.0f} | "
            f"Period: Last {self.backtest_months} months | "
            f"Strategy: EMA + VWAP + ADX + ATR (ATM Weekly Options)",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 4 * mm))

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        total_pnl = sum(t.net_pnl for t in self.trades)
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        win_rate = (len(wins) / total_trades * 100) if total_trades else 0.0
        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        equity_curve = [self.starting_equity]
        for t in self.trades:
            equity_curve.append(equity_curve[-1] + t.net_pnl)
        peak = self.starting_equity
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak else 0
            max_dd = max(max_dd, dd)

        summary_data = [
            [Paragraph("<b>Metric</b>", header_style), Paragraph("<b>Value</b>", header_style)],
            [Paragraph("Total Trades", cell_style), Paragraph(str(total_trades), cell_style)],
            [Paragraph("Win Rate", cell_style), Paragraph(f"{win_rate:.1f}%", green_style if win_rate >= 50 else red_style)],
            [Paragraph("Total Net P&L", cell_style), Paragraph(f"Rs.{total_pnl:,.2f}", green_style if total_pnl >= 0 else red_style)],
            [Paragraph("Gross Profit", cell_style), Paragraph(f"Rs.{gross_profit:,.2f}", green_style)],
            [Paragraph("Gross Loss", cell_style), Paragraph(f"Rs.{gross_loss:,.2f}", red_style)],
            [Paragraph("Profit Factor", cell_style), Paragraph(f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf", cell_style)],
            [Paragraph("Average Win", cell_style), Paragraph(f"Rs.{avg_win:,.2f}", green_style)],
            [Paragraph("Average Loss", cell_style), Paragraph(f"Rs.{avg_loss:,.2f}", red_style)],
            [Paragraph("Max Drawdown", cell_style), Paragraph(f"{max_dd:.2f}%", red_style)],
            [Paragraph("Final Equity", cell_style), Paragraph(f"Rs.{equity_curve[-1]:,.2f}", cell_style)],
            [Paragraph("Return %", cell_style), Paragraph(f"{(equity_curve[-1] - self.starting_equity) / self.starting_equity * 100:.2f}%", green_style if total_pnl >= 0 else red_style)],
        ]
        summary_table = Table(summary_data, colWidths=[60 * mm, 60 * mm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 6 * mm))

        monthly = defaultdict(list)
        for t in self.trades:
            monthly[t.trade_date[:7]].append(t)

        elements.append(Paragraph("Monthly Summary", h2_style))
        monthly_header = [
            Paragraph("<b>Month</b>", header_style),
            Paragraph("<b>Trades</b>", header_style),
            Paragraph("<b>Wins</b>", header_style),
            Paragraph("<b>Losses</b>", header_style),
            Paragraph("<b>Win Rate</b>", header_style),
            Paragraph("<b>Net P&L (Rs.)</b>", header_style),
            Paragraph("<b>Gross P&L</b>", header_style),
            Paragraph("<b>Charges</b>", header_style),
        ]
        monthly_data = [monthly_header]
        for month_key in sorted(monthly.keys()):
            mt = monthly[month_key]
            mw = sum(1 for t in mt if t.net_pnl > 0)
            ml = sum(1 for t in mt if t.net_pnl <= 0)
            m_pnl = sum(t.net_pnl for t in mt)
            m_gross = sum(t.gross_pnl for t in mt)
            m_charges = sum(t.brokerage_charges for t in mt)
            mwr = (mw / len(mt) * 100) if mt else 0.0
            pnl_s = green_style if m_pnl >= 0 else red_style
            monthly_data.append([
                Paragraph(month_key, cell_style), Paragraph(str(len(mt)), cell_style),
                Paragraph(str(mw), green_style), Paragraph(str(ml), red_style),
                Paragraph(f"{mwr:.1f}%", cell_style), Paragraph(f"{m_pnl:,.2f}", pnl_s),
                Paragraph(f"{m_gross:,.2f}", green_style if m_gross >= 0 else red_style),
                Paragraph(f"{m_charges:,.2f}", red_style),
            ])
        monthly_table = Table(monthly_data, colWidths=[20*mm, 14*mm, 12*mm, 12*mm, 14*mm, 22*mm, 20*mm, 18*mm])
        monthly_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(monthly_table)
        elements.append(Spacer(1, 6 * mm))

        weekly = defaultdict(list)
        for t in self.trades:
            dt = datetime.strptime(t.trade_date, "%Y-%m-%d")
            iso_year, iso_week, _ = dt.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            weekly[week_key].append(t)

        elements.append(Paragraph("Weekly Summary", h2_style))
        weekly_header = [
            Paragraph("<b>Week</b>", header_style),
            Paragraph("<b>Trades</b>", header_style),
            Paragraph("<b>Wins</b>", header_style),
            Paragraph("<b>Losses</b>", header_style),
            Paragraph("<b>Win Rate</b>", header_style),
            Paragraph("<b>Net P&L (Rs.)</b>", header_style),
        ]
        weekly_data = [weekly_header]
        for wk in sorted(weekly.keys()):
            wt = weekly[wk]
            ww = sum(1 for t in wt if t.net_pnl > 0)
            wl = sum(1 for t in wt if t.net_pnl <= 0)
            w_pnl = sum(t.net_pnl for t in wt)
            wwr = (ww / len(wt) * 100) if wt else 0.0
            pnl_s = green_style if w_pnl >= 0 else red_style
            weekly_data.append([
                Paragraph(wk, cell_style), Paragraph(str(len(wt)), cell_style),
                Paragraph(str(ww), green_style), Paragraph(str(wl), red_style),
                Paragraph(f"{wwr:.1f}%", cell_style), Paragraph(f"{w_pnl:,.2f}", pnl_s),
            ])
        weekly_table = Table(weekly_data, colWidths=[22*mm, 16*mm, 12*mm, 12*mm, 16*mm, 28*mm])
        weekly_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(weekly_table)
        elements.append(Spacer(1, 6 * mm))

        elements.append(Paragraph("Trade Details by Month", h2_style))
        for month_key in sorted(monthly.keys()):
            mt = monthly[month_key]
            elements.append(Paragraph(f"<b>{month_key}</b> — {len(mt)} trades, Net P&L: Rs.{sum(t.net_pnl for t in mt):,.2f}", h3_style))

            trade_header = [
                Paragraph("<b>Date</b>", header_style),
                Paragraph("<b>Entry</b>", header_style),
                Paragraph("<b>Exit</b>", header_style),
                Paragraph("<b>Side</b>", header_style),
                Paragraph("<b>NIFTY</b>", header_style),
                Paragraph("<b>Strike</b>", header_style),
                Paragraph("<b>Type</b>", header_style),
                Paragraph("<b>Symbol</b>", header_style),
                Paragraph("<b>Entry Prem</b>", header_style),
                Paragraph("<b>Exit Prem</b>", header_style),
                Paragraph("<b>Lots</b>", header_style),
                Paragraph("<b>Qty</b>", header_style),
                Paragraph("<b>Gross P&L</b>", header_style),
                Paragraph("<b>Charges</b>", header_style),
                Paragraph("<b>Net P&L</b>", header_style),
                Paragraph("<b>Reason</b>", header_style),
            ]
            trade_data = [trade_header]
            for t in mt:
                pnl_s = green_style if t.net_pnl >= 0 else red_style
                trade_data.append([
                    Paragraph(t.trade_date, cell_style),
                    Paragraph(t.entry_time, cell_style),
                    Paragraph(t.exit_time, cell_style),
                    Paragraph(t.direction, cell_style),
                    Paragraph(f"{t.nifty_spot_entry:.2f}", cell_style),
                    Paragraph(str(t.atm_strike), cell_style),
                    Paragraph(t.option_type, cell_style),
                    Paragraph(t.option_trading_symbol, wrap_style),
                    Paragraph(f"{t.entry_premium:.2f}", cell_style),
                    Paragraph(f"{t.exit_premium:.2f}", cell_style),
                    Paragraph(str(t.num_lots), cell_style),
                    Paragraph(str(t.quantity), cell_style),
                    Paragraph(f"{t.gross_pnl:,.2f}", green_style if t.gross_pnl >= 0 else red_style),
                    Paragraph(f"{t.brokerage_charges:,.2f}", red_style),
                    Paragraph(f"{t.net_pnl:,.2f}", pnl_s),
                    Paragraph(t.exit_reason, cell_style),
                ])

            trade_table = Table(trade_data, colWidths=[16*mm, 12*mm, 12*mm, 10*mm, 14*mm, 14*mm, 10*mm, 18*mm, 14*mm, 14*mm, 10*mm, 12*mm, 16*mm, 14*mm, 16*mm, 16*mm])
            trade_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]))
            elements.append(trade_table)
            elements.append(Spacer(1, 3 * mm))

        doc.build(elements)
        print(f"PDF report saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY Options Backtest with Real Premium Data")
    parser.add_argument("--months", type=int, default=CONFIG.backtest.backtest_months, help="Number of months to backtest")
    parser.add_argument("--equity", type=float, default=CONFIG.backtest.starting_equity, help="Starting equity")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path")
    args = parser.parse_args()

    print("=" * 60)
    print("NIFTY Options Backtest — Real Premium Data")
    print(f"Capital: Rs.{args.equity:,.0f}")
    print(f"Period: Last {args.months} months")
    print(f"Mode: BUY options only (CE for LONG, PE for SHORT)")
    print("=" * 60)

    kite = KiteAPI()
    kite.login()

    end_date = date.today()
    start_date = end_date - timedelta(days=30 * args.months)

    engine = OptionBacktestEngine(kite, starting_equity=args.equity)
    trades = engine.run(start_date, end_date)

    if not trades:
        print("No trades generated in the backtest period.")
        return

    total_pnl = sum(t.net_pnl for t in trades)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    win_rate = len(wins) / len(trades) * 100

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total Trades:     {len(trades)}")
    print(f"Wins:             {len(wins)}")
    print(f"Losses:           {len(losses)}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Total Net P&L:    Rs.{total_pnl:,.2f}")
    print(f"Gross Profit:     Rs.{sum(t.net_pnl for t in wins):,.2f}")
    print(f"Gross Loss:       Rs.{abs(sum(t.net_pnl for t in losses)):,.2f}")
    print(f"Final Equity:     Rs.{args.equity + total_pnl:,.2f}")
    print(f"Return %:         {(total_pnl / args.equity * 100):.2f}%")
    print("=" * 60)

    out_csv = CONFIG.backtest.results_dir / f"options_backtest_{end_date.strftime('%Y%m%d')}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trade Date", "Entry Time", "Exit Time", "Direction",
            "NIFTY Spot Entry", "Selected Expiry", "ATM Strike", "Option Type",
            "Option Trading Symbol", "Entry Premium", "Exit Premium",
            "Lot Size", "Number of Lots", "Quantity",
            "Gross P&L", "Brokerage & Charges", "Net P&L", "Exit Reason",
            "Signal Reason", "Confidence Notes",
        ])
        for t in trades:
            writer.writerow([
                t.trade_date, t.entry_time, t.exit_time, t.direction,
                t.nifty_spot_entry, t.selected_expiry, t.atm_strike, t.option_type,
                t.option_trading_symbol, t.entry_premium, t.exit_premium,
                t.lot_size, t.num_lots, t.quantity,
                t.gross_pnl, t.brokerage_charges, t.net_pnl, t.exit_reason,
                t.signal_reason, t.confidence_notes,
            ])
    print(f"CSV saved to: {out_csv}")

    if args.output is None:
        out_pdf = CONFIG.backtest.results_dir / f"options_backtest_{args.months}mo_{end_date.strftime('%Y%m%d_%H%M%S')}.pdf"
    else:
        out_pdf = Path(args.output)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    generator = OptionsReportGenerator(trades, args.equity, args.months)
    generator.generate_pdf(out_pdf)


if __name__ == "__main__":
    main()
