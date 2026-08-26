"""
backtest.py
===========
Replays historical NIFTY spot 5-min OHLCV data (CSV) through the SAME
strategy.py / risk_manager.py logic used live, so backtest results are
a faithful predictor of paper/live behaviour (no separate "backtest
strategy" reimplementation to drift out of sync).

Input CSV format (place under CONFIG.backtest.data_dir):
    datetime,open,high,low,close,volume
    2024-01-02 09:15:00,21750.5,21762.0,21745.0,21758.2,125000
    ...

Since historical option-chain premium data is rarely available cleanly,
P&L is computed in NIFTY INDEX POINTS by default and converted to
rupee P&L using an approximate premium-per-point multiplier
(CONFIG.backtest is extended here with a delta_approx). If you have
real historical option premium series, wire them into
`OptionPremiumFeed` instead and P&L will be exact.

Run:
    NIFTY_BOT_MODE=BACKTEST python backtest.py --file nifty_5min_2024.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

from config import CONFIG
from indicators import ema_series
from logger import get_logger
from risk_manager import RiskManager
from strategy import StrategyEngine
from utils import Candle, TradeSide, TRAIL_TRIGGER_POINTS

_log = get_logger("backtest")

# Rough at-the-money weekly-option delta used to translate index-point
# P&L into rupee P&L when no real premium series is supplied. ATM
# options run close to delta 0.5; tune via --delta if desired.
DEFAULT_ATM_DELTA_APPROX = 0.45
NIFTY_LOT_SIZE = 75  # update to current NSE lot size if it changes


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_index: float
    exit_index: float
    index_points_pnl: float
    rupee_pnl: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self, starting_equity: float) -> dict:
        if not self.trades:
            return {"trades": 0, "message": "No trades generated."}
        wins = [t for t in self.trades if t.rupee_pnl > 0]
        losses = [t for t in self.trades if t.rupee_pnl <= 0]
        gross_profit = sum(t.rupee_pnl for t in wins)
        gross_loss = abs(sum(t.rupee_pnl for t in losses))
        total_pnl = sum(t.rupee_pnl for t in self.trades)

        peak = starting_equity
        max_dd = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak else 0
            max_dd = max(max_dd, dd)

        win_rate = len(wins) / len(self.trades) * 100
        avg_win = mean([t.rupee_pnl for t in wins]) if wins else 0.0
        avg_loss = mean([t.rupee_pnl for t in losses]) if losses else 0.0
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "final_equity": round(starting_equity + total_pnl, 2),
            "return_pct": round(total_pnl / starting_equity * 100, 2),
        }


def load_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                ts=datetime.fromisoformat(row["datetime"]),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=int(float(row.get("volume", 0) or 0)),
            ))
    candles.sort(key=lambda c: c.ts)
    return candles


def resample(candles: list[Candle], minutes: int) -> list[Candle]:
    if minutes == CONFIG.session.primary_timeframe_min:
        return candles
    out: list[Candle] = []
    bucket_candles: list[Candle] = []
    cur_bucket = None
    for c in candles:
        b = c.ts.replace(minute=(c.ts.minute // minutes) * minutes, second=0, microsecond=0)
        if cur_bucket is not None and b != cur_bucket:
            out.append(_merge(bucket_candles))
            bucket_candles = []
        cur_bucket = b
        bucket_candles.append(c)
    if bucket_candles:
        out.append(_merge(bucket_candles))
    return out


def _merge(candles: list[Candle]) -> Candle:
    return Candle(
        ts=candles[0].ts, open=candles[0].open,
        high=max(c.high for c in candles), low=min(c.low for c in candles),
        close=candles[-1].close, volume=sum(c.volume for c in candles),
    )


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


def run_backtest(csv_path: Path, starting_equity: float, delta_approx: float, lot_size: int) -> BacktestResult:
    candles_5m = load_csv(csv_path)
    candles_15m = resample(candles_5m, CONFIG.session.higher_timeframe_min)
    vwap_series = compute_vwap_series(candles_5m)

    engine = StrategyEngine()
    risk = RiskManager(starting_equity=starting_equity)
    result = BacktestResult()
    equity = starting_equity

    open_trade = None  # dict tracking an active simulated trade
    ts_to_15m_idx = {}
    idx15 = -1
    for i, c in enumerate(candles_15m):
        ts_to_15m_idx[c.ts] = i

    last_15m_bucket = None
    visible_15m: list[Candle] = []

    for i in range(len(candles_5m)):
        cur_time = candles_5m[i].ts.time()

        # keep 15m series only up to bars fully closed before/at this point
        bucket = candles_5m[i].ts.replace(
            minute=(candles_5m[i].ts.minute // CONFIG.session.higher_timeframe_min) *
            CONFIG.session.higher_timeframe_min, second=0, microsecond=0)
        if bucket != last_15m_bucket:
            visible_15m = [c for c in candles_15m if c.ts < bucket]
            last_15m_bucket = bucket

        window_5m = candles_5m[: i + 1]

        # -------- manage open trade first --------
        if open_trade is not None:
            px = candles_5m[i].close
            hit_target = (
                px >= open_trade["target"] if open_trade["side"] == TradeSide.LONG else px <= open_trade["target"]
            )
            hit_sl = (
                px <= open_trade["sl"] if open_trade["side"] == TradeSide.LONG else px >= open_trade["sl"]
            )
            force_exit = cur_time >= CONFIG.session.force_square_off_time

            # Trailing SL update (matches generate_report.py / live order_manager)
            if not hit_target and not hit_sl and not force_exit:
                profit = (
                    px - open_trade["entry"]
                    if open_trade["side"] == TradeSide.LONG
                    else open_trade["entry"] - px
                )
                if profit >= TRAIL_TRIGGER_POINTS and open_trade["initial_sl"] > 0:
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
                exit_price = open_trade["target"] if hit_target else (open_trade["sl"] if hit_sl else px)
                reason = "TARGET" if hit_target else ("SL" if hit_sl else "FORCE_SQUARE_OFF")
                index_points = (exit_price - open_trade["entry"]) if open_trade["side"] == TradeSide.LONG \
                    else (open_trade["entry"] - exit_price)
                rupee_pnl = index_points * delta_approx * lot_size - CONFIG.backtest.commission_per_lot
                result.trades.append(BacktestTrade(
                    entry_time=open_trade["entry_time"], exit_time=candles_5m[i].ts,
                    side=open_trade["side"].value, entry_index=open_trade["entry"], exit_index=exit_price,
                    index_points_pnl=round(index_points, 2), rupee_pnl=round(rupee_pnl, 2), reason=reason,
                ))
                risk.record_trade_result(rupee_pnl)
                equity += rupee_pnl
                result.equity_curve.append(equity)
                open_trade = None
            continue  # one position at a time, matches max_trades_per_day gating too

        # -------- look for new entries --------
        if not (CONFIG.session.setup_scan_start <= cur_time <= CONFIG.session.last_entry_time):
            continue
        allowed, _ = risk.can_take_new_trade()
        if not allowed:
            continue
        if len(window_5m) < 60 or len(visible_15m) < 60:
            continue

        signal = engine.evaluate(window_5m, visible_15m, vwap_series[: i + 1], symbol="NIFTY")
        if signal.side == TradeSide.NONE:
            continue

        slippage = CONFIG.backtest.slippage_points
        entry_px = signal.entry_price + (slippage if signal.side == TradeSide.LONG else -slippage)
        initial_sl = abs(signal.entry_price - signal.stop_loss)
        open_trade = {
            "side": signal.side, "entry": entry_px, "sl": signal.stop_loss,
            "target": signal.target, "entry_time": candles_5m[i].ts,
            "initial_sl": initial_sl,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY weekly-options bot backtester")
    parser.add_argument("--file", required=True, help="CSV filename inside backtest data_dir, or full path")
    parser.add_argument("--equity", type=float, default=CONFIG.backtest.starting_equity)
    parser.add_argument("--delta", type=float, default=DEFAULT_ATM_DELTA_APPROX)
    parser.add_argument("--lot-size", type=int, default=NIFTY_LOT_SIZE)
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_absolute() and not path.exists():
        path = CONFIG.backtest.data_dir / args.file

    if not path.exists():
        _log.error("CSV not found: %s", path)
        return

    result = run_backtest(path, args.equity, args.delta, args.lot_size)
    summary = result.summary(args.equity)

    _log.info("=== BACKTEST RESULTS (%s) ===", path.name)
    for k, v in summary.items():
        _log.info("  %s: %s", k, v)

    out_path = CONFIG.backtest.results_dir / f"{path.stem}_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_time", "exit_time", "side", "entry_index", "exit_index",
                         "index_points_pnl", "rupee_pnl", "reason"])
        for t in result.trades:
            writer.writerow([t.entry_time, t.exit_time, t.side, t.entry_index, t.exit_index,
                             t.index_points_pnl, t.rupee_pnl, t.reason])
    _log.info("Trade log written to %s", out_path)


if __name__ == "__main__":
    main()