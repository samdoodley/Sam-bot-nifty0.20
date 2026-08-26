"""
main.py
=======
Paper trading entry point using real Kite live market data.

    NIFTY_BOT_MODE=PAPER python main.py

This mode connects to Kite for live spot/option prices and historical
data, but all orders are simulated locally (no real capital at risk).
For backtesting, use backtest.py instead - it replays historical CSVs.
For live trading with real money, use NIFTY_BOT_MODE=LIVE with
NIFTY_BOT_CONFIRM_LIVE=YES.

Self-healing:
  - Kite REST calls retry with backoff (kite_api.py).
  - WebSocket auto-reconnects via KiteTicker's built-in reconnect,
    using the subscribe-on-connect pattern so resubscription is safe.
  - The main loop itself is wrapped so an unexpected exception in one
    iteration is logged and the loop continues rather than crashing
    the whole process; force square-off logic runs independently in
    the position monitor thread so an entry-scan bug can't strand you
    in a naked position.
"""

from __future__ import annotations

import signal
import sys
import threading
import time as _time
from datetime import datetime

import dashboard
import daily_report
import indicators as ind
from config import CONFIG, TradingMode
from kite_api import KiteAPI
from logger import get_logger, log_decision, send_telegram_alert
from margin_manager import MarginManager
from market_data import CandleStore
from option_selector import OptionUniverse
from order_manager import OrderManager
from risk_manager import RiskManager
from strategy import StrategyEngine
from utils import (TradeSide, is_force_square_off_time, is_market_open,
                   now_ist_time,
                   )

_log = get_logger("main")

_shutdown = threading.Event()


def _handle_sigterm(signum, frame) -> None:
    _log.warning("Shutdown signal received - stopping gracefully.")
    _shutdown.set()


class TradingBot:
    def __init__(self) -> None:
        self.kite = KiteAPI()
        self.candles = CandleStore()
        self.strategy = StrategyEngine()
        self.risk = RiskManager(starting_equity=CONFIG.backtest.starting_equity
                                 if CONFIG.mode != TradingMode.LIVE else 0.0)
        self.option_universe = OptionUniverse(self.kite)
        self.margin_mgr = MarginManager(
            self.kite,
            paper_equity_override=None if CONFIG.mode == TradingMode.LIVE
            else CONFIG.backtest.starting_equity,
        )
        if CONFIG.mode == TradingMode.PAPER:
            self.kite.set_paper_equity(CONFIG.backtest.starting_equity)
        self.order_mgr = OrderManager(self.kite, get_ltp=self._get_option_ltp)
        self.order_mgr.on_exit(self._on_position_exit)

        self.spot_token: int | None = None
        self._latest_option_ltp: dict[str, float] = {}
        self._active_symbol: str | None = None
        self._active_option_token: int | None = None
        self._last_daily_report_date: str | None = None
        self._last_session_save: float = 0.0
        self._margin_blocked: bool = False

    # ------------------------------------------------------------
    # STARTUP
    # ------------------------------------------------------------

    def start(self) -> None:
        _log.info("Starting NIFTY bot in %s mode.", CONFIG.mode.value)
        if CONFIG.mode == TradingMode.PAPER:
            _log.info("PAPER TRADING: Using REAL Kite live market data | Orders are SIMULATED (no real capital at risk)")
        elif CONFIG.mode == TradingMode.LIVE:
            _log.info("LIVE TRADING: Using REAL Kite live market data | Orders are REAL (real capital at risk)")

        try:
            self.kite.login()
        except Exception:
            _log.exception("Kite login failed - aborting")
            return

        if CONFIG.mode == TradingMode.LIVE:
            margin = self.margin_mgr.get_available_margin()
            self.risk.equity = margin
            self.risk._starting_equity = margin
            _log.info("Live equity initialized from broker margin: Rs.%.2f", margin)

        try:
            profile = self.kite.kite.profile()
            _log.info("Pre-flight: user=%s margin=%.2f", profile.get("user_name"), margin if CONFIG.mode == TradingMode.LIVE else 0.0)
        except Exception:
            _log.warning("Pre-flight profile check failed - continuing")

        send_telegram_alert("BOT_START", f"Mode={CONFIG.mode.value} user={profile.get('user_name', 'unknown')}")

        self._load_session_state()

        self.spot_token = self.option_universe.get_spot_instrument_token()
        self._warm_up_history()

        tokens = [self.spot_token]
        self.kite.start_ticker(tokens, on_tick=self._on_ticks)
        if not self.kite.wait_until_connected(timeout=20):
            _log.error("WebSocket failed to connect within timeout - retrying in background.")

        gap_thread = threading.Thread(target=self._ws_gap_recovery_loop, daemon=True)
        gap_thread.start()

        dashboard.start_dashboard()

        signal.signal(signal.SIGINT, _handle_sigterm)
        signal.signal(signal.SIGTERM, _handle_sigterm)

        monitor_thread = threading.Thread(target=self._position_monitor_loop, daemon=True)
        monitor_thread.start()

        session_thread = threading.Thread(target=self._session_persistence_loop, daemon=True)
        session_thread.start()

        margin_thread = threading.Thread(target=self._margin_monitor_loop, daemon=True)
        margin_thread.start()

        self._main_scan_loop()

    def _warm_up_history(self) -> None:
        now = datetime.now()
        from_dt = now.replace(hour=9, minute=0, second=0, microsecond=0)
        try:
            bars_5m = self.kite.historical_data(self.spot_token, from_dt, now, "5minute")
            bars_15m = self.kite.historical_data(self.spot_token, from_dt, now, "15minute")
            self.candles.load_historical(self.spot_token, CONFIG.session.primary_timeframe_min, bars_5m)
            self.candles.load_historical(self.spot_token, CONFIG.session.higher_timeframe_min, bars_15m)
        except Exception:
            _log.exception("Historical warm-up fetch failed - continuing, will build from live ticks.")

    # ------------------------------------------------------------
    # TICK HANDLING
    # ------------------------------------------------------------

    def _on_ticks(self, ticks: list[dict]) -> None:
        for t in ticks:
            token = t.get("instrument_token")
            price = t.get("last_price")
            vol = t.get("last_traded_quantity", 0)
            if token == self.spot_token and price:
                self.candles.on_tick(token, price, vol)
            tsym = t.get("tradingsymbol")
            if tsym and price:
                self._latest_option_ltp[tsym] = price

    def _get_option_ltp(self, tradingsymbol: str) -> float | None:
        if tradingsymbol in self._latest_option_ltp:
            return self._latest_option_ltp[tradingsymbol]
        # fall back to a direct quote call if we don't have a live tick yet
        try:
            q = self.kite.ltp([f"{CONFIG.instrument.option_exchange}:{tradingsymbol}"])
            key = f"{CONFIG.instrument.option_exchange}:{tradingsymbol}"
            return q.get(key, {}).get("last_price")
        except Exception:
            _log.exception("Failed to fetch LTP for %s", tradingsymbol)
            return None

    # ------------------------------------------------------------
    # ACTIVE OPTION WEBSOCKET SUBSCRIPTION (tick-level LTP)
    # ------------------------------------------------------------

    def _subscribe_active_option(self, token: int) -> None:
        if self._active_option_token is not None and self._active_option_token != token:
            try:
                self.kite.remove_subscription(self._active_option_token)
            except Exception:
                _log.exception("Failed to unsubscribe previous option token %s", self._active_option_token)
        if self._active_option_token != token:
            try:
                self.kite.add_subscription(token)
                _log.info("Subscribed option token %s for tick-level LTP", token)
            except Exception:
                _log.exception("Failed to subscribe option token %s", token)
            self._active_option_token = token

    def _unsubscribe_active_option(self) -> None:
        if self._active_option_token is not None:
            try:
                self.kite.remove_subscription(self._active_option_token)
                _log.info("Unsubscribed option token %s", self._active_option_token)
            except Exception:
                _log.exception("Failed to unsubscribe option token %s", self._active_option_token)
            self._active_option_token = None

    # ------------------------------------------------------------
    # MAIN SCAN LOOP (entries)
    # ------------------------------------------------------------

    def _main_scan_loop(self) -> None:
        last_processed_bar_ts = None
        while not _shutdown.is_set():
            try:
                self._scan_iteration_guard(last_processed_bar_ts)
            except Exception:
                _log.exception("Unhandled error in scan loop iteration - continuing.")
            _time.sleep(CONFIG.session.scan_loop_interval_sec)

        self._shutdown_sequence()

    def _scan_iteration_guard(self, last_processed_bar_ts) -> None:
        if CONFIG.killswitch.killswitch_path.exists():
            _log.warning("Killswitch active at %s - blocking new entries", CONFIG.killswitch.killswitch_path)
            if is_market_open():
                self.order_mgr.force_square_off_all()
            return

        today_str = datetime.now().date().isoformat()
        if not is_market_open():
            if self._last_daily_report_date != today_str:
                try:
                    daily_report.save_daily_report()
                    daily_report.send_daily_report_telegram()
                    self._last_daily_report_date = today_str
                    _log.info("Daily report saved and sent via Telegram for %s", today_str)
                except Exception:
                    _log.exception("Failed to process daily report for %s", today_str)
            return

        closed_5m = self.candles.get_closed_candles(self.spot_token, CONFIG.session.primary_timeframe_min)
        if not closed_5m:
            return
        if len(closed_5m) < 45:
            return
        latest_bar_ts = closed_5m[-1].ts
        if latest_bar_ts == last_processed_bar_ts:
            return  # already evaluated this closed candle

        self._update_dashboard_indicators(closed_5m)

        if is_force_square_off_time():
            return  # position_monitor_loop handles the actual square-off

        if self.risk.stats.trades_taken >= CONFIG.strategy.max_trades_per_day:
            return  # daily trade cap reached

        if self.order_mgr.open_position_count() > 0:
            return  # one position at a time (sequential entries)

        if not self._check_margin_safety():
            return

        allowed, reason = self.risk.can_take_new_trade()
        if not allowed:
            if reason == "DAILY_LOSS_LIMIT_HIT":
                send_telegram_alert("DAILY_LOSS_LIMIT_HIT", f"daily_pnl={self.risk._daily_pnl:.2f}")
            return

        closed_15m = self.candles.get_closed_candles(self.spot_token, CONFIG.session.higher_timeframe_min)
        if len(closed_15m) < 20:
            return
        vwap_series = [self.candles.get_current_vwap(self.spot_token, CONFIG.session.primary_timeframe_min)] * 1
        # build an aligned vwap series matching closed_5m length using stored candles
        vwap_series = self._vwap_series_for(closed_5m)

        signal = self.strategy.evaluate(closed_5m, closed_15m, vwap_series, symbol="NIFTY")
        if signal.side == TradeSide.SHORT:
            if CONFIG.session.enable_time_based_entry_gate and now_ist_time() < CONFIG.session.first_entry_time:
                return
            self._execute_signal(signal)

    def _vwap_series_for(self, closed_5m) -> list[float]:
        # Recomputed cheaply from closed candles each pass; fine at 5-min cadence.
        vals, cum_pv, cum_vol, day = [], 0.0, 0.0, None
        for c in closed_5m:
            if day != c.ts.date():
                day, cum_pv, cum_vol = c.ts.date(), 0.0, 0.0
            v = max(c.volume, 1)
            cum_pv += c.close * v
            cum_vol += v
            vals.append(cum_pv / cum_vol)
        return vals

    def _execute_signal(self, signal) -> None:
        if not self._check_margin_safety():
            _log.warning("Entry blocked due to low margin")
            return

        try:
            spot_price = signal.entry_price
            contract = self.option_universe.resolve_atm_contract(spot_price, signal.side)
        except Exception:
            _log.exception("Failed to resolve ATM contract for signal")
            return

        premium = self._get_option_ltp(contract.tradingsymbol) or 0.0
        equity = self.risk.equity
        sizing = self.margin_mgr.size_position(
            equity=equity, entry_price=signal.entry_price, stop_loss_price=signal.stop_loss,
            option_premium=premium, lot_size=contract.lot_size,
            margin_per_lot_estimate=self._estimate_margin_per_lot(contract, premium),
        )
        if sizing.quantity <= 0:
            return

        delta_approx = 0.45
        sl_distance_index = abs(signal.entry_price - signal.stop_loss)
        target_distance_index = abs(signal.target - signal.entry_price)
        if signal.side == TradeSide.LONG:
            premium_sl = premium - delta_approx * sl_distance_index
            premium_target = premium + delta_approx * target_distance_index
        else:
            premium_sl = premium + delta_approx * sl_distance_index
            premium_target = premium - delta_approx * target_distance_index
        premium_sl = max(premium_sl, 0.05)

        self._active_symbol = contract.tradingsymbol
        self.order_mgr.enter_position(
            contract=contract, side=signal.side, quantity=sizing.quantity,
            entry_price=premium, stop_loss=premium_sl, target=premium_target,
            initial_sl=delta_approx * sl_distance_index,
        )
        self._subscribe_active_option(contract.instrument_token)
        send_telegram_alert("POSITION_OPEN", f"{contract.tradingsymbol} qty={sizing.quantity} side={signal.side.value}")

    def _estimate_margin_per_lot(self, contract, premium: float) -> float:
        # Buying options is fully paid (no leverage on the long premium side);
        # margin required ~= premium * lot_size for a long option buy.
        return max(premium, 0.05) * contract.lot_size

    # ------------------------------------------------------------
    # POSITION MONITOR LOOP (exits - runs independently of entry scan)
    # ------------------------------------------------------------

    def _position_monitor_loop(self) -> None:
        while not _shutdown.is_set():
            try:
                self.order_mgr.monitor_positions()
                if is_force_square_off_time():
                    self.order_mgr.force_square_off_all()
            except Exception:
                _log.exception("Unhandled error in position monitor loop - continuing.")
            _time.sleep(CONFIG.session.position_monitor_interval_sec)

    def _on_position_exit(self, symbol: str, result) -> None:
        self._unsubscribe_active_option()
        self.risk.record_trade_result(result.pnl)
        if CONFIG.mode != TradingMode.LIVE:
            self.risk.equity += result.pnl
            self.margin_mgr.update_paper_equity(self.risk.equity)
        send_telegram_alert("POSITION_CLOSE", f"{symbol} pnl={result.pnl:.2f} reason={result.reason}")
        dashboard.update_state(
            position=None, current_pnl=0, daily_pnl=round(self.risk.stats.realized_pnl, 2),
            trades_today=self.risk.stats.trades_taken, win_rate=round(self.risk.win_rate(), 1),
        )

    # ------------------------------------------------------------
    # KILL-SWITCH
    # ------------------------------------------------------------

    def _check_killswitch(self) -> bool:
        if CONFIG.killswitch.killswitch_path.exists():
            _log.warning("Killswitch active at %s", CONFIG.killswitch.killswitch_path)
            if is_market_open():
                self.order_mgr.force_square_off_all()
            return True
        return False

    # ------------------------------------------------------------
    # WEBSOCKET GAP RECOVERY
    # ------------------------------------------------------------

    def _ws_gap_recovery_loop(self) -> None:
        while not _shutdown.is_set():
            try:
                self._check_ws_gap()
            except Exception:
                _log.exception("Unhandled error in WS gap recovery loop - continuing.")
            _time.sleep(CONFIG.ws_recovery.gap_sync_interval_sec)

    def _check_ws_gap(self) -> None:
        if not self.spot_token:
            return
        closed = self.candles.get_closed_candles(self.spot_token, CONFIG.session.primary_timeframe_min)
        if not closed:
            return
        latest = closed[-1].ts
        missing_min = (datetime.now() - latest).total_seconds() / 60.0
        if missing_min >= CONFIG.ws_recovery.gap_candle_missing_min:
            _log.warning("WS gap detected: last candle is %.1f min old - syncing history", missing_min)
            try:
                self._warm_up_history()
            except Exception:
                _log.exception("WS gap recovery warm-up failed")

    # ------------------------------------------------------------
    # SESSION PERSISTENCE
    # ------------------------------------------------------------

    def _session_persistence_loop(self) -> None:
        while not _shutdown.is_set():
            try:
                now = _time.time()
                if now - self._last_session_save >= CONFIG.killswitch.session_save_interval_sec:
                    self._save_session_state()
                    self._last_session_save = now
            except Exception:
                _log.exception("Unhandled error in session persistence loop - continuing.")
            _time.sleep(5.0)

    def _save_session_state(self) -> None:
        state = self.order_mgr.export_session_state()
        if not state:
            return
        path = CONFIG.killswitch.session_state_path
        try:
            import json
            path.write_text(json.dumps(state, default=str), encoding="utf-8")
            _log.debug("Session state saved (%d positions)", len(state.get("positions", [])))
        except Exception:
            _log.exception("Failed to save session state")

    def _load_session_state(self) -> None:
        path = CONFIG.killswitch.session_state_path
        if not path.exists():
            return
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            self.order_mgr.import_session_state(data)
            _log.info("Session state restored (%d positions)", len(data.get("positions", [])))
            for pos in self.order_mgr.get_open_positions():
                try:
                    self.kite.add_subscription(pos.contract.instrument_token)
                    self._active_option_token = pos.contract.instrument_token
                    _log.info("Resubscribed restored option token %s", self._active_option_token)
                except Exception:
                    _log.exception("Failed to resubscribe restored option token")
        except Exception:
            _log.exception("Failed to load session state")

    # ------------------------------------------------------------
    # MARGIN SAFETY
    # ------------------------------------------------------------

    def _margin_monitor_loop(self) -> None:
        while not _shutdown.is_set():
            try:
                self._check_margin_safety()
            except Exception:
                _log.exception("Unhandled error in margin monitor loop - continuing.")
            _time.sleep(CONFIG.margin_safety.margin_check_interval_sec)

    def _check_margin_safety(self) -> bool:
        available = self.margin_mgr.get_available_margin()
        equity = self.risk.equity if self.risk.equity > 0 else available
        min_abs = CONFIG.margin_safety.min_available_margin
        min_pct = CONFIG.margin_safety.min_available_margin_pct
        threshold = max(min_abs, equity * min_pct / 100.0) if min_pct > 0 else min_abs

        if available < threshold:
            if not self._margin_blocked:
                self._margin_blocked = True
                _log.warning("LOW_MARGIN_ENTRY_BLOCK | available=%.2f threshold=%.2f", available, threshold)
                log_structured(
                    "LOW_MARGIN_ENTRY_BLOCK",
                    symbol="-",
                    side="-",
                    position_quantity=0,
                    exit_reason="LOW_MARGIN",
                    strategy_exit_price=0.0,
                    limit_price=0.0,
                    buffer_pct=0.0,
                    limit_order_id="-",
                    filled_quantity=0,
                    remaining_quantity=0,
                    timeout_seconds=0,
                    fallback_used=False,
                    final_exit_status="BLOCKED",
                    available_margin=round(available, 2),
                    used_margin=round(equity - available, 2),
                    margin_threshold=round(threshold, 2),
                )
                send_telegram_alert("LOW_MARGIN_ENTRY_BLOCK", f"available={available:.2f} threshold={threshold:.2f}")
            return False

        if self._margin_blocked:
            self._margin_blocked = False
            _log.info("MARGIN_RECOVERED | available=%.2f threshold=%.2f", available, threshold)
            log_structured(
                "MARGIN_RECOVERED",
                symbol="-",
                side="-",
                position_quantity=0,
                exit_reason="MARGIN_RECOVERED",
                strategy_exit_price=0.0,
                limit_price=0.0,
                buffer_pct=0.0,
                limit_order_id="-",
                filled_quantity=0,
                remaining_quantity=0,
                timeout_seconds=0,
                fallback_used=False,
                final_exit_status="RECOVERED",
                available_margin=round(available, 2),
                used_margin=round(equity - available, 2),
                margin_threshold=round(threshold, 2),
            )
            send_telegram_alert("MARGIN_RECOVERED", f"available={available:.2f} threshold={threshold:.2f}")

        return True

    # ------------------------------------------------------------
    # DASHBOARD FEED
    # ------------------------------------------------------------

    def _update_dashboard_indicators(self, closed_5m) -> None:
        closes = [c.close for c in closed_5m]
        ema20 = ind.latest_ema(closes, CONFIG.indicators.ema_fast)
        ema50 = ind.latest_ema(closes, CONFIG.indicators.ema_slow)
        adx_val = ind.adx(closed_5m, CONFIG.indicators.adx_period)
        atr_val = ind.atr(closed_5m, CONFIG.indicators.atr_period)
        vwap = self.candles.get_current_vwap(self.spot_token, CONFIG.session.primary_timeframe_min)
        positions = self.order_mgr.get_open_positions()
        pos = positions[0] if positions else None

        dashboard.update_state(
            nifty_spot=closes[-1] if closes else None,
            trend=("UP" if ema20 and ema50 and ema20 > ema50 else "DOWN" if ema20 and ema50 else None),
            ema20=round(ema20, 2) if ema20 else None,
            ema50=round(ema50, 2) if ema50 else None,
            vwap=round(vwap, 2) if vwap else None,
            adx=round(adx_val, 2) if adx_val else None,
            atr=round(atr_val, 2) if atr_val else None,
            position=pos.contract.tradingsymbol if pos else None,
            entry_price=pos.entry_price if pos else None,
            atm_strike=pos.contract.strike if pos else None,
            option_premium=self._get_option_ltp(pos.contract.tradingsymbol) if pos else None,
            available_margin=round(self.margin_mgr.get_available_margin(), 2),
            trades_today=self.risk.stats.trades_taken,
            daily_pnl=round(self.risk.stats.realized_pnl, 2),
            win_rate=round(self.risk.win_rate(), 1),
        )

    # ------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------

    def _shutdown_sequence(self) -> None:
        _log.info("Shutting down: squaring off any open positions and closing connections.")
        try:
            self.order_mgr.force_square_off_all()
        except Exception:
            _log.exception("Error during shutdown square-off")
        try:
            self.kite.stop_ticker()
        except Exception:
            pass
        try:
            daily_report.save_daily_report()
            report = daily_report.build_daily_report()
            today_str = datetime.now().date().isoformat()
            summary = report.get(today_str)
            if summary:
                _log.info("Day complete: PnL=%.2f Trades=%d WinRate=%.1f%%",
                          summary["net_pnl"], summary["trades_count"], summary["win_rate_pct"])
            _log.info("Final daily report saved on shutdown")
        except Exception:
            _log.exception("Failed to save daily report on shutdown")
        try:
            self._save_session_state()
        except Exception:
            _log.exception("Failed to save session state on shutdown")
        log_decision("BOT_SHUTDOWN", trades_today=self.risk.stats.trades_taken,
                     daily_pnl=round(self.risk.stats.realized_pnl, 2))
        send_telegram_alert("BOT_SHUTDOWN", f"Trades={self.risk.stats.trades_taken} PnL={self.risk.stats.realized_pnl:.2f}")
        sys.exit(0)


if __name__ == "__main__":
    bot = TradingBot()
    bot.start()