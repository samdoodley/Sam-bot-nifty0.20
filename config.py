"""
config.py
=========
Single source of truth for every configurable parameter in the bot.

Nothing in strategy.py / risk_manager.py / order_manager.py etc. should
ever hardcode a threshold - it must be read from here. This is what
lets you tune the bot from paper -> backtest -> live without touching
logic code.

Usage:
    from config import CONFIG
    if adx > CONFIG.strategy.adx_min: ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import time


# ============================================================
# MODE
# ============================================================

class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


# ============================================================
# BROKER / API
# ============================================================

@dataclass(frozen=True)
class KiteConfig:
    api_key: str = os.getenv("KITE_API_KEY", "np7uml4ruym4cf6y")
    api_secret: str = os.getenv("KITE_API_SECRET", "58g9z2nrdembv0v5092dknhokeyfef90")
    tokens_path: Path = Path.home() / ".nifty_bot" / "tokens.json"
    redirect_url: str = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:8000")
    # TOTP-based auto login (mirrors your WickFill setup)
    totp_secret: str = os.getenv("KITE_TOTP_SECRET", "")
    zerodha_user_id: str = os.getenv("KITE_USER_ID", "")
    zerodha_password: str = os.getenv("KITE_PASSWORD", "")

    request_timeout_sec: float = 8.0
    max_retries: int = 4
    retry_backoff_base_sec: float = 1.5   # exponential: base * 2^attempt
    retry_backoff_max_sec: float = 20.0
    variety_regular: str = "regular"

    # WebSocket
    ws_reconnect_max_tries: int = 300      # effectively "forever, but capped"
    ws_reconnect_backoff_sec: float = 3.0


# ============================================================
# CAPITAL / ACCOUNT
# ============================================================

@dataclass(frozen=True)
class CapitalConfig:
    configured_capital: float = 3_00_000.0   # informational only; margin_manager
                                              # always re-reads live broker margin
    product_type: str = "MIS"               # intraday only
    max_exposure_pct_of_margin: float = 0.90  # never use more than 90% of
                                               # available margin on one trade
    min_margin_buffer_rupees: float = 2_000.0  # always keep this much untouched


# ============================================================
# INSTRUMENT / UNIVERSE
# ============================================================

@dataclass(frozen=True)
class InstrumentConfig:
    underlying_symbol: str = "NIFTY 50"
    underlying_tradingsymbol_prefix: str = "NIFTY"
    exchange: str = "NSE"
    option_exchange: str = "NFO"
    strike_step: int = 50                   # NIFTY strikes are in steps of 50
    expiry_mode: str = "NEAREST_WEEKLY"      # always nearest weekly expiry
    option_type_ce: str = "CE"
    option_type_pe: str = "PE"


# ============================================================
# TIMEFRAME / TRADING WINDOW
# ============================================================

@dataclass(frozen=True)
class SessionConfig:
    market_open: time = time(9, 15)
    warm_up_until: time = time(9, 15)        # no warm-up delay
    setup_scan_start: time = time(9, 15)     # start scanning immediately
    last_entry_time: time = time(15, 15)     # allow entries until 15:15
    force_square_off_time: time = time(15, 15)
    market_close: time = time(15, 30)

    primary_timeframe_min: int = 5
    higher_timeframe_min: int = 15

    # loop cadence
    scan_loop_interval_sec: float = 5.0
    position_monitor_interval_sec: float = 1.0


# ============================================================
# INDICATORS
# ============================================================

@dataclass(frozen=True)
class IndicatorConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    htf_ema_fast: int = 20
    htf_ema_slow: int = 50
    adx_period: int = 14
    atr_period: int = 14
    vwap_reset_daily: bool = True
    volume_lookback: int = 1                # compare current vs previous candle


# ============================================================
# STRATEGY THRESHOLDS
# ============================================================

@dataclass(frozen=True)
class StrategyConfig:
    adx_min: float = 12.0
    atr_min_points: float = 0.5            # minimum NIFTY-points ATR to trade;
                                                  # below this = too quiet, skip
    ema_flat_slope_lookback: int = 5        # bars used to judge "EMA is flat"
    ema_flat_slope_min_points: float = 1.5  # min EMA20 movement over lookback
                                                  # bars to NOT be considered flat

    pullback_max_distance_from_ema20_points: float = 35.0
    vwap_cross_lookback_bars: int = 10
    vwap_max_crosses_allowed: int = 10      # more than this in lookback = choppy, skip

    gap_max_pct: float = 3.0                # skip day if opening gap > this %

    # confirmation candle
    strong_candle_body_to_range_min_pct: float = 0.25  # body >=25% of candle range
                                                          # to count as "strong"

    max_trades_per_day: int = 10
    max_consecutive_losses: int = 5

    manual_event_disable: bool = False      # flip True on RBI policy / budget days etc.


# ============================================================
# ENTRY / STOP / TARGET (in NIFTY INDEX POINTS - converted to
# option premium terms by option_selector / order_manager using
# the option's own ATR-scaled delta-equivalent move where needed)
# ============================================================

@dataclass(frozen=True)
class TradeManagementConfig:
    target_index_points: float = 10.0
    stop_loss_index_points_cap: float = 8.0   # SL = min(swing_sl, this)

    breakeven_trigger_index_points: float = 6.0
    breakeven_buffer_points: float = 0.5       # move SL to entry + this (buy side)

    enable_trailing_after_breakeven: bool = True
    trail_step_index_points: float = 2.0       # every +2 pts beyond breakeven,
    trail_lock_index_points: float = 1.0       # trail SL up by 1 pt

    # Option premium fallback target/SL if you prefer trading premium directly
    # instead of translating index points (used only if use_premium_targets=True)
    use_premium_targets: bool = False
    premium_target_points: float = 15.0
    premium_stop_loss_points: float = 12.0

    sl_order_type: str = "SL"                  # stop-limit, not SL-M (learned from
                                                 # WickFill live-slippage experience)
    sl_limit_offset_points: float = 1.0         # limit price = trigger +/- this
    sl_escalation_watchdog_sec: float = 5.0     # if SL-limit unfilled in this
                                                 # window, fire market exit


# ============================================================
# POSITION SIZING / MARGIN
# ============================================================

@dataclass(frozen=True)
class PositionSizingConfig:
    risk_per_trade_pct_of_equity: float = 3.0   # % of account equity risked per trade
    max_lots_per_trade: int = 10                 # hard ceiling regardless of margin
    round_lots_down: bool = True                  # never round up quantity


# ============================================================
# RISK MANAGEMENT (account level)
# ============================================================

@dataclass(frozen=True)
class RiskConfig:
    daily_loss_limit_pct_of_equity: float = 1.0
    daily_profit_target_pct_of_equity: float = 2.0
    stop_after_daily_loss_limit: bool = False
    stop_after_daily_profit_target: bool = False


# ============================================================
# ORDER MANAGEMENT
# ============================================================

@dataclass(frozen=True)
class OrderConfig:
    entry_order_type: str = "MARKET"
    fill_confirmation_poll_interval_sec: float = 0.5
    fill_confirmation_timeout_sec: float = 10.0
    max_order_retries: int = 3
    reject_retry_backoff_sec: float = 2.0


# ============================================================
# LOGGING
# ============================================================

@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path = Path.home() / ".nifty_bot" / "logs"
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    log_rotate_max_bytes: int = 10_000_000
    log_rotate_backups: int = 10
    decision_log_filename: str = "decisions.jsonl"   # structured, one JSON per line


# ============================================================
# DASHBOARD
# ============================================================

@dataclass(frozen=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_interval_sec: float = 2.0


# ============================================================
# BACKTEST
# ============================================================

@dataclass(frozen=True)
class BacktestConfig:
    data_dir: Path = Path.home() / ".nifty_bot" / "backtest_data"
    results_dir: Path = Path.home() / ".nifty_bot" / "backtest_results"
    starting_equity: float = 300_000.0
    commission_per_lot: float = 40.0        # round-trip brokerage + charges estimate
    slippage_points: float = 0.5            # applied to entries/exits in backtest

    # Options-specific backtest
    backtest_months: int = 8                # number of months to backtest
    atm_delta_approx: float = 0.5           # delta for translating index points to premium
    option_brokerage_per_order: float = 20.0  # Zerodha: ₹20 or 0.1% whichever is lower
    option_gst_pct: float = 0.18            # 18% GST on brokerage
    option_stamp_duty_per_lakh: float = 0.01  # approx stamp duty


# ============================================================
# ROOT CONFIG
# ============================================================

@dataclass(frozen=True)
class RootConfig:
    mode: TradingMode = TradingMode(os.getenv("NIFTY_BOT_MODE", "PAPER"))

    kite: KiteConfig = field(default_factory=KiteConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trade_mgmt: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        # ensure runtime dirs exist regardless of mode
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        self.kite.tokens_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == TradingMode.BACKTEST:
            self.backtest.data_dir.mkdir(parents=True, exist_ok=True)
            self.backtest.results_dir.mkdir(parents=True, exist_ok=True)

        # hard safety check: LIVE mode requires explicit env confirmation,
        # so a stray env var or default can never silently flip you live.
        if self.mode == TradingMode.LIVE and os.getenv("NIFTY_BOT_CONFIRM_LIVE") != "YES":
            raise RuntimeError(
                "Refusing to start in LIVE mode: set env var "
                "NIFTY_BOT_CONFIRM_LIVE=YES to confirm you intend to trade "
                "with real capital. (mode=PAPER or BACKTEST need no confirmation.)"
            )


CONFIG = RootConfig()


if __name__ == "__main__":
    # quick sanity dump when run directly
    import json
    from dataclasses import asdict

    def _default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, time):
            return o.isoformat()
        return str(o)

    print(f"Mode: {CONFIG.mode.value}")
    print(json.dumps(asdict(CONFIG), indent=2, default=_default))