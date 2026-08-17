"""
risk_manager.py
=================
Tracks trade outcomes, enforces daily loss/profit limits and
max-trades-per-day caps, and exposes a simple can_take_new_trade()
gate that the main scan loop checks before every entry attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import CONFIG
from logger import get_logger
from utils import TradeSide

_log = get_logger("risk_manager")


@dataclass
class RiskStats:
    realized_pnl: float = 0.0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0


class RiskManager:
    def __init__(self, starting_equity: float = 0.0) -> None:
        self.equity: float = starting_equity
        self._starting_equity: float = starting_equity
        self.stats: RiskStats = RiskStats()
        self._daily_pnl: float = 0.0

    def can_take_new_trade(self) -> tuple[bool, str]:
        if CONFIG.strategy.max_trades_per_day > 0 and self.stats.trades_taken >= CONFIG.strategy.max_trades_per_day:
            return False, "MAX_TRADES_REACHED"

        if CONFIG.risk.stop_after_daily_loss_limit:
            loss_limit = self._starting_equity * CONFIG.risk.daily_loss_limit_pct_of_equity / 100.0
            if self._daily_pnl <= -loss_limit:
                return False, "DAILY_LOSS_LIMIT_HIT"

        if CONFIG.risk.stop_after_daily_profit_target:
            profit_target = self._starting_equity * CONFIG.risk.daily_profit_target_pct_of_equity / 100.0
            if self._daily_pnl >= profit_target:
                return False, "DAILY_PROFIT_TARGET_HIT"

        if CONFIG.strategy.max_consecutive_losses > 0 and self.stats.consecutive_losses >= CONFIG.strategy.max_consecutive_losses:
            return False, "MAX_CONSECUTIVE_LOSSES"

        return True, "OK"

    def record_trade_result(self, pnl: float) -> None:
        self.stats.trades_taken += 1
        self._daily_pnl += pnl
        self.stats.realized_pnl += pnl

        if pnl > 0:
            self.stats.wins += 1
            self.stats.consecutive_losses = 0
        elif pnl < 0:
            self.stats.losses += 1
            self.stats.consecutive_losses += 1

        _log.info(
            "Trade result recorded: pnl=%.2f daily_pnl=%.2f consecutive_losses=%d",
            pnl, self._daily_pnl, self.stats.consecutive_losses,
        )

    def win_rate(self) -> float:
        if self.stats.trades_taken == 0:
            return 0.0
        return self.stats.wins / self.stats.trades_taken * 100.0

    def reset_daily_counters(self) -> None:
        self._daily_pnl = 0.0
        self.stats.trades_taken = 0
        self.stats.wins = 0
        self.stats.losses = 0
        self.stats.consecutive_losses = 0