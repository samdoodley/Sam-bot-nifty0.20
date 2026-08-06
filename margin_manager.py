"""
margin_manager.py
==================
Never uses a hardcoded capital figure for sizing. Always reads live
available margin from kite.margins() (in LIVE/PAPER-with-real-quotes
mode) and sizes the position so it:
  1. Risks no more than risk_per_trade_pct_of_equity of account equity
     given the stop-loss distance.
  2. Never exceeds max_exposure_pct_of_margin of available margin.
  3. Never exceeds max_lots_per_trade.
  4. Is always rounded DOWN to a whole number of lots.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG
from kite_api import KiteAPI
from logger import get_logger, log_decision
from utils import round_down_to_lot

_log = get_logger("margin_manager")


@dataclass
class SizingResult:
    quantity: int
    lots: int
    risked_rupees: float
    margin_required_estimate: float
    available_margin: float
    rejected_reason: str = ""


class MarginManager:
    def __init__(self, kite: KiteAPI, paper_equity_override: float | None = None) -> None:
        self.kite = kite
        # In PAPER/BACKTEST mode we still want realistic sizing, so we allow
        # a simulated equity figure. In LIVE mode this is ignored entirely -
        # live broker margin is authoritative.
        self._paper_equity_override = paper_equity_override

    def get_available_margin(self) -> float:
        if CONFIG.mode.value == "LIVE" or self._paper_equity_override is None:
            data = self.kite.margins()
            equity_segment = data.get("equity", {})
            available = equity_segment.get("available", {}).get("live_balance", 0.0)
            return float(available)
        return self._paper_equity_override

    def update_paper_equity(self, new_equity: float) -> None:
        self._paper_equity_override = new_equity

    def size_position(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        option_premium: float,
        lot_size: int,
        margin_per_lot_estimate: float,
    ) -> SizingResult:
        available_margin = self.get_available_margin()

        sl_distance_index_points = abs(entry_price - stop_loss_price)
        if sl_distance_index_points <= 0:
            return SizingResult(0, 0, 0.0, 0.0, available_margin, rejected_reason="ZERO_SL_DISTANCE")

        risk_rupees_allowed = equity * CONFIG.sizing.risk_per_trade_pct_of_equity / 100.0

        # We're buying options (never selling), so max loss per lot is
        # approximately (option premium - premium at SL) * lot_size. We
        # estimate premium-move-per-index-point via a simple ratio here;
        # order_manager will still enforce the hard SL regardless.
        premium_risk_per_lot = max(option_premium * 0.01, 0.05) * sl_distance_index_points * lot_size / \
            max(sl_distance_index_points, 1)
        # Fallback conservative estimate: assume premium risk roughly tracks
        # index-point risk 1:1 scaled by lot size if delta unknown.
        premium_risk_per_lot = max(premium_risk_per_lot, sl_distance_index_points * lot_size * 0.4)

        max_lots_by_risk = int(risk_rupees_allowed // premium_risk_per_lot) if premium_risk_per_lot > 0 else 0

        usable_margin = max(available_margin - CONFIG.capital.min_margin_buffer_rupees, 0.0) * \
            CONFIG.capital.max_exposure_pct_of_margin
        max_lots_by_margin = int(usable_margin // margin_per_lot_estimate) if margin_per_lot_estimate > 0 else 0

        max_lots = min(max_lots_by_risk, max_lots_by_margin, CONFIG.sizing.max_lots_per_trade)

        if max_lots <= 0:
            reason = "INSUFFICIENT_MARGIN" if max_lots_by_margin <= 0 else "RISK_LIMIT_TOO_TIGHT"
            log_decision("SIZING_REJECTED", reason=reason, available_margin=available_margin,
                         risk_rupees_allowed=round(risk_rupees_allowed, 2),
                         max_lots_by_risk=max_lots_by_risk, max_lots_by_margin=max_lots_by_margin)
            return SizingResult(0, 0, 0.0, 0.0, available_margin, rejected_reason=reason)

        quantity = max_lots * lot_size
        if CONFIG.sizing.round_lots_down:
            quantity = round_down_to_lot(quantity, lot_size)

        result = SizingResult(
            quantity=quantity,
            lots=max_lots,
            risked_rupees=premium_risk_per_lot * max_lots,
            margin_required_estimate=margin_per_lot_estimate * max_lots,
            available_margin=available_margin,
        )
        log_decision("SIZING_COMPUTED", quantity=quantity, lots=max_lots,
                     risked_rupees=round(result.risked_rupees, 2),
                     margin_used_estimate=round(result.margin_required_estimate, 2),
                     available_margin=round(available_margin, 2))
        return result