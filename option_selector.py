"""
option_selector.py
===================
Resolves "NIFTY spot is at X, I want to go long/short" into a concrete
tradeable option contract: nearest weekly expiry, ATM strike, correct
tradingsymbol + instrument_token + lot_size, sourced from Kite's NFO
instrument dump (never hardcoded).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from config import CONFIG
from kite_api import KiteAPI
from logger import get_logger
from utils import Candle, OptionContract, TradeSide, round_to_strike_step

_log = get_logger("option_selector")


class OptionUniverse:
    """Caches the NFO instrument dump for the day and answers ATM lookups."""

    def __init__(self, kite: KiteAPI) -> None:
        self.kite = kite
        self._instruments: list[dict] = []
        self._loaded_date: Optional[date] = None

    def refresh(self) -> None:
        today = date.today()
        if self._loaded_date == today and self._instruments:
            return
        self._instruments = self.kite.instruments(CONFIG.instrument.option_exchange)
        self._instruments = [
            i for i in self._instruments
            if i["name"] == CONFIG.instrument.underlying_tradingsymbol_prefix
            and i["segment"] == "NFO-OPT"
        ]
        self._loaded_date = today
        _log.info("Loaded %d NIFTY option instruments for %s", len(self._instruments), today)

    def nearest_weekly_expiry(self) -> date:
        self.refresh()
        today = date.today()
        expiries = sorted({i["expiry"] for i in self._instruments if i["expiry"] >= today})
        if not expiries:
            raise RuntimeError("No upcoming NIFTY weekly expiries found in instrument dump.")
        return expiries[0]

    def resolve_atm_contract(self, spot_price: float, side: TradeSide) -> OptionContract:
        """side=LONG -> buy CE (bullish), side=SHORT -> buy PE (bearish). We only ever BUY options."""
        self.refresh()
        expiry = self.nearest_weekly_expiry()
        strike = round_to_strike_step(spot_price, CONFIG.instrument.strike_step)
        option_type = CONFIG.instrument.option_type_ce if side == TradeSide.LONG else CONFIG.instrument.option_type_pe

        candidates = [
            i for i in self._instruments
            if i["expiry"] == expiry and i["strike"] == strike and i["instrument_type"] == option_type
        ]
        if not candidates:
            raise RuntimeError(
                f"No contract found for strike={strike} type={option_type} expiry={expiry}. "
                f"Spot may be far from any listed strike, or instrument dump is stale."
            )
        inst = candidates[0]
        contract = OptionContract(
            tradingsymbol=inst["tradingsymbol"],
            strike=int(inst["strike"]),
            option_type=option_type,
            expiry=str(expiry),
            instrument_token=int(inst["instrument_token"]),
            lot_size=int(inst["lot_size"]),
        )
        _log.info("Resolved ATM contract: %s (strike=%s, lot=%s)", contract.tradingsymbol,
                   contract.strike, contract.lot_size)
        return contract

    def get_spot_instrument_token(self) -> int:
        """NIFTY 50 index token lives on NSE, not NFO - fetched separately."""
        idx_instruments = self.kite.instruments(CONFIG.instrument.exchange)
        for inst in idx_instruments:
            if inst["tradingsymbol"] == CONFIG.instrument.underlying_symbol and inst["segment"] == "INDICES":
                return int(inst["instrument_token"])
        raise RuntimeError(f"Could not find instrument token for {CONFIG.instrument.underlying_symbol}")