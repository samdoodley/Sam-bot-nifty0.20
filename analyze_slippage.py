#!/usr/bin/env python3
"""
Slippage Analysis: Paper vs Live Market Data
Fetches 1-min candles from Kite and compares with paper trades.
Run: python analyze_slippage.py
"""

import json
import sys
from datetime import datetime, time
from pathlib import Path

# Add project to path
sys.path.insert(0, "/home/samkumarg/Nifty intra")

from kite_api import KiteAPI
from option_selector import OptionUniverse

LOG_DIR = Path.home() / ".nifty_bot" / "logs"
TRADES_FILE = LOG_DIR / "trades.jsonl"
OUTPUT_FILE = LOG_DIR / f"slippage_report_{datetime.now().strftime('%Y%m%d')}.csv"


def load_paper_trades(date_str: str):
    """Load today's paper trades from trades.jsonl"""
    trades = []
    if not TRADES_FILE.exists():
        print(f"❌ Trades file not found: {TRADES_FILE}")
        return trades
    
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get('event') == 'EXIT' and 'TESTCE' not in t.get('symbol', ''):
                    entry_time = t.get('entry_time', '')
                    if entry_time.startswith(date_str):
                        trades.append(t)
            except json.JSONDecodeError:
                continue
    return trades


def get_instrument_token(kite: KiteAPI, symbol: str):
    """Get instrument token for a symbol"""
    try:
        instruments = kite.instruments("NFO")
        for inst in instruments:
            if inst.get("tradingsymbol") == symbol:
                return inst.get("instrument_token")
    except Exception as e:
        print(f"⚠️  Error fetching instruments: {e}")
    return None


def fetch_1min_candles(kite: KiteAPI, token: int, date_str: str):
    """Fetch 1-minute candles for a token on a specific date"""
    try:
        from_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=9, minute=15)
        to_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=15, minute=30)
        data = kite.historical_data(token, from_dt, to_dt, "minute")
        return data
    except Exception as e:
        print(f"⚠️  Error fetching candles for token {token}: {e}")
        return []


def find_candle_at_time(candles, target_time: str):
    """Find 1-min candle matching target time (HH:MM:SS)"""
    # Parse target time (ISO format, naive = IST from journal)
    target_dt = datetime.fromisoformat(target_time.replace('Z', ''))
    target_minute = target_dt.replace(second=0, microsecond=0)
    
    for c in candles:
        candle_dt = c['date']
        # Candle is timezone-aware (IST), convert to naive for comparison
        if hasattr(candle_dt, 'tzinfo') and candle_dt.tzinfo is not None:
            candle_minute = candle_dt.replace(tzinfo=None, second=0, microsecond=0)
        elif hasattr(candle_dt, 'replace'):
            candle_minute = candle_dt.replace(second=0, microsecond=0)
        else:
            candle_minute = datetime.fromisoformat(str(candle_dt)).replace(second=0, microsecond=0)
        
        if candle_minute == target_minute:
            return c
    return None


def calculate_slippage(trade, entry_candle, exit_candle):
    """Calculate slippage vs 1-min candle close prices"""
    side = trade.get('side')
    paper_entry = trade.get('entry_price', 0)
    paper_exit = trade.get('exit_price', 0)
    qty = trade.get('quantity', 0)
    
    results = {
        'symbol': trade.get('symbol'),
        'side': side,
        'paper_entry': paper_entry,
        'paper_exit': paper_exit,
        'qty': qty,
        'paper_pnl': trade.get('pnl', 0),
        'reason': trade.get('reason'),
        'entry_time': trade.get('entry_time'),
        'exit_time': trade.get('exit_time'),
    }
    
    if entry_candle:
        live_entry_ref = entry_candle['close']
        entry_slip = paper_entry - live_entry_ref if side == 'LONG' else live_entry_ref - paper_entry
        results['live_entry_ref'] = live_entry_ref
        results['entry_slippage_pts'] = round(entry_slip, 2)
        results['entry_slippage_rs'] = round(entry_slip * qty, 2)
    else:
        results['live_entry_ref'] = None
        results['entry_slippage_pts'] = None
        results['entry_slippage_rs'] = None
    
    if exit_candle:
        live_exit_ref = exit_candle['close']
        exit_slip = live_exit_ref - paper_exit if side == 'LONG' else paper_exit - live_exit_ref
        results['live_exit_ref'] = live_exit_ref
        results['exit_slippage_pts'] = round(exit_slip, 2)
        results['exit_slippage_rs'] = round(exit_slip * qty, 2)
    else:
        results['live_exit_ref'] = None
        results['exit_slippage_pts'] = None
        results['exit_slippage_rs'] = None
    
    # Total slippage impact
    if results['entry_slippage_rs'] is not None and results['exit_slippage_rs'] is not None:
        results['total_slippage_rs'] = round(results['entry_slippage_rs'] + results['exit_slippage_rs'], 2)
        results['live_pnl_est'] = round(trade.get('pnl', 0) - results['total_slippage_rs'], 2)
    else:
        results['total_slippage_rs'] = None
        results['live_pnl_est'] = None
    
    return results


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"📊 Slippage Analysis for {date_str}")
    print("=" * 80)
    
    # Load paper trades
    trades = load_paper_trades(date_str)
    if not trades:
        print(f"❌ No paper trades found for {date_str}")
        return
    
    print(f"✅ Found {len(trades)} paper trades")
    
    # Initialize Kite
    print("🔐 Logging into Kite...")
    kite = KiteAPI()
    try:
        kite.login()
        print("✅ Kite login successful")
    except Exception as e:
        print(f"❌ Kite login failed: {e}")
        return
    
    # Get unique symbols
    symbols = list(set(t.get('symbol') for t in trades))
    print(f"📈 Symbols traded: {symbols}")
    
    # Fetch tokens and candles
    symbol_data = {}
    for sym in symbols:
        print(f"🔍 Fetching token for {sym}...")
        token = get_instrument_token(kite, sym)
        if not token:
            print(f"⚠️  Could not find token for {sym}")
            continue
        
        print(f"📥 Fetching 1-min candles for {sym} (token: {token})...")
        candles = fetch_1min_candles(kite, token, date_str)
        if candles:
            symbol_data[sym] = {'token': token, 'candles': candles}
            print(f"✅ Got {len(candles)} candles for {sym}")
        else:
            print(f"⚠️  No candles for {sym}")
    
    # Analyze each trade
    print("\n" + "=" * 80)
    print("📋 TRADE-BY-TRADE SLIPPAGE ANALYSIS")
    print("=" * 80)
    
    results = []
    for trade in trades:
        sym = trade.get('symbol')
        if sym not in symbol_data:
            print(f"⚠️  No live data for {sym}, skipping")
            continue
        
        candles = symbol_data[sym]['candles']
        
        entry_candle = find_candle_at_time(candles, trade.get('entry_time', ''))
        exit_candle = find_candle_at_time(candles, trade.get('exit_time', ''))
        
        result = calculate_slippage(trade, entry_candle, exit_candle)
        results.append(result)
        
        # Print summary
        print(f"\n{result['symbol']} | {result['side']} | Qty: {result['qty']}")
        print(f"  Paper: Entry ₹{result['paper_entry']:.2f} → Exit ₹{result['paper_exit']:.2f} | PnL: ₹{result['paper_pnl']:,.2f} | {result['reason']}")
        if result['live_entry_ref']:
            print(f"  Live Entry Ref: ₹{result['live_entry_ref']:.2f} | Slippage: {result['entry_slippage_pts']:+.2f} pts (₹{result['entry_slippage_rs']:+,.2f})")
        if result['live_exit_ref']:
            print(f"  Live Exit Ref:  ₹{result['live_exit_ref']:.2f} | Slippage: {result['exit_slippage_pts']:+.2f} pts (₹{result['exit_slippage_rs']:+,.2f})")
        if result['total_slippage_rs']:
            print(f"  📉 TOTAL SLIPPAGE: ₹{result['total_slippage_rs']:+,.2f} | Est Live PnL: ₹{result['live_pnl_est']:+,.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DAILY SUMMARY")
    print("=" * 80)
    
    total_paper = sum(r['paper_pnl'] for r in results)
    total_slippage = sum(r['total_slippage_rs'] for r in results if r['total_slippage_rs'])
    total_live_est = sum(r['live_pnl_est'] for r in results if r['live_pnl_est'])
    trades_with_data = len([r for r in results if r['total_slippage_rs'] is not None])
    
    print(f"Trades Analyzed: {trades_with_data}/{len(trades)}")
    print(f"Paper Net PnL:     ₹{total_paper:+,.2f}")
    print(f"Total Slippage:    ₹{total_slippage:+,.2f}")
    print(f"Est Live Net PnL:  ₹{total_live_est:+,.2f}")
    print(f"Difference:        ₹{total_live_est - total_paper:+,.2f}")
    
    # Win/Loss flip analysis
    flipped = 0
    for r in results:
        if r['live_pnl_est'] is not None:
            paper_win = r['paper_pnl'] > 0
            live_win = r['live_pnl_est'] > 0
            if paper_win != live_win:
                flipped += 1
                print(f"  🔄 FLIPPED: {r['symbol']} | Paper: ₹{r['paper_pnl']:+,.2f} → Live: ₹{r['live_pnl_est']:+,.2f}")
    
    print(f"Trades Flipped (W↔L): {flipped}")
    
    # Save CSV
    import csv
    if results:
        fieldnames = list(results[0].keys())
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()