#!/usr/bin/env python3
"""
Setup script for changing the ticker symbol used by all trading strategies.
This script updates the config.txt file which is read by all strategy files.
"""

import os

def get_current_ticker():
    """Get the current ticker from config.txt"""
    if os.path.exists('config.txt'):
        try:
            with open('config.txt', 'r') as f:
                return f.read().strip().upper()
        except Exception:
            return None
    return None

def set_ticker(new_ticker):
    """Set a new ticker symbol in config.txt"""
    try:
        with open('config.txt', 'w') as f:
            f.write(new_ticker.upper())
        print(f"✅ Ticker successfully changed to: {new_ticker.upper()}")
        return True
    except Exception as e:
        print(f"❌ Error updating ticker: {e}")
        return False

def main():
    print("🔧 Trading Strategies Ticker Setup")
    print("=" * 40)
    
    current_ticker = get_current_ticker()
    if current_ticker:
        print(f"Current ticker: {current_ticker}")
    else:
        print("No ticker currently set")
    
    print("\nEnter a new ticker symbol (e.g., AAPL, TSLA, SPY, etc.)")
    print("Or press Enter to keep current ticker")
    
    new_ticker = input("New ticker: ").strip()
    
    if not new_ticker:
        print("No changes made.")
        return
    
    # Basic validation
    if not new_ticker.isalpha() or len(new_ticker) > 10:
        print("❌ Invalid ticker symbol. Please use only letters (max 10 characters).")
        return
    
    if set_ticker(new_ticker):
        print(f"\n📊 All trading strategies will now use ticker: {new_ticker.upper()}")
        print("\nYou can now run any strategy file:")
        print("  python strategies/sma-crossover.py")
        print("  python strategies/macd-strategy.py")
        print("  python strategies/mean-reversion-zscore.py")
        print("  python strategies/rsi-divergence.py")
        print("  python strategies/bollinger-bands.py")

if __name__ == "__main__":
    main() 